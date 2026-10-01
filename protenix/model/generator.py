# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Optional

import torch

from protenix.model.utils import centre_random_augmentation, expand_at_dim
from protenix.tfg import parse_tfg_config, TFGEngine
from protenix.utils.logger import get_logger

logger = get_logger(__name__)

# How many atoms per epitope residue enter the contact penalty, chosen by
# smallest softmin distance to the antibody. Fixed at 1 -- only the single
# closest atom of each epitope residue is pulled towards the antibody. This is
# not configurable: it is the setting the released model was calibrated and
# benchmarked with, and it interacts with `epitope.guidance_alpha`.
CONTACT_TOP_K = 1


class TrainingNoiseSampler:
    """
    Sample the noise-level of training samples.

    Args:
        p_mean (float, optional): gaussian mean. Defaults to -1.2.
        p_std (float, optional): gaussian std. Defaults to 1.5.
        sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        return noise_level


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps).

    Args:
        s_max (float, optional): maximal noise level. Defaults to 160.0.
        s_min (float, optional): minimal noise level. Defaults to 4e-4.
        rho (float, optional): the exponent numerical part. Defaults to 7.
        sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    enable_efficient_fusion: bool = False,
    guidance_configs: Optional[dict[str, Any]] = None,
    epitope_configs: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        pair_z (torch.Tensor): pair feature embedding from InputFeatureEmbedder
            [..., N_tokens, N_tokens, c_z_inputs]
        p_lm (torch.Tensor): MSA embedding
            [..., N_tokens, c_p_lm]
        c_l (torch.Tensor): ligand embedding
            [..., N_tokens, c_c_l]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.
        enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.
        guidance_configs (Optional[dict[str, Any]]): training free guidance configs. Defaults to None.
        epitope_configs (Optional[dict[str, Any]]): SteerABLE reward-guided embedding-steering configs. When provided,
            trunk embeddings (s_τ, z_τ) are updated each step via the gradient of a hotspot-contact reward evaluated
            on the denoiser's raw x̂0 output. Expected keys: "mask_pairs" (from build_guidance_masks), plus optional
            "guidance_alpha" / "guidance_interval" / "alpha_init" / "alpha_trunc" / "d0" / "softmin_beta" /
            "lambda_clash" / "clash_tau" / "clash_start_step". "guidance_interval" (int >= 1, default 8)
            sets how OFTEN the reward + autograd + embedding update runs: steps 0, k, 2k, ... are steered and the
            rest are plain no-grad sampler steps on the embeddings as they stand. alpha is still evaluated at the
            raw step index, so the cosine schedule's truncation point and "clash_start_step" keep their step-space
            meaning; only the density of updates changes, and the total steering applied scales as ~1/k.
            Composes with TFG (`guidance_configs.enable=True`) — when both are on,
            SteerABLE pulls the grad-attached x̂0 out of tfg.step(return_x0=True) so the two mechanisms share one denoise
            call. Defaults to None (unguided).

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype
    tfg_cfg = parse_tfg_config(guidance_configs)
    if tfg_cfg.enable:
        logger.info("Guidance is enabled.")
        tfg = TFGEngine(tfg_cfg, device=device, dtype=dtype)

    # ---- SteerABLE epitope-reward-guided embedding steering ----
    epitope_on = epitope_configs is not None and bool(epitope_configs)
    if epitope_on:
        from protenix.model.steering import (
            contact_epitope_reward,
            epitope_clash_reward,
            rms_normalize,
        )
        g_mask_pairs = epitope_configs["mask_pairs"]
        _raw_alpha = epitope_configs.get("guidance_alpha", None)
        g_total_steps = len(noise_schedule) - 1
        g_alpha_const = float(_raw_alpha) if _raw_alpha is not None else None
        g_alpha_schedule_fn = None
        if g_alpha_const is None:
            from protenix.utils.alpha_schedule import make_cosine_trunc

            g_alpha_init = float(epitope_configs.get("alpha_init", 1.0))
            g_alpha_trunc = float(epitope_configs.get("alpha_trunc", 0.5))
            g_alpha_schedule_fn = make_cosine_trunc(
                omega_init=g_alpha_init, tau_trunc=g_alpha_trunc
            )
        g_d0 = float(epitope_configs.get("d0", 4.0))
        g_beta = float(epitope_configs.get("softmin_beta", 10.0))
        g_lambda_clash = float(epitope_configs.get("lambda_clash", 0.0))
        g_clash_tau = float(epitope_configs.get("clash_tau", 1.5))
        g_clash_start_step = int(epitope_configs.get("clash_start_step", 0))
        g_clash_on = g_lambda_clash > 0.0
        # Optional per-step, per-sample reward trace. The caller passes a list and
        # we append to it in place; it is dumped alongside the predictions so the
        # steering of each sample can be inspected (and compared across chunk
        # sizes) instead of only surviving as a log line every 50 steps.
        g_trace = epitope_configs.get("trace", None)
        # Reward/backprop cadence. 1 (default) = every step, which is the
        # pre-interval behaviour bit-for-bit: `step_i % 1 == 0` is True for
        # every int, so the gate below collapses to `cur_alpha != 0.0`.
        # k > 1 steers steps 0, k, 2k, ... and lets the rest fall through to
        # the frozen-embedding branch (one no-grad denoise, no backward).
        _raw_interval = epitope_configs.get("guidance_interval", 8)
        try:
            g_interval = int(_raw_interval)
        except (TypeError, ValueError):
            raise ValueError(
                "epitope.guidance_interval must be an integer >= 1, got "
                f"{_raw_interval!r}."
            )
        if g_interval < 1:
            # Raise, never clamp. 0 would raise ZeroDivisionError ~200 steps
            # into a GPU job, and -3 would run as exactly 3 with no error at
            # all (in Python, a % -b == 0 iff b divides a) -- a benchmark row
            # labelled with a value the run never used.
            raise ValueError(
                "epitope.guidance_interval must be >= 1 (1 = steer every step); "
                f"got {g_interval}. Use --gating.mode raw to disable steering."
            )
        # The steps that will actually apply the reward gradient. Evaluated with
        # the SAME predicate as the loop gate below, so the two cannot drift.
        # ceil(N/k) would be wrong: the cadence and the alpha schedule are
        # independent gates, and with `--epitope.guidance_alpha null` the cosine
        # truncates alpha to 0 from step tau_trunc*(N-1) onward -- at
        # alpha_trunc=0.5 that already halves the count at interval=1. Reporting
        # ceil(N/k) would overstate the steering density the operator is
        # calibrating against, on the default path too. N is a few hundred, so
        # materialising the list is free.
        g_steer_steps = [
            _i
            for _i in range(g_total_steps)
            if _i % g_interval == 0
            and (
                g_alpha_const
                if g_alpha_const is not None
                else g_alpha_schedule_fn(_i, g_total_steps)
            )
            != 0.0
        ]
        g_n_steer = len(g_steer_steps)
        logger.info(
            "Epitope embedding-guidance ON: d0=%.2f softmin_beta=%.2f "
            "λ_clash=%.3f alpha_const=%s (schedule=%s) tfg_compose=%s "
            "interval=%d (%d/%d steps steer)",
            g_d0, g_beta, g_lambda_clash,
            g_alpha_const, g_alpha_schedule_fn is not None, tfg_cfg.enable,
            g_interval, g_n_steer, g_total_steps,
        )
        if g_interval > 1:
            logger.warning(
                "epitope.guidance_interval=%d: only %d of %d steps apply the "
                "reward gradient. The per-update step size (guidance_alpha) is "
                "unchanged, so the TOTAL steering over the trajectory is ~1/%d "
                "of an interval=1 run. This is a guidance change, not only a "
                "speedup -- validate on DockQ before trusting the output.",
                g_interval, g_n_steer, g_total_steps, g_interval,
            )
        if g_clash_on:
            # The clash term is gated by BOTH `epitope_active` and
            # `step_i >= clash_start_step`, which cut along different axes. Their
            # intersection can be empty -- e.g. clash_start_step=190 with
            # interval=20 (last steered step 180), or a clash_start_step past the
            # last pre-truncation steered step. The run would then be identical
            # to lambda_clash=0.0 while still logging "λ_clash=0.100", so say so.
            g_n_clash = sum(1 for _i in g_steer_steps if _i >= g_clash_start_step)
            if g_n_clash == 0:
                logger.warning(
                    "epitope.lambda_clash=%.3f is set but the clash term will "
                    "NEVER be applied: no steered step is >= clash_start_step=%d "
                    "(steered steps run %s, interval=%d). This run is equivalent "
                    "to lambda_clash=0.0 -- lower clash_start_step or interval.",
                    g_lambda_clash, g_clash_start_step,
                    "none" if not g_steer_steps
                    else f"{g_steer_steps[0]}..{g_steer_steps[-1]}",
                    g_interval,
                )
            elif g_interval > 1:
                logger.info(
                    "  clash term applies on %d of %d steered steps.",
                    g_n_clash, g_n_steer,
                )
    else:
        g_trace = None

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe, sample_offset=0):
        # init noise
        # [..., N_sample, N_atom, 3]
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # NOTE: set seed in distributed training

        # ---- steerable local copies of trunk embeddings ----
        # When epitope guidance is on, s_tau/z_tau are updated each step by the
        # RMS-normalized reward gradient. Trunk-derived caches (pair_z/p_lm/c_l)
        # are invalidated because they depend on z_trunk which is now moving.
        if epitope_on:
            # Per-sample steering. Every sample in the chunk gets its OWN
            # (s_tau, z_tau) trajectory, carried on a dedicated leading axis.
            #
            # This is the whole point of the change: without that axis a chunk
            # shares one steered embedding driven by the chunk-AVERAGED reward,
            # so samples that want opposite corrections cancel each other and the
            # steering signal dies. That is why production pinned
            # sample_diffusion_chunk_size to 1 -- it bought per-sample steering by
            # giving up the batch dimension. With the axis present, backprop of
            # the SUMMED reward gives d(sum_i r_i)/d s_tau[j] = d r_j / d s_tau[j],
            # so the samples stay separated and the batch dimension comes back.
            if z_trunk is None:
                raise ValueError(
                    "Epitope steering needs the raw z_trunk to steer, but it was "
                    "None -- the caller passed a precomputed pair_z cache. The "
                    "steered branch must skip that cache (protenix.py: the "
                    "`and not steer` guard on enable_diffusion_shared_vars_cache)."
                )
            s_tau = expand_at_dim(
                s_trunk.detach(), dim=-3, n=chunk_n_sample
            ).contiguous()  # [..., N_sample, N_token, c_s]
            z_tau = expand_at_dim(
                z_trunk.detach(), dim=-4, n=chunk_n_sample
            ).contiguous()  # [..., N_sample, N_token, N_token, c_z]
            # Trunk-derived caches depend on z_trunk, which is now moving AND
            # per-sample, so they must not be reused.
            cur_pair_z = None
            cur_p_lm = None
            cur_c_l = None
        else:
            s_tau = s_trunk
            z_tau = z_trunk
            cur_pair_z = pair_z
            cur_p_lm = p_lm
            cur_c_l = c_l

        # Logging latch only: prints the guidance->frozen transition message
        # exactly once. It does NOT control freezing — that is decided per step
        # by `epitope_active` (cur_alpha != 0) below.
        truncation_logged = False

        for step_i, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # [..., N_sample, N_atom, 3]
            x_l = (
                centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                .squeeze(dim=-3)
                .to(dtype)
            )

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            # SteerABLE: decide this step's steering weight *before* the forward
            # pass. Once the cosine schedule truncates (cur_alpha == 0) there is
            # nothing to gain from the grad-enabled forward + autograd — the old
            # code still ran the full forward and merely scaled the update by 0.
            # Instead we fall through to the plain (frozen-embedding) run below.
            if epitope_on:
                cur_alpha = (
                    g_alpha_const
                    if g_alpha_const is not None
                    else g_alpha_schedule_fn(step_i, g_total_steps)
                )
                # Two gates, ANDed. (1) `cur_alpha != 0.0` is untouched -- the
                # cosine schedule may have truncated alpha, in which case there
                # is nothing to apply. (2) the new cadence gate. alpha is still
                # sampled at the RAW step_i (never at a guidance-event counter),
                # so the schedule's truncation point and `clash_start_step` keep
                # their step-space meaning; the interval only thins the steered
                # set to {0, k, 2k, ...}. Phase 0 (not k-1) means step 0 always
                # steers: the cosine peak is at s=0, and because s_tau/z_tau
                # persist, an early update conditions every remaining denoise.
                # g_interval == 1 makes the second conjunct a constant True, so
                # this is the same value AND the same bool type as before.
                epitope_active = cur_alpha != 0.0 and step_i % g_interval == 0
            else:
                epitope_active = False

            if epitope_active:
                # ==== SteerABLE epitope-reward-guided branch (composes with TFG) ====
                s_tau = s_tau.detach().requires_grad_(True)
                z_tau = z_tau.detach().requires_grad_(True)

                with torch.enable_grad():
                    if tfg_cfg.enable:
                        # TFG shares its Stage-2 denoise with us via return_x0.
                        # x0_pred has autograd graph → reward → grad wrt s_tau/z_tau.
                        # x_next is the fully TFG-processed x_{t-1} (PDM + refinement + Euler).
                        x_next, x0_pred = tfg.step(
                            denoise_net,
                            x=x_noisy,
                            t_hat=t_hat,
                            input_feature_dict=input_feature_dict,
                            s_inputs=s_inputs,
                            s_trunk=s_tau,
                            z_trunk=z_tau,
                            pair_z=cur_pair_z,
                            p_lm=cur_p_lm,
                            c_l=cur_c_l,
                            chunk_size=attn_chunk_size,
                            inplace_safe=False,
                            enable_efficient_fusion=False,
                            c_tau=c_tau,
                            step_i=step_i,
                            num_diffusion_steps=len(noise_schedule) - 1,
                            step_scale_eta=step_scale_eta,
                            return_x0=True,
                        )
                    else:
                        # Epitope-only branch: run the grad-enabled denoise directly
                        # and do the plain AF3 Euler step (no TFG projection).
                        x0_pred = denoise_net(
                            x_noisy=x_noisy,
                            t_hat_noise_level=t_hat,
                            input_feature_dict=input_feature_dict,
                            s_inputs=s_inputs,
                            s_trunk=s_tau,
                            z_trunk=z_tau,
                            pair_z=cur_pair_z,
                            p_lm=cur_p_lm,
                            c_l=cur_c_l,
                            chunk_size=attn_chunk_size,
                            inplace_safe=False,
                            enable_efficient_fusion=False,
                        )
                        delta = (x_noisy - x0_pred) / t_hat[..., None, None]
                        dt = c_tau - t_hat
                        x_next = x_noisy + step_scale_eta * dt[..., None, None] * delta

                    # Reward evaluated on grad-attached x̂0 (not on x_next which
                    # is Euler-stepped noisy waypoint). Backprop to s_tau/z_tau.
                    # Per-sample rewards: shape [N_sample], NOT a scalar. Keeping
                    # the sample axis is what stops one sample's reward from
                    # reaching another sample's embedding.
                    r_contact = contact_epitope_reward(
                        coords=x0_pred,
                        mask_pairs=g_mask_pairs,
                        d0=g_d0,
                        softmin_beta=g_beta,
                        top_k=CONTACT_TOP_K,
                        per_sample=True,
                    )
                    if g_clash_on and step_i >= g_clash_start_step:
                        r_clash = epitope_clash_reward(
                            coords=x0_pred,
                            mask_pairs=g_mask_pairs,
                            ref_element=input_feature_dict["ref_element"],
                            tau=g_clash_tau,
                            per_sample=True,
                        )
                        reward = r_contact + g_lambda_clash * r_clash
                    else:
                        r_clash = None
                        reward = r_contact

                    # Differentiating the SUM separates the samples exactly:
                    #   d(sum_i r_i)/d s_tau[j] = d r_j / d s_tau[j]
                    # because r_i depends only on s_tau[i]. No cross terms exist,
                    # so this is the serial (chunk_size=1) update, batched.
                    grads = torch.autograd.grad(
                        reward.sum(),
                        [s_tau, z_tau],
                        create_graph=False,
                        allow_unused=True,
                    )
                    g_s = grads[0] if grads[0] is not None else torch.zeros_like(s_tau)
                    g_z = grads[1] if grads[1] is not None else torch.zeros_like(z_tau)

                # Embedding update — RMS-normalised α-step (cur_alpha from above).
                # The RMS is taken PER SAMPLE; a global RMS would re-couple the
                # samples through the denominator and make a sample with a larger
                # gradient take a larger step than it does when run alone.
                s_tau = (
                    s_tau.detach() + cur_alpha * rms_normalize(g_s, sample_dim=-3)
                ).detach()
                z_tau = (
                    z_tau.detach() + cur_alpha * rms_normalize(g_z, sample_dim=-4)
                ).detach()
                x_l = x_next.detach()

                if g_trace is not None:
                    g_trace.append(
                        {
                            "step": step_i,
                            # index of this chunk's first sample in the full
                            # N_sample stack, so traces from different chunk
                            # sizes line up sample-for-sample
                            "sample_offset": sample_offset,
                            "alpha": cur_alpha,
                            "reward": reward.detach().flatten().tolist(),
                            "contact": r_contact.detach().flatten().tolist(),
                            "clash": (
                                None if r_clash is None
                                else r_clash.detach().flatten().tolist()
                            ),
                        }
                    )

                # First steered step of each 50-step window. At g_interval == 1
                # `step_i % 50 < 1` is `step_i % 50 == 0` over non-negative
                # ints, i.e. steps 0/50/100/150 exactly as before. Keeping the
                # old test would collapse the reward trace to multiples of
                # lcm(50, k): k=3 logs twice in 200 steps, k=4 twice, k=7 once --
                # unusable for comparing the very runs this knob exists for.
                if step_i % 50 < g_interval:
                    logger.info(
                        "  step %d: reward=%.4f%s alpha=%.6f |g_s|=%.4f |g_z|=%.4f "
                        "per_sample=%s",
                        step_i, float(reward.mean().item()),
                        "" if r_clash is None
                        else " (contact=%.4f clash=%.4f)"
                        % (r_contact.mean().item(), r_clash.mean().item()),
                        cur_alpha, g_s.norm().item(), g_z.norm().item(),
                        "[" + ", ".join(f"{v:.4f}" for v in reward.detach().flatten().tolist()) + "]",
                    )

            elif epitope_on:
                # ==== SteerABLE step with NO reward / autograd / update ====
                # Two disjoint reasons land here; the work is identical either
                # way, but they mean different things and must not be reported
                # as each other:
                #   (1) TRUNCATED    -- the cosine schedule hit tau_trunc, so
                #       alpha is 0 for this and every remaining step. Permanent.
                #   (2) OFF-CADENCE  -- guidance_interval = k > 1 and step_i is
                #       not a multiple of k. Temporary: more steered steps
                #       follow at the next multiple of k.
                # Either way: skip the grad-enabled forward + reward + autograd
                # and run the plain sampler on the current (already-steered)
                # embeddings. That is the runtime win -- one no-grad forward
                # instead of forward + reward + backprop.
                # NOTE: s_tau/z_tau are deliberately NOT reset to
                # s_trunk/z_trunk. The accumulated steering offset persists
                # across skipped steps, which is what makes the interval a
                # cadence knob rather than an alpha knob.
                s_tau = s_tau.detach()
                z_tau = z_tau.detach()
                if cur_alpha == 0.0 and not truncation_logged:
                    # One-time notice, and ONLY for a real truncation. An
                    # off-cadence step is not a truncation, so firing this here
                    # would print "truncated at step 1 (alpha=0)" on every
                    # interval>1 run (alpha is 0.1, nothing is truncated) and
                    # would then permanently suppress the real message on a
                    # cosine-schedule run. The cadence is reported once at
                    # setup instead. The freeze itself was decided above by
                    # epitope_active, not by this flag.
                    logger.info(
                        "Epitope guidance truncated at step %d (alpha=0); running "
                        "remaining steps on frozen steered embeddings (no autograd).",
                        step_i,
                    )
                    truncation_logged = True

                if tfg_cfg.enable:
                    # Hand the frozen s_tau/z_tau to TFG's own (unchanged) update.
                    x_l = tfg.step(
                        denoise_net,
                        x=x_noisy,
                        t_hat=t_hat,
                        input_feature_dict=input_feature_dict,
                        s_inputs=s_inputs,
                        s_trunk=s_tau,
                        z_trunk=z_tau,
                        pair_z=cur_pair_z,
                        p_lm=cur_p_lm,
                        c_l=cur_c_l,
                        chunk_size=attn_chunk_size,
                        inplace_safe=inplace_safe,
                        enable_efficient_fusion=enable_efficient_fusion,
                        c_tau=c_tau,
                        step_i=step_i,
                        num_diffusion_steps=len(noise_schedule) - 1,
                        step_scale_eta=step_scale_eta,
                    )
                    # Belt-and-braces detach. Grad is enabled process-wide
                    # while guidance is configured, and this is the only denoise
                    # site in the guided sampler whose result is neither wrapped
                    # in no_grad (unlike the else-branch below) nor detached
                    # (unlike branch (a)'s `x_l = x_next.detach()`). It is
                    # graph-free today only because tfg.step runs Stage 2 under
                    # no_grad when return_x0=False, detaches x0 before Stages
                    # 3-5, and detaches xt_shift -- accidents that hold at
                    # rho=0.0 and with analytic potentials. With an interval
                    # this branch becomes the majority path and, for any k that
                    # does not divide N_step-1, produces the FINAL x_l, where a
                    # stray graph makes the dumper fail on
                    # `pred_coordinate.cpu().numpy()`. Value-identical, so this
                    # cannot perturb the default path.
                    x_l = x_l.detach()
                else:
                    # Plain AF3 Euler step on the frozen steered embeddings.
                    # no_grad is what makes this the runtime win the comment
                    # above promises: grad is enabled process-wide while epitope
                    # guidance is configured, so without it the denoiser still
                    # builds a graph through its parameters — which also leaves
                    # x_l requiring grad and makes the dumper fail on
                    # `pred_coordinate.cpu().numpy()`.
                    with torch.no_grad():
                        x_denoised = denoise_net(
                            x_noisy=x_noisy,
                            t_hat_noise_level=t_hat,
                            input_feature_dict=input_feature_dict,
                            s_inputs=s_inputs,
                            s_trunk=s_tau,
                            z_trunk=z_tau,
                            pair_z=cur_pair_z,
                            p_lm=cur_p_lm,
                            c_l=cur_c_l,
                            chunk_size=attn_chunk_size,
                            inplace_safe=inplace_safe,
                            enable_efficient_fusion=enable_efficient_fusion,
                        )
                        delta = (x_noisy - x_denoised) / t_hat[..., None, None]
                        dt = c_tau - t_hat
                        x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

            elif tfg_cfg.enable:
                # ==== TFG-only branch (unchanged from upstream v2) ====
                x_l = tfg.step(
                    denoise_net,
                    x=x_noisy,
                    t_hat=t_hat,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    pair_z=pair_z,
                    p_lm=p_lm,
                    c_l=c_l,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    enable_efficient_fusion=enable_efficient_fusion,
                    c_tau=c_tau,
                    step_i=step_i,
                    num_diffusion_steps=len(noise_schedule) - 1,
                    step_scale_eta=step_scale_eta,
                )
            else:
                # ==== Baseline branch (unchanged from upstream v2) ====
                x_denoised = denoise_net(
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s_trunk,
                    z_trunk=z_trunk,
                    pair_z=pair_z,
                    p_lm=p_lm,
                    c_l=c_l,
                    chunk_size=attn_chunk_size,
                    inplace_safe=inplace_safe,
                    enable_efficient_fusion=enable_efficient_fusion,
                )

                delta = (x_noisy - x_denoised) / t_hat[
                    ..., None, None
                ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
                dt = c_tau - t_hat
                x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l = _chunk_sample_diffusion(
                chunk_n_sample,
                inplace_safe=inplace_safe,
                sample_offset=i * diffusion_chunk_size,
            )
            x_l.append(chunk_x_l)
        x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
    return x_l


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
    use_conditioning: bool = True,
    enable_efficient_fusion: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implements diffusion training as described in AF3 Appendix at page 23.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        noise_sampler (TrainingNoiseSampler): sampler for training noise-level.
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict[str, Any]) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        pair_z (torch.Tensor): pair feature embedding from InputFeatureEmbedder
            [..., N_tokens, N_tokens, c_z_inputs]
        p_lm (torch.Tensor): MSA embedding
            [..., N_tokens, c_p_lm]
        c_l (torch.Tensor): ligand embedding
            [..., N_tokens, c_c_l]
        N_sample (int): number of training samples
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        use_conditioning (bool): Whether to use conditioning. Defaults to True.
        enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x_gt_augment: the augmented ground-truth coordinates [..., N_sample, N_atom, 3]
            x_denoised: the denoised coordinates [..., N_sample, N_atom, 3]
            sigma: the sampled noise-level [..., N_sample]
    """
    batch_size_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    # Areate N_sample versions of the input structure by randomly rotating and translating
    x_gt_augment = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
    ).to(
        dtype
    )  # [..., N_sample, N_atom, 3]

    # Add independent noise to each structure
    # sigma: independent noise-level [..., N_sample]
    sigma = noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype)
    # noise: [..., N_sample, N_atom, 3]
    noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]

    # Get denoising outputs [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_gt_augment + noise,
            t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
        )
    else:
        x_denoised = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            x_denoised_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                use_conditioning=use_conditioning,
                enable_efficient_fusion=enable_efficient_fusion,
            )
            x_denoised.append(x_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)

    return x_gt_augment, x_denoised, sigma
