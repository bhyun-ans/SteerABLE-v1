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

import logging
import math
from typing import Any, Callable, Optional

import torch

from protenix.model.utils import centre_random_augmentation

logger = logging.getLogger(__name__)


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
    guidance_kwargs: Optional[dict[str, Any]] = None,
    save_trajectory: bool = False,
    reward_buffer: Optional[list] = None,
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
        guidance_kwargs (Optional[dict[str, Any]]): If provided, enables reward-guided
            embedding steering. Expected keys: mask_pairs (list of
            (residue_groups, partner_mask) per Ag chain), guidance_alpha, d0,
            softmin_beta.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    from protenix.model.guidance import (
        contact_epitope_reward,
        epitope_clash_reward,
        rms_normalize,
    )

    logger = logging.getLogger(__name__)
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    # ---- resolve guidance parameters ----
    guidance_on = guidance_kwargs is not None and bool(guidance_kwargs)
    if guidance_on:
        g_mask_pairs = guidance_kwargs["mask_pairs"]
        _raw_alpha = guidance_kwargs.get("guidance_alpha", None)
        g_total_steps = len(noise_schedule) - 1

        # Alpha resolution priority:
        #   1. guidance_alpha (constant float) — overrides the schedule
        #   2. truncated raised-cosine schedule (flow-matching paper form), default
        g_alpha_const = float(_raw_alpha) if _raw_alpha is not None else None
        g_alpha_schedule_fn = None
        if g_alpha_const is not None:
            _alpha_desc = f"alpha={g_alpha_const:.4f} (constant)"
        else:
            from protenix.utils.alpha_schedule import make_cosine_trunc
            g_alpha_init = float(guidance_kwargs.get("alpha_init", 1.0))
            g_alpha_trunc = float(guidance_kwargs.get("alpha_trunc", 0.5))
            g_alpha_schedule_fn = make_cosine_trunc(
                omega_init=g_alpha_init, tau_trunc=g_alpha_trunc
            )
            # First step at which alpha is forced to 0 (s = step/(total-1) >= tau).
            _off_step = math.ceil(g_alpha_trunc * (g_total_steps - 1)) if g_total_steps > 1 else 0
            _alpha_desc = (
                f"alpha=cosine_trunc(omega_init={g_alpha_init:.3f}, "
                f"tau_trunc={g_alpha_trunc:.3f}; off at step {_off_step}/{g_total_steps})"
            )

        g_d0 = float(guidance_kwargs.get("d0", 4.0))
        g_beta = float(guidance_kwargs.get("softmin_beta", 10.0))
        g_top_k = guidance_kwargs.get("top_k", None)
        g_lambda_clash = float(guidance_kwargs.get("lambda_clash", 0.0))
        g_clash_tau = float(guidance_kwargs.get("clash_tau", 1.5))
        g_clash_start_step = int(guidance_kwargs.get("clash_start_step", 0))
        g_clash_on = g_lambda_clash > 0.0
        logger.info(
            "Guidance ON: %s, d0=%.1f, softmin_beta=%.1f, lambda_clash=%.3f, clash_tau=%.2f, clash_start_step=%d",
            _alpha_desc, g_d0, g_beta, g_lambda_clash, g_clash_tau, g_clash_start_step,
        )

    total_steps = len(noise_schedule) - 1

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        trajectory_frames = [] if save_trajectory else None
        trajectory_rotations = [] if save_trajectory else None
        # Per-chunk reward list; appended to reward_buffer at end.
        chunk_rewards = [] if (guidance_on and reward_buffer is not None) else None
        # ---- steerable local copies of trunk embeddings ----
        if guidance_on:
            s_tau = s_trunk.detach().clone()
            z_tau = z_trunk.detach().clone()
            # When guidance is on, pair_z / p_lm / c_l are None so the
            # diffusion module recomputes them from z_tau each step.
            cur_pair_z = None
            cur_p_lm = None
            cur_c_l = None
        else:
            s_tau = s_trunk
            z_tau = z_trunk
            cur_pair_z = pair_z
            cur_p_lm = p_lm
            cur_c_l = c_l

        # init noise  [..., N_sample, N_atom, 3]
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )

        # Cumulative rotation applied by centre_random_augmentation across steps.
        # Kept in fp32 to avoid orthogonality drift over hundreds of matmuls.
        if save_trajectory:
            R_cum = (
                torch.eye(3, device=device, dtype=torch.float32)
                .expand(*batch_shape, chunk_n_sample, 3, 3)
                .contiguous()
            )
        else:
            R_cum = None

        # Logging latch only: prints the guidance->frozen transition message
        # exactly once. It does NOT control freezing — that is decided per step
        # by `guidance_active` (cur_alpha != 0) below.
        truncation_logged = False

        for step_idx, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # [..., N_sample, N_atom, 3]
            if save_trajectory:
                x_l, R_step = centre_random_augmentation(
                    x_input_coords=x_l, N_sample=1, return_transform=True,
                )
                x_l = x_l.squeeze(dim=-3).to(dtype)
                # R_step: [..., 1, 3, 3] → [..., chunk_n_sample, 3, 3]
                R_step = R_step.squeeze(dim=-3).to(torch.float32).detach()
                R_cum = torch.matmul(R_step, R_cum)
            else:
                x_l = (
                    centre_random_augmentation(x_input_coords=x_l, N_sample=1)
                    .squeeze(dim=-3)
                    .to(dtype)
                )

            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            # Decide this step's steering weight *before* the forward pass. Once
            # the cosine schedule truncates (cur_alpha == 0) there is nothing to
            # gain from the grad-enabled forward + autograd — skip it and fall
            # through to the plain (frozen-embedding) run below.
            if guidance_on:
                if g_alpha_const is not None:
                    cur_alpha = g_alpha_const
                else:
                    cur_alpha = g_alpha_schedule_fn(step_idx, g_total_steps)
                guidance_active = cur_alpha != 0.0
            else:
                guidance_active = False

            if guidance_active:
                # ---- guided step: grad-enabled forward ----
                s_tau = s_tau.detach().requires_grad_(True)
                z_tau = z_tau.detach().requires_grad_(True)

                with torch.enable_grad():
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
                        inplace_safe=False,
                        enable_efficient_fusion=False,
                    )

                    # cur_alpha computed above (before the forward pass).
                    reward_contact = contact_epitope_reward(
                        coords=x_denoised,
                        mask_pairs=g_mask_pairs,
                        d0=g_d0,
                        softmin_beta=g_beta,
                        top_k=g_top_k,
                    )
                    if g_clash_on and step_idx >= g_clash_start_step:
                        reward_clash = epitope_clash_reward(
                            coords=x_denoised,
                            mask_pairs=g_mask_pairs,
                            ref_element=input_feature_dict["ref_element"],
                            tau=g_clash_tau,
                        )
                        reward = reward_contact + g_lambda_clash * reward_clash
                    else:
                        reward_clash = None
                        reward = reward_contact

                    grads = torch.autograd.grad(
                        outputs=reward,
                        inputs=[s_tau, z_tau],
                        create_graph=False,
                        allow_unused=True,
                    )
                    g_s = grads[0] if grads[0] is not None else torch.zeros_like(s_tau)
                    g_z = grads[1] if grads[1] is not None else torch.zeros_like(z_tau)

                # RMS-normalise and update embeddings
                g_s_bar = rms_normalize(g_s)
                g_z_bar = rms_normalize(g_z)
                s_tau = (s_tau.detach() + cur_alpha * g_s_bar).detach()
                z_tau = (z_tau.detach() + cur_alpha * g_z_bar).detach()

                x_denoised = x_denoised.detach()

                # Capture scalar reward for this step (chunk-level, averaged over samples)
                if chunk_rewards is not None:
                    chunk_rewards.append(float(reward.item()))

                if step_idx % 50 == 0:
                    if reward_clash is not None:
                        logger.info(
                            "  step %d: reward=%.4f (contact=%.4f, clash=%.4f, λ_clash=%.3f), alpha=%.6f, |g_s|=%.6f, |g_z|=%.6f, top_k=%s",
                            step_idx, reward.item(), reward_contact.item(), reward_clash.item(),
                            g_lambda_clash, cur_alpha, g_s.norm().item(), g_z.norm().item(),
                            g_top_k,
                        )
                    else:
                        logger.info(
                            "  step %d: reward=%.4f, alpha=%.6f, |g_s|=%.6f, |g_z|=%.6f, top_k=%s",
                            step_idx, reward.item(), cur_alpha, g_s.norm().item(), g_z.norm().item(),
                            g_top_k,
                        )
            else:
                # ---- plain path: genuinely unguided, OR guidance truncated ----
                # When guidance_on and we reach here, the cosine schedule has
                # truncated (cur_alpha == 0): skip the grad forward + autograd and
                # run a single plain denoise on the *frozen* steered embeddings.
                if guidance_on:
                    s_tau = s_tau.detach()
                    z_tau = z_tau.detach()
                    if not truncation_logged:
                        logger.info(
                            "Guidance truncated at step %d (alpha=0); running "
                            "remaining steps on frozen steered embeddings (no autograd).",
                            step_idx,
                        )
                        truncation_logged = True
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

            # Euler step: coordinate update
            delta = (x_noisy - x_denoised) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe is a typo.
            dt = c_tau - t_hat
            x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta

            # ---- trajectory capture (every step) ----
            if trajectory_frames is not None:
                trajectory_frames.append(x_l.detach().cpu())
                trajectory_rotations.append(R_cum.detach().cpu())

        # Side-channel: push this chunk's reward list into the caller-provided buffer.
        # The return signature below stays unchanged.
        if chunk_rewards is not None:
            reward_buffer.append(chunk_rewards)

        if trajectory_frames is not None:
            return x_l, trajectory_frames, trajectory_rotations
        return x_l

    if diffusion_chunk_size is None:
        result = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l_chunks = []
        traj_chunks = [] if save_trajectory else None
        rot_chunks = [] if save_trajectory else None
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            result = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            if save_trajectory:
                chunk_x_l, chunk_traj, chunk_rot = result
                x_l_chunks.append(chunk_x_l)
                traj_chunks.append(chunk_traj)
                rot_chunks.append(chunk_rot)
            else:
                x_l_chunks.append(result)
        x_l = torch.cat(x_l_chunks, -3)  # [..., N_sample, N_atom, 3]
        if save_trajectory:
            # merge chunk trajectories + rotations along sample axis (-3 for coords,
            # -3 for R whose shape is [..., N_sample, 3, 3]).
            n_frames = len(traj_chunks[0])
            merged_traj = []
            merged_rot = []
            for f in range(n_frames):
                merged_traj.append(torch.cat([tc[f] for tc in traj_chunks], dim=-3))
                merged_rot.append(torch.cat([rc[f] for rc in rot_chunks], dim=-3))
            result = (x_l, merged_traj, merged_rot)
        else:
            result = x_l
    return result


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
