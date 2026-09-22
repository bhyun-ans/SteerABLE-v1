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

import copy
import random
import time
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.model import sample_confidence
from protenix.model.gating import (
    compute_epitope_enrichment,
    is_steered_branch,
    MODE_BOTH,
    MODE_ROUTE,
    resolve_branches,
    steered_branch_names,
    trunk_fingerprint,
)
from protenix.model.generator import (
    InferenceNoiseScheduler,
    sample_diffusion,
    sample_diffusion_training,
    TrainingNoiseSampler,
)
from protenix.model.modules.confidence import ConfidenceHead
from protenix.model.modules.diffusion import DiffusionModule
from protenix.model.modules.embedders import (
    ConstraintEmbedder,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from protenix.model.modules.head import DistogramHead
from protenix.model.modules.pairformer import (
    MSAModule,
    PairformerStack,
    TemplateEmbedder,
)
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.triangular.layers import LayerNorm
from protenix.model.utils import simple_merge_dict_list
from protenix.utils.logger import get_logger
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator

logger = get_logger(__name__)


def _stage_time() -> float:
    """Wall-clock time after a CUDA sync.

    Stage timings are compared against each other (the routing gate is judged on
    how much sampling time it saves), so they must not be skewed by kernels that
    are still queued when `time.time()` is read.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def _summarize_steering_trace(
    trace: list[dict[str, Any]], n_sample: int
) -> dict[str, Any]:
    """Turn the sampler's per-step reward trace into a per-sample summary.

    The sampler appends one entry per guided step, each holding the reward of
    every sample in that chunk plus the chunk's offset into the full N_sample
    stack.  Reassembling by absolute sample index makes traces from different
    `sample_diffusion_chunk_size` settings directly comparable.

    The field that matters is `samples_share_one_curve`.  When epitope steering
    runs per sample, each sample sees a reward computed from its own structure,
    so the curves must differ.  If they are identical the chunk collapsed to a
    single shared steering trajectory driven by the chunk-averaged reward --
    the failure mode that forced chunk_size=1 in the first place.
    """
    if not trace:
        return {}

    curves: dict[int, list[float]] = {i: [] for i in range(n_sample)}
    steps: dict[int, list[int]] = {i: [] for i in range(n_sample)}
    spreads: list[float] = []
    for entry in trace:
        off = int(entry.get("sample_offset", 0))
        rewards = entry.get("reward") or []
        for k, r in enumerate(rewards):
            idx = off + k
            if idx in curves:
                curves[idx].append(float(r))
                steps[idx].append(int(entry["step"]))
        if len(rewards) > 1:
            spreads.append(float(max(rewards) - min(rewards)))

    lengths = {i: len(v) for i, v in curves.items()}
    n_steps = max(lengths.values()) if lengths else 0
    aligned = [curves[i] for i in range(n_sample) if len(curves[i]) == n_steps]
    identical = False
    max_step_spread = None
    if len(aligned) > 1:
        # Compare every sample's curve to sample 0 at matching steps.
        identical = all(
            all(abs(a - b) == 0.0 for a, b in zip(aligned[0], c)) for c in aligned[1:]
        )
        max_step_spread = max(
            (max(vals) - min(vals) for vals in zip(*aligned)), default=None
        )

    return {
        "n_guided_steps": n_steps,
        "n_samples_traced": len(aligned),
        # THE diagnostic: True means steering degenerated to one shared curve.
        "samples_share_one_curve": identical,
        # Largest spread between samples at any single step; 0.0 also means shared.
        "max_reward_spread_across_samples": max_step_spread,
        "mean_within_chunk_spread": (
            sum(spreads) / len(spreads) if spreads else None
        ),
        "reward_first_step": [c[0] for c in aligned] if aligned else [],
        "reward_last_step": [c[-1] for c in aligned] if aligned else [],
        # Full per-sample curves, sub-sampled to keep the JSON small.
        "reward_curves": {
            str(i): curves[i][:: max(1, n_steps // 40)] for i in range(n_sample)
        },
        "curve_step_index": (
            steps[0][:: max(1, n_steps // 40)] if steps.get(0) else []
        ),
    }


def _amp_is_reducing_precision() -> bool:
    """Whether autocast is currently casting anything down.

    `torch.autocast(dtype=torch.float32)` — what `--dtype fp32` produces — still
    reports itself as enabled on CUDA while casting nothing, so the enablement
    flag on its own would claim amp is active for an fp32 run.
    """
    if not torch.is_autocast_enabled():
        return False
    get_dtype = getattr(torch, "get_autocast_dtype", None)
    dtype = get_dtype("cuda") if get_dtype else torch.get_autocast_gpu_dtype()
    return dtype != torch.float32


def update_input_feature_dict(input_feature_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Lines 1-3 of Algorithm 5 compute d_lm, v_lm, and pad_info utilized in the AtomAttentionEncoder.
    Args:
            input_feature_dict (dict[str, Any]): input features
    Returns:
            input_feature_dict (dict[str, Any]): input features
    """
    from protenix.model.modules.transformer import rearrange_qk_to_dense_trunk

    with torch.no_grad():
        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=32,
            n_keys=128,
            compute_mask=True,
        )
        # Compute atom pair feature
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, 1]
        input_feature_dict["d_lm"] = d_lm
        input_feature_dict["v_lm"] = v_lm
        input_feature_dict["pad_info"] = pad_info
        return input_feature_dict


class Protenix(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs: Any) -> None:
        super(Protenix, self).__init__()
        self.configs = configs
        torch.backends.cuda.matmul.allow_tf32 = self.configs.enable_tf32
        # Some constants
        self.enable_diffusion_shared_vars_cache = (
            self.configs.enable_diffusion_shared_vars_cache
        )
        self.enable_efficient_fusion = self.configs.enable_efficient_fusion
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.train_confidence_only = configs.train_confidence_only
        if self.train_confidence_only:  # the final finetune stage
            assert configs.loss.weight.alpha_diffusion == 0.0
            assert configs.loss.weight.alpha_distogram == 0.0

        # Diffusion scheduler
        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        esm_configs = configs.get("esm", {})  # This is used in InputFeatureEmbedder
        self.input_embedder = InputFeatureEmbedder(
            **configs.model.input_embedder, esm_configs=esm_configs
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.constraint_embedder = ConstraintEmbedder(
            **configs.model.constraint_embedder
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)
        self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)
        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        mc_dropout: bool = False,
        mc_dropout_rate: float = 0.4,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        if self.train_confidence_only:
            self.input_embedder.eval()
            self.template_embedder.eval()
            self.msa_module.eval()
            self.pairformer_stack.eval()

        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]
        z_constraint = None

        if "constraint_feature" in input_feature_dict:
            z_constraint = self.constraint_embedder(
                input_feature_dict["constraint_feature"]
            )

        s_init = self.linear_no_bias_sinit(s_inputs)  # [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  # [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict["relp"])
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init += z_constraint
        else:
            z_init = z_init + self.relative_position_encoding(
                input_feature_dict["relp"]
            )
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init = z_init + z_constraint
        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                if mc_dropout:
                    z = z_init + F.dropout(
                        self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z)),
                        p=self.configs.mc_dropout_rate,
                    )
                else:
                    z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        if self.train_confidence_only:
            self.input_embedder.train()
            self.template_embedder.train()
            self.msa_module.train()
            self.pairformer_stack.train()

        return s_inputs, s, z

    def sample_diffusion(
        self, chunk_size_override: Optional[int] = None, **kwargs: Any
    ) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Args:
            chunk_size_override: Replaces `infer_setting.sample_diffusion_chunk_size`
                for this call. The steered and unguided branches have very
                different constraints -- the unguided one holds no steering state
                and runs under no_grad, so it can batch every sample at once
                regardless of what the steered branch can afford.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    (
                        chunk_size_override
                        if chunk_size_override is not None
                        else self.configs.infer_setting.sample_diffusion_chunk_size
                    )
                    if not self.training
                    else None
                ),
            }
        )
        # `epitope_configs` is injected at the caller site (_main_inference_loop)
        # after the mask_pairs are built from the live input_feature_dict, so
        # it arrives in **kwargs. We do not read it from self.configs directly.
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def run_confidence_head(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.
            mc_dropout_apply_rate (float): Only for inference mode

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        # For backward compatibility, if N_model_seed > 1, process multiple seeds here
        # But in evaluation mode, this should be handled externally
        if N_model_seed > 1 and mode in ["inference"]:
            # Multi-seed merging concatenates one prediction dict per seed, which
            # has no meaning once a seed returns several sampling branches.
            gate_mode = str(
                (self.configs.to_dict().get("gating", {}) or {}).get("mode", MODE_BOTH)
            )
            if gate_mode in (MODE_BOTH, MODE_ROUTE) and getattr(
                self.configs, "epitope_residue", None
            ):
                raise ValueError(
                    f"N_model_seed={N_model_seed} cannot be combined with "
                    f"gating.mode='{gate_mode}': the merged sample stack cannot "
                    "record which branch each seed sampled (with 'both' there "
                    "are even two per seed). Use gating.mode='steerable'/'raw', or "
                    "run one seed at a time."
                )
            pred_dicts = []
            log_dicts = []
            time_trackers = []
            for _ in range(N_model_seed):
                pred_dict, log_dict, time_tracker = self._main_inference_loop(
                    input_feature_dict=(
                        copy.deepcopy(input_feature_dict)
                        if (N_model_seed > 1 and mode == "inference")
                        else input_feature_dict
                    ),  # the input_feature_dict is modified when mode is "inference"
                    label_dict=label_dict,
                    N_cycle=N_cycle,
                    mode=mode,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                    symmetric_permutation=symmetric_permutation,
                    mc_dropout=random.random() < mc_dropout_apply_rate,
                )
                pred_dicts.append(pred_dict)
                log_dicts.append(log_dict)
                time_trackers.append(time_tracker)

            # Combine outputs of multiple models
            def _cat(dict_list, key):
                return torch.cat([x[key] for x in dict_list], dim=0)

            def _list_join(dict_list, key):
                return sum([x[key] for x in dict_list], [])

            all_pred_dict = {
                "coordinate": _cat(pred_dicts, "coordinate"),
                "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
                "full_data": _list_join(pred_dicts, "full_data"),
                "plddt": _cat(pred_dicts, "plddt"),
                "pae": _cat(pred_dicts, "pae"),
                "pde": _cat(pred_dicts, "pde"),
                "resolved": _cat(pred_dicts, "resolved"),
            }

            # The gate report is trunk-level and this path runs one trunk per
            # seed, so keep the first seed's (the contact map with it) and list
            # every seed's enrichment rather than dropping the side-car.
            gate_reports = [p["gating"] for p in pred_dicts if "gating" in p]
            if gate_reports:
                all_pred_dict["gating"] = {
                    **gate_reports[0],
                    "N_model_seed": N_model_seed,
                    "per_model_seed_enrichment": [
                        g.get("enrichment") for g in gate_reports
                    ],
                }
                for pred in pred_dicts:
                    if "gating_arrays" in pred:
                        all_pred_dict["gating_arrays"] = pred["gating_arrays"]
                        break

            all_log_dict = simple_merge_dict_list(log_dicts)
            all_time_dict = simple_merge_dict_list(time_trackers)
            return all_pred_dict, all_log_dict, all_time_dict
        else:
            # Single seed inference - delegate to _main_inference_loop
            return self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                mc_dropout=random.random() < mc_dropout_apply_rate,
            )

    def _get_dynamic_chunk_size(self, N_token: int) -> Optional[int]:
        """
        Get dynamic chunk_size based on token count

        Args:
            N_token (int): Number of tokens

        Returns:
            Optional[int]: Optimal chunk_size for the given token count
        """
        if not hasattr(self.configs.infer_setting, "chunk_size_thresholds"):
            return self.configs.infer_setting.chunk_size

        thresholds = self.configs.infer_setting.chunk_size_thresholds

        # Convert string keys to integers and sort in ascending order
        threshold_pairs = [(int(k), v) for k, v in thresholds.items()]
        sorted_thresholds = sorted(threshold_pairs, key=lambda x: x[0])

        # Find the appropriate chunk_size for the given token count
        for threshold, chunk_size in sorted_thresholds:
            if N_token <= threshold:
                return None if chunk_size == -1 else chunk_size

        # For token counts larger than the largest threshold, use smallest chunk_size
        return 32  # extreme case for very large proteins

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.
        mc_dropout: do not use by default

        The trunk runs once and every sampling branch shares its output (see
        `protenix.model.gating`): with `gating.mode="both"` the same pairformer
        output feeds a steered and a no-guidance diffusion run, and the returned
        dict carries both under "branches" (primary first) instead of being a
        prediction dict itself.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]

        # Apply dynamic chunk_size if enabled (otherwise keep the passed chunk_size)
        if (
            hasattr(self.configs.infer_setting, "dynamic_chunk_size")
            and self.configs.infer_setting.dynamic_chunk_size
        ):
            chunk_size = self._get_dynamic_chunk_size(N_token)
        # If dynamic chunking is disabled, chunk_size keeps its original value from the function parameter

        log_dict = {}
        time_tracker = {}

        # Recorded in the gate report: the trunk is shared by every branch, so
        # its precision decides what the `pairformer` timing below is worth and
        # whether the run is numerically comparable to another one.
        trunk_amp_enabled = _amp_is_reducing_precision()

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            mc_dropout=mc_dropout,
        )

        keys_to_delete = []
        for key in input_feature_dict.keys():
            if "template_" in key or key in [
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
                # "token_bonds",
            ]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del input_feature_dict[key]
        step_trunk = _stage_time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        # Every sampling branch below (one per epitope set, plus raw) must start
        # from THIS trunk output. Fingerprint it now and re-check before each
        # branch; see trunk_fingerprint for why the check exists.
        trunk_fp = trunk_fingerprint(s, z)
        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )

        # Distogram logits: log contact_probs only, to reduce the dimension.
        # Computed here, ahead of diffusion, because the routing gate scores it.
        # Sampling never writes to `z` (epitope steering steers local copies),
        # so this is the same tensor the old post-diffusion call produced.
        with torch.no_grad():
            contact_probs = autocasting_disable_decorator(True)(
                sample_confidence.compute_contact_prob
            )(
                distogram_logits=self.distogram_head(z),
                **sample_confidence.get_bin_params(self.configs.loss.distogram),
            )  # [N_token, N_token]
        step_distogram = _stage_time()
        # Not gate overhead: contact_probs is dumped in full_data either way.
        time_tracker.update({"distogram": step_distogram - step_trunk})

        # ---- SteerABLE epitope-guidance setup ----
        # One entry per epitope SET: `--epitope_residue "B:1,B:2;B:40,B:41"`
        # gives two. Each set is steered by its own diffusion branch off the one
        # trunk computed above, so K sets cost one pairformer pass, not K runs.
        (
            epitope_configs_list,
            epitope_set_strs,
            epitope_residues_list,
            ab_chain_ids,
        ) = self._build_epitope_configs(input_feature_dict)
        epitope_on = epitope_configs_list is not None
        n_epitope_sets = len(epitope_configs_list) if epitope_on else 0

        # ---- routing gate: how concentrated is the trunk on each epitope? ----
        gating_cfg = self.configs.to_dict().get("gating", {}) or {}
        gate_mode = str(gating_cfg.get("mode", MODE_BOTH))
        gate_threshold = gating_cfg.get("threshold", None)
        gate_stats_list: list[dict[str, Any]] = []
        gate_arrays = None
        if epitope_on:
            t_gate = _stage_time()
            arrays_list = []
            for epitope_residues in epitope_residues_list:
                stats, arrays = compute_epitope_enrichment(
                    contact_probs=contact_probs,
                    asym_id=input_feature_dict["asym_id"],
                    residue_index=input_feature_dict["residue_index"],
                    epitope_residues=epitope_residues,
                    ab_chain_ids=ab_chain_ids,
                )
                gate_stats_list.append(stats)
                arrays_list.append(arrays)
            time_tracker.update({"enrichment": _stage_time() - t_gate})
            if gating_cfg.get("save_contact_probs", True):
                # The contact map, e_j, Ab mask and token ids come from the
                # trunk and are identical for every set; only the epitope mask
                # is per set. One set keeps the [N_token] mask unchanged,
                # several sets stack it to [K, N_token] in CLI order.
                gate_arrays = dict(arrays_list[0])
                if n_epitope_sets > 1:
                    gate_arrays["epitope_token_mask"] = torch.stack(
                        [a["epitope_token_mask"] for a in arrays_list], dim=0
                    )

        branch_names, routed_branch = resolve_branches(
            mode=gate_mode,
            epitope_on=epitope_on,
            threshold=gate_threshold,
            enrichment=(
                gate_stats_list[0]["enrichment"] if n_epitope_sets == 1 else None
            ),
            n_epitope_sets=max(n_epitope_sets, 1),
        )
        # Steered branch k steers epitope set k, in CLI order.
        branch_to_set = {
            name: k
            for k, name in enumerate(steered_branch_names(max(n_epitope_sets, 1)))
        }
        multi_branch = len(branch_names) > 1
        if multi_branch and (
            label_dict is not None or symmetric_permutation is not None
        ):
            raise ValueError(
                f"gating.mode='{gate_mode}' samples {len(branch_names)} diffusion "
                "branches from one trunk, which is inference-only: it cannot be "
                "combined with labels / symmetric permutation. Use "
                "gating.mode='steerable' or 'raw' for eval runs."
            )
        logger.info(
            "Sampling branches: %s (gating.mode=%s%s)",
            branch_names,
            gate_mode,
            (
                ""
                if routed_branch is None
                else f", gate routed to {routed_branch}"
            ),
        )

        def _sample_branch(branch: str) -> tuple[dict[str, Any], dict[str, float]]:
            """Diffusion + confidence for one branch, on the shared trunk output.

            Steered branches (`steerable`, or `steerable_<k>` with several epitope
            sets) steer epitope set `branch_to_set[branch]`; `raw` steers nothing.

            steer=True: SteerABLE epitope-guided sampling. Autograd must reach the
            denoiser so the reward can steer (s, z), and the trunk-derived
            diffusion caches are disabled because the steered z moves every step.

            steer=False: plain Protenix sampling — no_grad, shared-vars cache,
            in-place ops. That is exactly what a run without `epitope_residue`
            does, so the branch's wall-clock is the honest "what would the gate
            save" number even though guidance is configured for this process.

            Both branches inherit the caller's autocast state; the sampler itself
            is kept in fp32 by autocasting_disable_decorator(skip_amp.
            sample_diffusion), which is what the reward backward needs.
            """
            steer = is_steered_branch(branch)
            epitope_configs = (
                epitope_configs_list[branch_to_set[branch]] if steer else None
            )
            branch_pred: dict[str, Any] = {}
            if steer:
                branch_inplace_safe = inplace_safe
            else:
                # The caller passes inplace_safe=False whenever guidance is
                # configured, because grad is enabled for the whole forward.
                # This branch re-enters no_grad and is always sampled last, so
                # in-place ops are safe again.
                branch_inplace_safe = True if epitope_on else inplace_safe
            grad_ctx = nullcontext() if steer else torch.no_grad()

            t_branch_st = _stage_time()
            with grad_ctx:
                cache = {"pair_z": None, "p_lm/c_l": [None, None]}
                if self.enable_diffusion_shared_vars_cache and not steer:
                    # line 1-5 of algorithm 21 calculate z in diffusion conditioning
                    cache["pair_z"] = autocasting_disable_decorator(
                        self.configs.skip_amp.sample_diffusion
                    )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                        input_feature_dict["relp"], z, False
                    )
                    cache["p_lm/c_l"] = autocasting_disable_decorator(
                        self.configs.skip_amp.sample_diffusion
                    )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                        ref_pos=input_feature_dict["ref_pos"],
                        ref_charge=input_feature_dict["ref_charge"],
                        ref_mask=input_feature_dict["ref_mask"],
                        ref_element=input_feature_dict["ref_element"],
                        ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                        atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                        d_lm=input_feature_dict["d_lm"],
                        v_lm=input_feature_dict["v_lm"],
                        pad_info=input_feature_dict["pad_info"],
                        r_l=True,
                        z=cache["pair_z"],
                        inplace_safe=False,
                    )
                t_cache = _stage_time()
                branch_pred["coordinate"] = self.sample_diffusion(
                    denoise_net=self.diffusion_module,
                    input_feature_dict=input_feature_dict,
                    s_inputs=s_inputs,
                    s_trunk=s,
                    z_trunk=None if cache["pair_z"] is not None else z,
                    pair_z=cache["pair_z"],
                    p_lm=cache["p_lm/c_l"][0],
                    c_l=cache["p_lm/c_l"][1],
                    N_sample=N_sample,
                    noise_schedule=noise_schedule,
                    inplace_safe=branch_inplace_safe,
                    enable_efficient_fusion=self.enable_efficient_fusion,
                    epitope_configs=epitope_configs,
                    # The unguided branch keeps s_trunk/z_trunk read-only for the
                    # whole trajectory, so chunking is the pure memory knob it is
                    # upstream and does not change its results -- batch it fully
                    # whatever the steered branch is limited to.
                    chunk_size_override=None if steer else N_sample,
                )
                t_diffusion = _stage_time()
                branch_pred["contact_probs"] = contact_probs

                # Confidence logits. Under no_grad: nothing downstream needs a
                # graph here, and keeping one OOMs the steered branch on large
                # targets (N_token >~ 700).
                with torch.no_grad():
                    (
                        branch_pred["plddt"],
                        branch_pred["pae"],
                        branch_pred["pde"],
                        branch_pred["resolved"],
                    ) = self.run_confidence_head(
                        input_feature_dict=input_feature_dict,
                        s_inputs=s_inputs,
                        s_trunk=s,
                        z_trunk=z,
                        pair_mask=None,
                        x_pred_coords=branch_pred["coordinate"],
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=branch_inplace_safe,
                        chunk_size=chunk_size,
                    )
                t_confidence = _stage_time()

            return branch_pred, {
                "diffusion_cache": t_cache - t_branch_st,
                "diffusion": t_diffusion - t_cache,
                "confidence": t_confidence - t_diffusion,
                "total": t_confidence - t_branch_st,
            }

        # Every branch has to start from the same RNG state: the branches are
        # then a paired comparison (identical x_T and per-step noise, so a DockQ
        # difference is guidance and not a different draw), and each branch
        # samples exactly what a single-branch run with this seed would sample.
        # Three streams matter: torch CPU, torch CUDA (the noise in
        # sample_diffusion) and numpy — centre_random_augmentation draws its
        # per-step rotation from scipy's Rotation.random, i.e. numpy's global RNG.
        if multi_branch:
            cpu_rng_state = torch.get_rng_state()
            numpy_rng_state = np.random.get_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )

        branch_results: list[tuple[str, dict[str, Any]]] = []
        branch_times: dict[str, dict[str, float]] = {}
        for branch in branch_names:
            if trunk_fingerprint(s, z) != trunk_fp:
                raise RuntimeError(
                    f"Trunk embeddings changed before branch '{branch}': an "
                    "earlier branch wrote into the shared (s, z), so this branch "
                    "would not start from the same trunk. Steered branches must "
                    "run with in-place ops disabled and 'raw' must run last."
                )
            if multi_branch:
                torch.set_rng_state(cpu_rng_state)
                np.random.set_state(numpy_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_state)
            branch_pred, branch_time = _sample_branch(branch)
            branch_times[branch] = branch_time
            if multi_branch:
                time_tracker.update(
                    {
                        f"diffusion_{branch}": branch_time["diffusion"],
                        f"confidence_{branch}": branch_time["confidence"],
                    }
                )
            else:
                time_tracker.update(
                    {
                        "diffusion": branch_time["diffusion"],
                        "confidence": branch_time["confidence"],
                    }
                )
                # Permutation: when label is given, permute coordinates and other heads
                if label_dict is not None and symmetric_permutation is not None:
                    t_permutation = _stage_time()
                    (
                        branch_pred,
                        log_dict,
                    ) = symmetric_permutation.permute_inference_pred_dict(
                        input_feature_dict=input_feature_dict,
                        pred_dict=branch_pred,
                        label_dict=label_dict,
                        permute_by_pocket=("pocket_mask" in label_dict)
                        and ("interested_ligand_mask" in label_dict),
                    )
                    time_tracker.update({"permutation": _stage_time() - t_permutation})

            # Summary Confidence & Full Data
            # Computed after coordinates and logits are permuted
            self._compute_summary_confidence(
                pred_dict=branch_pred,
                input_feature_dict=input_feature_dict,
                label_dict=None if multi_branch else label_dict,
                N_cycle=N_cycle,
                mode=mode,
            )
            if multi_branch:
                # Only coordinate / summary_confidence / full_data are dumped,
                # so release the per-sample logits instead of making the next
                # branch share memory with them.
                for key in ("plddt", "pae", "pde", "resolved"):
                    branch_pred.pop(key, None)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            branch_results.append((branch, branch_pred))

        if multi_branch:
            # Not a prediction dict: the caller dumps each branch separately.
            pred_dict: dict[str, Any] = {"branches": branch_results}
        else:
            pred_dict = branch_results[0][1]
        time_tracker.update({"model_forward": _stage_time() - step_st})

        if epitope_on:
            report: dict[str, Any] = {}
            if n_epitope_sets == 1:
                # One set: the flat single-epitope report, unchanged.
                report.update(gate_stats_list[0])
            report.update({
                # compute_contact_prob's cutoff for "in contact"
                "contact_cutoff_angstrom": 8.0,
                "epitope_residue": getattr(self.configs, "epitope_residue", None),
                "ab_chains": getattr(self.configs, "ab_chains", None),
                "n_epitope_sets": n_epitope_sets,
                "mode": gate_mode,
                "threshold": gate_threshold,
                "branches_run": list(branch_names),
                "routed_branch": routed_branch,
                "N_token": int(N_token),
                "N_sample": int(N_sample),
                "N_step": int(N_step),
                # Provenance for the interval sweep. Without it, two runs that
                # differ only in --epitope.guidance_interval produce
                # metadata-identical output dirs while differing in both the
                # coordinates and time.<branch>.diffusion, which makes the
                # resulting wall-clock/quality table unreconstructable.
                "epitope_guidance_interval": int(
                    (self.configs.to_dict().get("epitope") or {}).get(
                        "guidance_interval", 8
                    )
                ),
                "trunk_amp_enabled": trunk_amp_enabled,
                # Every branch was checked (before it sampled) to start from the
                # fingerprinted trunk; a mismatch raises, so reaching this line
                # means every branch shares one original (s, z) / distogram.
                "shared_trunk_verified": True,
            })
            if n_epitope_sets == 1:
                report["steering"] = _summarize_steering_trace(
                    epitope_configs_list[0].get("trace") or [], int(N_sample)
                )
            else:
                # Several sets: the per-epitope stats move into `epitopes`, one
                # block per set with the branch that steered it, so a branch
                # directory can be traced back to its residues and enrichment.
                # `n_ab_tokens` is the only trunk-level stat and stays flat.
                report["n_ab_tokens"] = gate_stats_list[0]["n_ab_tokens"]
                steered = steered_branch_names(n_epitope_sets)
                report["epitopes"] = [
                    {
                        "branch": steered[k],
                        "epitope_residue": epitope_set_strs[k],
                        **gate_stats_list[k],
                        "steering": _summarize_steering_trace(
                            epitope_configs_list[k].get("trace") or [],
                            int(N_sample),
                        ),
                    }
                    for k in range(n_epitope_sets)
                ]
            # pairformer + distogram + enrichment + the branches sum to
            # model_forward, so any gating scenario can be re-costed offline.
            report["time"] = {
                "pairformer": time_tracker["pairformer"],
                "distogram": time_tracker["distogram"],
                "enrichment": time_tracker.get("enrichment", 0.0),
                "model_forward": time_tracker["model_forward"],
                **{name: branch_times[name] for name in branch_names},
            }
            pred_dict["gating"] = report
            if gate_arrays is not None:
                pred_dict["gating_arrays"] = gate_arrays

        return pred_dict, log_dict, time_tracker

    def _build_epitope_configs(
        self, input_feature_dict: dict[str, Any]
    ) -> tuple[
        Optional[list[dict[str, Any]]],
        Optional[list[str]],
        Optional[list[list[tuple[int, int]]]],
        Optional[set[int]],
    ]:
        """Build the runtime epitope-guidance configs for this input.

        When `epitope_residue` is set in the top-level configs, mask_pairs are
        derived from the live input_feature_dict (which is why this cannot be
        static config), and the parsed epitope residues / Ab chains are handed
        back so the routing gate can score exactly the residues being steered.

        `epitope_residue` may hold several epitope SETS separated by ';'
        (see `protenix.model.steering.parse_epitope_sets`). Each set gets its
        own configs dict with its own mask_pairs and its own reward trace; the
        lists below are index-aligned, in CLI order.

        Returns:
            `(epitope_configs_list, epitope_set_strs, epitope_residues_list,
            ab_chain_ids)`, all None when `epitope_residue` is unset, i.e.
            guidance is off.
        """
        epitope_str = getattr(self.configs, "epitope_residue", None)
        if not epitope_str:
            return None, None, None, None

        ab_chains_str = getattr(self.configs, "ab_chains", None)
        if not ab_chains_str:
            raise ValueError(
                "epitope_residue is set but ab_chains is missing; "
                "provide ab_chains (e.g. 'A,B') so we can restrict the "
                "partner atom mask to antibody chains."
            )
        from protenix.model.steering import (
            build_guidance_masks,
            parse_chain_list,
            parse_epitope_residue,
            parse_epitope_sets,
            validate_and_infer_ag_chains,
        )

        ab_chain_ids = parse_chain_list(ab_chains_str)
        validate_and_infer_ag_chains(input_feature_dict, ab_chain_ids)
        _ecfg = self.configs.to_dict().get("epitope", {}) or {}
        epitope_set_strs = parse_epitope_sets(epitope_str)
        n_sets = len(epitope_set_strs)
        epitope_residues_list: list[list[tuple[int, int]]] = []
        epitope_configs_list: list[dict[str, Any]] = []
        for k, set_str in enumerate(epitope_set_strs):
            if n_sets > 1:
                logger.info("Epitope set %d/%d: %s", k, n_sets, set_str)
            epitope_residues = parse_epitope_residue(set_str)
            mask_pairs = build_guidance_masks(
                input_feature_dict, epitope_residues, ab_chain_ids
            )
            epitope_residues_list.append(epitope_residues)
            epitope_configs_list.append(self._epitope_configs_for(mask_pairs, _ecfg))
        return (
            epitope_configs_list,
            epitope_set_strs,
            epitope_residues_list,
            ab_chain_ids,
        )

    @staticmethod
    def _epitope_configs_for(
        mask_pairs: list[tuple[list[torch.Tensor], torch.Tensor]],
        _ecfg: dict[str, Any],
    ) -> dict[str, Any]:
        """The sampler-side configs dict for one epitope set.

        Called once per set so each set carries its own `mask_pairs` and its
        own (fresh, empty) `trace` list; the α / geometry / clash knobs are
        shared by every set.
        """
        epitope_configs = {
            "mask_pairs": mask_pairs,
            "guidance_alpha": _ecfg.get("guidance_alpha", None),
            "alpha_init": _ecfg.get("alpha_init", 1.0),
            "alpha_trunc": _ecfg.get("alpha_trunc", 0.5),
            "d0": _ecfg.get("d0", 4.0),
            "softmin_beta": _ecfg.get("softmin_beta", 10.0),
            "lambda_clash": _ecfg.get("lambda_clash", 0.0),
            "clash_tau": _ecfg.get("clash_tau", 1.5),
            "clash_start_step": _ecfg.get("clash_start_step", 0),
            # Filled in place by the sampler: one entry per guided step holding
            # the per-sample reward. Dumped into <PDB>_gating.json so the steering
            # of each individual sample is inspectable after the run.
            "trace": [],
            # NOTE: this dict is an explicit allow-list, not a splat of _ecfg.
            # A key added to configs/configs_inference.py but not copied here
            # is accepted by argparse, appears in the dumped config, and then
            # silently never reaches sample_diffusion.
            "guidance_interval": _ecfg.get("guidance_interval", 8),
        }
        return epitope_configs

    def _compute_summary_confidence(
        self,
        pred_dict: dict[str, Any],
        input_feature_dict: dict[str, Any],
        label_dict: Optional[dict[str, Any]],
        N_cycle: int,
        mode: str,
    ) -> None:
        """Fill `summary_confidence` / `full_data` into pred_dict, in place."""
        if label_dict is None:
            interested_atom_mask = None
        else:
            interested_atom_mask = label_dict.get("interested_ligand_mask", None)
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=pred_dict["pae"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=True,
            mol_id=(input_feature_dict["mol_id"] if mode != "inference" else None),
            elements_one_hot=(
                input_feature_dict["ref_element"] if mode != "inference" else None
            ),
        )

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        symmetric_permutation: SymmetricPermutation,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training loop for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            N_cycle (int): Number of cycles.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        log_dict = {}
        pred_dict = {}

        cache = dict()
        if self.enable_diffusion_shared_vars_cache:
            cache["pair_z"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                input_feature_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                ref_pos=input_feature_dict["ref_pos"],
                ref_charge=input_feature_dict["ref_charge"],
                ref_mask=input_feature_dict["ref_mask"],
                ref_element=input_feature_dict["ref_element"],
                ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                d_lm=input_feature_dict["d_lm"],
                v_lm=input_feature_dict["v_lm"],
                pad_info=input_feature_dict["pad_info"],
                r_l=True,
                z=cache["pair_z"],
                inplace_safe=False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        # Mini-rollout: used for confidence and label permutation
        with torch.no_grad():
            # [..., 1, N_atom, 3]
            N_sample_mini_rollout = self.configs.sample_diffusion[
                "N_sample_mini_rollout"
            ]  # =1
            N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]
            self.diffusion_module.eval()  # use eval mode for mini-rollout
            coordinate_mini = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs.detach(),
                s_trunk=s.detach(),
                z_trunk=None if cache["pair_z"] is not None else z.detach(),
                pair_z=None if cache["pair_z"] is None else cache["pair_z"].detach(),
                p_lm=(
                    None
                    if cache["p_lm/c_l"][0] is None
                    else cache["p_lm/c_l"][0].detach()
                ),
                c_l=(
                    None
                    if cache["p_lm/c_l"][1] is None
                    else cache["p_lm/c_l"][1].detach()
                ),
                N_sample=N_sample_mini_rollout,
                noise_schedule=self.inference_noise_scheduler(
                    N_step=N_step_mini_rollout,
                    device=s_inputs.device,
                    dtype=s_inputs.dtype,
                ),
                enable_efficient_fusion=self.enable_efficient_fusion,
            )
            self.diffusion_module.train()
            coordinate_mini.detach_()
            pred_dict["coordinate_mini"] = coordinate_mini

            # Permute ground truth to match mini-rollout prediction
            (
                label_dict,
                perm_log_dict,
            ) = symmetric_permutation.permute_label_to_match_mini_rollout(
                coordinate_mini,
                input_feature_dict,
                label_dict,
                label_full_dict,
            )
            log_dict.update(perm_log_dict)

        # Confidence: use mini-rollout prediction, and detach token embeddings
        drop_embedding = (
            random.random() < self.configs.model.confidence_embedding_drop_rate
        )
        plddt_pred, pae_pred, pde_pred, resolved_pred = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=coordinate_mini,
            use_embedding=not drop_embedding,
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        pred_dict.update(
            {
                "plddt": plddt_pred,
                "pae": pae_pred,
                "pde": pde_pred,
                "resolved": resolved_pred,
            }
        )

        if self.train_confidence_only:
            # Skip diffusion loss and distogram loss. Return now.
            return pred_dict, label_dict, log_dict

        # Denoising: use permuted coords to generate noisy samples and perform denoising
        # x_denoised: [..., N_sample, N_atom, 3]
        # x_noise_level: [..., N_sample]
        N_sample = self.diffusion_batch_size
        drop_conditioning = (
            random.random() < self.configs.model.condition_embedding_drop_rate
        )
        _, x_denoised, x_noise_level = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            noise_sampler=self.train_noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            diffusion_chunk_size=self.configs.diffusion_chunk_size,
            use_conditioning=not drop_conditioning,
            enable_efficient_fusion=self.enable_efficient_fusion,
        )
        pred_dict.update(
            {
                "distogram": autocasting_disable_decorator(True)(self.distogram_head)(
                    z
                ),
                # [..., N_sample=48, N_atom, 3]: diffusion loss
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
            }
        )

        # Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        (
            pred_dict,
            perm_log_dict,
            _,
            _,
        ) = symmetric_permutation.permute_diffusion_sample_to_match_label(
            input_feature_dict, pred_dict, label_dict, stage="train"
        )
        log_dict.update(perm_log_dict)
        log_dict.update({"noise_level": x_noise_level})

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
        disable_inplace: bool = False,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "eval", "inference"]
        not_use_gradient = not (self.training or torch.is_grad_enabled())
        inplace_safe = not_use_gradient and (not disable_inplace)

        input_feature_dict = self.relative_position_encoding.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            assert self.training
            assert label_dict is not None
            assert symmetric_permutation is not None

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=None,
            )
            log_dict["N_cycle"] = N_cycle
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})
        elif mode == "eval":
            if label_dict is not None:
                assert (
                    label_dict["coordinate"].size()
                    == label_full_dict["coordinate"].size()
                )
                label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=1,
                symmetric_permutation=symmetric_permutation,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
