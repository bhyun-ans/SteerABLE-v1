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

# pylint: disable=C0114
import os
from pathlib import Path

from protenix.config.extend_types import (
    DefaultNoneWithType,
    ListValue,
    RequiredValue,
    ValueMaybeNone,
)

PROTENIX_ROOT_DIR = os.environ.get("PROTENIX_ROOT_DIR", str(Path.home()))
inference_configs = {
    "model_name": "protenix_base_default_v1.0.0",  # inference model selection
    "seeds": ListValue([101]),
    "dump_dir": "./output",
    "need_atom_confidence": False,
    "sorted_by_ranking_score": True,
    "input_json_path": RequiredValue(str),
    "load_checkpoint_dir": os.path.join(PROTENIX_ROOT_DIR, "checkpoint"),
    "num_workers": 0,
    "use_msa": True,
    "enable_tf32": True,
    "enable_efficient_fusion": True,
    "enable_diffusion_shared_vars_cache": True,
    "msa_pair_as_unpair": True,
    "use_template": False,
    "use_rna_msa": False,
    "use_seeds_in_json": False,
    # -------------------------------------------------------------------
    # SteerABLE epitope-guidance controls
    # -------------------------------------------------------------------
    # `epitope_residue` acts as the on/off switch: when provided, epitope-guided
    # embedding steering is automatically active with the defaults below. Set
    # `ab_chains` to declare which chains are the antibody (heavy+light); the
    # antigen (partner) chains are inferred as the complement.
    #
    # The `epitope.*` defaults below are SteerABLE's recommended setting and are
    # what the released benchmarks were produced with. Two further parts of that
    # recommended setting live in configs shared with upstream Protenix and are
    # therefore NOT defaulted here -- pass them on the command line:
    #
    #   --sample_diffusion.guidance.enable true    # TFG (upstream default: off)
    #   --dtype fp32                               # (upstream default: bf16)
    #
    # See examples/steerable/7yds/run.sh for a ready-to-run invocation.
    #
    # Example CLI:
    #   --epitope_residue "C:45,C:48,D:52" --ab_chains "A,B"
    #   --epitope_residue "C:45,C:48;D:52,D:53" --ab_chains "A,B"
    #                                          # ';' separates epitope SETS.
    #                                          # The trunk (pairformer +
    #                                          # distogram) runs ONCE; each set
    #                                          # gets its own steered diffusion
    #                                          # branch (steerable_0,
    #                                          # steerable_1, ..) from that same
    #                                          # trunk, RNG-paired with each
    #                                          # other and with raw.
    #   [optional overrides]
    #   --epitope.guidance_alpha 0.05         # constant alpha (default 0.1)
    #   --epitope.guidance_alpha null --epitope.alpha_init 0.1 --epitope.alpha_trunc 0.5
    #                                          # switch to cosine-truncated schedule
    #   --epitope.lambda_clash 0.0            # disable the Ab-Ag clash penalty
    #   --epitope.guidance_interval 1         # steer on every diffusion step
    "epitope_residue": DefaultNoneWithType(str),   # e.g. "C:45,C:48,D:52"; None disables guidance
    "ab_chains": DefaultNoneWithType(str),          # e.g. "A,B" (required if epitope_residue is set)
    "epitope": {
        # guidance strength -- non-None => constant alpha; None => cosine schedule
        # Pass e.g. `--epitope.guidance_alpha null` on the CLI to nullify.
        # alpha = 1.0 (the flow-matching paper value) over-steers by ~10x in
        # this RMS-normalized regime; 0.1 is the calibrated value.
        "guidance_alpha": ValueMaybeNone(0.1),
        # cosine schedule params (used only when guidance_alpha is None)
        "alpha_init": 0.1,
        "alpha_trunc": 0.5,     # turns alpha off at step (alpha_trunc * N_step)
        # contact reward geometry
        "d0": 4.0,               # target contact distance (A)
        "softmin_beta": 10.0,    # sharper -> picks the single closest partner atom
        # Ab-Ag clash penalty. 0 disables it.
        "lambda_clash": 0.1,
        "clash_tau": 1.5,        # clash softness (A)
        "clash_start_step": 0,   # start step for clash penalty
        # ---- how OFTEN the reward gradient is computed and applied ----
        # k runs the grad-enabled denoise + reward + autograd.grad + (s_tau,
        # z_tau) update only on steps 0, k, 2k, ...; the steps in between reuse
        # the embeddings left by the last steered step and cost one plain
        # no-grad denoise instead of forward + backward. That is the runtime
        # win: the reward backward through the denoiser is the single most
        # expensive thing in the guided sampler. At the default k=8 and the
        # usual N_step=200, 25 of the 200 steps steer.
        #
        # NOT purely a wall-clock knob: the per-update step size stays
        # `guidance_alpha`, so the TOTAL steering applied over the trajectory
        # also scales as ~1/k. k=8 with alpha=0.1 is the calibrated pair -- if
        # you change one, re-validate rather than assuming the other still
        # holds. k=1 steers on every step (the original, ~8x more total
        # steering and a much slower run). Must be an int >= 1; `null`, `0`,
        # `-3` and `2.5` all fail loudly.
        "guidance_interval": 8,
    },
    # -------------------------------------------------------------------
    # Distogram routing gate (raw Protenix vs SteerABLE steering)
    # -------------------------------------------------------------------
    # The pairformer trunk already decides where the antibody sits, and the
    # distogram exposes it: score every antigen token by its best antibody
    # contact probability, e_j = max_{i in Ab} C_ij, then
    #     enrichment = mean(e_j | j in epitope) / mean(e_j | j elsewhere)
    # ~1 => no epitope signal in the trunk, >1 => the signal is already there
    # (and raw Protenix tends to be good on its own).
    #
    # Enrichment is computed and dumped whenever `epitope_residue` is set --
    # `mode` only decides which diffusion branch(es) actually run:
    #   steerable (default) steered only. One diffusion per epitope set.
    #   both      run BOTH the steered and the no-guidance diffusion on the
    #             same trunk and keep both outputs. The steered run keeps the
    #             normal output layout; the no-guidance run is dumped under
    #             <dump_dir>/raw/. No routing decision is made. Costs ~2x.
    #   route     run exactly ONE branch: enrichment >= threshold
    #             => raw Protenix, else SteerABLE steering. Requires `threshold`.
    #   raw       no-guidance only (still reports enrichment).
    # Without `epitope_residue` there is nothing to steer or score, so the run
    # behaves exactly like upstream Protenix whatever `mode` says.
    #
    # Each run dir gets predictions/<name>_gating.json (enrichment + per-branch
    # wall-clock) and, while `save_contact_probs` is on, predictions/
    # <name>_contact_probs.npz with the [N_token, N_token] contact map, the
    # per-token e_j and the Ab / epitope token masks.
    #
    # With several epitope sets (';' in `epitope_residue`) and more than one
    # branch, EVERY branch is dumped under its own sub-directory
    # (<dump_dir>/steerable_<k>/... per set, <dump_dir>/raw/...) and the
    # target's root directory holds only the trunk-level side-cars: the gate
    # report with a per-set `epitopes` list (branch, residues, enrichment,
    # steering trace) and the contact map, whose `epitope_token_mask` is then
    # [K, N_token].
    "gating": {
        "mode": "steerable",                       # steerable | both | route | raw
        "threshold": DefaultNoneWithType(float),   # required by mode=route
        "save_contact_probs": True,                # dump the N x N contact map
    },
    # -------------------------------------------------------------------
    # Activation checkpointing safety net
    # -------------------------------------------------------------------
    # `--blocks_per_ckpt null` keeps activations instead of recomputing them,
    # which buys 1.4-2.3x but costs memory: measured on a 48 GB A6000 it fits at
    # 729 tokens and does not at 936. With this flag on, a target above
    # ~800 tokens gets checkpointing restored up front so it completes instead
    # of OOM-ing. Set it to false to reproduce the published runs, which had no
    # such restore (see the reproduction section of the README).
    "auto_restore_activation_checkpointing": True,
}
