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

"""Distogram routing gate: raw Protenix vs SteerABLE epitope steering.

The pairformer trunk already commits to an Ab-Ag binding mode before diffusion
starts, and the distogram exposes it.  Turning the distogram into contact
probabilities ``C`` and scoring each antigen token by its best antibody contact

    e_j = max_{i in Ab} C_ij                          (j over antigen tokens)
    enrichment = mean_{j in H} e_j / mean_{j not in H} e_j

with ``H`` the requested epitope tokens gives one scalar per target:

  * ``enrichment == 1`` — the epitope looks like any other patch of antigen
    surface, i.e. the trunk carries no epitope signal.
  * ``enrichment > 1``  — contacts already concentrate on the requested
    epitope, so raw Protenix sampling tends to land on it unaided and steering
    has little left to add.

Because it is a *ratio*, a single sharp peak inside ``H`` is enough to raise it.

The gate reads the ``contact_probs`` tensor the model already computes for
``full_data`` (8 A cutoff), so it costs one max-reduction over an
[N_ab, N_token] block.
"""

import logging
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

# Branch names. The secondary branch also names the sub-directory it is dumped
# into when both branches run (``mode="both"``).
BRANCH_RAW = "raw"
BRANCH_STEERED = "steerable"


def steered_branch_names(n_epitope_sets: int) -> list[str]:
    """Name the steered branch(es) for ``n_epitope_sets`` epitope sets.

    One set uses the plain name ``"steerable"`` and keeps the normal output
    layout.  Several sets get one branch each, ``"steerable_0"``,
    ``"steerable_1"``, ... in the order the sets were given on the CLI; the
    index is the only thing linking a branch directory back to its epitope set,
    so the gate report spells the mapping out.
    """
    if n_epitope_sets < 1:
        raise ValueError(f"n_epitope_sets must be >= 1, got {n_epitope_sets}")
    if n_epitope_sets == 1:
        return [BRANCH_STEERED]
    return [f"{BRANCH_STEERED}_{k}" for k in range(n_epitope_sets)]


def is_steered_branch(name: str) -> bool:
    """True for every steered branch (``steerable``, ``steerable_<k>``), False for ``raw``."""
    return name != BRANCH_RAW


@torch.no_grad()
def trunk_fingerprint(*tensors: torch.Tensor) -> tuple[float, ...]:
    """Cheap identity check for the shared trunk embeddings.

    Every sampling branch must start from the SAME trunk output.  Steering
    works on detached per-sample copies and the steered branches run with
    in-place ops disabled, so nothing should ever write into ``s`` / ``z``
    between branches -- but the no-guidance branch's confidence head does
    ``z_trunk *= 0`` when it is allowed in-place ops, which is exactly why it
    has to run last.  This fingerprint (fp32 sum and L2 norm per tensor) is
    taken once after the trunk and re-checked before each branch, so a branch
    that starts from a modified trunk fails loudly instead of quietly
    producing a "steered" structure of a different model.

    Sum and norm are plain deterministic reductions, so the same tensor gives
    the same fingerprint within a process; two identical fingerprints of
    tensors this size are, for the purpose here, the same tensor.
    """
    out: list[float] = []
    for t in tensors:
        out.append(float(t.sum(dtype=torch.float32).item()))
        out.append(float(torch.linalg.vector_norm(t, dtype=torch.float32).item()))
    return tuple(out)

# `gating.mode` values.
MODE_BOTH = "both"  # run both branches, no routing (for comparison runs)
MODE_ROUTE = "route"  # route by enrichment vs threshold (deployment)
MODE_STEERABLE = "steerable"  # steered only (default)
MODE_RAW = "raw"  # no-guidance only
GATING_MODES = (MODE_BOTH, MODE_ROUTE, MODE_STEERABLE, MODE_RAW)


def compute_epitope_enrichment(
    contact_probs: torch.Tensor,
    asym_id: torch.Tensor,
    residue_index: torch.Tensor,
    epitope_residues: list[tuple[int, int]],
    ab_chain_ids: set[int],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Measure how strongly the trunk's distogram favours the given epitope.

    Antibody / antigen is a binary partition of the input chains: every chain
    not in ``ab_chain_ids`` is antigen.  Epitope tokens are matched by
    ``(asym_id, residue_index)`` exactly as
    :func:`protenix.model.steering.build_guidance_masks` matches them, so the
    gate scores the same residues the reward steers towards.

    Note for homo-multimeric antigens: the background is *all* antigen tokens
    outside ``H``, which includes the symmetry-equivalent copies of the epitope
    on the other antigen chains.  Those copies legitimately carry contact
    signal, so enrichment is diluted for homomers.

    Args:
        contact_probs: [N_token, N_token] contact probabilities (leading
            singleton batch dims are accepted and dropped).
        asym_id: [N_token] 0-based chain index per token.
        residue_index: [N_token] residue number per token.
        epitope_residues: ``(asym_id, residue_number)`` pairs from
            :func:`protenix.model.steering.parse_epitope_residue`.
        ab_chain_ids: 0-based asym_ids of the antibody chains.

    Returns:
        ``(stats, arrays)``.  ``stats`` holds plain Python scalars
        (JSON-dumpable); its ``enrichment`` is ``None`` when there are no
        background antigen tokens to divide by.  ``arrays`` holds the tensors
        the scalars were reduced from — the contact map, the per-token ``e_j``
        and the token masks — for dumping / offline re-analysis.

    Raises:
        ValueError: If no antibody token or no epitope token is present.
    """
    cp = contact_probs
    if cp.dim() > 2:
        # Inference runs one complex at a time; drop leading batch dims.
        cp = cp.reshape(-1, cp.shape[-2], cp.shape[-1])[0]
    cp = cp.detach().float()

    ab_ids = torch.tensor(
        sorted(ab_chain_ids), device=asym_id.device, dtype=asym_id.dtype
    )
    ab_token_mask = torch.isin(asym_id, ab_ids)  # [N_token] bool
    ag_token_mask = ~ab_token_mask

    epitope_token_mask = torch.zeros_like(ab_token_mask)
    n_matched = 0
    for chain_id, res_num in epitope_residues:
        if chain_id in ab_chain_ids:
            # Epitopes live on the antigen; build_guidance_masks warns already.
            continue
        hit = (asym_id == chain_id) & (residue_index == res_num)
        if not bool(hit.any()):
            continue
        n_matched += 1
        epitope_token_mask |= hit
    epitope_token_mask &= ag_token_mask
    background_token_mask = ag_token_mask & ~epitope_token_mask

    n_ab = int(ab_token_mask.sum().item())
    n_epitope = int(epitope_token_mask.sum().item())
    n_background = int(background_token_mask.sum().item())

    if n_ab == 0:
        raise ValueError(
            f"ab_chains={sorted(ab_chain_ids)} matches no token; "
            f"present asym_ids: {asym_id.unique().tolist()}."
        )
    if n_epitope == 0:
        raise ValueError(
            f"None of the epitope residues {epitope_residues} matched an "
            "antigen token, so enrichment is undefined."
        )

    # e_j: best contact probability between token j and any antibody token.
    e_j = cp[ab_token_mask].max(dim=0).values  # [N_token]
    epitope_mean = float(e_j[epitope_token_mask].mean().item())
    background_mean = (
        float(e_j[background_token_mask].mean().item()) if n_background > 0 else 0.0
    )
    enrichment = epitope_mean / background_mean if background_mean > 0.0 else None

    stats = {
        "enrichment": enrichment,
        "epitope_contact_mean": epitope_mean,
        "background_contact_mean": background_mean,
        "n_ab_tokens": n_ab,
        "n_epitope_tokens": n_epitope,
        "n_background_tokens": n_background,
        "n_epitope_residues_requested": len(epitope_residues),
        "n_epitope_residues_matched": n_matched,
    }
    logger.info(
        "Epitope contact enrichment = %s (epitope mean %.4f over %d tokens / "
        "background mean %.4f over %d tokens; %d Ab tokens, %d/%d epitope "
        "residues matched)",
        "n/a" if enrichment is None else f"{enrichment:.3f}",
        epitope_mean,
        n_epitope,
        background_mean,
        n_background,
        n_ab,
        n_matched,
        len(epitope_residues),
    )
    arrays = {
        "contact_probs": cp,  # [N_token, N_token]
        "epitope_contact_score": e_j,  # [N_token] e_j = max_{i in Ab} C_ij
        "ab_token_mask": ab_token_mask,
        "epitope_token_mask": epitope_token_mask,
        "asym_id": asym_id,
        "residue_index": residue_index,
    }
    return stats, arrays


def resolve_branches(
    mode: str,
    epitope_on: bool,
    threshold: Optional[float] = None,
    enrichment: Optional[float] = None,
    n_epitope_sets: int = 1,
) -> tuple[list[str], Optional[str]]:
    """Decide which sampling branch(es) to run.

    Args:
        mode: One of :data:`GATING_MODES` (``gating.mode``).
        epitope_on: Whether epitope guidance is configured at all.  With no
            epitope there is nothing to steer, so every mode collapses to a
            single unguided run.
        threshold: ``gating.threshold``; only read by ``mode="route"``.
        enrichment: Value from :func:`compute_epitope_enrichment`; only read by
            ``mode="route"``.
        n_epitope_sets: How many epitope sets were given (see
            :func:`protenix.model.steering.parse_epitope_sets`).  Each set gets
            its own steered branch, named by :func:`steered_branch_names`.

    Returns:
        ``(branch_names, routed_branch)``.  ``routed_branch`` is the gate's
        decision and is ``None`` unless ``mode="route"`` — the other modes take
        no decision.  The steered branches are listed first so the cheap
        ``raw`` branch is the last to touch the trunk embeddings (it runs with
        in-place ops enabled).

    Raises:
        ValueError: On an unknown mode, ``mode="route"`` without a threshold,
            or ``mode="route"`` with several epitope sets (the gate takes one
            decision per trunk, which several sets do not define).
    """
    if mode not in GATING_MODES:
        raise ValueError(
            f"Unknown gating.mode '{mode}'. Expected one of {list(GATING_MODES)}."
        )
    if not epitope_on or mode == MODE_RAW:
        return [BRANCH_RAW], None
    steered = steered_branch_names(n_epitope_sets)
    if mode == MODE_STEERABLE:
        return steered, None
    if mode == MODE_BOTH:
        return [*steered, BRANCH_RAW], None
    # mode == MODE_ROUTE
    if n_epitope_sets > 1:
        raise ValueError(
            f"gating.mode='route' takes one raw-vs-steered decision per trunk, "
            f"which {n_epitope_sets} epitope sets do not define. Use "
            "gating.mode='both' (every set + raw) or 'steerable' (every set)."
        )
    if threshold is None:
        raise ValueError(
            "gating.mode='route' needs gating.threshold. Leave mode='both' "
            "while calibrating: it runs both branches so a threshold can be "
            "swept against DockQ afterwards."
        )
    if enrichment is None:
        logger.warning(
            "Enrichment is undefined (no background antigen token); routing to %s.",
            BRANCH_STEERED,
        )
        return [BRANCH_STEERED], BRANCH_STEERED
    if enrichment >= float(threshold):
        return [BRANCH_RAW], BRANCH_RAW
    return [BRANCH_STEERED], BRANCH_STEERED
