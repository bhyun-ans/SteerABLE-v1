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

"""
Reward-guided embedding steering for diffusion sampling (Ab–Ag specialised).

Core idea: during reverse diffusion, steer trunk embeddings (s, z) via the
gradient of a contact-based reward that encourages epitope (antigen)
residues to make contacts with antibody atoms.  Unlike the generic
hotspot-on-any-chain mode, the partner-atom mask is restricted to the
antibody chains supplied via ``ab_chain_ids``, so antigen-antigen
inter-chain contacts are excluded — necessary for homo-multimeric antigens.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def parse_chain_list(chain_str: str) -> set[int]:
    """Parse comma-separated chain letters into a set of 0-based asym_ids.

    Format: ``"A,B"`` → ``{0, 1}``.  Chain letters map to 0-based asym_id
    in the usual A=0, B=1, C=2, … scheme — i.e. the position of the
    corresponding chain in the input JSON's ``sequences`` list.

    Args:
        chain_str: Comma-separated single-letter chain identifiers.

    Returns:
        Set of asym_id integers.

    Raises:
        ValueError: On malformed input or empty result.
    """
    chains: set[int] = set()
    for c in chain_str.split(","):
        c = c.strip()
        if not c:
            continue
        if len(c) != 1 or not c.isalpha():
            raise ValueError(
                f"Invalid chain '{c}' in '{chain_str}'. "
                "Expected a single letter (A, B, C, ...)."
            )
        chains.add(ord(c.upper()) - ord("A"))
    if not chains:
        raise ValueError(f"No chains parsed from '{chain_str}'.")
    return chains


def validate_and_infer_ag_chains(
    input_feature_dict: dict[str, Any],
    ab_chain_ids: set[int],
) -> set[int]:
    """Validate ``ab_chain_ids`` against the input and return the Ag complement.

    Ab/Ag is a binary partition — every chain in the input that is **not**
    in ``ab_chain_ids`` is treated as antigen.  This helper just makes the
    partition explicit and catches user mistakes:

      * ``ab_chain_ids`` must be non-empty.
      * Every asym_id in ``ab_chain_ids`` must actually exist in the input
        (catches typos like declaring chain "G" when input has only A-F).
      * At least one antigen chain must remain after removing Ab
        (otherwise there's nothing for the epitope reward to point at).

    Logs the resolved Ab and Ag chain letters at INFO so the user can
    sanity-check the partition visually.

    Args:
        input_feature_dict: Must contain ``asym_id`` [N_token].
        ab_chain_ids: 0-based asym_ids declared as antibody chains.

    Returns:
        ``ag_chain_ids`` — the complement of ``ab_chain_ids`` within the
        input's chain set.

    Raises:
        ValueError: On any of the conditions above.
    """
    if not ab_chain_ids:
        raise ValueError("ab_chain_ids is empty — at least one antibody chain required.")

    asym_id = input_feature_dict["asym_id"]
    present = {int(x) for x in asym_id.unique().tolist()}

    extra = ab_chain_ids - present
    if extra:
        chains = ", ".join(chr(ord("A") + c) for c in sorted(extra))
        raise ValueError(
            f"ab_chains references chain(s) {chains} (asym_ids {sorted(extra)}) "
            f"not present in input (present: {sorted(present)})."
        )

    ag_chain_ids = present - ab_chain_ids
    if not ag_chain_ids:
        raise ValueError(
            f"All input chains were declared as Ab ({sorted(ab_chain_ids)}); "
            "no antigen chains remain."
        )

    ab_str = ", ".join(chr(ord("A") + c) for c in sorted(ab_chain_ids))
    ag_str = ", ".join(chr(ord("A") + c) for c in sorted(ag_chain_ids))
    logger.info("Chain partition — Ab: %s | Ag: %s", ab_str, ag_str)

    return ag_chain_ids


def parse_epitope_residue(epitope_str: str) -> list[tuple[int, int]]:
    """Parse epitope residue string into (chain_asym_id, residue_number) pairs.

    Format: ``"C:45,C:48,D:52"``.  Chain letter maps to 0-based asym_id:
    A=0, B=1, C=2, …  Each epitope residue lives on an antigen chain
    (see :func:`parse_chain_list` and :func:`build_guidance_masks` for
    Ab/Ag separation semantics).

    Args:
        epitope_str: Comma-separated ``"CHAIN:RESIDUE"`` entries.

    Returns:
        List of ``(asym_id, residue_index)`` tuples.

    Raises:
        ValueError: On malformed input.
    """
    epitopes: list[tuple[int, int]] = []
    for entry in epitope_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"Invalid epitope format '{entry}'. "
                "Expected 'CHAIN:RESIDUE' (e.g., 'C:45')."
            )
        chain_str, res_str = entry.split(":", 1)
        chain_str = chain_str.strip()
        res_str = res_str.strip()

        if len(chain_str) != 1 or not chain_str.isalpha():
            raise ValueError(
                f"Invalid chain '{chain_str}' in '{entry}'. "
                "Expected a single letter (A, B, C, ...)."
            )
        try:
            res_num = int(res_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid residue number '{res_str}' in '{entry}'. "
                "Expected an integer."
            ) from exc

        asym_id = ord(chain_str.upper()) - ord("A")
        epitopes.append((asym_id, res_num))

    if not epitopes:
        raise ValueError(f"No valid epitope residues parsed from '{epitope_str}'.")
    return epitopes


def build_guidance_masks(
    input_feature_dict: dict[str, Any],
    epitope_residues: list[tuple[int, int]],
    ab_chain_ids: set[int],
) -> list[tuple[list[torch.Tensor], torch.Tensor]]:
    """Build per-Ag-chain epitope residue groups with an antibody partner mask.

    Epitope residues live on antigen chains; reward is computed against
    antibody atoms only.  The partner mask is therefore the **same**
    across every Ag chain group — every atom whose asym_id is in
    ``ab_chain_ids``.  Ag-Ag inter-chain contacts (e.g. homo-tetrameric
    antigens) are excluded by construction.

    Epitope residues are still grouped by chain so that the downstream
    reward can be aggregated per residue / per Ag chain.

    Epitope entries whose chain is in ``ab_chain_ids`` are skipped with a
    warning, since by definition the epitope is on the antigen.

    Args:
        input_feature_dict: Must contain ``asym_id`` [N_token],
            ``residue_index`` [N_token], ``atom_to_token_idx`` [N_atom].
        epitope_residues: Output of :func:`parse_epitope_residue`.
        ab_chain_ids: 0-based asym_ids representing antibody chains.  Atoms
            in these chains form the partner pool against which the
            reward is computed.

    Returns:
        List of ``(residue_groups, partner_atom_mask)`` pairs, one per
        Ag chain that has at least one matched epitope residue.
        ``residue_groups`` is ``list[LongTensor]`` where each tensor holds
        the atom indices for one epitope residue.  Every pair shares the
        same ``partner_atom_mask`` (the antibody-atom mask).

    Raises:
        ValueError: If ``ab_chain_ids`` selects no atoms, or if zero
            epitope residues match the feature dict.
    """
    asym_id = input_feature_dict["asym_id"]  # [N_token]
    residue_index = input_feature_dict["residue_index"]  # [N_token]
    atom_to_token_idx = input_feature_dict["atom_to_token_idx"]  # [N_atom]

    atom_chain = asym_id[atom_to_token_idx]  # [N_atom]

    # Antibody-atom mask (shared across all Ag chain groups).
    ab_id_tensor = torch.tensor(
        sorted(ab_chain_ids), device=asym_id.device, dtype=atom_chain.dtype
    )
    partner_atom_mask = torch.isin(atom_chain, ab_id_tensor)  # [N_atom] bool
    n_ab_atoms = int(partner_atom_mask.sum().item())
    if n_ab_atoms == 0:
        raise ValueError(
            f"ab_chain_ids={sorted(ab_chain_ids)} matches no atoms. "
            f"Available asym_ids: {asym_id.unique().tolist()}."
        )

    # Group epitope residues by chain (Ag chains only).
    chain_to_residues: dict[int, list[int]] = {}
    for chain_id, res_num in epitope_residues:
        if chain_id in ab_chain_ids:
            logger.warning(
                "Epitope residue on antibody chain %s (asym_id=%d) skipped — "
                "epitopes must live on antigen chains.",
                chr(ord("A") + chain_id), chain_id,
            )
            continue
        chain_to_residues.setdefault(chain_id, []).append(res_num)

    mask_pairs: list[tuple[list[torch.Tensor], torch.Tensor]] = []
    total_matched = 0

    for chain_id, res_nums in chain_to_residues.items():
        residue_groups: list[torch.Tensor] = []
        chain_matched = 0
        for res_num in res_nums:
            token_match = (asym_id == chain_id) & (residue_index == res_num)
            if not token_match.any():
                continue
            chain_matched += 1
            # Atoms whose token belongs to this residue
            is_atom_in_residue = token_match[atom_to_token_idx]  # [N_atom] bool
            atom_indices = torch.nonzero(is_atom_in_residue, as_tuple=False).squeeze(-1)
            residue_groups.append(atom_indices)

        if chain_matched == 0:
            logger.warning(
                "Ag chain %s (asym_id=%d): none of residues %s matched.",
                chr(ord("A") + chain_id), chain_id, res_nums,
            )
            continue

        total_matched += chain_matched

        n_h_total = sum(int(g.numel()) for g in residue_groups)
        atom_counts = [int(g.numel()) for g in residue_groups]
        logger.info(
            "Ag chain %s (asym_id=%d): %d epitope residues (atoms per residue: %s, total %d), %d Ab partner atoms.",
            chr(ord("A") + chain_id), chain_id, chain_matched, atom_counts, n_h_total, n_ab_atoms,
        )
        mask_pairs.append((residue_groups, partner_atom_mask))

    if total_matched == 0:
        raise ValueError(
            f"No epitope residues matched. Requested: {epitope_residues}. "
            f"Available asym_ids: {asym_id.unique().tolist()}, "
            f"residue_index range: [{residue_index.min().item()}, "
            f"{residue_index.max().item()}]."
        )
    if total_matched < len(epitope_residues):
        logger.warning(
            "Only %d / %d epitope residues matched the input features.",
            total_matched, len(epitope_residues),
        )

    return mask_pairs


def _single_chain_reward(
    coords_flat: torch.Tensor,
    residue_groups: list[torch.Tensor],
    partner_mask: torch.Tensor,
    d0: float,
    softmin_beta: float,
    eps: float,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Reward for one Ag chain group, aggregating per residue.  Internal helper.

    For each epitope residue:
      1. Compute softmin distance to partner atoms for each atom in the residue.
      2. Select the ``top_k`` atoms with the smallest softmin distance.
         If ``top_k`` is ``None`` or larger than the residue's atom count,
         use all atoms.
      3. Penalty contribution = ``Σ softplus(softmin_d - d0)²`` over selected atoms.

    Args:
        top_k: How many atoms per residue to include in the penalty.
            ``None`` = use all atoms (original behaviour).
            Falls back to "all" if a residue has fewer atoms than ``top_k``.
    """
    if not residue_groups or not partner_mask.any():
        return torch.tensor(0.0, device=coords_flat.device, dtype=coords_flat.dtype)

    p_coords = coords_flat[:, partner_mask]  # [B, N_p, 3]
    B = coords_flat.shape[0]

    total_penalty = torch.zeros(B, device=coords_flat.device, dtype=coords_flat.dtype)

    for atom_indices in residue_groups:
        if atom_indices.numel() == 0:
            continue

        # Atoms of this residue: [B, n_atoms, 3]
        r_coords = coords_flat[:, atom_indices]
        n_atoms = int(atom_indices.shape[0])

        # Pairwise distances to partner atoms [B, n_atoms, N_p]
        diff = r_coords.unsqueeze(-2) - p_coords.unsqueeze(-3)
        dists = torch.sqrt((diff * diff).sum(dim=-1) + eps)

        # Softmin over partner atoms for each atom [B, n_atoms]
        softmin_d = -1.0 / softmin_beta * torch.logsumexp(
            -softmin_beta * dists, dim=-1
        )

        # Select top-k atoms with smallest softmin distance.
        # Fall back to "use all" when k is None or exceeds n_atoms.
        if top_k is not None and top_k < n_atoms:
            selected, _ = torch.topk(softmin_d, k=top_k, dim=-1, largest=False)  # [B, k]
        else:
            selected = softmin_d  # [B, n_atoms]

        penalty = F.softplus(selected - d0)
        total_penalty = total_penalty + (penalty ** 2).sum(dim=-1)  # [B]

    return -total_penalty.mean()


def contact_epitope_reward(
    coords: torch.Tensor,
    mask_pairs: list[tuple[list[torch.Tensor], torch.Tensor]],
    d0: float = 4.0,
    softmin_beta: float = 10.0,
    eps: float = 1e-8,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Differentiable contact reward for epitope ↔ antibody atoms.

    Each Ag chain group in ``mask_pairs`` shares the same partner mask
    (antibody atoms) constructed by :func:`build_guidance_masks`, so the
    reward only encourages epitope-Ab contacts — not Ag-Ag inter-chain
    contacts, which is essential for multi-Ag complexes.

    Per residue:
        For each atom a in the epitope residue:
            softmin_d(a) = -1/β · logsumexp(-β · ‖x_a - x_p‖,  p ∈ Ab atoms)
        Select the ``top_k`` atoms with smallest softmin_d (or all if
        the residue has fewer atoms).
        R_residue = - Σ_{a ∈ topk} softplus(softmin_d(a) - d0)²

    Total reward = Σ over epitope residues across all Ag chain groups.

    Args:
        coords: Atom coordinates [..., N_atom, 3].
        mask_pairs: List of ``(residue_groups, partner_atom_mask)`` tuples,
            one per Ag chain group (output of :func:`build_guidance_masks`).
            ``residue_groups`` is ``list[LongTensor]``: one tensor of atom
            indices per epitope residue.
        d0: Contact threshold in Ångströms.
        softmin_beta: Inverse temperature for soft-minimum.
        eps: Numerical stability constant.
        top_k: How many atoms per epitope residue to include in the
            penalty (selected by smallest softmin distance).
            ``None`` = use all atoms in each residue.
            Falls back to "all" if a residue has fewer atoms than ``top_k``.

    Returns:
        Scalar reward (higher = better contact).
    """
    if not mask_pairs:
        return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

    orig_shape = coords.shape
    coords_flat = coords.reshape(-1, orig_shape[-2], orig_shape[-1])  # [B, N_atom, 3]

    total_reward = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
    for residue_groups, partner_mask in mask_pairs:
        total_reward = total_reward + _single_chain_reward(
            coords_flat, residue_groups, partner_mask, d0, softmin_beta, eps, top_k,
        )
    return total_reward


def _single_chain_clash_penalty(
    coords_flat: torch.Tensor,
    residue_groups: list[torch.Tensor],
    partner_mask: torch.Tensor,
    vdw_radii: torch.Tensor,
    tau: float,
    eps: float,
) -> torch.Tensor:
    """Epitope-local inter-chain clash penalty for one Ag chain group (C2).

    For each epitope residue, for each atom a in the residue, measure how
    far inside the vdW-overlap region each partner (Ab) atom p sits:
        arg(a, p) = (r_a + r_p − τ) − d(a, p)       # > 0 ⇔ overlap
        penalty  += Σ_{a, p} softplus(arg(a, p))²

    ``softplus`` mirrors the existing contact reward: smooth hinge, grows
    roughly linearly on the violating side and decays toward 0 on the
    safe side.  The squared wrap scales penalty with violation depth
    while keeping a C¹ surface everywhere.

    Returns a scalar ≥ 0.  Caller negates it to form a reward.
    """
    if not residue_groups or not partner_mask.any():
        return torch.tensor(0.0, device=coords_flat.device, dtype=coords_flat.dtype)

    p_coords = coords_flat[:, partner_mask]                          # [B, N_p, 3]
    p_radii = vdw_radii[partner_mask].to(coords_flat.dtype)          # [N_p]
    B = coords_flat.shape[0]

    total = torch.zeros(B, device=coords_flat.device, dtype=coords_flat.dtype)

    for atom_indices in residue_groups:
        if atom_indices.numel() == 0:
            continue

        r_coords = coords_flat[:, atom_indices]                       # [B, n_a, 3]
        r_radii = vdw_radii[atom_indices].to(coords_flat.dtype)       # [n_a]

        diff = r_coords.unsqueeze(-2) - p_coords.unsqueeze(-3)        # [B, n_a, N_p, 3]
        dists = torch.sqrt((diff * diff).sum(dim=-1) + eps)           # [B, n_a, N_p]

        vdw_sum = r_radii.unsqueeze(-1) + p_radii.unsqueeze(0)        # [n_a, N_p]
        penalty = F.softplus(vdw_sum - tau - dists) ** 2              # [B, n_a, N_p]

        total = total + penalty.sum(dim=(-2, -1))                     # [B]

    return total.mean()


def epitope_clash_reward(
    coords: torch.Tensor,
    mask_pairs: list[tuple[list[torch.Tensor], torch.Tensor]],
    ref_element: torch.Tensor,
    tau: float = 1.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable epitope-local clash reward (C2).

    Penalises overlap between epitope-residue atoms and antibody atoms
    when ``d_ij < r_i + r_j − τ``.  Per-atom vdW radii are looked up inside
    this function from the rdkit table already built into
    ``protenix.metrics.clash``.

    Args:
        coords: Atom coordinates [..., N_atom, 3].
        mask_pairs: Output of :func:`build_guidance_masks`.
        ref_element: Per-atom one-hot element feature [N_atom, 128], as in
            ``input_feature_dict["ref_element"]``.
        tau: Overlap tolerance in Ångströms (default 1.5, AF3-style).
        eps: Numerical stability for distance sqrt.

    Returns:
        Scalar reward (≤ 0; higher = less clash).
    """
    if not mask_pairs:
        return torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

    from protenix.metrics.clash import get_vdw_radii
    vdw_radii = get_vdw_radii(ref_element).to(coords.device)          # [N_atom]

    orig_shape = coords.shape
    coords_flat = coords.reshape(-1, orig_shape[-2], orig_shape[-1])  # [B, N_atom, 3]

    total_penalty = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
    for residue_groups, partner_mask in mask_pairs:
        total_penalty = total_penalty + _single_chain_clash_penalty(
            coords_flat, residue_groups, partner_mask, vdw_radii, tau, eps,
        )
    return -total_penalty


def rms_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """RMS-normalise a tensor (global RMS across all elements).

    Args:
        x: Tensor of arbitrary shape.
        eps: Stability constant.

    Returns:
        Tensor of the same shape with unit RMS.
    """
    rms = torch.sqrt(torch.mean(x * x) + eps)
    return x / rms


