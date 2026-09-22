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

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from biotite.structure import AtomArray

from protenix.data.utils import save_structure_cif
from protenix.utils.file_io import save_json
from protenix.utils.torch_utils import round_values


def get_clean_full_confidence(full_confidence_dict: dict) -> dict:
    """
    Clean and format the full confidence dictionary by removing
    unnecessary keys and rounding values.

    Args:
        full_confidence_dict (dict): The dictionary containing full confidence data.

    Returns:
        dict: The cleaned and formatted dictionary.
    """
    # Remove atom_coordinate
    full_confidence_dict.pop("atom_coordinate")
    # Remove atom_is_polymer
    full_confidence_dict.pop("atom_is_polymer")
    # Keep two decimal places
    full_confidence_dict = round_values(full_confidence_dict)
    return full_confidence_dict


class DataDumper:
    """
    Class for dumping prediction data, including structure coordinates and confidence scores.

    Args:
        base_dir (str): Base directory for saving dumped data.
        need_atom_confidence (bool): Whether to save detailed atom-level confidence data.
        sorted_by_ranking_score (bool): Whether to sort output files by ranking score.
    """

    def __init__(
        self,
        base_dir: str,
        need_atom_confidence: bool = False,
        sorted_by_ranking_score: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.need_atom_confidence = need_atom_confidence
        self.sorted_by_ranking_score = sorted_by_ranking_score

    def dump(
        self,
        dataset_name: str,
        pdb_id: str,
        seed: int,
        pred_dict: dict,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
    ):
        """
        Dump the predictions and related data to the specified directory.

        Args:
            dataset_name (str): The name of the dataset.
            pdb_id (str): The PDB ID of the sample.
            seed (int): The seed used for randomization.
            pred_dict (dict): The dictionary containing the predictions.
            atom_array (AtomArray): The AtomArray object containing the structure data.
            entity_poly_type (dict[str, str]): The entity poly type information.
        """
        dump_dir = self._get_dump_dir(dataset_name, pdb_id, seed)
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

        self.dump_predictions(
            pred_dict=pred_dict,
            dump_dir=dump_dir,
            pdb_id=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            seed=seed,
        )

    def _get_dump_dir(self, dataset_name: str, sample_name: str, seed: int) -> str:
        """
        Generate the directory path for dumping data based on the dataset
        name, sample name, and seed.
        """
        dump_dir = os.path.join(
            self.base_dir, dataset_name, sample_name, f"seed_{seed}"
        )
        return dump_dir

    def dump_predictions(
        self,
        pred_dict: dict,
        dump_dir: str,
        pdb_id: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        seed: int,
    ):
        """
        Dump raw predictions from the model.

        Args:
            pred_dict (dict): Prediction results.
            dump_dir (str): Directory where to save the predictions.
            pdb_id (str): PDB ID or sample name.
            atom_array (AtomArray): Reference atom array for structure formatting.
            entity_poly_type (dict[str, str]): Dictionary mapping entity IDs to their polymer types.
            seed (int): Random seed used for the prediction.
        """
        prediction_save_dir = os.path.join(dump_dir, "predictions")
        os.makedirs(prediction_save_dir, exist_ok=True)

        # Dump structure
        b_factor = None
        if "full_data" in pred_dict:
            all_atom_plddt = []
            # len(pred_dict["full_data"]) == N_sample
            for each_sample_dict in pred_dict["full_data"]:
                if "atom_plddt" in each_sample_dict:
                    # atom_plddt.shape == [N_atom]
                    atom_plddt = each_sample_dict["atom_plddt"]
                    if atom_plddt.dtype == torch.bfloat16:
                        atom_plddt = atom_plddt.to(torch.float32)
                    all_atom_plddt.append(atom_plddt.cpu().numpy() * 100.0)

            if len(all_atom_plddt) == len(pred_dict["full_data"]):
                b_factor = all_atom_plddt
        sorted_indices = self._get_ranker_indices(data=pred_dict)
        self._save_structure(
            pred_coordinates=pred_dict["coordinate"],
            prediction_save_dir=prediction_save_dir,
            sample_name=pdb_id,
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            seed=seed,
            sorted_indices=sorted_indices,
            b_factor=b_factor,
        )
        # Dump trajectory
        if "trajectory" in pred_dict and pred_dict["trajectory"]:
            self._save_trajectory(
                trajectory=pred_dict["trajectory"],
                trajectory_rotations=pred_dict.get("trajectory_rotations"),
                dump_dir=dump_dir,
                sample_name=pdb_id,
                atom_array=atom_array,
                sorted_indices=sorted_indices,
                epitope_residue=pred_dict.get("epitope_residue"),
            )

        # Dump confidence
        self._save_confidence(
            data=pred_dict,
            prediction_save_dir=prediction_save_dir,
            sample_name=pdb_id,
            seed=seed,
            sorted_indices=sorted_indices,
        )

    def _save_structure(
        self,
        pred_coordinates: torch.Tensor,
        prediction_save_dir: str,
        sample_name: str,
        atom_array: AtomArray,
        entity_poly_type: dict[str, str],
        seed: int,
        sorted_indices: Optional[List[int]],
        b_factor: Optional[List[np.ndarray]] = None,
    ):
        """
        Save predicted structures to CIF files.

        Args:
            pred_coordinates (torch.Tensor): Predicted coordinates [N_sample, N_atom, 3].
            prediction_save_dir (str): Directory where to save the structures.
            sample_name (str): Sample name.
            atom_array (AtomArray): Template atom array.
            entity_poly_type (dict[str, str]): Entity polymer types.
            seed (int): Prediction seed.
            sorted_indices (Optional[List[int]]): Indices for ranking.
            b_factor (Optional[List[np.ndarray]]): Predicted LDDT scores to be saved as B-factors.
        """
        assert atom_array is not None
        N_sample = pred_coordinates.shape[0]
        if sorted_indices is None:
            sorted_indices = range(N_sample)  # do not rank the output file
        for idx, rank in enumerate(sorted_indices):
            output_fpath = os.path.join(
                prediction_save_dir,
                f"{sample_name}_sample_{rank}.cif",
            )
            if b_factor is not None:
                # b_factor.shape == [N_sample, N_atom]
                atom_array.set_annotation("b_factor", np.round(b_factor[idx], 2))

            save_structure_cif(
                atom_array=atom_array,
                pred_coordinate=pred_coordinates[idx],
                output_fpath=output_fpath,
                entity_poly_type=entity_poly_type,
                pdb_id=sample_name,
            )

    def _save_trajectory(
        self,
        trajectory: list,
        dump_dir: str,
        sample_name: str,
        atom_array: AtomArray,
        trajectory_rotations: Optional[list] = None,
        sorted_indices: Optional[List[int]] = None,
        epitope_residue: Optional[str] = None,
    ):
        """Save diffusion trajectory as a multi-MODEL PDB per sample.

        Each diffusion step becomes one MODEL block. The per-step cumulative
        rotation applied by ``centre_random_augmentation`` is undone via
        ``R_cum^T`` so all MODELs live in a common canonical frame
        (removes the global spin seen in prior GIF renders).

        When ``epitope_residue`` is given (format ``"C:13,C:18,D:56"``), the
        atoms of those residues get b_factor=1.0 (non-epitope=0.0) and a
        companion ``sample_<rank>_view.pml`` script is written that loads the
        PDB and colors the epitope red so the viewer shows it in a distinct
        color out of the box.
        """
        from biotite.structure import AtomArrayStack
        from biotite.structure.io.pdb import PDBFile

        traj_dir = os.path.join(dump_dir, "traj")
        os.makedirs(traj_dir, exist_ok=True)

        n_frames = len(trajectory)
        N_sample = trajectory[0].shape[-3]
        N_atom = atom_array.array_length()
        if sorted_indices is None:
            sorted_indices = list(range(N_sample))

        # Build an epitope atom mask from the atom_array annotations. Shape [N_atom], bool.
        epitope_mask = None
        epitope_sel_pml = None
        if epitope_residue:
            chain_to_resi: dict[str, set[int]] = {}
            for entry in epitope_residue.split(","):
                entry = entry.strip()
                if ":" not in entry:
                    continue
                ch, resi = entry.split(":", 1)
                try:
                    chain_to_resi.setdefault(ch.strip(), set()).add(int(resi.strip()))
                except ValueError:
                    continue
            if chain_to_resi:
                chains = atom_array.chain_id
                resids = atom_array.res_id
                epitope_mask = np.zeros(N_atom, dtype=bool)
                for ch, resi_set in chain_to_resi.items():
                    ch_match = chains == ch
                    for r in resi_set:
                        epitope_mask |= ch_match & (resids == r)
                epitope_sel_pml = " or ".join(
                    f"(chain {ch} and resi {'+'.join(str(r) for r in sorted(rs))})"
                    for ch, rs in chain_to_resi.items()
                )

        for idx, rank in enumerate(sorted_indices):
            coords = np.stack(
                [trajectory[f][..., idx, :, :].numpy().astype(np.float32)
                 for f in range(n_frames)],
                axis=0,
            )  # [n_frames, N_atom, 3]
            if trajectory_rotations is not None:
                rots = np.stack(
                    [trajectory_rotations[f][..., idx, :, :].numpy().astype(np.float32)
                     for f in range(n_frames)],
                    axis=0,
                )  # [n_frames, 3, 3]
                # x_world = R_cum @ x_canonical → x_canonical = R_cum^T @ x_world.
                # For row-vectors [N_atom, 3], that's coords @ R_cum.
                coords = np.einsum('fai,fij->faj', coords, rots)

            coords = coords - coords.mean(axis=1, keepdims=True)
            np.clip(coords, -999.0, 9999.0, out=coords)

            stack = AtomArrayStack(depth=n_frames, length=N_atom)
            for cat in atom_array.get_annotation_categories():
                stack.set_annotation(cat, atom_array.get_annotation(cat))
            if atom_array.bonds is not None:
                stack.bonds = atom_array.bonds
            stack.coord = coords
            if epitope_mask is not None:
                b_marker = np.where(epitope_mask, 1.0, 0.0).astype(np.float32)
                stack.set_annotation("b_factor", b_marker)

            out_path = os.path.join(traj_dir, f"sample_{rank}.pdb")
            pdb = PDBFile()
            pdb.set_structure(stack)
            pdb.write(out_path)

            if epitope_sel_pml is not None:
                pml_path = os.path.join(traj_dir, f"sample_{rank}_view.pml")
                with open(pml_path, "w") as f:
                    f.write(
                        f"load {os.path.basename(out_path)}, traj\n"
                        "hide everything\n"
                        "show cartoon\n"
                        "util.cbc\n"
                        f"select epitope,{epitope_sel_pml}\n"
                        "color red, epitope\n"
                        "show sticks, epitope\n"
                        "set stick_radius, 0.25, epitope\n"
                        "show spheres, epitope and name CA\n"
                        "set sphere_scale, 1.2, epitope and name CA\n"
                        "orient\n"
                        "mplay\n"
                    )

    def _get_ranker_indices(self, data: dict) -> List[int]:
        """
        Get indices for ranking predictions based on their confidence scores.

        Args:
            data (dict): Prediction results containing summary confidence.

        Returns:
            List[int]: List of indices sorted by ranking score.
        """
        N_sample = len(data["summary_confidence"])
        if self.sorted_by_ranking_score:
            value = torch.tensor(
                [
                    data["summary_confidence"][i]["ranking_score"]
                    for i in range(N_sample)
                ]
            )
            sorted_indices = [
                i for i in torch.argsort(torch.argsort(value, descending=True))
            ]
        else:
            sorted_indices = [i for i in range(N_sample)]
        return sorted_indices

    def _save_confidence(
        self,
        data: dict,
        prediction_save_dir: str,
        sample_name: str,
        seed: int,
        sorted_indices: Optional[List[int]],
    ):
        """
        Save confidence data to JSON files.

        Args:
            data (dict): Prediction results containing confidence scores.
            prediction_save_dir (str): Directory where to save the files.
            sample_name (str): Sample name.
            seed (int): Prediction seed.
            sorted_indices (Optional[List[int]]): Indices for ranking.
        """
        N_sample = len(data["summary_confidence"])
        for idx in range(N_sample):
            if self.need_atom_confidence:
                data["full_data"][idx] = get_clean_full_confidence(
                    data["full_data"][idx]
                )
        if sorted_indices is None:
            sorted_indices = range(N_sample)
        reward_history_all = data.get("reward_history")
        for idx, rank in enumerate(sorted_indices):
            output_fpath = os.path.join(
                prediction_save_dir,
                f"{sample_name}_summary_confidence_sample_{rank}.json",
            )
            summary = dict(data["summary_confidence"][idx])  # shallow copy to avoid mutating source
            if reward_history_all is not None and idx < len(reward_history_all):
                summary["reward_history"] = reward_history_all[idx]
            save_json(summary, output_fpath, indent=4)
            if self.need_atom_confidence:
                output_fpath = os.path.join(
                    prediction_save_dir,
                    f"{sample_name}_full_data_sample_{rank}.json",
                )
                save_json(data["full_data"][idx], output_fpath, indent=None)
