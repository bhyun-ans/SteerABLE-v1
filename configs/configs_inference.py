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

from protenix.config.extend_types import DefaultNoneWithType, ListValue, RequiredValue

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
    # Trajectory visualization
    "save_trajectory": False,  # save diffusion trajectory as GIF (every 10 steps)
    # Reward-guided embedding steering (Ab–Ag mode)
    "ab_chains": DefaultNoneWithType(str),  # comma-separated antibody chain letters, e.g. "A,B"; Ag = all other input chains
    "epitope_residue": DefaultNoneWithType(str),  # epitope residues on Ag chains, e.g. "C:45,C:48,D:52"
    "guidance": {
        # Constant-alpha override: None = use the cosine schedule below;
        # float (e.g. 0.1) = constant alpha for every step (overrides the schedule).
        "guidance_alpha": DefaultNoneWithType(float),
        # --- Truncated raised-cosine alpha schedule (flow-matching paper form) ---
        # Active whenever guidance_alpha is None. See protenix/utils/alpha_schedule.py.
        #   omega(s) = alpha_init * 0.5*(1 + cos(pi*s))  for s < alpha_trunc, else 0
        # with s in [0,1] generation progress (s=0 noise/start, s=1 clean/end).
        "alpha_init": 1.0,    # omega_init: peak weight at s=0 (paper grid-search value)
        "alpha_trunc": 0.5,   # tau_trunc: progress fraction where alpha -> 0 ("when to turn alpha off"; 0.5 = step 100/200)
        "d0": 4.0,
        "softmin_beta": 10.0,
        "top_k": DefaultNoneWithType(int),  # None = all atoms per epitope residue; positive int = top-k atoms per residue (k=1 = closest atom only)
        "lambda_clash": 0.0,  # 0 = clash term off; positive float = weight of epitope-local clash penalty
        "clash_tau": 1.5,  # vdW overlap tolerance in Ångströms (AF3-style)
        "clash_start_step": 0,  # diffusion step at which lambda_clash kicks in (0 = from the beginning)
    },
}
