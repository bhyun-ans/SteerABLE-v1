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

# ==============================================================================
# SteerABLE inference examples
#
# Run from the repository root. Every command below uses the bundled 7yds
# example, so they work as written after:
#
#     bash examples/steerable/7yds/prepare.sh
#
# The recommended setting is two flags on top of the defaults, because both are
# shared with upstream Protenix and we leave upstream's defaults alone:
#     --use_tfg_guidance true      (upstream default: off)
#     --dtype fp32                 (upstream default: bf16)
# The epitope knobs already carry their recommended values -- guidance_alpha
# 0.1, lambda_clash 0.1, guidance_interval 8, gating.mode steerable -- so they
# only appear below where an example deliberately changes one.
#
# Argument summary (`steerable pred -h` for the full list):
#   -i, --input                    Input JSON file or directory.
#   -o, --out_dir                  Output directory.            [./output]
#   -s, --seeds                    Comma-separated seeds.       [101]
#   -e, --sample                   Samples per seed.            [5]
#   -d, --dtype                    bf16 | fp32.                 [bf16]
#   -n, --model_name               Model checkpoint.            [protenix_base_default_v1.0.0]
#   --use_tfg_guidance             Training-Free Guidance.      [false]
#   --ab_chains                    Antibody/binder chains, e.g. "A,B".
#   --epitope_residue              Epitope, e.g. "C:19,C:51"; ';' separates sets.
#   --epitope_guidance_alpha       Steering step size.          [0.1]
#   --epitope_lambda_clash         Ab-Ag clash penalty weight.  [0.1]
#   --epitope_guidance_interval    Steer every k-th step.       [8]
#   --gating_mode                  steerable | both | route | raw. [steerable]
#
# ENVIRONMENT
#   PROTENIX_ROOT_DIR   where checkpoints are downloaded to (default ~/).
#   LAYERNORM_TYPE=torch   skip the fused CUDA layernorm kernel if it will not
#                          build on your toolchain (see docs/kernels.md).
# ==============================================================================

set -euo pipefail

EX=examples/steerable/7yds
AB="$(cat $EX/ab_chains.txt)"
EPITOPE="$(cat $EX/epitope.txt)"

bash $EX/prepare.sh

# ------------------------------------------------------------------------------
# 1. The recommended setting
# ------------------------------------------------------------------------------
steerable pred \
    -i $EX/7yds.json \
    -o ./test_outputs/steerable/7yds \
    -s 101 \
    -n protenix-v2 \
    --dtype fp32 \
    --use_tfg_guidance true \
    --ab_chains "$AB" \
    --epitope_residue "$EPITOPE"

# ------------------------------------------------------------------------------
# 2. Several epitope hypotheses off ONE shared trunk
#
# ';' separates sets. Each becomes its own branch in steerable_<k>/, all sharing
# the trunk, the initial noise and the per-step noise.
# ------------------------------------------------------------------------------
steerable pred \
    -i $EX/7yds.json \
    -o ./test_outputs/steerable/7yds_multi \
    -s 101 \
    -n protenix-v2 \
    --dtype fp32 \
    --use_tfg_guidance true \
    --ab_chains "A,B" \
    --epitope_residue "C:19,C:51,C:52;C:111,C:112,C:113;C:60,C:61"

# ------------------------------------------------------------------------------
# 3. Steered AND unguided off the same trunk, for a paired comparison
#
# The unguided run lands in raw/. Costs about twice as much.
# ------------------------------------------------------------------------------
steerable pred \
    -i $EX/7yds.json \
    -o ./test_outputs/steerable/7yds_both \
    -s 101 \
    -n protenix-v2 \
    --dtype fp32 \
    --use_tfg_guidance true \
    --gating_mode both \
    --ab_chains "$AB" \
    --epitope_residue "$EPITOPE"

# ------------------------------------------------------------------------------
# 4. No epitope: plain Protenix, unchanged
# ------------------------------------------------------------------------------
steerable pred \
    -i $EX/7yds.json \
    -o ./test_outputs/baseline/7yds \
    -s 101 \
    -n protenix-v2 \
    --dtype fp32

# ------------------------------------------------------------------------------
# 5. Straight to the runner, which exposes every config key
#
# Use this for anything the CLI does not reach, and to reproduce the published
# runs (see the reproduction section of the README).
# ------------------------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python3 runner/inference.py \
    --model_name protenix-v2 \
    --seeds 101 \
    --dump_dir ./test_outputs/steerable/7yds_runner \
    --input_json_path $EX/7yds.json \
    --model.N_cycle 10 \
    --sample_diffusion.N_sample 5 \
    --sample_diffusion.N_step 200 \
    --dtype fp32 \
    --sample_diffusion.guidance.enable true \
    --ab_chains "$AB" \
    --epitope_residue "$EPITOPE"

# ------------------------------------------------------------------------------
# 6. Reproducing the published runs
# ------------------------------------------------------------------------------
python3 runner/inference.py \
    --model_name protenix-v2 \
    --seeds 101 \
    --dump_dir ./test_outputs/steerable/7yds_repro \
    --input_json_path $EX/7yds.json \
    --model.N_cycle 10 \
    --sample_diffusion.N_sample 5 \
    --sample_diffusion.N_step 200 \
    --dtype fp32 \
    --sample_diffusion.guidance.enable true \
    --blocks_per_ckpt null \
    --auto_restore_activation_checkpointing false \
    --gating.save_contact_probs false \
    --ab_chains "$AB" \
    --epitope_residue "$EPITOPE"

echo "All inference examples completed."

# Multi-GPU inference via DDP:
# torchrun \
#     --nproc_per_node $NPROC \
#     --master_addr $WORKER_0_HOST --master_port $WORKER_0_PORT \
#     --node_rank=$ID --nnodes=$WORKER_NUM \
#     runner/inference.py \
#     --seeds 101 --dump_dir ./out --input_json_path $EX/7yds.json \
#     --model.N_cycle 10 --sample_diffusion.N_sample 5 --sample_diffusion.N_step 200
