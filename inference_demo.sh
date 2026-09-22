#!/usr/bin/env bash
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
# SteerABLE-v1 inference examples
#
# Run from the repository root. Every command below uses the bundled 7yds
# example, so they work as written after:
#
#     bash examples/steerable/7yds/prepare.sh
#
# The epitope knobs already carry their defaults -- guidance_alpha 0.1,
# lambda_clash 0.1, guidance_interval 8, gating.mode steerable -- so they only
# appear below where an example deliberately changes one. Protenix-v1 has no
# Training-Free Guidance, so there is no TFG switch; x-hat-0 comes straight
# from the DiffusionModule.
#
# Argument summary (`steerable-v1 pred -h` for the full list):
#   -i, --input                    Input JSON file or directory.
#   -o, --out_dir                  Output directory.            [./output]
#   -s, --seeds                    Comma-separated seeds.       [101]
#   -e, --sample                   Samples per seed.            [5]
#   -d, --dtype                    bf16 | fp32.                 [bf16]
#   -n, --model_name               Model checkpoint.            [protenix_base_default_v1.0.0]
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
MODEL=protenix_base_default_v1.0.0
AB="$(cat $EX/ab_chains.txt)"
EPITOPE="$(cat $EX/epitope.txt)"

bash $EX/prepare.sh

# ------------------------------------------------------------------------------
# 1. The defaults
# ------------------------------------------------------------------------------
steerable-v1 pred \
    -i $EX/7yds.json \
    -o ./test_outputs/steerable/7yds \
    -s 101 \
    -n $MODEL \
    --ab_chains "$AB" \
    --epitope_residue "$EPITOPE"

# ------------------------------------------------------------------------------
# 2. Several epitope hypotheses off ONE shared trunk
#
# ';' separates sets. Each becomes its own branch in steerable_<k>/, all sharing
# the trunk, the initial noise and the per-step noise.
# ------------------------------------------------------------------------------
steerable-v1 pred \
    -i $EX/7yds.json \
    -o ./test_outputs/steerable/7yds_multi \
    -s 101 \
    -n $MODEL \
    --ab_chains "A,B" \
    --epitope_residue "C:19,C:51,C:52;C:111,C:112,C:113;C:60,C:61"

# ------------------------------------------------------------------------------
# 3. Steered AND unguided off the same trunk, for a paired comparison
#
# The unguided run lands in raw/. Costs about twice as much.
# ------------------------------------------------------------------------------
steerable-v1 pred \
    -i $EX/7yds.json \
    -o ./test_outputs/steerable/7yds_both \
    -s 101 \
    -n $MODEL \
    --gating_mode both \
    --ab_chains "$AB" \
    --epitope_residue "$EPITOPE"

# ------------------------------------------------------------------------------
# 4. No epitope: plain Protenix-v1, unchanged
# ------------------------------------------------------------------------------
steerable-v1 pred \
    -i $EX/7yds.json \
    -o ./test_outputs/baseline/7yds \
    -s 101 \
    -n $MODEL

# ------------------------------------------------------------------------------
# 5. Straight to the runner, which exposes every config key
#
# Use this for anything the CLI does not reach (e.g. --epitope.alpha_trunc,
# --gating.threshold, --blocks_per_ckpt).
# ------------------------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

python3 runner/inference.py \
    --model_name $MODEL \
    --seeds 101 \
    --dump_dir ./test_outputs/steerable/7yds_runner \
    --input_json_path $EX/7yds.json \
    --model.N_cycle 10 \
    --sample_diffusion.N_sample 5 \
    --sample_diffusion.N_step 200 \
    --ab_chains "$AB" \
    --epitope_residue "$EPITOPE"

# ------------------------------------------------------------------------------
# 6. Reproducing the v1 (EmAbAg) benchmarks
#
# Those runs steered on EVERY diffusion step, one sample per chunk, in bf16,
# without templates, at alpha 0.1 and lambda_clash 0.1 (or 0.0 for the
# no-clash arm). guidance_interval 8 -- today's default -- was calibrated on
# Protenix-v2 and has not been re-validated on v1, so pass interval 1 here.
# ------------------------------------------------------------------------------
python3 runner/inference.py \
    --model_name $MODEL \
    --seeds 101 \
    --dump_dir ./test_outputs/steerable/7yds_v1bench \
    --input_json_path $EX/7yds.json \
    --model.N_cycle 10 \
    --sample_diffusion.N_sample 5 \
    --sample_diffusion.N_step 200 \
    --infer_setting.sample_diffusion_chunk_size 1 \
    --use_template false \
    --dtype bf16 \
    --epitope.guidance_interval 1 \
    --epitope.guidance_alpha 0.1 \
    --epitope.lambda_clash 0.1 \
    --ab_chains "$AB" \
    --epitope_residue "$EPITOPE"

# ------------------------------------------------------------------------------
# 7. Faster steering: no activation checkpointing
#
# --blocks_per_ckpt null keeps the denoiser's activations for the reward
# backward instead of recomputing them (1.4-2.3x on the steered branch in our
# v2 measurements). It costs memory; the OOM backoff halves the sample chunk
# and, as a last resort, restores checkpointing, so a large target completes
# slowly rather than failing. Results are unchanged.
# ------------------------------------------------------------------------------
python3 runner/inference.py \
    --model_name $MODEL \
    --seeds 101 \
    --dump_dir ./test_outputs/steerable/7yds_nockpt \
    --input_json_path $EX/7yds.json \
    --model.N_cycle 10 \
    --sample_diffusion.N_sample 5 \
    --sample_diffusion.N_step 200 \
    --blocks_per_ckpt null \
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
