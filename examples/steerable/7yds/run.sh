#!/usr/bin/env bash
#
# SteerABLE-v1 on 7yds -- the default setting, end to end.
#
# 7yds is a Fab (chains A and B) bound to a 147-residue antigen (chain C).
# 375 tokens in total. The epitope in epitope.txt is the Ca-Ca 10 A footprint
# of the true interface, i.e. this run asks "given where the antibody binds,
# can the model build that complex" rather than "can it find the epitope".
#
#   bash examples/steerable/7yds/prepare.sh     # once, unpacks the MSAs
#   bash examples/steerable/7yds/run.sh         # this script
#
# RUN IT FROM THE REPOSITORY ROOT: the paths inside 7yds.json are relative.
# On a cluster, submit this through your scheduler rather than running it on a
# login node -- it needs a GPU and takes minutes, not seconds.
#
# ---------------------------------------------------------------------------
# Everything is already the default value out of the box
# (configs/configs_inference.py):
#
#   --epitope_guidance_alpha 0.1        steering step size
#   --epitope_lambda_clash 0.1          antibody-antigen clash penalty
#   --epitope_guidance_interval 8       steer on every 8th of the 200 steps
#   --gating_mode steerable             sample the steered branch only
#   --dtype bf16                        upstream Protenix-v1 default
#
# Protenix-v1 has no Training-Free Guidance, so there is no TFG switch.
#
# Pass a knob explicitly if you want to change it. alpha and the interval were
# calibrated together on Protenix-v2 -- the total steering applied over the
# trajectory scales as ~1/interval. The v1 benchmarks steered on every step;
# add --epitope_guidance_interval 1 to reproduce that regime (see the README).
# ---------------------------------------------------------------------------
set -euo pipefail

EX="examples/steerable/7yds"
OUT="${OUT:-./output/7yds_steerable}"
SEEDS="${SEEDS:-101}"

if [ ! -s "$EX/msa/A/paired.a3m" ]; then
    echo "MSAs are still packed. Run: bash $EX/prepare.sh" >&2
    exit 1
fi

AB="$(cat "$EX/ab_chains.txt")"
EPITOPE="$(cat "$EX/epitope.txt")"

echo "target   : 7yds (Fab $AB + antigen C, 375 tokens)"
echo "epitope  : $(echo "$EPITOPE" | tr ',' '\n' | wc -l) residues"
echo "seeds    : $SEEDS"
echo "output   : $OUT"
echo

# ---- installed CLI ----------------------------------------------------------
steerable-v1 pred \
    --input "$EX/7yds.json" \
    --out_dir "$OUT" \
    --model_name protenix_base_default_v1.0.0 \
    --seeds "$SEEDS" \
    --ab_chains "$AB" \
    --epitope_residue "$EPITOPE"

# ---- the equivalent call straight to the runner ----------------------------
# Use this one to reach settings the CLI does not expose (and to reproduce the
# v1 benchmarks -- see the reproduction section of the README):
#
# python runner/inference.py \
#     --model_name protenix_base_default_v1.0.0 \
#     --input_json_path "$EX/7yds.json" \
#     --dump_dir "$OUT" \
#     --seeds "$SEEDS" \
#     --ab_chains "$AB" \
#     --epitope_residue "$EPITOPE"
