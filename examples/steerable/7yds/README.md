# Example: 7yds

A Fab bound to a 147-residue antigen. This is the example to run first: it is
small enough to finish in minutes on one GPU, and steering changes the answer
enough that you can see whether it worked.

| | |
|---|---|
| chains | `A` heavy, `B` light, `C` antigen |
| tokens | 375 |
| epitope | 30 residues on chain `C`, the Cα–Cα 10 Å footprint of the true interface |

The epitope is supplied as an **input**, the way a real run would take it from
epitope mapping, cross-linking, phage display or an upstream epitope predictor.
Here it is the crystallographic answer, which makes this a clean check that the
machinery works rather than a blind prediction.

## Run it

From the **repository root** — the paths inside `7yds.json` are relative:

```bash
bash examples/steerable/7yds/prepare.sh   # once: unpacks the MSAs
bash examples/steerable/7yds/run.sh
```

`prepare.sh` gunzips the bundled MSAs (5.4 MB packed, 24.6 MB unpacked).
`run.sh` uses the defaults and explains every knob in comments.

The model checkpoint downloads itself on first use. Set `PROTENIX_ROOT_DIR` if
you want it somewhere other than `~/checkpoint`.

On a cluster, submit `run.sh` through your scheduler. It needs a GPU.

Override the defaults with environment variables:

```bash
OUT=./my_output SEEDS=101,102,103 bash examples/steerable/7yds/run.sh
```

## What you get

```
output/7yds_steerable/7yds/seed_101/predictions/
├── 7yds_sample_0.cif ... _sample_4.cif        5 structures
├── 7yds_summary_confidence_sample_*.json      per-sample confidence
├── 7yds_gating.json                           ← read this one
└── 7yds_contact_probs.npz                     trunk contact map
```

`7yds_gating.json` is the record of what the steering actually did. Check:

- `epitope_guidance_interval` is `8`, and the steering trace covers 25 of the
  200 diffusion steps (the v1 benchmarks used interval 1, i.e. all 200). The run log says the same thing:
  `interval=8 (25/200 steps steer)`.
- `enrichment` — how much the *unsteered* pairformer trunk already favoured
  your epitope. Roughly 1 means the trunk had no opinion, so steering has the
  most to add. A large value means raw Protenix was already heading there.
- the per-step, per-sample reward trace, if you want to see when each sample
  locked onto the epitope.

## Expected result

Steering should place the Fab on the named epitope. Against the deposited
structure, five samples from one seed are a small draw, so treat a single seed
as a smoke test rather than a measurement — sample-to-sample spread on this
target is wide, and the published numbers pool 100 structures per target.

## Sanity check without MSAs

To confirm the install works before committing to a real run, drop the MSAs.
It is much faster and much less accurate:

```bash
steerable-v1 pred -i examples/steerable/7yds/7yds.json -o ./output/smoke \
    --model_name protenix_base_default_v1.0.0 --use_msa false --seeds 101 \
    --ab_chains "$(cat examples/steerable/7yds/ab_chains.txt)" \
    --epitope_residue "$(cat examples/steerable/7yds/epitope.txt)"
```

## Using your own target

1. Write a JSON in the same shape: one `proteinChain` per chain, and either
   `pairedMsaPath`/`unpairedMsaPath` or no MSA fields at all (then use
   `steerable-v1 msa` to generate them). Protenix-v1 letters the chains A, B,
   C, … in the order they appear in `sequences`; it does not read a per-chain
   `id` field.
2. Name the antibody chains in `--ab_chains`. Everything else is the antigen.
3. Write the epitope as `<chain>:<position>` pairs, comma separated, using
   **1-based positions into the sequence you put in the JSON** — not author or
   PDB numbering. Mismatched numbering is the most common way to get a run that
   completes and steers towards nothing.

Separate several epitope hypotheses with `;` to sample each as its own branch
off one shared trunk:

```bash
--epitope_residue "C:19,C:51,C:52;C:111,C:112;C:60,C:61"
```

Each branch lands in its own `steerable_<k>/` directory and they share the
trunk, the initial noise and the per-step noise, so they differ only in what
they were steered towards.
