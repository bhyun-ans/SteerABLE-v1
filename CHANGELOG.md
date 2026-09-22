# Changelog

All notable changes to SteerABLE-v1 are documented here. Changes inherited from
upstream Protenix are not restated; see the
[Protenix changelog](https://github.com/bytedance/Protenix) for those.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]

First release. Fork point: Protenix-v1 (upstream 1.0.5, checkpoint
`protenix_base_default_v1.0.0`), by way of EmAbAg, the epitope-steering fork
that produced the v1 benchmarks. SteerABLE-v1 is the Protenix-v1 counterpart of
[SteerABLE](https://github.com/bhyun-ans/SteerABLE), which is built on
Protenix-v2; the two share one interface and one steering algorithm.

### Added
- **Embedding-space epitope steering.** A differentiable contact reward on the
  denoiser's clean-structure prediction back-propagates to the pairformer trunk
  embeddings `(s, z)`, which are carried as steerable state across the diffusion
  trajectory. `--epitope_residue` turns it on; `--ab_chains` says which side of
  the interface is the antibody. Protenix-v1 has no Training-Free Guidance, so
  x̂₀ is read straight from the DiffusionModule and the coordinate update is
  the plain AF3 Euler step.
- **Several epitope hypotheses off one trunk.** `--epitope_residue "a;b;c"`
  computes the pairformer trunk and its distogram once, then samples one steered
  branch per set into `steerable_<k>/`. Branches share the trunk, the initial
  noise and the per-step noise, and the trunk embeddings are fingerprinted
  before each branch so none can start from a modified `(s, z)`.
- **Per-sample steering.** Each sample in a diffusion chunk is steered by its
  own reward gradient, so `sample_diffusion_chunk_size > 1` no longer averages
  opposing corrections into nothing.
- **Distogram routing gate.** Enrichment of the epitope in the unsteered trunk's
  contact map is computed and dumped on every steered run
  (`<name>_gating.json`), and can route between steered and raw
  (`--gating.mode route`).
- **Clash penalty.** An optional antibody × antigen van der Waals overlap term,
  on by default at `lambda_clash = 0.1`.
- `--epitope.guidance_interval`: apply the steering gradient on every k-th step.
- `--auto_restore_activation_checkpointing` and an OOM backoff that halves the
  diffusion sample chunk, then restores activation checkpointing, instead of
  losing a target. `--blocks_per_ckpt null` (no activation checkpointing) is
  therefore safe to use for speed.
- `examples/steerable/7yds/`: a complete runnable example with bundled MSAs.

### Changed
- The distribution and console command are `steerable-v1`. The importable
  package is still `protenix`, so this tree keeps diffing cleanly against
  upstream Protenix-v1.
- Epitope defaults mirror SteerABLE: `guidance_alpha` 0.1, `lambda_clash` 0.1,
  `guidance_interval` 8, `gating.mode` `steerable`. Note that the v1 benchmarks
  (EmAbAg) were produced with the gradient applied on every step; see the
  README's reproduction section.
- The epitope options live under `--epitope.*` (was `--guidance.*` in EmAbAg).

### Removed
- The training and fine-tuning stack, training-data preparation scripts and
  the upstream benchmark reports (inference-only release).
- EmAbAg's `--guidance.top_k` knob: the contact penalty always uses the single
  closest atom of each epitope residue (k = 1).
- EmAbAg's `--save_trajectory` (multi-model PDB trajectory + PyMOL view) and the
  per-sample `reward_history` in the summary-confidence JSON. The per-step,
  per-sample reward trace now lives in `<name>_gating.json`.
