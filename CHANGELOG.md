# Changelog

All notable changes to SteerABLE are documented here. Changes inherited from
upstream Protenix are not restated; see the
[Protenix changelog](https://github.com/bytedance/Protenix) for those.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0]

First public release. Fork point: Protenix-v2 (2.0.0). The accompanying
preprint is in preparation.

### Added
- **Embedding-space epitope steering.** A differentiable contact reward on the
  denoiser's clean-structure prediction back-propagates to the pairformer trunk
  embeddings `(s, z)`, which are carried as steerable state across the diffusion
  trajectory. `--epitope_residue` turns it on; `--ab_chains` says which side of
  the interface is the antibody.
- **Composition with Training-Free Guidance.** `TFGEngine.step(return_x0=True)`
  runs its Stage-2 denoise under `enable_grad` and returns the graph-attached
  `x̂₀`, so coordinate-space TFG and embedding-space steering share a single
  denoiser forward instead of paying for two.
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
- `--auto_restore_activation_checkpointing`: set false to reproduce the
  published runs, which had no large-target checkpointing safety net.
- OOM backoff that halves the diffusion sample chunk instead of losing a target.
- `examples/steerable/7yds/`: a complete runnable example with bundled MSAs.

### Changed
- The distribution and console command are `steerable`. The importable package
  is still `protenix`, so this tree keeps diffing cleanly against upstream.
- Epitope defaults now carry the recommended setting: `guidance_alpha` 0.1,
  `lambda_clash` 0.1, `guidance_interval` 8, `gating.mode` `steerable`.
- Settings shared with upstream Protenix keep upstream's defaults. The two that
  the recommended setting changes, TFG (`--use_tfg_guidance true`) and precision
  (`--dtype fp32`), must be passed explicitly. See `examples/steerable/7yds/run.sh`.
- `predict()` in the CLI accepts the epitope options; previously they existed
  only on `runner/inference.py`.

### Removed
- The top-k atom selection knob. The contact penalty always uses the single
  closest atom of each epitope residue (k = 1), which is what the released
  model was calibrated and benchmarked with.
