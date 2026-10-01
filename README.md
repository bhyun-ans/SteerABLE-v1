# SteerABLE

Epitope-steered antibody–antigen structure prediction, built on
[Protenix-v2](https://github.com/bytedance/Protenix).

Tell the model where the antibody binds, and it builds that complex. SteerABLE
turns a known or predicted epitope into a differentiable reward and uses it to
steer the pairformer trunk embeddings at every few diffusion steps. No
retraining, no change to the weights — the steering happens entirely at
inference time.

Without `--epitope_residue` this repository behaves exactly like upstream
Protenix.

---

## Quickstart

```bash
pip install -e .

bash examples/steerable/7yds/prepare.sh   # unpack the bundled MSAs
bash examples/steerable/7yds/run.sh       # steer a Fab onto a known epitope
```

The model checkpoint downloads itself on first use, into `~/checkpoint` (or
`$PROTENIX_ROOT_DIR/checkpoint`). The example needs one GPU and finishes in
minutes. [`examples/steerable/7yds/README.md`](examples/steerable/7yds/README.md)
walks through what comes out.

Your own target, in one command:

```bash
steerable pred \
    --input my_complex.json --out_dir ./out \
    --model_name protenix-v2 \
    --dtype fp32 --use_tfg_guidance true \
    --ab_chains "A,B" \
    --epitope_residue "C:45,C:48,C:52,C:120,C:121"
```

`--ab_chains` names the antibody chains; everything else is the antigen.
Epitope residues are `<chain>:<position>` with **1-based positions into the
sequence you put in the JSON**, not author or PDB numbering.

## Recommended settings

Most of the recommendation is already the default. Two knobs are not, because
they are shared with upstream Protenix and we leave upstream's defaults alone:

| setting | recommended | already the default? |
|---|---|---|
| `--use_tfg_guidance true` (`--sample_diffusion.guidance.enable`) | on | **no** — pass it |
| `--dtype fp32` | fp32 | **no** — upstream default is `bf16` |
| `--epitope_guidance_alpha` | `0.1` | yes |
| `--epitope_lambda_clash` | `0.1` | yes |
| `--epitope_guidance_interval` | `8` | yes |
| `--gating_mode` | `steerable` | yes |
| top-k atom selection | `k = 1` | fixed, not configurable |

`examples/steerable/7yds/run.sh` is the canonical invocation and explains every
line.

**`guidance_alpha` and `guidance_interval` were calibrated together.** The
per-update step size stays `guidance_alpha` whatever the interval, so the total
steering applied over the trajectory scales as roughly `1/interval`. Changing
one without the other moves you off the validated setting; treat any new pair
as something to re-validate, not a free speed knob.

## Installation

```bash
pip install -e .
```

Runtime and checkpoint dependencies follow upstream Protenix. See
[docs/installation_and_inference.md](docs/installation_and_inference.md)
for the full setup guide, including MSA and template search and Docker.

No MSAs for your sequences? Generate them against a public server:

```bash
steerable msa --input my_complex.fasta --out_dir ./msa_out
```

### If the fused layernorm kernel will not build

Inherited from upstream Protenix, a custom CUDA layernorm kernel is JIT-compiled
on first use. On many clusters `nvcc` refuses the host compiler
(`error -- unsupported GNU version! gcc versions later than 11 are not
supported`) or the built extension cannot find its CUDA runtime
(`libcudart.so.11.0: cannot open shared object file`).

Either point the build at a compiler and CUDA runtime it accepts:

```bash
export PATH=/path/to/gcc-9/bin:$PATH
export CC=/path/to/gcc-9/bin/gcc CXX=/path/to/gcc-9/bin/g++
export LD_LIBRARY_PATH=/path/to/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

or skip the kernel entirely and use PyTorch's own layernorm:

```bash
export LAYERNORM_TYPE=torch
```

The second costs some speed and nothing else. See [docs/kernels.md](docs/kernels.md)
for the other kernel switches.

## How it works

SteerABLE adds two orthogonal, composable inference-time guidance mechanisms.
They act on different tensors — trunk conditioning `(s, z)` versus coordinates
`x` — and therefore compose without interfering.

1. **Embedding-space epitope steering** (this repository). A reward gradient
   nudges the pairformer trunk embeddings toward the epitope residues you name.
2. **Coordinate-space Training-Free Guidance** (TFG, upstream Protenix-v2).
   Projection- and gradient-based corrections enforce hard geometric
   constraints — chirality, bond distances, clashes — on denoised coordinates.

### Why an antibody–antigen model needs both

Ab–Ag complexes are among the hardest targets for structure prediction: the
interface is small, the paratope is diverse, and confidence-based sample
selection is unreliable. Two complementary levers help.

Prior knowledge of the epitope is often available — from cross-linking,
epitope mapping, phage display, or a companion epitope predictor. Using that
prior as a differentiable reward at inference time is cheap and modular.

Separately, physically implausible outputs — flipped chirality, stretched
bonds, atomic clashes — appear rarely but disproportionately damage interface
geometry. TFG catches these, and embedding-side gradients cannot: chirality and
bond-length violations need a projection-style solver, not gradient descent.

### 1. Embedding-space epitope steering

At every steered step of the reverse diffusion trajectory, shift the pairformer
trunk output `(s_τ, z_τ)` so that the denoiser's clean-structure prediction
`x̂₀` places the epitope residues into contact with the antibody.

**Reward.** For each epitope residue `h`, take the softmin distance from its
single closest atom to the antibody atom set:

```
d_h = -(1/β) · log Σ_{a ∈ antibody} exp(-β · ‖x_h - x_a‖)
```

with `β = 10`. The per-residue penalty is flat-bottomed with contact distance
`d₀ = 4 Å`:

```
penalty(h) = softplus(d_h - d₀)²
Reward R   = -Σ_h penalty(h)
```

For homomeric antigens the mask includes every non-antibody chain, so a hotspot
on any antigen copy is admissible. An additive clash reward based on
antibody × antigen vdW overlap is layered on at `lambda_clash = 0.1`.

**Update rule.** The trunk embeddings are treated as steerable state maintained
across diffusion steps:

```
s_τ, z_τ ← detach(s_trunk), detach(z_trunk)         # once, before the loop
for τ = T, T-1, …, 1:
    if τ is a steered step:
        s_τ.requires_grad_(True); z_τ.requires_grad_(True)
        with torch.enable_grad():
            x̂₀   = denoise(x_noisy_τ, t̂, s_τ, z_τ, features)
            R_τ  = Reward(x̂₀)
            g_s, g_z = ∇_{(s_τ, z_τ)} R_τ
        s_τ ← detach(s_τ) + α · RMS_normalize(g_s)
        z_τ ← detach(z_τ) + α · RMS_normalize(g_z)
    x_l ← Euler_step(x_noisy_τ, x̂₀, t̂, c_τ)
```

Two things are worth emphasising.

**The reward is evaluated on `x̂₀`, the denoiser's clean-structure prediction,
not on the noisy `x_{t-1}` that leaves the Euler step.** `x̂₀` is noise-free
throughout the trajectory, it is what the model *thinks* the final structure
looks like, and its gradient direction does not depend on the sampler's
step-size hyperparameters.

**The gradient magnitude is RMS-normalised**, so the effective step size `α` is
scale-invariant. Constant `α = 0.1` works best; `α = 1.0`, the flow-matching
paper value, over-steers by roughly 10× in this normalisation. A truncated
raised-cosine schedule is also available
(`--epitope.guidance_alpha null --epitope.alpha_init 0.1 --epitope.alpha_trunc 0.5`).

Every sample in a diffusion chunk is steered by its own reward gradient.
Differentiating the *sum* of per-sample rewards separates them exactly, because
`r_i` depends only on `s_τ[i]`, so there are no cross terms — a batched chunk
gives the same trajectories as running the samples one at a time.

### 2. Coordinate-space Training-Free Guidance

TFG is upstream to Protenix-v2 (Ye et al. 2024). After the denoiser produces
`x̂₀`, TFG replaces the plain AF3 update with a five-stage pipeline. Two
potentials override the base `_project` hook and actually move atoms:

| Potential | Constraint | How it moves atoms |
| --- | --- | --- |
| `ChiralAtomPotential` | R/S torsion sign | Linearised solver flips only atoms whose CIP dihedral is on the wrong side of the buffer |
| `PairwiseDistancePotential` | Bond-length, bond-angle, steric-clash bounds | Projects violated pair distances back into `[lower, upper]`; angles first, then bonds |

Seven further potentials contribute only to the soft `μ`-refinement pass and
never move atoms directly.

### 3. Composition — one denoise call, two guidance signals

Running both together is mechanically awkward: TFG's Stage-2 denoise runs under
`torch.no_grad()`, which stops a reward computed on that `x̂₀` from
back-propagating to `s_τ, z_τ`. The naive workaround — a second, grad-enabled
denoise — doubles the wall time.

The fix is a single semantic change to the TFG engine. `TFGEngine.step()` gains
a `return_x0` flag; when set, Stage 2 runs under `torch.enable_grad()` (with
`inplace_safe=False` to preserve the graph) and the raw `x̂₀` is returned
alongside `x_next`:

```python
# tfg/engine.py — inside step()
stage2_needs_grad = return_x0 and outer == 0
stage2_ctx = torch.enable_grad() if stage2_needs_grad else torch.no_grad()
with stage2_ctx:
    x0 = denoise_net(x_work + xt_shift, ..., s_trunk=s_trunk, z_trunk=z_trunk, ...)
if stage2_needs_grad:
    x0_for_return = x0                 # keep the graph-attached reference

x0_ref = x0.detach()                   # TFG's own downstream work uses a detached copy
# ... projection, μ-refinement, Euler step ...

if return_x0:
    return x_next, x0_for_return
return x_next
```

`x0.detach()` makes a new tensor for TFG's internal use without destroying the
graph attached to the original. The generator then forks one forward pass into
two consumers:

```python
# model/generator.py — inside the diffusion loop
s_τ.requires_grad_(True); z_τ.requires_grad_(True)

with torch.enable_grad():
    x_next, x̂₀ = tfg.step(..., s_trunk=s_τ, z_trunk=z_τ, return_x0=True)
    R = contact_epitope_reward(x̂₀, mask_pairs, d0, β, top_k=1)
    g_s, g_z = torch.autograd.grad(R, [s_τ, z_τ])

s_τ = (s_τ.detach() + α · RMS_normalize(g_s)).detach()
z_τ = (z_τ.detach() + α · RMS_normalize(g_z)).detach()
x_l = x_next.detach()
```

One denoiser forward, two independent update paths. TFG projects and refines
coordinates; SteerABLE computes reward gradients and updates embeddings.
Neither sees the other's output; the shared `x̂₀` is the single tangential point.

**Cost.** At `guidance_interval = 8`, 25 of the usual 200 diffusion steps carry
the reward backward and the other 175 are plain no-grad sampler steps on the
embeddings as they stand.

| Mode | Denoise calls/step | Backward passes/step | Wall time vs. vanilla v2 |
| --- | :---: | :---: | :---: |
| Baseline | 1 | 0 | 1.0× |
| TFG only | 1 | (0, or the `ρ`-path if enabled) | ~1.3× |
| Steering only, every step | 1 | 1 (reward) | ~1.7× |
| Steering + TFG, every step | **1** | 1 (reward) | ~1.7× |

The combined mode costs essentially nothing over steering alone: TFG's
projection and refinement use analytic, not autograd, gradients.

## Usage

### Several epitope hypotheses from one trunk

Separate epitope sets with `;` to steer towards each of them in one run:

```bash
steerable pred \
    -i input.json -o ./out -n protenix-v2 \
    --dtype fp32 --use_tfg_guidance true \
    --ab_chains "A,B" \
    --epitope_residue "C:45,C:48,C:52;D:120,D:121;C:10,C:11,C:14"
```

The pairformer trunk and its distogram are computed **once**; each set is then
sampled as its own diffusion branch (`steerable_0`, `steerable_1`, … in CLI
order) from that same trunk. Before each branch the trunk embeddings are
re-checked against a fingerprint taken right after the trunk, so a branch can
never start from a modified `(s, z)`; the RNG state is restored per branch, so
all branches share the same initial noise and per-step noise and differ only in
what they were steered towards.

This matters because trunks are not reproducible across processes. On one
target, the same seed and input gave trunk enrichments of 18.5, 20.8, 25.3 and
29.5 in four separate runs — epitopes compared across separate processes would
have been compared across different trunks. Verified the other way too: three
branches steered towards the *same* set are bit-identical under
`--deterministic true`.

Output layout with several sets: each branch gets its own sub-directory
(`./out/steerable_<k>/<name>/seed_<s>/predictions/`, `./out/raw/…`), and the
target's root directory holds only the trunk-level side-cars —
`<name>_gating.json` with one `epitopes` entry per set, and
`<name>_contact_probs.npz` whose `epitope_token_mask` becomes `[K, N_token]`.
A single set keeps the plain layout.

### Comparing against no guidance

`--gating_mode both` samples the steered branch *and* an unguided branch off the
same trunk, with matched noise, and writes the unguided one to `./out/raw/`.
That is the paired comparison you want when deciding whether steering helped on
a given target. It costs about twice as much.

### Reading the gate report

Every steered run writes `predictions/<name>_gating.json`. Besides the steering
trace and per-branch wall-clock, it reports **enrichment**: score each antigen
token by its best antibody contact probability in the unsteered trunk's
distogram, `e_j = max_{i ∈ Ab} C_ij`, then

```
enrichment = mean(e_j | j ∈ epitope) / mean(e_j | j elsewhere)
```

Around 1 means the trunk had no epitope signal of its own, so steering has the
most to add. A large value means raw Protenix was already heading there.
`--gating_mode route` turns this into an automatic decision against
`--gating.threshold`.

### Baseline and TFG-only

```bash
# plain Protenix
steerable pred -i input.json -o ./out -n protenix-v2

# TFG only, no epitope
steerable pred -i input.json -o ./out -n protenix-v2 --use_tfg_guidance true
```

### The runner entry point

`runner/inference.py` exposes the entire config tree as `--a.b.c value`, which
the CLI's curated option list does not. Use it for anything the CLI does not
reach, and for reproduction:

```bash
python runner/inference.py \
    --model_name protenix-v2 \
    --input_json_path input.json --dump_dir ./out \
    --seeds 101 --dtype fp32 \
    --sample_diffusion.guidance.enable true \
    --ab_chains "A,B" --epitope_residue "C:45,C:48,C:52"
```

Every knob is documented in
[`configs/configs_inference.py`](configs/configs_inference.py).

## Reproducing the published runs

The benchmarks were produced with the settings above plus three that the
released defaults deliberately do not carry:

```bash
python runner/inference.py \
    ... \
    --dtype fp32 \
    --sample_diffusion.guidance.enable true \
    --blocks_per_ckpt null \
    --auto_restore_activation_checkpointing false \
    --gating.save_contact_probs false
```

`--blocks_per_ckpt null` keeps activations instead of recomputing them, worth
1.4–2.3× in wall time at the cost of memory: on a 48 GB card it fits at 729
tokens and does not at 936. The released default restores checkpointing for
targets above ~800 tokens so that a large target completes slowly rather than
failing; `--auto_restore_activation_checkpointing false` removes that safety
net, which is what the published runs did.

Two further things to know when reproducing numbers:

- **Inference is deterministic but depends on the whole RNG consumption
  history.** One process that handles several targets draws a different MSA
  subset from the second target onward, so its trunk differs from a
  one-process-per-target run. Match the process layout, not just the seed.
- **Do not pool across GPU architectures.** Identical configurations on two
  different cards differed by 0.32 DockQ standard deviation in our runs.
  Compare arms by paired deltas within a target.

## Repository layout

Guidance-related files. Everything outside this list is upstream Protenix-v2's
inference machinery, unchanged.

SteerABLE is **inference-only**: upstream's training and fine-tuning stack, its
training-data preparation scripts, and its benchmark reports for the other
Protenix checkpoints are not part of this repository. Use
[Protenix](https://github.com/bytedance/Protenix) if you need them.

| Path | Role |
| --- | --- |
| `protenix/model/steering.py` | Epitope contact reward, clash reward, mask builder, RMS normaliser, epitope-string parsing |
| `protenix/model/gating.py` | Distogram enrichment, branch naming and the raw-vs-steered routing gate |
| `protenix/model/generator.py` | Three-way dispatch inside `sample_diffusion()`: baseline / TFG-only / steered (± TFG) |
| `protenix/model/protenix.py` | Parses `epitope_residue` and `ab_chains`, builds `mask_pairs`, runs one trunk and one branch per epitope set |
| `protenix/tfg/engine.py` | `TFGEngine.step(return_x0=…)`, the shared-denoise hook. Byte-identical to upstream when `return_x0=False` |
| `protenix/utils/alpha_schedule.py` | Truncated raised-cosine α-schedule |
| `configs/configs_inference.py` | `epitope_residue`, `ab_chains`, the `epitope.{…}` sub-config and the gating options |
| `runner/inference.py` | Drops `@torch.no_grad()` on `predict()`, conditions autocast on the epitope flag, OOM backoff |
| `runner/batch_inference.py` | The `steerable` CLI, including the epitope options |
| `protenix/metrics/clash.py` | van der Waals radii behind the clash reward (also used by upstream's confidence path) |
| `docs/per_sample_batching.md` | Why per-sample steering needs a sample axis, and two RNG traps it exposes |
| `examples/steerable/7yds/` | A complete runnable example with bundled MSAs |

The importable Python package is still called `protenix`. That is deliberate:
it keeps this tree diffable and re-baseable against upstream. Only the
distribution and the console command are named `steerable`.

## Citation

The SteerABLE preprint is in preparation; a citation will be added here once it
is posted. In the meantime, please cite Protenix and AlphaFold 3, whose model
and inference machinery SteerABLE builds on.

```bibtex
@article{Zhang2026.02.05.703733,
    author  = {Zhang, Yuxuan and Gong, Chengyue and Zhang, Hanyu and Ma, Wenzhi and Liu, Zhenyu and Chen, Xinshi and Guan, Jiaqi and Wang, Lan and Yang, Yanping and Xia, Yu and Xiao, Wenzhi},
    title   = {Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction},
    journal = {bioRxiv},
    year    = {2026},
    doi     = {10.64898/2026.02.05.703733}
}

@article{abramson2024accurate,
    title   = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
    author  = {Abramson, Josh and Adler, Jonas and Dunger, Jack and others},
    journal = {Nature},
    volume  = {630},
    pages   = {493--500},
    year    = {2024}
}

@article{ye2024tfg,
    title   = {TFG: Unified Training-Free Guidance for Diffusion Models},
    author  = {Ye, Haotian and Lin, Haowei and Han, Jiaqi and others},
    journal = {arXiv preprint arXiv:2409.15723},
    year    = {2024}
}
```

## Attribution

SteerABLE is built on top of Protenix-v2 (ByteDance) and inherits its
architecture, weights and inference machinery. For LayerNorm and pairformer
implementations it inherits code from
[OneFlow](https://github.com/Oneflow-Inc/oneflow),
[FastFold](https://github.com/hpcaitech/FastFold) and
[OpenFold](https://github.com/aqlaboratory/openfold); see
[upstream Protenix](https://github.com/bytedance/Protenix) for details.

## License

Apache 2.0, inherited from upstream Protenix. See [LICENSE](LICENSE).
