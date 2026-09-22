# Per-sample epitope steering, and two things that will bite you

> These notes were written while developing SteerABLE (the Protenix-v2 release)
> and apply unchanged to SteerABLE-v1: the sampler, the reward code and the
> trunk's RNG behaviour are the same. The timings and memory ceilings quoted
> below were measured on Protenix-v2 (c_z=256); the v1 trunk is narrower, so
> expect more headroom, not less.

Development notes on why steering needs a sample axis, and on the two ways a
run's randomness will silently change your results. Read this before trusting a
benchmark you produced after touching anything in the trunk.

## What the change does

`sample_diffusion_chunk_size` was pinned to 1 in production because a batched
chunk shared one steered `(s_tau, z_tau)` driven by the chunk-**averaged**
reward: samples wanting opposite corrections cancelled and the steering died.

That is an artifact of the trunk embeddings having no sample axis, not a
property of batching. Give them one and differentiating the **summed**
per-sample reward separates the samples exactly:

```
∂(Σᵢ rᵢ)/∂s_tau[j] = ∂r_j/∂s_tau[j]        no cross terms
```

Three things must change together or the samples silently recouple:

1. `generator.py` — `s_tau`/`z_tau` get a leading `N_sample` axis
2. `steering.py` — rewards return `[N_sample]`, not a scalar (`per_sample=True`)
3. `steering.py` — `rms_normalize(sample_dim=...)`; a global RMS re-couples the
   samples through the denominator even when the gradients are separate

The load-bearing edit is `DiffusionConditioning.forward`: skipping
`single_s.unsqueeze(-3)` for a per-sample trunk. Without it the result is
`[B,S,S,N,c_s]` **with no exception raised** — every sample mixed with every
other sample's noise level. There is a shape assert there now.

`transformer.py` needed no changes; it was already rank-generic.

Verification lives in `tests/test_per_sample_steering.py`: the reward is
block-diagonal in the sample axis (`∂r_j/∂coords[i] == 0` for `i≠j`) and the
batched gradient equals the serial one at `atol=0`. At run time,
`<PDB>_gating.json` carries a `steering` block whose
`samples_share_one_curve` must be `False`.

Measured on a 48 GB A6000 (fp32, N_sample=5, N_step=200, gating.mode=both),
steerable branch seconds:

| target | N_token | chunk 1 | chunk 5 | +`--blocks_per_ckpt null` | both |
|---|---|---|---|---|---|
| 8dtn | 147 | 146.8 | 52.1 (2.82x) | 64.9 (2.26x) | 35.3 (**4.16x**) |
| 8gy5 | 349 | 255.6 | 210.1 (1.22x) | 173.9 (1.47x) | 152.8 (**1.67x**) |
| 8evd | 729 | 838.7 | OOM | 604.4 (**1.39x**) | OOM |

The two levers do not multiply — both attack per-step fixed overhead. Batching
is strongest on small targets (the GPU is 23 % utilised at 147 tokens);
`blocks_per_ckpt null` is flatter and is the only lever left on big ones.

The unguided branch holds no steering state, so it batches fully at any size
(1.7–4.5x).

A flag that let one process steer several targets was tried and reverted. It
amortised the ~97 s startup, worth about 13 % of wall
clock, but the trunk depends on total RNG consumption order: with one process
per target every target starts from the same state, whereas the second target in
a process draws later numbers and lands on a different MSA subset. Measured on
11 targets, only the 3 that ran first in their process reproduced the baseline
enrichment. Keep one process per target for anything that has to compare against
existing results.

## Landmine 1: inference is stochastic, and the trunk is where it hurts

Two upstream mechanisms make a single forward sample one of many models. Both
run **once per target, in the trunk**, so all N_sample diffusion samples share
the outcome — the "5 samples" average over diffusion noise only.

**MSA subsampling.** `MSAModule.forward` starts with
`sample_msa_feature_dict_random_without_replacement`, and `sample_indices`
draws the *depth* uniformly:

```python
sample_size = torch.randint(low=min(lower_bound, n), high=n + 1).item()
indices = torch.randperm(n)[:sample_size]
```

With `min_size.test = 1` and `sample_cutoff.test = 16384` that is uniform over
`[1, 16384]`, re-drawn every recycling cycle. Traced on 8dtn (n=16384, 10
cycles), the kept depths were:

```
1362, 8007, 5696, 600, 6142, 12364, 4080, 15754, 974, 5673
```

**MC dropout.** `protenix.py` draws `random.random() < mc_dropout_apply_rate`
(stock 0.4) once per forward. When it lands, `F.dropout` runs with p=0.4 on the
pair embedding at every recycling pass — and `F.dropout` defaults to
`training=True`, so `model.eval()` does not stop it. It also consumes torch RNG,
which shifts every MSA depth drawn afterwards.

Both are deliberate: the `mc_dropout_apply_rate` docstring says "Only for
inference mode", `sample_indices` documents "Sample msa indices k from
uniform[1,n]", and `seeds` takes a list. This is the AF3 pattern — generate a
diverse ensemble, rank it by confidence. It only pays off if you actually run
several seeds; with one seed you get a single draw and no ensemble.

## Landmine 2: results depend on total RNG consumption order

Runs are reproducible — the same code, seed and command give bit-identical
output, verified across days, jobs and nodes. But the trunk depends on the
*entire* RNG history before it, so anything that shifts consumption changes the
answer. One process per target is stable because every target starts from the
same state. A shard is not: the second target draws later numbers.

Two plausible-looking "fixes" for this were tried during development and both
were wrong. The measurement that condemned them is one target's trunk
enrichment, which is 3.4052 for the whole line of optimisation work:

| attempt | enrichment on the probe target |
|---|---|
| re-seed before every prediction attempt | 1.8928 |
| force Monte-Carlo dropout off for inference | 1.8928 |
| neither (what ships) | **3.4052** |

Re-seeding reset the Python RNG right before `predict`, so the MC dropout coin
became draw #1 (0.5812, off) instead of the real draw (0.2036, **on**) — the
baseline had MC dropout **on** for this target, the opposite of what the change
assumed. Turning the coin off then removed the dropout's torch RNG consumption,
shifting every MSA depth: the traced sequence looked intact but was offset by
two positions. Both branches of the gate degraded together, which is the
signature of a changed trunk rather than changed steering.

**If you change anything that draws random numbers before or inside the trunk,
check `enrichment` on a known target before trusting a benchmark.** The startup
log now prints the effective `mc_dropout_apply_rate` so a run's regime is never
a mystery.

## Sizing

Per-sample `z_tau` scales the `N_token² · c_z` tensors with the chunk, so the
chunk that fits depends on target size. Calibrated on 48 GB: 349 fits at chunk
5, 395 does not; 490 fits at 3; 655 does not fit at 2. `--blocks_per_ckpt null`
fits at 729 and does not at 936. `update_inference_configs` applies those
ceilings and `_predict_with_oom_backoff` halves the chunk on OOM, restoring
activation checkpointing as a last resort before giving up.

Note the backoff must hand each attempt a fresh copy of the feature dict:
`predict` consumes it, so a retry on the original dies with `KeyError: 'profile'`
and the OOM is masked as a data error.

## Landmine 3 (Protenix-v1 only): the in-place flag changes the trunk

Measured on SteerABLE-v1 with `8c3l_D_#_C` (176 tokens, bf16, `--deterministic
true`), running the pairformer trunk under different contexts and comparing
`(s, z)` and the contact map bit for bit
(`benchmarks/.../steerablev1_sanity/analysis/trunk_grad_probe.py`):

| contexts compared | v1 max Δs / Δz / Δcontact | v2 (SteerABLE) |
|---|---|---|
| repeat of the same run | 0 / 0 / 0 | 0 / 0 / 0 |
| in-place ops on vs off, both `no_grad` | **16 / 40 / 0.31** | 0 / 0 / 0 |
| grad on vs off, in-place off in both | 0 / 0 / 0 | 0 / 0 / 0 |
| activation checkpointing on vs off, grad on | 0 / 0 / 0 | 0 / 0 / 0 |

`Protenix.forward` sets `inplace_safe = not torch.is_grad_enabled()`, so any
run with an epitope configured computes its trunk with in-place ops off. On v2
that is numerically invisible. On v1 it is not: 200 diffusion steps turn the
rounding difference into different samples (12 Å apart on this target), so a
gated run's `raw` branch is bit-identical to a `--gating.mode raw` run and to
the steered branch's trunk, but **not** to a run without `--epitope_residue`.
Compare steered against `raw` from the same `--gating.mode both` run, not
against a separately produced baseline. This also applies to the historical
EmAbAg benchmarks, whose baseline arm was a plain run.
