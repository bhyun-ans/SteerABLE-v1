# EmAbAg

**Em**bedding-steered **Ab**–**Ag** docking — inference-time **hotspot/epitope steering** for antibody–antigen structure prediction, built on [Protenix](https://github.com/bytedance/Protenix) (an open-source AlphaFold3-style model).

> Give EmAbAg the **antigen epitope** you care about, and it steers the structure module **during diffusion** so the antibody docks onto that epitope — yielding higher **DockQ**, **without retraining the AF3 trunk**.

<img src="assets/emabag_figure.png" style="width: 100%; height: auto;" alt="EmAbAg — hotspot-steered diffusion">

---

## 🎯 What EmAbAg does

Antibody–antigen (Ab–Ag) complexes are among the hardest targets for AF3-style predictors: the model frequently docks the antibody onto the **wrong surface** of the antigen, so even a structurally reasonable sample scores poorly on DockQ.

EmAbAg fixes the *placement* problem at inference time. You provide the **epitope** — the antigen residues the antibody is supposed to contact — together with which chains are the **antibody**. During reverse diffusion, EmAbAg nudges the trunk embeddings along the gradient of a **contact reward** that pulls antibody atoms toward the specified epitope. The samples that come out have the antibody seated on the intended epitope, which translates directly into better **DockQ** / interface quality.

Key properties:

- **No retraining.** The Protenix/AF3 trunk and diffusion weights are frozen. EmAbAg is a guidance term added to the sampling loop — pure inference-time **embedding steering**.
- **Epitope in, docked complex out.** The only extra inputs are `ab_chains` and `epitope_residue` (passed on the command line); the input JSON stays a standard Protenix input.
- **Steers embeddings, not coordinates.** At each step the *single* (`s_i`) and *pair* (`z_ij`) trunk embeddings are perturbed; the steered `z` feeds the next denoising step (see figure, bottom track).

## 🔬 How it works

At every reverse-diffusion step of the AF3 sampler (Algorithm 18), EmAbAg inserts a guidance update:

1. **Differentiate the embeddings.** The current `s_τ`, `z_τ` are detached and made differentiable; the denoiser is run under `enable_grad`.
2. **Contact reward.** For each epitope residue, EmAbAg takes the *soft-minimum* distance from each of its atoms to the nearest antibody atoms and applies a smooth squared-hinge penalty for atoms farther than a contact threshold `d0` (default **4 Å**):

   ```
   reward = − Σ_residues Σ_atoms  softplus( softmin_dist(atom → antibody) − d0 )²
   ```

   Higher reward ⇔ the epitope sits in close contact with the antibody. `top_k` restricts the penalty to the *k* closest atoms per residue (`k=1` = closest-atom only).
3. **Optional clash penalty.** A van-der-Waals overlap term between epitope and antibody atoms (weight `lambda_clash`, AF3-style tolerance `clash_tau`) discourages interpenetrating poses. It can be switched on only after a chosen step via `clash_start_step`.
4. **Steer.** The gradient `∂reward/∂(s,z)` is RMS-normalized and added back:

   ```
   s_τ ← s_τ + α · ĝ_s        z_τ ← z_τ + α · ĝ_z
   ```

   where **α** (guidance strength) follows a **truncated raised-cosine schedule** (from flow matching): `α(s) = α_init · ½(1 + cos(πs))` for `s < τ_trunc`, else `0`. Here `s ∈ [0,1]` is generation progress (`s=0` at pure noise/start, `s=1` clean/end), so guidance is strongest early and switches off at `s = τ_trunc` (the `alpha_trunc` knob — e.g. `0.5` = step 100/200). Setting `guidance_alpha` to a float overrides the schedule with a constant α.

**Ab/Ag is a binary partition.** You name the antibody chains via `ab_chains`; every other input chain is treated as antigen — there is no separate antigen argument. The reward targets only **epitope ↔ antibody** contacts, so antigen–antigen contacts (e.g. for homo-multimeric antigens) are excluded by construction.

When guidance is on, the diffusion shared-variable cache is disabled (the pair embedding changes every step), and the confidence head is run under `no_grad` to keep memory bounded on large targets.

## 🚀 Quick start

EmAbAg is a fork of Protenix and runs from source through the same entry point. Follow the upstream environment/installation steps in [`docs/`](docs/) (e.g. [Training and Inference Instructions](docs/training_inference_instructions.md)) and [`inference_demo.sh`](inference_demo.sh), then run **guided** inference by adding the EmAbAg flags:

```bash
python3 runner/inference.py \
    --model_name protenix_base_default_v1.0.0 \
    --seeds 101 \
    --dump_dir ./output/guided_run \
    --input_json_path ./examples/abag.json \
    --sample_diffusion.N_sample 5 \
    --sample_diffusion.N_step 200 \
    --ab_chains "A,B" \
    --epitope_residue "C:45,C:48,D:52" \
    --guidance.guidance_alpha 0.1 \
    --guidance.top_k 1 \
    --guidance.lambda_clash 0.1
```

Setting `--epitope_residue` is what turns guidance **on**; leaving it unset runs stock (unguided) Protenix. Chain letters refer to the order of the `sequences` entries in the input JSON (A = first, B = second, …).

## ⚙️ Configuration

Guidance is configured entirely from the command line (no input-JSON changes). Flags live in [`configs/configs_inference.py`](configs/configs_inference.py):

| Flag | Default | Meaning |
| :--- | :---: | :--- |
| `--ab_chains` | `None` | Comma-separated antibody chain letters, e.g. `"A,B"`. Antigen = all other input chains. |
| `--epitope_residue` | `None` | Epitope residues on antigen chains, `"CHAIN:RESID,…"`, e.g. `"C:45,C:48"`. **Setting this enables guidance.** |
| `--guidance.guidance_alpha` | `None` | Constant guidance strength α. If set (e.g. `0.1`), overrides the cosine schedule below. |
| `--guidance.alpha_init` | `1.0` | `α_init`: peak weight at `s=0` for the truncated raised-cosine schedule (used when `guidance_alpha` is `None`). |
| `--guidance.alpha_trunc` | `0.5` | `τ_trunc`: generation-progress fraction `∈ (0,1]` at which α drops to `0` ("when to turn α off"; `0.5` = step 100/200). |
| `--guidance.d0` | `4.0` | Contact threshold (Å) for the reward. |
| `--guidance.softmin_beta` | `10.0` | Soft-min inverse temperature. |
| `--guidance.top_k` | `None` | Atoms per epitope residue in the penalty (`None` = all, `1` = closest only). |
| `--guidance.lambda_clash` | `0.0` | Clash-penalty weight (`0` = off). |
| `--guidance.clash_tau` | `1.5` | vdW overlap tolerance (Å). |
| `--guidance.clash_start_step` | `0` | Diffusion step at which the clash term turns on. |
| `--save_trajectory` | `False` | Save the per-step diffusion trajectory (see Outputs). |

## 📦 Outputs

Predictions are written under `<dump_dir>/<pdb_id>/seed_<seed>/`:

- `predictions/<pdb_id>_sample_<rank>.cif` — one structure per sample, ranked by confidence (B-factor = per-atom pLDDT).
- `predictions/<pdb_id>_summary_confidence_sample_<rank>.json` — confidence summary; under guidance this also carries the **per-step `reward_history`** so you can inspect how the contact/clash reward evolved during sampling.
- With `--save_trajectory`, a `traj/` folder holds a multi-model PDB per sample (one MODEL per diffusion step, re-aligned to a common frame). When an epitope is set, epitope atoms are flagged and a companion PyMOL `*_view.pml` is written that highlights the epitope.

## 🙏 Acknowledgements

EmAbAg is built directly on **[Protenix](https://github.com/bytedance/Protenix)** (ByteDance), itself a reimplementation in the spirit of **AlphaFold3**. All trunk/diffusion model code and weights are Protenix's; EmAbAg adds the inference-time epitope-steering guidance described above. Please cite the upstream work when using EmAbAg:

```bibtex
@article{Zhang2026Protenix,
  author  = {Zhang, Yuxuan and Gong, Chengyue and Zhang, Hanyu and Ma, Wenzhi and Liu, Zhenyu and Chen, Xinshi and Guan, Jiaqi and Wang, Lan and Yang, Yanping and Xia, Yu and Xiao, Wenzhi},
  title   = {Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.02.05.703733}
}
@article{abramson2024accurate,
  title   = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  author  = {Abramson, Josh and Adler, Jonas and Dunger, Jack and others},
  journal = {Nature}, volume = {630}, pages = {493--500}, year = {2024}
}
```

## 📄 License

Inherited from Protenix: released under the [Apache 2.0 License](./LICENSE).
