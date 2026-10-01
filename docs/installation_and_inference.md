# Installation and inference

How to install SteerABLE and run it. For the epitope-steering options specifically,
see the main [README](../README.md); this document covers the inference machinery
SteerABLE inherits from Protenix-v2.

SteerABLE is inference-only. The upstream training and fine-tuning stack is not
part of this repository; if you need it, use
[Protenix](https://github.com/bytedance/Protenix) directly.

## 🛠 Installation

### From source
```bash
git clone https://github.com/bhyun-ans/SteerABLE.git
cd SteerABLE
pip3 install -e .
```

This installs the `steerable` command. SteerABLE is not published on PyPI.

### Docker
Check the detailed guide: [<u> Docker Installation</u>](./docker_installation.md).

### External Dependencies
For features such as **Template search** and **RNA MSA search**, additional system tools are required:
- **kalign**: Used for sequence alignment.
- **hmmer**: Used for sequence profile searches.

**Note**:
- **Docker Users**: These dependencies are already pre-installed in the upstream Protenix Docker image, which SteerABLE uses unchanged.
- **Non-Docker Users**: You must install them manually. On Ubuntu/Debian, run:
  ```bash
  apt-get update && apt-get install -y kalign hmmer
  ```
  Or, you can provide the paths to the binaries built from source via command-line arguments (e.g.,`--kalign_binary_path`, `--hmmsearch_binary_path`, `--hmmbuild_binary_path`, `--nhmmer_binary_path`, etc.).
  For more information, refer to `steerable pred -h`.


## 🚀 Inference & CLI Usage

SteerABLE provides a unified CLI for structure prediction, data preprocessing, and database searching, under the `steerable` command.

### CLI Commands Overview
| Command | Alias | Description |
|---------|-------|-------------|
| `predict` | `pred` | Perform model inference on JSON input(s). |
| `tojson` | `json` | Convert PDB or CIF files to Protenix-compatible JSON. |
| `msa` | `msa` | Generate Multiple Sequence Alignments (MSA) for proteins. |
| `msatemplate` | `mt` | Run sequential MSA and template search. |
| `inputprep` | `prep` | Full preprocessing: MSA, Template, and RNA MSA search. |

### 1. Data Conversion (`tojson`)
Convert structural files into the required JSON format.
```bash
# Convert PDB/CIF to JSON
steerable json --input ./your_structure.pdb --out_dir ./output --altloc first

# Advanced: Specify assembly ID for biological assemblies
wget https://files.rcsb.org/download/7pzb.cif
steerable json --input ./7pzb.cif --out_dir ./output --altloc first

# Advanced: Keep discontinuous polymer-polymer bonds (e.g. cyclic-peptide)
steerable json --input ./your_cyclic_peptide.cif --out_dir ./output --altloc first --include_discont_poly_poly_bonds
```

### 2. Input Preprocessing (`prep`, `mt`, `msa`)
Protenix requires MSA and template information for optimal accuracy.
```bash
# Full preprocessing (Protein MSA + Template + RNA MSA)
steerable prep --input your_input.json --out_dir ./output

# Sequential Protein MSA and Template search
steerable mt --input your_input.json --out_dir ./output

# Independent MSA search (supports JSON or Protein FASTA)
steerable msa --input your_sequences.fasta --out_dir ./output --msa_server_mode protenix
```

> **Note**: For `prep` and `mt`, you may need to specify paths to external databases (e.g., `--seqres_database_path`) and HMMER binaries if they are not in your system PATH.

### 3. Model Inference (`predict`)
Run the prediction engine with customizable configurations.
The bundled example is `examples/steerable/7yds/7yds.json`; run
`bash examples/steerable/7yds/prepare.sh` once to unpack its MSAs.

```bash
# Epitope-steered prediction -- the recommended SteerABLE setting
steerable pred -i examples/steerable/7yds/7yds.json -o ./output \
    -s 101 -n protenix-v2 --dtype fp32 --use_tfg_guidance true \
    --ab_chains "$(cat examples/steerable/7yds/ab_chains.txt)" \
    --epitope_residue "$(cat examples/steerable/7yds/epitope.txt)"

# No epitope: plain Protenix-v2
steerable pred -i examples/steerable/7yds/7yds.json -o ./output -s 101 -n protenix-v2

# Seeds taken from the input JSON
steerable pred -i your_input.json --use_seeds_in_json true

# Disable MSA (much faster, much less accurate -- use for smoke tests only)
steerable pred -i your_input.json --use_msa false --enable_cache true
```

#### Key Inference Flags
- `--seeds`: Comma-separated list of random seeds (e.g., `101,102`).
- `--model_name`: Model checkpoint. Use `protenix-v2`, which is what SteerABLE was calibrated and benchmarked on. The other upstream Protenix checkpoints still load but have not been validated with epitope steering.
- `--use_default_params`: (Default: `true`) Automatically configures cycles and steps based on the selected model. Set to `false` to manually override `--cycle` and `--step`.
- `--use_tfg_guidance`: Enable Training-Free Guidance (TFG) for refined sampling. Off by default, as upstream; SteerABLE recommends `true`.
- `--ab_chains` / `--epitope_residue`: Turn on epitope steering. See the [README](../README.md).
- `--use_msa` / `--use_template` / `--use_rna_msa`: (Default: `true`/`false`/`false`) Toggle specific features for inference.
- `--dtype`: Set data type to `bf16` (default) or `fp32`.
- `--trimul_kernel` / `--triatt_kernel`: Choose specialized kernels (e.g., `cuequivariance`, `triattention`) for hardware acceleration.
- `--enable_cache` / `--enable_fusion`: Enable memory/speed optimizations (recommended for GPU).

### Worked examples

`inference_demo.sh` at the repository root runs the bundled 7yds example through
every mode -- the recommended setting, several epitope sets off one trunk, the
paired steered-vs-unguided comparison, plain Protenix, the runner entry point,
and the published-run reproduction settings. Read it as the reference for what
each flag does:

```bash
bash inference_demo.sh
```

> **Performance Tip**: By default, specialized CUDA kernels are enabled. For significant speedups on NVIDIA GPUs, follow the [**Kernels Setup Guide**](./kernels.md).

## Inference cost

By default, the model performs inference in mixed-precision (BF16). However, the `SampleDiffusion` and `ConfidenceHead` modules are executed in full-precision (FP32) to maintain numerical stability and prediction accuracy.

The table below provides benchmark data for GPU memory utilization and inference latency across various input sizes.

| `N_token` | `N_atom` | Peak Memory (GB) | Latency (s) |
| :--- | :--- | :---: | :---: |
| 500 | 5000 | 6.1 | 17 |
| 1000 | 10000 | 18.2 | 59 |
| 2000 | 20000 | 66.6 | 226 |
| 3000 | 30000 | 60.8 | 935 |
| 4000 | 40000 | 78.1 | 1424 |

To mitigate potential Out-of-Memory (OOM) issues during large-scale inference, the inference script ([runner/inference.py](../runner/inference.py)) dynamically adjusts the precision for `SampleDiffusion` and `ConfidenceHead` based on the token count (`N_token`):
```python
def update_inference_configs(configs: Any, n_token: int) -> Any:
    """
    Adjust inference configurations based on the number of tokens to manage memory usage and prevent OOM.
    
    Args:
        configs (Any): Original configurations.
        n_token (int): Number of tokens in the sample.

    Returns:
        Any: Updated configurations.
    """
    if n_token > 3840:
        # Enable AMP for both modules to save memory for extremely large sequences
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = False
    elif n_token > 2560:
        # Enable AMP only for ConfidenceHead
        configs.skip_amp.confidence_head = False
        configs.skip_amp.sample_diffusion = True
    else:
        # Default: Disable AMP for both (run in FP32) to prioritize accuracy
        configs.skip_amp.confidence_head = True
        configs.skip_amp.sample_diffusion = True

    return configs
```
