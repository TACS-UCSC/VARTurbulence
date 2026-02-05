# VQ-VAE for 2D Turbulence

JAX/Equinox implementation of a Vector Quantized VAE for compressing 2D turbulence (vorticity) fields.

## Architecture

- **Input**: 256x256 single-channel vorticity snapshots from `.mat` files
- **Encoder**: 16x downsampling (256->16) via strided convolutions + residual blocks
- **Quantizer**: Vector quantization with EMA codebook updates, straight-through gradients
- **Decoder**: 16x upsampling via nearest-neighbor interpolation + convolutions
- **Output**: 16x16 grid of codebook indices (discrete latent representation)

Two model variants are available:
- **VQVAE2d**: Standard single-scale VQ-VAE
- **VARVQVAE2d**: Multi-scale residual quantizer (VAR-style) with progressive quantization at increasing resolutions (1x1 -> 2x2 -> 4x4 -> 8x8 -> 16x16)

## Installation

```bash
pip install -r requirements.txt
```

For GPU support, install JAX with CUDA:
```bash
pip install --upgrade "jax[cuda12]"
```

## Usage

### Training

```bash
# Standard VQ-VAE
python train_vqvae.py --data_dir /path/to/mat/files --epochs 100 --batch_size 16

# VAR multi-scale VQ-VAE
python train_vqvae.py --data_dir /path/to/mat/files --var_mode --epochs 100
```

### Key Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden_dim` | 512 | Hidden dimension |
| `--codebook_dim` | 64 | Codebook embedding dimension |
| `--vocab_size` | 512 | Number of codebook vectors |
| `--commitment_weight` | 0.25 | Commitment loss weight (beta) |
| `--base_channels` | 128 | Base channel count |
| `--channel_mult` | 1,2,4,4 | Channel multipliers per stage |
| `--num_res_blocks` | 2 | ResBlocks per resolution stage |
| `--var_mode` | False | Enable VAR multi-scale quantization |
| `--scales` | 1,2,4,8,16 | Scales for VAR mode |

### Data Format

Expects `.mat` files containing a variable named `Omega` with shape `(256, 256)` representing vorticity fields.

## Files

| File | Description |
|------|-------------|
| `models.py` | Model definitions (VQVAE2d, VARVQVAE2d, Encoder2d, Decoder2d, Quantizer2d) |
| `trainer.py` | Training step with loss computation and EMA codebook updates |
| `dataloaders.py` | Data loading from MATLAB files with prefetching |
| `train_vqvae.py` | Main training script with wandb integration |
| `tokenizer.py` | Tokenization pipeline for AR model training |
| `nsp_model.py` | Next-Scale Prediction model (NextScalePredictor, block-causal attention) |
| `train_nsp.py` | NSP training script with per-scale loss and wandb logging |
| `eval_nsp.py` | NSP evaluation: iterative generation, VQ-VAE decoding, metrics |

## Tokenization

The `tokenizer.py` module wraps a trained VQ-VAE to prepare data for autoregressive model training.

**VQVAETokenizer class**:
- Loads trained VQ-VAE checkpoint (supports VQVAE2d and VARVQVAE2d)
- `fit()` scans dataset to collect used codebook indices
- **Standard VQ-VAE**: Remaps sparse codebook usage to consecutive indices
- **VAR mode**: Builds unified per-scale codebook with contiguous index ranges per scale

**Usage**:
```bash
# Show tokenizer stats
python tokenizer.py info --checkpoint model.eqx --data_dir /path/to/data --var_mode

# Tokenize and save to file
python tokenizer.py save --checkpoint model.eqx --data_dir /path/to/data --output tokens.npz --var_mode
```

**Output format** (`.npz`):
- `indices_flat`: [N, total_tokens] discrete token indices
- `vectors_flat`: [N, total_tokens, codebook_dim] corresponding codebook vectors
- `codebook`: unified codebook for AR model embedding layer
- `scale_offsets`: [n_scales] start of each scale's range in unified vocab (VAR mode)
- `scale_vocab_sizes`: [n_scales] unique tokens per scale (VAR mode)

## Next-Scale Prediction (NSP) Model

The NSP model predicts all tokens in a scale simultaneously, conditioned on all coarser scales via block-causal attention.

**Architecture**:
- Block-causal attention: each scale can only attend to strictly earlier scales
- Frozen codebook embedding with learnable projection + mask token
- Per-scale classification heads (one per trainable scale)
- Scales 0-3 (1x1 through 4x4) are deterministic; scales 4-9 are predicted
- Temporal modeling: t1 is predicted conditioned on t0 (full context) and coarser scales of t1

**Training**:
```bash
python train_nsp.py --tokens_path tokens.npz --vqvae_checkpoint model.eqx --epochs 100
```

Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--n_layer` | 6 | Number of transformer layers |
| `--n_head` | 8 | Number of attention heads |
| `--n_embd` | 256 | Embedding dimension |
| `--optimizer` | lion | Optimizer (lion, adamw, adafactor) |
| `--lr` | 1e-4 | Learning rate |

**Evaluation**:
```bash
python eval_nsp.py --nsp_checkpoint nsp_model.eqx --vqvae_checkpoint model.eqx \
    --tokens_path tokens.npz --n_samples 4 --n_steps 100
```

Generates autoregressive video rollouts (t0 -> t100), decodes through VQ-VAE, and produces:
- Side-by-side videos (Generated vs GT)
- Per-scale token distribution histograms
- Pixel-space comparison grids (if raw data provided)
- JS divergence and TV distance metrics

## Logging

Training metrics are logged to [Weights & Biases](https://wandb.ai) if installed. Disable by uninstalling wandb or running offline.

## Dependencies

- JAX
- Equinox
- Optax
- scipy (for loading .mat files)
- h5py (for loading HDF5 datasets)
- PyTorch CPU (`torch`, for HDF5 dataloader)
- matplotlib (for visualization)
- wandb (optional, for logging)
