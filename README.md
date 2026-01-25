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

## Logging

Training metrics are logged to [Weights & Biases](https://wandb.ai) if installed. Disable by uninstalling wandb or running offline.
