# VQ-VAE for 2D Turbulence

JAX/Equinox implementation of a Vector Quantized VAE for compressing 2D turbulence (vorticity) fields.

## Architecture

- **Input**: 256×256 single-channel vorticity snapshots from `.mat` files
- **Encoder**: 16× downsampling (256→16) via strided convolutions + residual blocks
- **Quantizer**: Vector quantization with EMA codebook updates, straight-through gradients
- **Decoder**: 16× upsampling via nearest-neighbor interpolation + convolutions
- **Output**: 16×16 grid of codebook indices (discrete latent representation)

## Files

- `models.py` - Model definitions (VQVAE2d, Encoder2d, Decoder2d, Quantizer2d, ResBlock2d)
- `trainer.py` - Training step with loss computation and EMA codebook updates
- `dataloaders.py` - Data loading from MATLAB files
- `train_vqvae.py` - Main training script with wandb integration

## Training

```bash
python train_vqvae.py --data_dir /path/to/mat/files --epochs 100 --batch_size 16
```

Key hyperparameters: `--hidden_dim`, `--codebook_dim`, `--vocab_size`, `--commitment_weight`

## Dependencies

- JAX
- Equinox
- Optax
- scipy (for loading .mat files)
- wandb (optional, for logging)
