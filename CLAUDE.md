# VQ-VAE for 2D Turbulence

JAX/Equinox implementation of a Vector Quantized VAE for compressing 2D turbulence (vorticity) fields.

## Architecture

- **Input**: C×H×W fields (multi-channel, non-square supported) from `.mat` files
- **Encoder**: 16× downsampling via strided convolutions + residual blocks, configurable `in_channels`
- **Quantizer**: Vector quantization with EMA codebook updates, straight-through gradients. Per-scale codebooks are channel-agnostic (encoder projects all channels into `codebook_dim`).
- **Decoder**: 16× upsampling via nearest-neighbor interpolation + convolutions, configurable `out_channels`
- **Output**: Multi-scale grid of codebook indices (discrete latent representation)
- **Scales**: Specified as `(h, w)` tuples, e.g. `((1,1), (2,2), (4,4), (8,8), (16,16))` for square, or `((1,1), (2,1), (4,2), (8,4), (16,8))` for non-square latents

## Files

- `models.py` - Model definitions (VQVAE2d, Encoder2d, Decoder2d, Quantizer2d, ResBlock2d)
- `trainer.py` - Training step with loss computation and EMA codebook updates
- `dataloaders.py` - Data loading from MATLAB files
- `train_vqvae.py` - Main training script with wandb integration
- `tokenizer.py` - Tokenization pipeline for AR model training
- `nsp_model.py` - Next-Scale Prediction model (NextScalePredictor, block-causal attention)
- `train_nsp.py` - NSP training script with per-scale loss and wandb logging
- `eval_nsp.py` - NSP evaluation: iterative generation, VQ-VAE decoding, metrics

## Training

```bash
python train_vqvae.py --data_dir /path/to/mat/files --epochs 100 --batch_size 16
```

Key hyperparameters: `--hidden_dim`, `--codebook_dim`, `--vocab_size`, `--commitment_weight`

## Tokenization

The `tokenizer.py` module wraps a trained VQ-VAE to prepare data for autoregressive model training.

**VQVAETokenizer class**:
- Loads trained VQ-VAE checkpoint (supports VQVAE2d and VARVQVAE2d)
- `fit()` scans dataset to collect used codebook indices
- **Standard VQ-VAE**: Remaps sparse codebook usage to consecutive indices
- **VAR mode**: Builds unified per-scale codebook with contiguous index ranges per scale. Each scale gets its own range in the unified vocabulary (e.g., scale 1 → [0, n0), scale 2 → [n0, n0+n1), etc.), so the AR model can distinguish tokens from different scales
- `encode()` / `encode_batch()` convert images to discrete tokens
- `decode_indices()` reconstructs from tokens

**Usage**:
```bash
# Show tokenizer stats
python tokenizer.py info --checkpoint model.eqx --data_dir /path/to/data --var_mode

# Tokenize and save to file
python tokenizer.py save --checkpoint model.eqx --data_dir /path/to/data --output tokens.npz --var_mode
```

**Output format** (`.npz`):
- `indices_flat`: [N, total_tokens] discrete token indices (unified indices for VAR mode)
- `vectors_flat`: [N, total_tokens, codebook_dim] corresponding codebook vectors
- `codebook`: unified codebook for AR model embedding layer (all scales concatenated)
- `scale_offsets`: [n_scales] start of each scale's range in unified vocab (VAR mode)
- `scale_vocab_sizes`: [n_scales] unique tokens per scale (VAR mode)
- `unified_to_scale`: [unified_vocab] scale index per unified entry (VAR mode)
- `unified_to_original`: [unified_vocab] original codebook index per entry (VAR mode)

## Next-Scale Prediction (NSP) Model

The NSP model predicts all tokens in a scale simultaneously, conditioned on all coarser scales via block-causal attention.

**Architecture**:
- Block-causal attention: each scale can only attend to strictly earlier scales
- Frozen codebook embedding with learnable projection + mask token
- Per-scale classification heads (one per trainable scale)
- Scales 0-3 (1x1 through 4x4) are deterministic; scales 4-9 (5x5 through 16x16) are predicted
- ~6M parameters with default config (6 layers, 8 heads, 256 embed dim)

**Training**:
```bash
python train_nsp.py --tokens_path tokens.npz --vqvae_checkpoint model.eqx --epochs 100
```
Each step randomly samples one of 6 trainable scales. Per-scale loss and accuracy are tracked separately.

**Evaluation**:
```bash
python eval_nsp.py --nsp_checkpoint nsp_model.eqx --vqvae_checkpoint model.eqx \
    --tokens_path tokens.npz --n_samples 4 --n_steps 100 --temperature 1.0
```
Generates autoregressive video rollouts (t0 -> t_n), decodes through VQ-VAE, and produces:
- Side-by-side MP4 videos comparing generated vs GT trajectories
- Token trajectories saved as .npz files
- Per-scale token distribution histograms with JS divergence and TV distance metrics
- Pixel-space comparison grids (if `--data_dir` provided for raw .mat files)

## Dependencies

- JAX
- Equinox
- Optax
- scipy (for loading .mat files)
- wandb (optional, for logging)
