"""
Training script for Autoregressive Next-Scale Prediction (NSP).

Trains the model to predict t1 given t0 (context) and coarser scales of t1.
Data is loaded as pairs [t0, t1].
"""

import argparse
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed.")

from nsp_model import (
    NextScalePredictor, NextScalePredConfig,
    build_temporal_mask, load_nsp_model_from_tokenized_data,
)
from tokenizer import VQVAETokenizer, load_vqvae_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train AR-NSP model")
    # Data
    parser.add_argument("--tokens_path", type=str, default="tokens.npz")
    parser.add_argument("--vqvae_checkpoint", type=str, required=True)
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Lion needs lower LR than AdamW)")
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="lion", choices=["adamw", "lion", "adafactor"],
                        help="Optimizer to use (default: lion)")
    # Model
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    # Logging
    parser.add_argument("--wandb_project", type=str, default="turbulence-ar-nsp")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--vis_every", type=int, default=200)
    parser.add_argument("--no_vis", action="store_true")
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_ar")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_tokenized_data(path: str) -> dict:
    import json
    data = dict(np.load(path, allow_pickle=True))
    result = {
        "indices_flat": data["indices_flat"],
        "vectors_flat": data["vectors_flat"],
        "codebook": data["codebook"],
        "effective_vocab_size": int(data["effective_vocab_size"]),
        "vocab_size": int(data["vocab_size"]),
        "codebook_dim": int(data["codebook_dim"]),
        "config": json.loads(str(data["config_json"])),
        "var_mode": bool(data["var_mode"]),
    }
    if result["var_mode"]:
        result["scales"] = tuple(data["scales"].tolist())
        result["scale_offsets"] = data["scale_offsets"]
        result["scale_vocab_sizes"] = data["scale_vocab_sizes"]
        result["unified_to_scale"] = data["unified_to_scale"]
        result["unified_to_original"] = data["unified_to_original"]
        scale_old_to_unified = []
        for k in range(len(result["scales"])):
            key = f"scale_old_to_unified_{k}"
            if key in data:
                scale_old_to_unified.append(data[key])
        result["scale_old_to_unified"] = scale_old_to_unified
    else:
        result["old_to_new"] = data["old_to_new"]
        result["new_to_old"] = data["new_to_old"]
    return result


def create_paired_dataloader(data: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 0):
    """
    Yields batches of [B, 2 * tokens_per_frame] by pairing data[i] and data[i+1].
    """
    n_samples = len(data) - 1  # Need pairs
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for i in range(0, n_samples - batch_size + 1, batch_size):
        batch_indices = indices[i:i + batch_size]
        
        # Load t0 and t1
        t0 = data[batch_indices]
        t1 = data[batch_indices + 1]
        
        # Concatenate: [B, 2*680]
        yield np.concatenate([t0, t1], axis=1)


def make_train_step(config: NextScalePredConfig, target_scale_idx: int,
                    attn_bias: jax.Array):
    """Create JIT-compiled train step for t1 prediction at a specific scale."""
    
    boundaries = config.scale_boundaries
    padded_len = config.padded_seq_len
    
    # Scale ranges within the SINGLE frame
    scale_start = boundaries[target_scale_idx]
    scale_end = boundaries[target_scale_idx + 1]
    
    offset = config.scale_offsets[target_scale_idx]
    head_idx = target_scale_idx - config.first_trainable_scale

    # Precompute mask: Mask t1 target scale and beyond
    # t0 is context (never masked)
    mask_positions = jnp.zeros(2 * padded_len, dtype=jnp.bool_)
    
    # Offset into t1 part of sequence
    t1_offset = padded_len
    mask_positions = mask_positions.at[t1_offset + scale_start : t1_offset + config.tokens_per_frame].set(True)

    @eqx.filter_jit
    def step(model, opt_state, batch_tokens, optimizer):
        # batch_tokens: [B, 2*tokens_per_frame] (unpadded)

        def loss_fn(model):
            B = batch_tokens.shape[0]
            
            # Split batch back into t0, t1 to pad separately
            t0 = batch_tokens[:, :config.tokens_per_frame]
            t1 = batch_tokens[:, config.tokens_per_frame:]

            # Pad both to aligned length
            t0_pad = jnp.pad(t0, ((0,0), (0, padded_len - config.tokens_per_frame)))
            t1_pad = jnp.pad(t1, ((0,0), (0, padded_len - config.tokens_per_frame)))
            
            # Re-concat: [B, 2*padded_len]
            tokens_in = jnp.concatenate([t0_pad, t1_pad], axis=1)

            # Forward pass
            # hidden: [B, 2*padded_len, n_embd]
            hidden = jax.vmap(
                lambda t: model(t, mask_positions, attn_bias)
            )(tokens_in)

            # Slice t1 hidden states for this scale
            # t1 starts at index `padded_len`
            h_scale = hidden[:, t1_offset + scale_start : t1_offset + scale_end, :]

            # Apply Head
            logits = jax.vmap(jax.vmap(model.scale_heads[head_idx]))(h_scale)

            # Targets (from original unpadded t1)
            targets = t1[:, scale_start:scale_end] - offset

            # Loss
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(
                log_probs, targets[:, :, None], axis=-1
            ).squeeze(-1)
            
            loss = -jnp.mean(target_log_probs)
            
            # Acc
            preds = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean(preds == targets)
            
            return loss, accuracy

        (loss, accuracy), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, accuracy

    return step


def decode_and_visualize(t0_tokens, t1_gt_tokens, t1_gen_tokens, tokenizer):
    """Visualize: Context(t0) -> Generated(t1) vs GT(t1)."""
    t0_img = np.array(tokenizer.decode_flat_indices(t0_tokens)[0])
    t1_gt_img = np.array(tokenizer.decode_flat_indices(t1_gt_tokens)[0])
    t1_gen_img = np.array(tokenizer.decode_flat_indices(t1_gen_tokens)[0])

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    vmin, vmax = -10, 10
    
    axes[0].imshow(t0_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title("Context (t0)")
    axes[0].axis('off')

    axes[1].imshow(t1_gt_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth (t1)")
    axes[1].axis('off')

    axes[2].imshow(t1_gen_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[2].set_title("Generated (t1)")
    axes[2].axis('off')

    diff = t1_gen_img - t1_gt_img
    axes[3].imshow(diff, cmap='RdBu_r', vmin=-5, vmax=5)
    axes[3].set_title(f"Diff (MSE={np.mean(diff**2):.4f})")
    axes[3].axis('off')

    plt.tight_layout()
    return fig


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)

    print(f"Loading data from {args.tokens_path}...")
    token_data = load_tokenized_data(args.tokens_path)
    indices = token_data["indices_flat"]
    codebook = jnp.array(token_data["codebook"])
    scales = token_data["scales"]
    tokens_per_frame = sum(s * s for s in scales)

    # Setup Model Config
    config = NextScalePredConfig(
        tokens_per_frame=tokens_per_frame,
        scales=scales,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        codebook_dim=token_data["codebook_dim"],
        unified_vocab_size=token_data["effective_vocab_size"],
        scale_vocab_sizes=tuple(int(x) for x in token_data["scale_vocab_sizes"]),
        scale_offsets=tuple(int(x) for x in token_data["scale_offsets"]),
        first_trainable_scale=4,
    )

    # Initialize Model
    key, model_key = jax.random.split(key)
    model = NextScalePredictor(config, codebook, model_key)
    if args.resume:
        model = eqx.tree_deserialise_leaves(args.resume, model)
    """
    print("Casting model parameters to bfloat16 for speed...")
    model = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16) if eqx.is_array(x) and x.dtype == jnp.float32 else x, 
        model
    )
    """
    # Build Mask (2*L, 2*L)
    attn_bias = build_temporal_mask(config.scales, config.padded_seq_len)
    print(f"Temporal Mask Shape: {attn_bias.shape}")

    # Compile Train Steps
    trainable_indices = config.trainable_scale_indices
    train_steps = {}
    for scale_idx in trainable_indices:
        train_steps[scale_idx] = make_train_step(config, scale_idx, attn_bias)
        print(f"Compiled step for scale index {scale_idx}")

    # Optimizer
    n_samples = len(indices) - 1
    steps_per_epoch = n_samples // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, args.warmup_steps, total_steps, args.lr * 0.01
    )
    
    if args.optimizer == "lion":
        print(f"Using Lion optimizer (lr={args.lr})")
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.lion(learning_rate=schedule, weight_decay=args.weight_decay)
        )
    elif args.optimizer == "adamw":
        print(f"Using AdamW optimizer (lr={args.lr})")
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adamw(schedule, weight_decay=args.weight_decay)
        )
    elif args.optimizer == "adafactor":
        print(f"Using Adafactor optimizer")
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.grad_clip),
            optax.adafactor(learning_rate=schedule)
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
        
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    if WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # Load Tokenizer for Vis
    vqvae_config = token_data["config"]
    key, vqvae_key = jax.random.split(key)
    vqvae_model = load_vqvae_checkpoint(args.vqvae_checkpoint, vqvae_config, vqvae_key)
    tokenizer = VQVAETokenizer(vqvae_model)
    if token_data["var_mode"]:
        tokenizer.set_mapping(
            scale_old_to_unified=[jnp.array(m) for m in token_data["scale_old_to_unified"]],
            unified_to_scale=jnp.array(token_data["unified_to_scale"]),
            unified_to_original=jnp.array(token_data["unified_to_original"]),
            unified_codebook=jnp.array(token_data["codebook"]),
            scale_offsets=np.array(token_data["scale_offsets"]),
            scale_vocab_sizes=np.array(token_data["scale_vocab_sizes"]),
        )

    # --- Training Loop ---
    global_step = args.start_epoch * steps_per_epoch

    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        key, loader_key = jax.random.split(key)
        
        # Track accuracy per scale for this epoch
        per_scale_accs = {idx: [] for idx in trainable_indices}

        # New Paired Dataloader
        dataloader = create_paired_dataloader(indices, args.batch_size, seed=int(loader_key[0]))
        
        for batch_idx, batch in enumerate(dataloader):
            batch = jnp.array(batch) # [B, 2*tokens_per_frame]

            # Sample Scale
            key, sk = jax.random.split(key)
            target_idx = trainable_indices[int(jax.random.randint(sk, (), 0, len(trainable_indices)))]
            
            model, opt_state, loss, acc = train_steps[target_idx](model, opt_state, batch, optimizer)
            
            # Record acc
            per_scale_accs[target_idx].append(float(acc))

            if batch_idx % args.log_every == 0:
                print(f"  Step {batch_idx}: loss={loss:.4f} acc={acc:.4f} (Scale {config.scales[target_idx]})")
                if WANDB_AVAILABLE:
                    wandb.log({"loss": float(loss), "acc": float(acc), "step": global_step})

            if not args.no_vis and batch_idx % args.vis_every == 0 and batch_idx > 0:
                # Viz: Use t0 from batch[0], predict t1
                row = batch[0]
                t0_tokens = row[:tokens_per_frame]
                t1_gt_tokens = row[tokens_per_frame:]
                
                # Deterministic Seed for t1 (Scales 0-3)
                det_len = config.scale_boundaries[config.first_trainable_scale]
                seed_tokens = t1_gt_tokens[:det_len]
                
                key, gen_key = jax.random.split(key)
                t1_gen = model.generate(codebook, t0_tokens, seed_tokens, attn_bias, key=gen_key)
                
                fig = decode_and_visualize(t0_tokens, t1_gt_tokens, t1_gen, tokenizer)
                if WANDB_AVAILABLE:
                    wandb.log({"vis": wandb.Image(fig)})
                plt.close(fig)

            global_step += 1

        # End of Epoch Summary
        epoch_log = {"epoch": epoch + 1}
        print(f"--- Epoch {epoch+1} Summary ---")
        for idx in trainable_indices:
            scale_name = config.scales[idx]
            if len(per_scale_accs[idx]) > 0:
                avg_scale_acc = np.mean(per_scale_accs[idx])
                print(f"  Scale {scale_name}: {avg_scale_acc:.4f}")
                epoch_log[f"acc/scale_{scale_name}"] = avg_scale_acc
            else:
                print(f"  Scale {scale_name}: No steps sampled")

        if WANDB_AVAILABLE:
            wandb.log(epoch_log)

        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            path = os.path.join(args.checkpoint_dir, f"nsp_ar_epoch_{epoch+1}.eqx")
            eqx.tree_serialise_leaves(path, model)
            print(f"Saved {path}")

    # Final Save
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    path = os.path.join(args.checkpoint_dir, "nsp_ar_final.eqx")
    eqx.tree_serialise_leaves(path, model)
    print("Done.")

if __name__ == "__main__":
    main()