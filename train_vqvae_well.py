from importlib import reload
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for logging.")

import models
reload(models)
from models import VQVAE2d, VARVQVAE2d
from trainer import make_step, make_step_var
import dataloaders
reload(dataloaders)
from dataloaders import load_turbulence_data_mat, load_turbulence_data_rb_convection, create_turbulence_dataloader, well_dataset_loader
# from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ-VAE on 2D turbulence data")
    parser.add_argument("--data_dir", type=str, default="/glade/derecho/scratch/llupinji/data/the_well_datasets/datasets/rayleigh_benard/data/train",
                        help="Directory containing .hdf5 files")
    parser.add_argument("--data_type", type=str, default="rb_convection", help="Type of data to load")
    # parser.add_argument("--data_loc", type=str, default="/glade/derecho/scratch/llupinji/data/the_well_datasets/datasets/rayleigh_benard/data/test/rayleigh_benard_Rayleigh_1e6_Prandtl_1e-1.hdf5", help="Location of the HDF5 file (rb convection tested)")
    parser.add_argument("--start_idx", type=int, default=50, help="Start index for data files")
    parser.add_argument("--stop_idx", type=int, default=-1, help="Stop index for data files")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--codebook_dim", type=int, default=128, help="Codebook embedding dimension")
    parser.add_argument("--vocab_size", type=int, default=1028, help="Number of codebook vectors")
    parser.add_argument("--commitment_weight", type=float, default=0.25, help="Commitment loss weight (beta)")
    parser.add_argument("--decay", type=float, default=0.99, help="EMA decay for codebook")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--normalize", action="store_true", help="Normalize data")
    parser.add_argument("--wandb_project", type=str, default="vqvae-turbulence", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    # VAR mode arguments
    parser.add_argument("--var_mode", action="store_true", help="Enable VAR multi-scale quantization")
    parser.add_argument("--scales", type=str, default="1,4,16", help="Comma-separated scales for VAR mode")
    # Encoder/decoder capacity arguments
    parser.add_argument("--base_channels", type=int, default=128,
                        help="Base channel count (multiplied by channel_mult)")
    parser.add_argument("--channel_mult", type=str, default="1,2,4,4",
                        help="Comma-separated channel multipliers per stage")
    parser.add_argument("--num_res_blocks", type=int, default=2,
                        help="Number of ResBlocks per resolution stage")
    parser.add_argument("--use_attention", action="store_true", default=True,
                        help="Use self-attention at bottleneck")
    parser.add_argument("--no_attention", action="store_true",
                        help="Disable self-attention at bottleneck")
    parser.add_argument("--use_norm", action="store_true", default=True,
                        help="Use GroupNorm in ResBlocks")
    parser.add_argument("--no_norm", action="store_true",
                        help="Disable GroupNorm in ResBlocks")
    parser.add_argument("--attention_heads", type=int, default=4,
                        help="Number of attention heads")
    return parser.parse_args()


def plot_reconstruction(inputs, outputs, fields):
    """
    Plot input vs reconstruction for multiple fields.
    
    Args:
        inputs: np.ndarray, shape (N, C, H, W)
        outputs: np.ndarray, shape (N, C, H, W)
        fields: list of str or str, channel names (len(fields) == C)
    """
    if isinstance(fields, str):
        fields = [fields]

    N, C, H, W = inputs.shape
    max_fields = min(C, 4)
    max_samples = min(4, N)
    fig, axes = plt.subplots(max_samples, max_fields * 2, figsize=(5 * max_fields * 2, 4 * max_samples))
    if max_samples == 1:
        axes = np.expand_dims(axes, 0)
    if max_fields * 2 == 2:
        axes = np.expand_dims(axes, 1)
    instd = inputs.std()
    for i in range(max_samples):
        for c in range(max_fields):
            input_img = inputs[i, c]
            output_img = outputs[i, c]
            # Input
            ax_in = axes[i, c * 2]
            im = ax_in.imshow(input_img.T, cmap='coolwarm', vmin=-2*instd, vmax=2*instd)
            ax_in.set_title(f"Input {i} - {fields[c] if c < len(fields) else f'Field {c}'}")
            ax_in.axis('off')
            # Reconstruction
            ax_out = axes[i, c * 2 + 1]
            im = ax_out.imshow(output_img.T, cmap='coolwarm', vmin=-2*instd, vmax=2*instd)
            ax_out.set_title(f"Recon {i} - {fields[c] if c < len(fields) else f'Field {c}'}")
            ax_out.axis('off')

    plt.suptitle("Input vs Reconstruction Comparison\nFields: " + ", ".join(fields))
    plt.tight_layout()
    return fig


def plot_codebook_usage(indices, vocab_size):
    """Plot histogram of codebook usage and return figure."""
    flat_indices = indices.flatten()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(np.array(flat_indices), bins=vocab_size, range=(0, vocab_size), density=True)
    ax.axhline(y=1/vocab_size, color='r', linestyle='--', label='Uniform')
    ax.set_xlabel("Codebook Index")
    ax.set_ylabel("Frequency")
    ax.set_title("Codebook Usage Distribution")
    ax.legend()
    plt.tight_layout()
    return fig


def save_checkpoint(model, opt_state, epoch, checkpoint_dir, var_mode=False):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    prefix = "var_vqvae" if var_mode else "vqvae"
    checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_epoch_{epoch}.eqx")
    eqx.tree_serialise_leaves(checkpoint_path, model)
    print(f"Saved checkpoint to {checkpoint_path}")

# def main():
args = parse_args()

# Set random seed
key = jax.random.PRNGKey(args.seed)

# Load data
if args.data_type == "rb_convection":
    if args.data_dir is None:
        data_dir = "/glade/derecho/scratch/llupinji/data/the_well_datasets/datasets/rayleigh_benard/data/train"
    else:
        data_dir = args.data_dir
    fields = ["buoyancy", "pressure", "velocity_x", "velocity_y"]
else:
    raise ValueError(f"Invalid data type: {args.data_type}")

print("Loading turbulence data...")
if args.data_dir is None:
    data_locs = None
else:
    data_locs = os.listdir(args.data_dir)
    data_locs = [os.path.join(args.data_dir, loc) for loc in data_locs]

# Calculate total channels
total_channels = 0
for f in fields:
    if f == "velocity":
        total_channels += 2
    else:
        total_channels += 1
print(f"Total channels: {total_channels}")

# Create PyTorch DataLoader
# data_loader = well_dataset_loader(
data_loader = well_dataset_loader(
    data_locs,
    dataset_name=args.data_type,
    start_idx=args.start_idx,
    stop_idx=args.stop_idx,
    normalize=args.normalize,
    fields=fields,
    dt=None,  # No temporal sequences, just individual samples
    batch_size=args.batch_size,
    shuffle=True,
    seed=args.seed
)

print(f"Loaded {len(data_loader)} samples")

# Create PyTorch DataLoader
# data_loader = DataLoader(
#     rb_dataset,
#     batch_size=args.batch_size,
#     shuffle=True,
#     num_workers=0,  # Keep 0 for HDF5 data to avoid multiprocessing issues
#     drop_last=True  # Drop last incomplete batch
# )

# Parse scales for VAR mode
scales = tuple(int(s) for s in args.scales.split(","))

# Parse channel multipliers
channel_mult = tuple(int(m) for m in args.channel_mult.split(","))

# Handle attention/norm flags
use_attention = args.use_attention and not args.no_attention
use_norm = args.use_norm and not args.no_norm

# Initialize model
key, model_key = jax.random.split(key)
if args.var_mode:
    print(f"Using VAR multi-scale VQ-VAE with scales: {scales}")
    print(f"  base_channels={args.base_channels}, channel_mult={channel_mult}")
    print(f"  num_res_blocks={args.num_res_blocks}, attention={use_attention}, norm={use_norm}")
    model = VARVQVAE2d(
        channels=total_channels,
        hidden_dim=args.hidden_dim,
        codebook_dim=args.codebook_dim,
        vocab_size=args.vocab_size,
        scales=scales,
        decay=args.decay,
        base_channels=args.base_channels,
        channel_mult=channel_mult,
        num_res_blocks=args.num_res_blocks,
        use_attention=use_attention,
        use_norm=use_norm,
        attention_heads=args.attention_heads,
        key=model_key,
    )
    step_fn = make_step_var
else:
    print("Using standard VQ-VAE")
    print(f"  base_channels={args.base_channels}, channel_mult={channel_mult}")
    print(f"  num_res_blocks={args.num_res_blocks}, attention={use_attention}, norm={use_norm}")
    model = VQVAE2d(
        channels=total_channels,
        hidden_dim=args.hidden_dim,
        codebook_dim=args.codebook_dim,
        vocab_size=args.vocab_size,
        decay=args.decay,
        base_channels=args.base_channels,
        channel_mult=channel_mult,
        num_res_blocks=args.num_res_blocks,
        use_attention=use_attention,
        use_norm=use_norm,
        attention_heads=args.attention_heads,
        key=model_key,
    )
    step_fn = make_step

# Test forward pass
test_input = jnp.zeros((1, total_channels, 256, 256))
if args.var_mode:
    z_e, z_q, _, indices_list, _, y = jax.vmap(model)(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Latent shape (z_e): {z_e.shape}")
    print(f"Indices shapes: {[idx.shape for idx in indices_list]}")
    total_tokens = sum(s * s for s in scales)
    print(f"Total tokens per sample: {total_tokens}")
    print(f"Output shape: {y.shape}")
else:
    z_e, z_q, _, indices, y = jax.vmap(model)(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Latent shape (z_e): {z_e.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Output shape: {y.shape}")

# Calculate total training steps for LR schedule
steps_per_epoch = len(data_loader)  # DataLoader already accounts for batching
total_steps = steps_per_epoch * args.epochs
warmup_steps = min(1000, total_steps // 10)  # 10% of training or 1000 steps, whichever is smaller

print(f"Steps per epoch: {steps_per_epoch}")
print(f"LR schedule: {warmup_steps} warmup steps, {total_steps} total steps")

# Initialize optimizer with warmup + cosine decay
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=args.lr,
    warmup_steps=warmup_steps,
    decay_steps=total_steps,
    end_value=args.lr * 0.01,
)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adamw(schedule, weight_decay=1e-4),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# Initialize wandb
if WANDB_AVAILABLE:
    config = {
        "hidden_dim": args.hidden_dim,
        "codebook_dim": args.codebook_dim,
        "vocab_size": args.vocab_size,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "commitment_weight": args.commitment_weight,
        "decay": args.decay,
        "epochs": args.epochs,
        "seed": args.seed,
        "normalize": args.normalize,
        "var_mode": args.var_mode,
        # Capacity hyperparameters
        "base_channels": args.base_channels,
        "channel_mult": channel_mult,
        "num_res_blocks": args.num_res_blocks,
        "use_attention": use_attention,
        "use_norm": use_norm,
        "attention_heads": args.attention_heads,
    }
    if args.var_mode:
        config["scales"] = scales
        config["total_tokens"] = sum(s * s for s in scales)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=config
    )

# Training loop
global_step = 0
for epoch in range(args.epochs):
    print(f"\nEpoch {epoch + 1}/{args.epochs}")

    epoch_losses = []
    epoch_recon_losses = []
    epoch_commit_losses = []

    # Create dataloader for this epoch
    key, loader_key = jax.random.split(key)

    for batch_idx, batch_data in enumerate(data_loader):
        key, step_key = jax.random.split(key)
        
        # Convert PyTorch tensor to numpy
        # batch_data is (B, C, H, W)
        inputs = np.array(batch_data.numpy())
        inputs = jnp.array(inputs)

        model, opt_state, total_loss, recon_loss, commit_loss, indices_out, outputs = step_fn(
            model, optimizer, opt_state, inputs, step_key, args.commitment_weight
        )

        epoch_losses.append(float(total_loss))
        epoch_recon_losses.append(float(recon_loss))
        epoch_commit_losses.append(float(commit_loss))

        # Count unique codes used
        if args.var_mode:
            # indices_out is a list of arrays for each scale
            all_indices = np.concatenate([np.array(idx).flatten() for idx in indices_out])
            unique_codes = len(np.unique(all_indices))
        else:
            unique_codes = len(np.unique(np.array(indices_out)))

        # Log to wandb
        if WANDB_AVAILABLE:
            wandb.log({
                "loss/total": float(total_loss),
                "loss/reconstruction": float(recon_loss),
                "loss/commitment": float(commit_loss),
                "codebook/unique_codes": unique_codes,
                "step": global_step,
            })

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}: Loss={total_loss:.4f}, Recon={recon_loss:.4f}, Commit={commit_loss:.4f}, Codes={unique_codes}")

            # Log reconstructions and codebook usage
            if WANDB_AVAILABLE:
                recon_fig = plot_reconstruction(np.array(inputs), np.array(outputs), fields)
                if args.var_mode:
                    # Concatenate all indices for histogram
                    all_idx = np.concatenate([np.array(idx).flatten() for idx in indices_out])
                    usage_fig = plot_codebook_usage(all_idx, args.vocab_size)
                else:
                    usage_fig = plot_codebook_usage(indices_out, args.vocab_size)
                wandb.log({
                    "reconstructions": wandb.Image(recon_fig),
                    "codebook_usage": wandb.Image(usage_fig),
                })
                plt.close(recon_fig)
                plt.close(usage_fig)

        global_step += 1

    # Epoch summary
    avg_loss = np.mean(epoch_losses)
    avg_recon = np.mean(epoch_recon_losses)
    avg_commit = np.mean(epoch_commit_losses)
    print(f"Epoch {epoch + 1} Summary: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Commit={avg_commit:.4f}")

    if WANDB_AVAILABLE:
        wandb.log({
            "epoch/loss": avg_loss,
            "epoch/reconstruction": avg_recon,
            "epoch/commitment": avg_commit,
            "epoch": epoch + 1,
        })

    # Save checkpoint
    if (epoch + 1) % args.save_every == 0:
        save_checkpoint(model, opt_state, epoch + 1, args.checkpoint_dir, args.var_mode)

# Save final checkpoint
save_checkpoint(model, opt_state, args.epochs, args.checkpoint_dir, args.var_mode)

# Also save final weights to wandb run directory
if WANDB_AVAILABLE and wandb.run is not None:
    prefix = "var_vqvae" if args.var_mode else "vqvae"
    wandb_checkpoint_path = os.path.join(wandb.run.dir, f"{prefix}_final.eqx")
    eqx.tree_serialise_leaves(wandb_checkpoint_path, model)
    print(f"Saved final weights to {wandb_checkpoint_path}")
    wandb.finish()
print("\nTraining complete!")

# if __name__ == "__main__":
#     main()
