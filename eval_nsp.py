"""
Evaluation script for NSP Autoregressive Video Rollout.

Generates trajectories (t0 -> t100) and saves them as side-by-side MP4 videos
to visualize dynamics over time. Also saves token trajectories and computes
histogram statistics comparing generated vs ground truth token distributions.
"""

import argparse
import json
import os
import jax
import jax.numpy as jnp
from jax import lax
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from nsp_model import (
    NextScalePredictor, NextScalePredConfig,
    build_temporal_mask,
)
from tokenizer import VQVAETokenizer, load_vqvae_checkpoint
from dataloaders import load_turbulence_data


def parse_args():
    parser = argparse.ArgumentParser(description="NSP Video Rollout")

    # Checkpoints
    parser.add_argument("--nsp_checkpoint", type=str, required=True,
                        help="Path to trained NSP model checkpoint")
    parser.add_argument("--vqvae_checkpoint", type=str, required=True,
                        help="Path to VQ-VAE checkpoint for decoding")
    parser.add_argument("--tokens_path", type=str, default="tokens.npz",
                        help="Path to tokens.npz")

    # Rollout Config
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of video trajectories to generate")
    parser.add_argument("--n_steps", type=int, default=100,
                        help="Number of autoregressive steps (frames)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Dataset index to start from")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for output video")

    # Sampling
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)

    # Model Params (Must match training)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)

    parser.add_argument("--output_dir", type=str, default="./eval_video")
    parser.add_argument("--seed", type=int, default=42)

    # Pixel-space comparison
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to raw .mat files for pixel-space comparison")
    parser.add_argument("--data_start_idx", type=int, default=10000,
                        help="Start index used when creating tokens.npz")

    return parser.parse_args()


def load_tokenized_data(path: str) -> dict:
    """Load tokenized data from npz file."""
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
        result["original_codebook"] = data["original_codebook"]
    return result


def construct_deterministic_seed(config: NextScalePredConfig) -> jax.Array:
    """Construct constant tokens for scales 0-3."""
    seed_tokens = []
    for i in range(config.first_trainable_scale):
        s = config.scales[i]
        val = config.scale_offsets[i] 
        seed_tokens.extend([val] * (s * s))
    return jnp.array(seed_tokens, dtype=jnp.int32)


def generate_trajectory(model, config, codebook, start_frame, attn_bias,
                        n_steps, key, temperature, top_k):
    """Autoregressive rollout: t0 -> t1 -> ... -> t_n."""
    trajectory = [start_frame]
    current_context = start_frame
    seed_tokens = construct_deterministic_seed(config)

    print(f"  Starting rollout ({n_steps} steps)...")
    
    for t in range(1, n_steps + 1):
        key, gen_key = jax.random.split(key)
        
        next_frame = model.generate(
            codebook, current_context, seed_tokens, attn_bias,
            key=gen_key, temperature=temperature, top_k=top_k
        )
        
        trajectory.append(next_frame)
        current_context = next_frame
        
        if t % 25 == 0:
            print(f"    Step {t}/{n_steps}")

    return jnp.stack(trajectory)


def decode_trajectory(tokens, tokenizer):
    """Decode a sequence of tokens into a sequence of images [T, H, W]."""
    # tokens shape: [T, tokens_per_frame]
    images = []
    for i in range(len(tokens)):
        # Decode returns [1, H, W], we take [0]
        img = tokenizer.decode_flat_indices(tokens[i])[0]
        images.append(np.array(img))
    return np.stack(images)


def save_token_trajectory(gen_tokens, gt_tokens, output_path, scales, scale_offsets):
    """Save token trajectories to npz file."""
    # Compute scale boundaries for splitting tokens
    scale_boundaries = []
    pos = 0
    for s in scales:
        scale_boundaries.append((pos, pos + s * s))
        pos += s * s

    np.savez(
        output_path,
        gen_tokens=np.array(gen_tokens),
        gt_tokens=np.array(gt_tokens),
        scales=np.array(scales),
        scale_offsets=np.array(scale_offsets),
        scale_boundaries=np.array(scale_boundaries),
    )
    print(f"  Saved token trajectory to {output_path}")


def compute_scale_histograms(gen_tokens, gt_tokens, scales, scale_offsets, scale_vocab_sizes):
    """
    Compute per-scale and overall histograms of token distributions.

    Returns dict with histogram data for each scale and overall.
    """
    # Compute scale boundaries (start, end) indices within a frame
    scale_boundaries = []
    pos = 0
    for s in scales:
        scale_boundaries.append((pos, pos + s * s))
        pos += s * s

    results = {}
    all_gen_tokens = []
    all_gt_tokens = []

    for scale_idx, (start, end) in enumerate(scale_boundaries):
        # Extract tokens for this scale across all frames
        gen_scale = gen_tokens[:, start:end].flatten()
        gt_scale = gt_tokens[:, start:end].flatten()

        all_gen_tokens.append(gen_scale)
        all_gt_tokens.append(gt_scale)

        # Get vocab range for this scale
        vocab_start = scale_offsets[scale_idx]
        vocab_end = vocab_start + scale_vocab_sizes[scale_idx]
        bins = np.arange(vocab_start, vocab_end + 1)

        # Compute histograms (normalized to probability)
        gen_hist, _ = np.histogram(gen_scale, bins=bins, density=True)
        gt_hist, _ = np.histogram(gt_scale, bins=bins, density=True)

        results[f"scale_{scale_idx}"] = {
            "scale_size": scales[scale_idx],
            "vocab_start": vocab_start,
            "vocab_end": vocab_end,
            "bins": bins[:-1],  # bin centers (left edges)
            "gen_hist": gen_hist,
            "gt_hist": gt_hist,
            "gen_tokens": gen_scale,
            "gt_tokens": gt_scale,
        }

    # Overall histogram (all scales combined)
    all_gen = np.concatenate(all_gen_tokens)
    all_gt = np.concatenate(all_gt_tokens)
    total_vocab = scale_offsets[-1] + scale_vocab_sizes[-1]
    bins_all = np.arange(0, total_vocab + 1)

    gen_hist_all, _ = np.histogram(all_gen, bins=bins_all, density=True)
    gt_hist_all, _ = np.histogram(all_gt, bins=bins_all, density=True)

    results["overall"] = {
        "bins": bins_all[:-1],
        "gen_hist": gen_hist_all,
        "gt_hist": gt_hist_all,
    }

    return results


def plot_histograms(hist_data, scales, output_path):
    """Plot per-scale and overall histograms comparing gen vs gt."""
    n_scales = len(scales)
    n_cols = 3
    n_rows = (n_scales + 1 + n_cols - 1) // n_cols  # +1 for overall

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # Plot per-scale histograms
    for scale_idx in range(n_scales):
        ax = axes[scale_idx]
        data = hist_data[f"scale_{scale_idx}"]

        bins = data["bins"]
        width = 0.4

        # Use bar plot for cleaner visualization
        ax.bar(bins - width/2, data["gt_hist"], width=width, alpha=0.7, label="GT", color="blue")
        ax.bar(bins + width/2, data["gen_hist"], width=width, alpha=0.7, label="Gen", color="orange")

        ax.set_title(f"Scale {scale_idx} ({data['scale_size']}×{data['scale_size']})")
        ax.set_xlabel("Token Index")
        ax.set_ylabel("Density")
        ax.legend()

        # Limit x-axis to non-zero region for clarity
        nonzero_mask = (data["gt_hist"] > 0) | (data["gen_hist"] > 0)
        if nonzero_mask.any():
            nonzero_bins = bins[nonzero_mask]
            margin = max(1, len(nonzero_bins) // 10)
            ax.set_xlim(nonzero_bins.min() - margin, nonzero_bins.max() + margin)

    # Plot overall histogram
    ax_overall = axes[n_scales]
    data_overall = hist_data["overall"]
    ax_overall.bar(data_overall["bins"], data_overall["gt_hist"], alpha=0.5, label="GT", color="blue")
    ax_overall.bar(data_overall["bins"], data_overall["gen_hist"], alpha=0.5, label="Gen", color="orange")
    ax_overall.set_title("Overall (All Scales)")
    ax_overall.set_xlabel("Token Index")
    ax_overall.set_ylabel("Density")
    ax_overall.legend()

    # Hide unused axes
    for idx in range(n_scales + 1, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved histogram plot to {output_path}")


def compute_histogram_metrics(hist_data, scales):
    """Compute quantitative metrics comparing gen vs gt distributions."""
    metrics = {}

    for scale_idx in range(len(scales)):
        data = hist_data[f"scale_{scale_idx}"]
        gen_hist = data["gen_hist"]
        gt_hist = data["gt_hist"]

        # Add small epsilon to avoid division by zero
        eps = 1e-10
        gen_hist = gen_hist + eps
        gt_hist = gt_hist + eps

        # Normalize to ensure valid probability distributions
        gen_hist = gen_hist / gen_hist.sum()
        gt_hist = gt_hist / gt_hist.sum()

        # KL divergence: KL(GT || Gen)
        kl_div = np.sum(gt_hist * np.log(gt_hist / gen_hist))

        # Jensen-Shannon divergence (symmetric)
        m = 0.5 * (gen_hist + gt_hist)
        js_div = 0.5 * np.sum(gen_hist * np.log(gen_hist / m)) + 0.5 * np.sum(gt_hist * np.log(gt_hist / m))

        # Total variation distance
        tv_dist = 0.5 * np.sum(np.abs(gen_hist - gt_hist))

        metrics[f"scale_{scale_idx}"] = {
            "kl_divergence": float(kl_div),
            "js_divergence": float(js_div),
            "tv_distance": float(tv_dist),
        }

    # Overall metrics
    data_overall = hist_data["overall"]
    gen_hist = data_overall["gen_hist"] + 1e-10
    gt_hist = data_overall["gt_hist"] + 1e-10
    gen_hist = gen_hist / gen_hist.sum()
    gt_hist = gt_hist / gt_hist.sum()

    m = 0.5 * (gen_hist + gt_hist)
    metrics["overall"] = {
        "kl_divergence": float(np.sum(gt_hist * np.log(gt_hist / gen_hist))),
        "js_divergence": float(0.5 * np.sum(gen_hist * np.log(gen_hist / m)) + 0.5 * np.sum(gt_hist * np.log(gt_hist / m))),
        "tv_distance": float(0.5 * np.sum(np.abs(gen_hist - gt_hist))),
    }

    return metrics


def compute_pixel_histograms(gen_pixels, recon_gt_pixels, raw_gt_pixels, n_bins=100):
    """
    Compute normalized histograms for pixel value distributions.

    Args:
        gen_pixels: Flattened array of generated pixel values
        recon_gt_pixels: Flattened array of reconstructed GT pixel values
        raw_gt_pixels: Flattened array of raw GT pixel values
        n_bins: Number of histogram bins

    Returns:
        dict with histogram data for all three distributions
    """
    # Determine bin range from raw GT (ground truth data range)
    bin_min = np.min(raw_gt_pixels)
    bin_max = np.max(raw_gt_pixels)
    # Add small margin to ensure all values are captured
    margin = (bin_max - bin_min) * 0.01
    bins = np.linspace(bin_min - margin, bin_max + margin, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Compute histograms (normalized to density)
    gen_hist, _ = np.histogram(gen_pixels, bins=bins, density=True)
    recon_gt_hist, _ = np.histogram(recon_gt_pixels, bins=bins, density=True)
    raw_gt_hist, _ = np.histogram(raw_gt_pixels, bins=bins, density=True)

    return {
        "bins": bins,
        "bin_centers": bin_centers,
        "gen_hist": gen_hist,
        "recon_gt_hist": recon_gt_hist,
        "raw_gt_hist": raw_gt_hist,
    }


def plot_pixel_histograms(hist_data, output_path):
    """
    Plot overlaid histograms comparing three pixel distributions.

    Args:
        hist_data: dict from compute_pixel_histograms
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bin_centers = hist_data["bin_centers"]
    width = (bin_centers[1] - bin_centers[0]) * 0.8

    # Plot as step histograms for clarity
    ax.step(bin_centers, hist_data["raw_gt_hist"], where='mid',
            label="Raw GT", color="blue", linewidth=2, alpha=0.8)
    ax.step(bin_centers, hist_data["recon_gt_hist"], where='mid',
            label="Reconstructed GT", color="green", linewidth=2, alpha=0.8)
    ax.step(bin_centers, hist_data["gen_hist"], where='mid',
            label="Generated", color="orange", linewidth=2, alpha=0.8)

    ax.set_xlabel("Pixel Value (Vorticity)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pixel Value Distribution Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved pixel histogram plot to {output_path}")


def compute_pixel_metrics(hist_data):
    """
    Compute JS divergence and TV distance between pixel distributions.

    Returns:
        dict with metrics for: Gen vs Raw GT, Gen vs Recon GT, Recon GT vs Raw GT
    """
    eps = 1e-10

    # Normalize histograms to valid probability distributions
    gen = hist_data["gen_hist"] + eps
    recon_gt = hist_data["recon_gt_hist"] + eps
    raw_gt = hist_data["raw_gt_hist"] + eps

    gen = gen / gen.sum()
    recon_gt = recon_gt / recon_gt.sum()
    raw_gt = raw_gt / raw_gt.sum()

    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

    def tv_distance(p, q):
        return float(0.5 * np.sum(np.abs(p - q)))

    return {
        "gen_vs_raw_gt": {
            "js_divergence": js_divergence(gen, raw_gt),
            "tv_distance": tv_distance(gen, raw_gt),
        },
        "gen_vs_recon_gt": {
            "js_divergence": js_divergence(gen, recon_gt),
            "tv_distance": tv_distance(gen, recon_gt),
        },
        "recon_gt_vs_raw_gt": {
            "js_divergence": js_divergence(recon_gt, raw_gt),
            "tv_distance": tv_distance(recon_gt, raw_gt),
        },
    }


def save_pixel_comparison_grid(raw_gt_imgs, recon_gt_imgs, gen_imgs, output_path):
    """
    Save a comparison grid showing Raw GT | Reconstructed GT | Generated at selected timesteps.

    Args:
        raw_gt_imgs: [T, H, W] raw ground truth images
        recon_gt_imgs: [T, H, W] reconstructed ground truth images
        gen_imgs: [T, H, W] generated images
        output_path: Path to save the PNG
    """
    # Fixed timesteps to show (0-indexed)
    timesteps = [0, 1, 4, 9, 99, 499, 999]  # t=1, 2, 5, 10, 100, 500, 1000

    # Filter to valid timesteps
    max_t = min(len(raw_gt_imgs), len(recon_gt_imgs), len(gen_imgs))
    valid_timesteps = [t for t in timesteps if t < max_t]

    if not valid_timesteps:
        print(f"  Warning: No valid timesteps for comparison grid (max_t={max_t})")
        return

    n_rows = len(valid_timesteps)
    n_cols = 3

    # Determine consistent color range across all images
    all_imgs = np.concatenate([
        raw_gt_imgs[valid_timesteps].flatten(),
        recon_gt_imgs[valid_timesteps].flatten(),
        gen_imgs[valid_timesteps].flatten()
    ])
    vmin, vmax = np.percentile(all_imgs, [1, 99])
    # Use symmetric range centered at 0 for vorticity
    vmax_abs = max(abs(vmin), abs(vmax))
    vmin, vmax = -vmax_abs, vmax_abs

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Raw GT", "Reconstructed GT", "Generated"]

    for row_idx, t in enumerate(valid_timesteps):
        images = [raw_gt_imgs[t], recon_gt_imgs[t], gen_imgs[t]]

        for col_idx, (img, title) in enumerate(zip(images, col_titles)):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax.axis('off')

            # Add column titles on first row
            if row_idx == 0:
                ax.set_title(title, fontsize=12)

            # Add timestep label on first column
            if col_idx == 0:
                ax.text(-0.15, 0.5, f"t={t+1}", transform=ax.transAxes,
                        fontsize=11, verticalalignment='center', fontweight='bold')

    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Vorticity')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved comparison grid to {output_path}")


def save_video(gen_imgs, gt_imgs, output_path, fps=10):
    """Save side-by-side video of Generated vs GT."""
    print(f"  Rendering video to {output_path}...")
    
    # Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    vmin, vmax = -10, 10
    
    # Initial plot
    im_gt = axes[0].imshow(gt_imgs[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    im_gen = axes[1].imshow(gen_imgs[0], cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title("Generated (Autoregressive)")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    def update(frame_idx):
        # Update data if frame exists
        if frame_idx < len(gt_imgs):
            im_gt.set_data(gt_imgs[frame_idx])
        else:
            # If GT is shorter than Gen (shouldn't happen with current logic)
            im_gt.set_data(np.zeros_like(gt_imgs[0]))
            
        if frame_idx < len(gen_imgs):
            im_gen.set_data(gen_imgs[frame_idx])
            
        fig.suptitle(f"Frame {frame_idx}", fontsize=14)
        return im_gt, im_gen

    # Create Animation
    num_frames = max(len(gen_imgs), len(gt_imgs))
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=1000/fps, blit=False
    )
    
    # Save
    try:
        anim.save(output_path, writer='ffmpeg', fps=fps)
    except Exception as e:
        print(f"Warning: MP4 save failed (ffmpeg installed?), trying GIF. Error: {e}")
        gif_path = output_path.replace(".mp4", ".gif")
        anim.save(gif_path, writer='pillow', fps=fps)
        print(f"Saved GIF to {gif_path}")
        
    plt.close(fig)


def main():
    args = parse_args()

    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    videos_dir = os.path.join(args.output_dir, "videos")
    tokens_dir = os.path.join(args.output_dir, "tokens")
    stats_dir = os.path.join(args.output_dir, "stats")
    comparisons_dir = os.path.join(args.output_dir, "comparisons")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(tokens_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    if args.data_dir:
        os.makedirs(comparisons_dir, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)

    # 1. Load Data
    print(f"Loading data from {args.tokens_path}...")
    token_data = load_tokenized_data(args.tokens_path)
    all_indices = token_data["indices_flat"]
    codebook = jnp.array(token_data["codebook"])
    scales = token_data["scales"]
    tokens_per_frame = sum(s*s for s in scales)
    
    # 2. Config
    config = NextScalePredConfig(
        tokens_per_frame=tokens_per_frame,
        scales=scales,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
        codebook_dim=token_data["codebook_dim"],
        unified_vocab_size=token_data["effective_vocab_size"],
        scale_vocab_sizes=tuple(int(x) for x in token_data["scale_vocab_sizes"]),
        scale_offsets=tuple(int(x) for x in token_data["scale_offsets"]),
        first_trainable_scale=4,
    )

    # 3. Model
    print(f"Loading NSP model from {args.nsp_checkpoint}...")
    key, model_key = jax.random.split(key)
    model = NextScalePredictor(config, codebook, model_key)
    model = eqx.tree_deserialise_leaves(args.nsp_checkpoint, model)
    attn_bias = build_temporal_mask(config.scales, config.padded_seq_len)

    # 4. Tokenizer
    print(f"Loading VQ-VAE from {args.vqvae_checkpoint}...")
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

    # 5. Load raw data if data_dir provided
    raw_data = None
    if args.data_dir:
        # Compute required range of raw data indices
        # Each sample starts at token index: args.start_idx + i * (args.n_steps + 100)
        # and spans n_steps + 1 frames
        max_sample_start = args.start_idx + (args.n_samples - 1) * (args.n_steps + 100)
        max_raw_idx = args.data_start_idx + max_sample_start + args.n_steps

        print(f"\nLoading raw .mat files from {args.data_dir}...")
        print(f"  Range: {args.data_start_idx} to {max_raw_idx}")
        raw_data = load_turbulence_data(
            args.data_dir,
            start_idx=args.data_start_idx,
            stop_idx=max_raw_idx
        )

    # 6. Loop & Generate Videos
    print(f"\nGenerating {args.n_samples} video trajectories...")

    # Accumulate all trajectories for aggregate statistics
    all_gen_trajectories = []
    all_gt_trajectories = []

    # Accumulate pixel values for pixel-space histograms (only if raw data available)
    all_gen_pixels = []
    all_recon_gt_pixels = []
    all_raw_gt_pixels = []

    for i in range(args.n_samples):
        # Select start index
        idx = args.start_idx + i * (args.n_steps + 100)
        if idx + args.n_steps >= len(all_indices):
            print("Dataset exhausted.")
            break

        print(f"\n--- Sample {i+1}/{args.n_samples} (Start Index {idx}) ---")

        # Get data
        t0_frame = jnp.array(all_indices[idx])
        gt_traj_tokens = jnp.array(all_indices[idx : idx + args.n_steps + 1])

        # Generate
        key, traj_key = jax.random.split(key)
        gen_traj_tokens = generate_trajectory(
            model, config, codebook, t0_frame, attn_bias,
            n_steps=args.n_steps, key=traj_key,
            temperature=args.temperature, top_k=args.top_k
        )

        # Accumulate for statistics
        all_gen_trajectories.append(np.array(gen_traj_tokens))
        all_gt_trajectories.append(np.array(gt_traj_tokens))

        # Save token trajectory
        token_out = os.path.join(tokens_dir, f"tokens_sample_{i}.npz")
        save_token_trajectory(
            gen_traj_tokens, gt_traj_tokens, token_out,
            scales, config.scale_offsets
        )

        # Decode to images [T, H, W]
        print("  Decoding frames...")
        gen_imgs = decode_trajectory(gen_traj_tokens, tokenizer)
        recon_gt_imgs = decode_trajectory(gt_traj_tokens, tokenizer)

        # Save Video
        out_name = os.path.join(videos_dir, f"rollout_sample_{i}.mp4")
        save_video(gen_imgs, recon_gt_imgs, out_name, fps=args.fps)

        # Pixel-space comparison (if raw data available)
        if raw_data is not None:
            # Load raw GT images for this trajectory
            raw_gt_imgs = []
            for t in range(args.n_steps + 1):
                raw_idx = args.data_start_idx + idx + t
                if raw_idx in raw_data:
                    raw_gt_imgs.append(raw_data[raw_idx])
                else:
                    print(f"  Warning: Missing raw data index {raw_idx}")
                    break
            raw_gt_imgs = np.stack(raw_gt_imgs)

            # Save comparison grid
            comparison_path = os.path.join(comparisons_dir, f"comparison_sample_{i}.png")
            save_pixel_comparison_grid(raw_gt_imgs, recon_gt_imgs, gen_imgs, comparison_path)

            # Accumulate pixel values for aggregate histograms
            all_gen_pixels.append(gen_imgs.flatten())
            all_recon_gt_pixels.append(recon_gt_imgs.flatten())
            all_raw_gt_pixels.append(raw_gt_imgs.flatten())

    # 7. Compute aggregate token statistics across all samples
    if all_gen_trajectories:
        print("\n--- Computing Token Histogram Statistics ---")

        # Concatenate all trajectories [total_frames, tokens_per_frame]
        all_gen = np.concatenate(all_gen_trajectories, axis=0)
        all_gt = np.concatenate(all_gt_trajectories, axis=0)

        print(f"  Total frames: {len(all_gen)} generated, {len(all_gt)} ground truth")

        # Compute histograms
        hist_data = compute_scale_histograms(
            all_gen, all_gt, scales,
            np.array(config.scale_offsets),
            np.array(config.scale_vocab_sizes)
        )

        # Plot histograms
        hist_plot_path = os.path.join(stats_dir, "token_histograms.png")
        plot_histograms(hist_data, scales, hist_plot_path)

        # Compute and save metrics
        token_metrics = compute_histogram_metrics(hist_data, scales)
        metrics_path = os.path.join(stats_dir, "token_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(token_metrics, f, indent=2)
        print(f"  Saved token metrics to {metrics_path}")

        # Print summary
        print("\n--- Token Histogram Metrics Summary ---")
        for scale_idx in range(len(scales)):
            m = token_metrics[f"scale_{scale_idx}"]
            print(f"  Scale {scale_idx} ({scales[scale_idx]}×{scales[scale_idx]}): "
                  f"JS={m['js_divergence']:.4f}, TV={m['tv_distance']:.4f}")
        m = token_metrics["overall"]
        print(f"  Overall: JS={m['js_divergence']:.4f}, TV={m['tv_distance']:.4f}")

    # 8. Compute aggregate pixel statistics (if raw data available)
    if all_raw_gt_pixels:
        print("\n--- Computing Pixel Histogram Statistics ---")

        # Concatenate all pixel values
        gen_pixels = np.concatenate(all_gen_pixels)
        recon_gt_pixels = np.concatenate(all_recon_gt_pixels)
        raw_gt_pixels = np.concatenate(all_raw_gt_pixels)

        print(f"  Total pixels: {len(gen_pixels):,} per distribution")

        # Compute pixel histograms
        pixel_hist_data = compute_pixel_histograms(gen_pixels, recon_gt_pixels, raw_gt_pixels)

        # Plot pixel histograms
        pixel_hist_path = os.path.join(stats_dir, "pixel_histograms.png")
        plot_pixel_histograms(pixel_hist_data, pixel_hist_path)

        # Compute and save pixel metrics
        pixel_metrics = compute_pixel_metrics(pixel_hist_data)
        pixel_metrics_path = os.path.join(stats_dir, "pixel_metrics.json")
        with open(pixel_metrics_path, "w") as f:
            json.dump(pixel_metrics, f, indent=2)
        print(f"  Saved pixel metrics to {pixel_metrics_path}")

        # Print summary
        print("\n--- Pixel Histogram Metrics Summary ---")
        print(f"  Gen vs Raw GT:     JS={pixel_metrics['gen_vs_raw_gt']['js_divergence']:.4f}, "
              f"TV={pixel_metrics['gen_vs_raw_gt']['tv_distance']:.4f}")
        print(f"  Gen vs Recon GT:   JS={pixel_metrics['gen_vs_recon_gt']['js_divergence']:.4f}, "
              f"TV={pixel_metrics['gen_vs_recon_gt']['tv_distance']:.4f}")
        print(f"  Recon GT vs Raw GT: JS={pixel_metrics['recon_gt_vs_raw_gt']['js_divergence']:.4f}, "
              f"TV={pixel_metrics['recon_gt_vs_raw_gt']['tv_distance']:.4f}")

    print(f"\nDone. Results saved to {args.output_dir}/")
    print(f"  videos/       - Side-by-side rollout videos")
    print(f"  tokens/       - Token trajectories (.npz)")
    print(f"  stats/        - Histograms and metrics")
    if args.data_dir:
        print(f"  comparisons/  - Raw GT | Recon GT | Gen comparison grids")

if __name__ == "__main__":
    main()