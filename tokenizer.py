"""Tokenization pipeline for AR model training.

This module provides a tokenizer that wraps a trained VQ-VAE model to prepare
data for autoregressive model training. It supports both streaming (live) and
file-saving modes.

Key features:
- Load trained VQ-VAE checkpoint with matching config
- Encode data to discrete indices and codebook vectors (z_q)
- Remap sparse codebook usage to consecutive indices
- VAR mode: unified per-scale codebook with contiguous index ranges per scale
- Stream tokenized data or save to file for reuse
"""

import argparse
import json
import os
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from models import VARVQVAE2d, VQVAE2d


def load_vqvae_checkpoint(checkpoint_path: str, config: dict, key) -> eqx.Module:
    """Load a VQ-VAE model from checkpoint.

    Args:
        checkpoint_path: Path to the .eqx checkpoint file
        config: Dictionary with model configuration:
            - var_mode: bool - Use VARVQVAE2d if True, else VQVAE2d
            - hidden_dim, codebook_dim, vocab_size, decay
            - base_channels, channel_mult, num_res_blocks
            - use_attention, use_norm, attention_heads
            - scales (only for var_mode)
        key: JAX random key for model initialization

    Returns:
        Loaded model with checkpoint weights
    """
    var_mode = config.get("var_mode", False)

    if var_mode:
        model = VARVQVAE2d(
            hidden_dim=config.get("hidden_dim", 512),
            codebook_dim=config.get("codebook_dim", 64),
            vocab_size=config.get("vocab_size", 4096),
            scales=config.get("scales", ((1, 1), (2, 2), (4, 4), (8, 8), (16, 16))),
            in_channels=config.get("in_channels", 1),
            decay=config.get("decay", 0.99),
            base_channels=config.get("base_channels", 128),
            channel_mult=config.get("channel_mult", (1, 2, 4, 4)),
            num_res_blocks=config.get("num_res_blocks", 2),
            use_attention=config.get("use_attention", True),
            use_norm=config.get("use_norm", True),
            attention_heads=config.get("attention_heads", 8),
            key=key,
        )
    else:
        model = VQVAE2d(
            hidden_dim=config.get("hidden_dim", 512),
            codebook_dim=config.get("codebook_dim", 64),
            vocab_size=config.get("vocab_size", 512),
            decay=config.get("decay", 0.99),
            base_channels=config.get("base_channels", 128),
            channel_mult=config.get("channel_mult", (1, 2, 4, 4)),
            num_res_blocks=config.get("num_res_blocks", 2),
            use_attention=config.get("use_attention", True),
            use_norm=config.get("use_norm", True),
            attention_heads=config.get("attention_heads", 8),
            in_channels=config.get("in_channels", 1),
            key=key,
        )

    model = eqx.tree_deserialise_leaves(checkpoint_path, model)
    return model


def create_index_mapping(unique_indices: jnp.ndarray, vocab_size: int):
    """Create bidirectional mapping between sparse and consecutive indices.

    Args:
        unique_indices: Array of unique codebook indices actually used
        vocab_size: Total vocabulary size of the codebook

    Returns:
        old_to_new: Array of shape [vocab_size] mapping original -> remapped
                    Unused indices map to -1
        new_to_old: Array of shape [effective_vocab] mapping remapped -> original
    """
    unique_sorted = jnp.sort(unique_indices)
    effective_vocab = len(unique_sorted)

    # new_to_old: consecutive index -> original codebook index
    new_to_old = unique_sorted

    # old_to_new: original codebook index -> consecutive index (or -1 if unused)
    old_to_new = jnp.full(vocab_size, -1, dtype=jnp.int32)
    old_to_new = old_to_new.at[unique_sorted].set(jnp.arange(effective_vocab))

    return old_to_new, new_to_old


def flatten_multiscale_indices(indices_list):
    """Flatten multi-scale indices to a single 1D array.

    Args:
        indices_list: List of arrays with shapes [sh, sw] for each scale (sh, sw)

    Returns:
        Flattened 1D array of all indices concatenated
    """
    return jnp.concatenate([idx.flatten() for idx in indices_list])


def unflatten_to_scales(flat_indices, scales):
    """Unflatten 1D indices back to multi-scale list.

    Args:
        flat_indices: 1D array of concatenated indices
        scales: Tuple of (h, w) tuples (e.g., ((1,1), (2,2), (4,4), (8,8), (16,16)))

    Returns:
        List of arrays with shapes [sh, sw] for each scale (sh, sw)
    """
    indices_list = []
    offset = 0
    for (sh, sw) in scales:
        size = sh * sw
        idx = flat_indices[offset : offset + size].reshape(sh, sw)
        indices_list.append(idx)
        offset += size
    return indices_list


class VQVAETokenizer:
    """Tokenizer wrapping a trained VQ-VAE for AR model data preparation.

    This tokenizer:
    1. Encodes images to discrete codebook indices
    2. Remaps sparse codebook usage to consecutive indices
    3. In VAR mode, builds a unified per-scale codebook where each scale
       gets its own contiguous index range
    4. Provides access to codebook vectors for AR input embeddings

    Attributes:
        model: The trained VQ-VAE model (VQVAE2d or VARVQVAE2d)
        var_mode: Whether model is VARVQVAE2d
        scales: Tuple of scales for VAR mode (None for standard VQ-VAE)

        Standard VQ-VAE mapping (non-VAR):
            old_to_new: Mapping from original to remapped indices
            new_to_old: Mapping from remapped to original indices

        VAR mode unified mapping:
            scale_old_to_unified: List of [K] arrays, original -> unified index per scale
            unified_to_scale: [unified_vocab] scale index for each unified entry
            unified_to_original: [unified_vocab] original codebook index per entry
            unified_codebook: [unified_vocab, D] concatenated codebook
            scale_offsets: [n_scales] start of each scale's range in unified vocab
            scale_vocab_sizes: [n_scales] number of unique tokens per scale
    """

    def __init__(
        self,
        model: eqx.Module,
        old_to_new: Optional[jnp.ndarray] = None,
        new_to_old: Optional[jnp.ndarray] = None,
        first_trainable_scale: Optional[int] = None,
    ):
        self.model = model
        self.old_to_new = old_to_new
        self.new_to_old = new_to_old

        # Detect model type
        self.var_mode = isinstance(model, VARVQVAE2d)
        if self.var_mode:
            self.scales = model.quantizer.scales
        else:
            self.scales = None

        # Deterministic vs trainable scale tracking (VAR mode only)
        self.first_trainable_scale = first_trainable_scale
        self.deterministic_scales = None

        # Unified per-scale codebook mapping (VAR mode only, set by fit() or set_mapping())
        self.scale_old_to_unified = None
        self.unified_to_scale = None
        self.unified_to_original = None
        self.unified_codebook = None
        self.scale_offsets = None
        self.scale_vocab_sizes = None

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config: dict, key):
        """Load tokenizer from VQ-VAE checkpoint.

        Args:
            checkpoint_path: Path to .eqx checkpoint
            config: Model configuration dict
            key: JAX random key

        Returns:
            VQVAETokenizer instance (unfitted)
        """
        model = load_vqvae_checkpoint(checkpoint_path, config, key)
        return cls(
            model,
            first_trainable_scale=config.get("first_trainable_scale"),
        )

    @property
    def is_fitted(self) -> bool:
        """Whether the tokenizer has been fitted with index mappings."""
        if self.var_mode:
            return self.scale_old_to_unified is not None
        return self.old_to_new is not None

    @property
    def codebook(self) -> jnp.ndarray:
        """Original codebook [vocab_size, codebook_dim].

        For VARVQVAE2d with per-scale codebooks, returns the first scale's codebook.
        Use codebooks property to access all per-scale codebooks.
        """
        if self.var_mode:
            return self.model.quantizer.codebooks[0]
        return self.model.quantizer.codebook

    @property
    def codebooks(self):
        """Per-scale codebooks (VAR mode only). Tuple of [vocab_size, codebook_dim] arrays."""
        if not self.var_mode:
            raise ValueError("codebooks property only available in VAR mode")
        return self.model.quantizer.codebooks

    @property
    def codebook_dim(self) -> int:
        """Dimension of codebook vectors."""
        return self.model.quantizer.D

    @property
    def vocab_size(self) -> int:
        """Original vocabulary size (per-scale codebook size)."""
        return self.model.quantizer.K

    @property
    def effective_vocab_size(self) -> int:
        """Number of actually used codebook entries after fit().

        For VAR mode, this is the unified vocabulary size (sum of per-scale uniques).
        """
        if self.var_mode:
            if self.unified_codebook is None:
                raise ValueError("Tokenizer not fitted. Call fit() first.")
            return len(self.unified_codebook)
        if self.new_to_old is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        return len(self.new_to_old)

    @property
    def remapped_codebook(self) -> jnp.ndarray:
        """Codebook reordered by remapped indices [effective_vocab, codebook_dim].

        For VAR mode, returns the unified codebook (all scales concatenated
        with contiguous index ranges per scale).
        """
        if self.var_mode:
            if self.unified_codebook is None:
                raise ValueError("Tokenizer not fitted. Call fit() first.")
            return self.unified_codebook
        if self.new_to_old is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        return self.codebook[self.new_to_old]

    @property
    def remapped_codebooks(self):
        """Per-scale slices of the unified codebook (VAR mode only).

        Returns:
            Tuple of [scale_vocab_size, codebook_dim] arrays, one per scale.
        """
        if not self.var_mode:
            raise ValueError("remapped_codebooks property only available in VAR mode")
        if self.unified_codebook is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        result = []
        for k in range(len(self.scales)):
            start = int(self.scale_offsets[k])
            end = start + int(self.scale_vocab_sizes[k])
            result.append(self.unified_codebook[start:end])
        return tuple(result)

    @property
    def tokens_per_sample(self) -> int:
        """Total number of tokens per sample."""
        if self.var_mode:
            return sum(sh * sw for sh, sw in self.scales)
        else:
            # Standard VQ-VAE: 16x16 latent grid
            return 16 * 16

    def _collect_unique_indices_batch(self, batch: jnp.ndarray) -> jnp.ndarray:
        """Encode a batch and collect unique indices (non-VAR mode only)."""
        _, indices = jax.vmap(self.model.encode)(batch)
        all_indices = indices.reshape(-1)
        return jnp.unique(all_indices)

    def fit(self, data: np.ndarray, batch_size: int = 32):
        """First pass: collect unique indices and build mapping.

        For VAR mode, builds a unified per-scale codebook where each scale
        gets its own contiguous index range in the unified vocabulary.

        Args:
            data: numpy array of shape (N, 1, H, W) from load_turbulence_data_mat
            batch_size: Batch size for processing

        Returns:
            self (for chaining)
        """
        print("Fitting tokenizer: collecting unique codebook indices...")

        all_data = data  # already (N, 1, H, W)
        n_samples = len(all_data)

        n_batches = (n_samples + batch_size - 1) // batch_size

        if self.var_mode:
            # Collect per-scale unique indices
            n_scales = len(self.scales)
            per_scale_unique = [[] for _ in range(n_scales)]

            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_samples)
                batch = jnp.array(all_data[start:end])

                _, indices_list = jax.vmap(self.model.encode)(batch)
                # indices_list is list of [B, s, s] arrays
                for k in range(n_scales):
                    unique_k = jnp.unique(indices_list[k].reshape(-1))
                    per_scale_unique[k].append(np.array(unique_k))

                if (i + 1) % 50 == 0 or i == n_batches - 1:
                    print(f"  Processed {end}/{n_samples} samples")

            # Build unified mapping
            unified_offset = 0
            self.scale_old_to_unified = []
            unified_to_scale_parts = []
            unified_to_original_parts = []
            unified_codebook_parts = []
            scale_offsets = []
            scale_vocab_sizes = []

            for k in range(n_scales):
                combined = np.unique(np.concatenate(per_scale_unique[k]))
                unique_sorted = jnp.array(combined)
                n_unique = len(unique_sorted)

                scale_offsets.append(unified_offset)
                scale_vocab_sizes.append(n_unique)

                # Per-scale mapping: original index -> unified index
                old_to_unified_k = jnp.full(self.vocab_size, -1, dtype=jnp.int32)
                old_to_unified_k = old_to_unified_k.at[unique_sorted].set(
                    jnp.arange(n_unique, dtype=jnp.int32) + unified_offset
                )
                self.scale_old_to_unified.append(old_to_unified_k)

                # Reverse mappings
                unified_to_scale_parts.append(jnp.full(n_unique, k, dtype=jnp.int32))
                unified_to_original_parts.append(unique_sorted.astype(jnp.int32))

                # Unified codebook part from this scale's codebook
                unified_codebook_parts.append(self.codebooks[k][unique_sorted])

                sh, sw = self.scales[k]
                print(f"  Scale {sh}x{sw}: {n_unique:4d} unique codes "
                      f"(unified [{unified_offset}, {unified_offset + n_unique}))")

                unified_offset += n_unique

            self.scale_offsets = np.array(scale_offsets, dtype=np.int64)
            self.scale_vocab_sizes = np.array(scale_vocab_sizes, dtype=np.int64)
            self.unified_to_scale = jnp.concatenate(unified_to_scale_parts)
            self.unified_to_original = jnp.concatenate(unified_to_original_parts)
            self.unified_codebook = jnp.concatenate(unified_codebook_parts, axis=0)

            # Auto-detect deterministic scales (scales where only 1 unique code is used)
            if self.first_trainable_scale is None:
                last_deterministic = -1
                for k in range(n_scales):
                    if scale_vocab_sizes[k] == 1:
                        last_deterministic = k
                    else:
                        break
                self.first_trainable_scale = last_deterministic + 1
            self.deterministic_scales = list(range(self.first_trainable_scale))

            if self.first_trainable_scale > 0:
                det_names = [f"{sh}x{sw}" for sh, sw in self.scales[:self.first_trainable_scale]]
                print(f"Deterministic scales: {', '.join(det_names)} (indices 0-{self.first_trainable_scale - 1})")
                sh, sw = self.scales[self.first_trainable_scale]
                print(f"First trainable scale: {sh}x{sw} (index {self.first_trainable_scale})")
            else:
                print("No deterministic scales detected")

            print(f"Fit complete: {self.effective_vocab_size} unified vocab "
                  f"(from {self.vocab_size} per-scale codebook, {n_scales} scales)")
        else:
            # Standard VQ-VAE: shared mapping
            all_unique = []
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_samples)
                batch = jnp.array(all_data[start:end])

                unique = self._collect_unique_indices_batch(batch)
                all_unique.append(np.array(unique))

                if (i + 1) % 50 == 0 or i == n_batches - 1:
                    print(f"  Processed {end}/{n_samples} samples")

            all_unique_combined = np.unique(np.concatenate(all_unique))
            unique_indices = jnp.array(all_unique_combined)

            self.old_to_new, self.new_to_old = create_index_mapping(
                unique_indices, self.vocab_size
            )

            print(f"Fit complete: {self.effective_vocab_size}/{self.vocab_size} codes used "
                  f"({100 * self.effective_vocab_size / self.vocab_size:.1f}%)")

        return self

    def set_mapping(self, old_to_new=None, new_to_old=None, **kwargs):
        """Set index mapping manually (e.g., from loaded tokenized data).

        For standard VQ-VAE:
            old_to_new: [vocab_size] mapping array
            new_to_old: [effective_vocab] mapping array

        For VAR mode (unified per-scale codebook), pass as kwargs:
            scale_old_to_unified: list of [K] arrays per scale
            unified_to_scale: [unified_vocab] int array
            unified_to_original: [unified_vocab] int array
            unified_codebook: [unified_vocab, D] array
            scale_offsets: [n_scales] int array
            scale_vocab_sizes: [n_scales] int array
        """
        self.old_to_new = old_to_new
        self.new_to_old = new_to_old
        for attr in ('scale_old_to_unified', 'unified_to_scale', 'unified_to_original',
                     'unified_codebook', 'scale_offsets', 'scale_vocab_sizes'):
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])

    def encode(self, x: jnp.ndarray):
        """Encode a single sample.

        Args:
            x: Single input [1, H, W]

        Returns:
            remapped_indices: Remapped discrete indices
                - VAR mode: list of [s, s] arrays (unified indices per scale)
                - Standard: [H', W'] array
            z_q: Quantized vectors [D, H', W']
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        z_q, indices = self.model.encode(x)

        if self.var_mode:
            remapped = [self.scale_old_to_unified[k][idx] for k, idx in enumerate(indices)]
        else:
            remapped = self.old_to_new[indices]

        return remapped, z_q

    def encode_batch(self, batch: jnp.ndarray):
        """Encode a batch of samples.

        Args:
            batch: Batch of inputs [B, 1, H, W]

        Returns:
            remapped_indices: Remapped discrete indices
                - VAR mode: list of [B, s, s] arrays (unified indices per scale)
                - Standard: [B, H', W'] array
            z_q: Quantized vectors [B, D, H', W']
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        z_q, indices = jax.vmap(self.model.encode)(batch)

        if self.var_mode:
            remapped = [self.scale_old_to_unified[k][idx] for k, idx in enumerate(indices)]
        else:
            remapped = self.old_to_new[indices]

        return remapped, z_q

    def encode_batch_flat(self, batch: jnp.ndarray):
        """Encode a batch and return flattened indices and vectors.

        Args:
            batch: Batch of inputs [B, 1, H, W]

        Returns:
            indices_flat: [B, total_tokens] remapped discrete indices (unified for VAR)
            vectors_flat: [B, total_tokens, codebook_dim] corresponding vectors
        """
        remapped, z_q = self.encode_batch(batch)

        if self.var_mode:
            # remapped is list of [B, s, s] arrays with unified indices
            B = batch.shape[0]

            # Flatten indices: [B, total_tokens]
            indices_flat = jnp.concatenate(
                [idx.reshape(B, -1) for idx in remapped], axis=1
            )

            # Vectors from unified codebook directly
            vectors_flat = self.unified_codebook[indices_flat]  # [B, total_tokens, D]
        else:
            B = batch.shape[0]
            indices_flat = remapped.reshape(B, -1)  # [B, H'*W']
            original_flat = self.new_to_old[indices_flat]
            vectors_flat = self.codebook[original_flat]  # [B, H'*W', D]

        return indices_flat, vectors_flat

    def decode_indices(self, remapped_indices):
        """Convert remapped indices back to original and decode.

        Args:
            remapped_indices: Remapped indices
                - VAR mode: list of [s, s] arrays (unified indices per scale)
                - Standard: [H', W'] array

        Returns:
            Reconstruction [1, H, W]
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        if self.var_mode:
            original_indices = [self.unified_to_original[idx] for idx in remapped_indices]
            return self.model.decode_indices(original_indices)
        else:
            original_indices = self.new_to_old[remapped_indices]
            return self.model.decode_indices(original_indices)

    def decode_flat_indices(self, flat_remapped_indices: jnp.ndarray):
        """Decode from flattened remapped indices.

        Args:
            flat_remapped_indices: [total_tokens] flattened remapped indices

        Returns:
            Reconstruction [1, H, W]
        """
        if self.var_mode:
            indices_list = unflatten_to_scales(flat_remapped_indices, self.scales)
            return self.decode_indices(indices_list)
        else:
            # Standard VQ-VAE: 16x16 latent
            indices_2d = flat_remapped_indices.reshape(16, 16)
            return self.decode_indices(indices_2d)


def create_tokenized_dataloader(
    tokenizer: VQVAETokenizer,
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
):
    """Generator that tokenizes on-the-fly during iteration.

    Yields:
        indices: [B, total_tokens] remapped discrete indices
        vectors: [B, total_tokens, codebook_dim] corresponding codebook vectors
    """
    all_data = data  # already (N, 1, H, W)
    n_samples = len(all_data)

    # Shuffle if requested
    if shuffle:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_samples)
        all_data = all_data[perm]

    n_batches = n_samples // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch = jnp.array(all_data[start:end])

        indices, vectors = tokenizer.encode_batch_flat(batch)
        yield indices, vectors


def save_tokenized_data(
    path: str,
    tokenizer: VQVAETokenizer,
    data: np.ndarray,
    config: dict,
    batch_size: int = 32,
):
    """Tokenize and save entire dataset to NPZ.

    Args:
        path: Output path for .npz file
        tokenizer: Fitted VQVAETokenizer
        data: numpy array of shape (N, 1, H, W) from load_turbulence_data_mat
        config: Model configuration dict
        batch_size: Batch size for processing
    """
    if not tokenizer.is_fitted:
        raise ValueError("Tokenizer not fitted. Call fit() first.")

    print(f"Tokenizing and saving to {path}...")

    all_data = data  # already (N, 1, H, W)
    n_samples = len(all_data)

    # Collect all tokenized data
    all_indices = []
    all_vectors = []

    # Per-scale indices for VAR mode (keyed by scale index)
    per_scale_indices = {k: [] for k in range(len(tokenizer.scales or []))}

    n_batches = (n_samples + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)
        batch = jnp.array(all_data[start:end])

        indices_flat, vectors_flat = tokenizer.encode_batch_flat(batch)
        all_indices.append(np.array(indices_flat))
        all_vectors.append(np.array(vectors_flat))

        # Also collect per-scale indices for VAR mode
        if tokenizer.var_mode:
            remapped, _ = tokenizer.encode_batch(batch)
            for scale_idx in range(len(tokenizer.scales)):
                per_scale_indices[scale_idx].append(np.array(remapped[scale_idx]))

        if (i + 1) % 50 == 0 or i == n_batches - 1:
            print(f"  Processed {end}/{n_samples} samples")

    # Concatenate all batches
    all_indices = np.concatenate(all_indices, axis=0)
    all_vectors = np.concatenate(all_vectors, axis=0)

    # Prepare save dict (common fields)
    save_dict = {
        "codebook": np.array(tokenizer.remapped_codebook),
        "effective_vocab_size": np.array(tokenizer.effective_vocab_size),
        "vocab_size": np.array(tokenizer.vocab_size),
        "codebook_dim": np.array(tokenizer.codebook_dim),
        # Tokens (both representations)
        "indices_flat": all_indices,
        "vectors_flat": all_vectors,
        # Config as JSON string
        "config_json": np.array(json.dumps(config)),
        "var_mode": np.array(tokenizer.var_mode),
    }

    if tokenizer.var_mode:
        # Unified per-scale codebook mapping
        save_dict["scales"] = np.array(tokenizer.scales)
        save_dict["scale_offsets"] = np.array(tokenizer.scale_offsets)
        save_dict["scale_vocab_sizes"] = np.array(tokenizer.scale_vocab_sizes)
        save_dict["unified_to_scale"] = np.array(tokenizer.unified_to_scale)
        save_dict["unified_to_original"] = np.array(tokenizer.unified_to_original)
        if tokenizer.first_trainable_scale is not None:
            save_dict["first_trainable_scale"] = np.array(tokenizer.first_trainable_scale)

        for k, (sh, sw) in enumerate(tokenizer.scales):
            save_dict[f"indices_scale_{sh}x{sw}"] = np.concatenate(per_scale_indices[k], axis=0)
            save_dict[f"original_codebook_scale_{k}"] = np.array(tokenizer.codebooks[k])
            save_dict[f"scale_old_to_unified_{k}"] = np.array(tokenizer.scale_old_to_unified[k])
    else:
        # Standard VQ-VAE mapping
        save_dict["old_to_new"] = np.array(tokenizer.old_to_new)
        save_dict["new_to_old"] = np.array(tokenizer.new_to_old)
        save_dict["original_codebook"] = np.array(tokenizer.codebook)

    np.savez_compressed(path, **save_dict)
    print(f"Saved tokenized data: {all_indices.shape[0]} samples, "
          f"{tokenizer.effective_vocab_size} effective vocab, "
          f"{all_indices.shape[1]} tokens per sample")


def load_tokenized_data(path: str) -> dict:
    """Load tokenized data for AR training.

    Args:
        path: Path to .npz file

    Returns:
        Dictionary with:
            - indices_flat: [N, total_tokens] discrete targets
            - vectors_flat: [N, total_tokens, D] input embeddings
            - codebook: [effective_vocab, D] remapped codebook (unified for VAR)
            - effective_vocab_size: int
            - vocab_size: int
            - codebook_dim: int
            - config: dict (parsed from JSON)
            - var_mode: bool

            Standard VQ-VAE only:
            - original_codebook: [vocab_size, D] original codebook
            - old_to_new: [vocab_size] mapping
            - new_to_old: [effective_vocab] mapping

            VAR mode only:
            - scales: tuple of (h, w) tuples
            - scale_offsets: [n_scales] start of each scale's unified range
            - scale_vocab_sizes: [n_scales] unique codes per scale
            - unified_to_scale: [unified_vocab] scale index per entry
            - unified_to_original: [unified_vocab] original codebook index per entry
            - scale_old_to_unified: list of [K] per-scale mapping arrays
            - indices_scale_{sh}x{sw}: [N, sh, sw] per-scale indices
            - original_codebooks: list of [vocab_size, D] per-scale original codebooks
            - first_trainable_scale: int (if available)
    """
    data = dict(np.load(path, allow_pickle=True))

    # Convert scalar arrays to Python types
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
        # Unified per-scale codebook fields
        result["scales"] = tuple(tuple(s) for s in data["scales"].tolist())
        n_scales = len(result["scales"])

        result["scale_offsets"] = data["scale_offsets"]
        result["scale_vocab_sizes"] = data["scale_vocab_sizes"]
        result["unified_to_scale"] = data["unified_to_scale"]
        result["unified_to_original"] = data["unified_to_original"]

        if "first_trainable_scale" in data:
            result["first_trainable_scale"] = int(data["first_trainable_scale"])

        scale_old_to_unified = []
        original_codebooks = []
        for k, (sh, sw) in enumerate(result["scales"]):
            key = f"indices_scale_{sh}x{sw}"
            if key in data:
                result[key] = data[key]
            mapping_key = f"scale_old_to_unified_{k}"
            if mapping_key in data:
                scale_old_to_unified.append(data[mapping_key])
            ocb_key = f"original_codebook_scale_{k}"
            if ocb_key in data:
                original_codebooks.append(data[ocb_key])
        result["scale_old_to_unified"] = scale_old_to_unified
        if original_codebooks:
            result["original_codebooks"] = original_codebooks
    else:
        # Standard VQ-VAE fields
        result["original_codebook"] = data["original_codebook"]
        result["old_to_new"] = data["old_to_new"]
        result["new_to_old"] = data["new_to_old"]

    return result


def print_tokenizer_info(tokenizer: VQVAETokenizer, data: np.ndarray):
    """Print statistics about the tokenizer and data.

    Args:
        tokenizer: Fitted VQVAETokenizer
        data: numpy array of shape (N, 1, H, W)
    """
    print("\n" + "=" * 60)
    print("TOKENIZER INFO")
    print("=" * 60)

    print(f"\nModel type: {'VARVQVAE2d' if tokenizer.var_mode else 'VQVAE2d'}")
    print(f"Original vocab size (per-scale): {tokenizer.vocab_size}")
    print(f"Effective vocab size: {tokenizer.effective_vocab_size}")
    print(f"Codebook dimension: {tokenizer.codebook_dim}")

    if tokenizer.var_mode:
        scale_strs = [f"{sh}x{sw}" for sh, sw in tokenizer.scales]
        print(f"\nScales: {', '.join(scale_strs)}")
        print(f"Tokens per scale: {[sh*sw for sh, sw in tokenizer.scales]}")
        print(f"Total tokens per sample: {tokenizer.tokens_per_sample}")

        if tokenizer.first_trainable_scale is not None:
            print(f"First trainable scale: index {tokenizer.first_trainable_scale}")
            if tokenizer.deterministic_scales:
                det_names = [f"{sh}x{sw}" for sh, sw in tokenizer.scales[:tokenizer.first_trainable_scale]]
                print(f"Deterministic scales: {', '.join(det_names)}")

        # Unified codebook breakdown
        print(f"\nUnified codebook breakdown:")
        for k, (sh, sw) in enumerate(tokenizer.scales):
            offset = int(tokenizer.scale_offsets[k])
            n = int(tokenizer.scale_vocab_sizes[k])
            print(f"  Scale {sh}x{sw}: {n:4d} unique codes "
                  f"(unified [{offset}, {offset + n}), "
                  f"{sh*sw} positions/frame)")
    else:
        print(f"Codebook utilization: {100 * tokenizer.effective_vocab_size / tokenizer.vocab_size:.1f}%")
        print(f"\nTokens per sample: {tokenizer.tokens_per_sample} (16x16 grid)")

    print(f"\nDataset size: {len(data)} samples")
    print(f"Total tokens in dataset: {len(data) * tokenizer.tokens_per_sample}")

    # Codebook usage histogram
    if tokenizer.var_mode:
        for k, (sh, sw) in enumerate(tokenizer.scales):
            print(f"\nCodebook usage distribution (scale {sh}x{sw}):")
            usage = np.array(tokenizer.model.quantizer.cluster_sizes[k])
            total_usage = usage.sum()
            if total_usage > 0:
                nonzero_mask = usage > 0
                print(f"  Codes with usage > 0: {nonzero_mask.sum()}")
                print(f"  Max usage: {usage.max():.0f} ({100*usage.max()/total_usage:.2f}%)")
                print(f"  Min nonzero usage: {usage[nonzero_mask].min():.0f} ({100*usage[nonzero_mask].min()/total_usage:.4f}%)")
                print(f"  Mean usage: {usage.mean():.1f}")
    else:
        print("\nCodebook usage distribution:")
        usage = np.array(tokenizer.model.quantizer.cluster_size)
        total_usage = usage.sum()
        if total_usage > 0:
            nonzero_mask = usage > 0
            print(f"  Codes with usage > 0: {nonzero_mask.sum()}")
            print(f"  Max usage: {usage.max():.0f} ({100*usage.max()/total_usage:.2f}%)")
            print(f"  Min nonzero usage: {usage[nonzero_mask].min():.0f} ({100*usage[nonzero_mask].min()/total_usage:.4f}%)")
            print(f"  Mean usage: {usage.mean():.1f}")

    print("=" * 60 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize turbulence data using trained VQ-VAE"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--checkpoint", type=str, required=True, help="Path to VQ-VAE checkpoint"
    )
    common.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing .mat files"
    )
    common.add_argument("--start_idx", type=int, default=10000, help="Start index")
    common.add_argument("--stop_idx", type=int, default=19999, help="Stop index")
    common.add_argument("--batch_size", type=int, default=32, help="Batch size")
    common.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model config arguments
    common.add_argument("--var_mode", action="store_true", help="Use VAR mode")
    common.add_argument("--hidden_dim", type=int, default=512)
    common.add_argument("--codebook_dim", type=int, default=64)
    common.add_argument("--vocab_size", type=int, default=2048)
    common.add_argument("--decay", type=float, default=0.99)
    common.add_argument("--base_channels", type=int, default=64)
    common.add_argument(
        "--channel_mult", type=str, default="2,2,4,8", help="Comma-separated"
    )
    common.add_argument("--num_res_blocks", type=int, default=2)
    common.add_argument("--use_attention", action="store_true", default=True)
    common.add_argument("--no_attention", action="store_true")
    common.add_argument("--use_norm", action="store_true", default=True)
    common.add_argument("--no_norm", action="store_true")
    common.add_argument("--attention_heads", type=int, default=8)
    common.add_argument(
        "--scales", type=str, default="1x1,2x2,3x3,4x4,5x5,6x6,8x8,10x10,13x13,16x16",
        help="Comma-separated HxW scales (e.g. 1x1,2x2,4x4,8x8,16x16)",
    )
    common.add_argument(
        "--first_trainable_scale", type=int, default=None,
        help="Index of first trainable scale (auto-detected if not set)",
    )

    # Save command
    save_parser = subparsers.add_parser(
        "save", parents=[common], help="Tokenize and save to file"
    )
    save_parser.add_argument(
        "--output", type=str, required=True, help="Output .npz file path"
    )

    # Info command
    subparsers.add_parser(
        "info", parents=[common], help="Show tokenizer stats without saving"
    )

    return parser.parse_args()


def build_config_from_args(args) -> dict:
    """Build config dict from parsed arguments."""
    channel_mult = tuple(int(m) for m in args.channel_mult.split(","))
    scales = tuple(
        tuple(int(d) for d in s.split("x")) for s in args.scales.split(",")
    )

    config = {
        "var_mode": args.var_mode,
        "hidden_dim": args.hidden_dim,
        "codebook_dim": args.codebook_dim,
        "vocab_size": args.vocab_size,
        "decay": args.decay,
        "base_channels": args.base_channels,
        "channel_mult": channel_mult,
        "num_res_blocks": args.num_res_blocks,
        "use_attention": args.use_attention and not args.no_attention,
        "use_norm": args.use_norm and not args.no_norm,
        "attention_heads": args.attention_heads,
        "scales": scales,
    }

    if args.first_trainable_scale is not None:
        config["first_trainable_scale"] = args.first_trainable_scale

    return config


def main():
    args = parse_args()

    if args.command is None:
        print("Error: No command specified. Use 'save' or 'info'.")
        return

    # Build config
    config = build_config_from_args(args)

    # Load data
    from dataloaders import load_turbulence_data_mat

    print("Loading turbulence data...")
    data = load_turbulence_data_mat(
        args.data_dir, start_idx=args.start_idx, stop_idx=args.stop_idx
    )
    print(f"Loaded {len(data)} samples")

    # Load tokenizer
    print(f"Loading VQ-VAE from {args.checkpoint}...")
    key = jax.random.PRNGKey(args.seed)
    tokenizer = VQVAETokenizer.from_checkpoint(args.checkpoint, config, key)

    # Fit tokenizer
    tokenizer.fit(data, batch_size=args.batch_size)

    if args.command == "info":
        print_tokenizer_info(tokenizer, data)

        # Verify round-trip
        print("Verifying round-trip encoding/decoding...")
        sample = jnp.array(data[0])  # [1, H, W]

        remapped, z_q = tokenizer.encode(sample)
        recon = tokenizer.decode_indices(remapped)

        mse = float(jnp.mean((sample - recon) ** 2))
        print(f"Round-trip MSE: {mse:.6f}")

    elif args.command == "save":
        save_tokenized_data(
            args.output, tokenizer, data, config, batch_size=args.batch_size
        )

        # Print info after saving
        print_tokenizer_info(tokenizer, data)


if __name__ == "__main__":
    main()
