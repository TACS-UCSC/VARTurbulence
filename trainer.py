import equinox as eqx
import jax
import jax.numpy as jnp
import typing as tp


def update_codebook_ema(model, updates: tuple, codebook_indices, key=None):
    """Update codebook using EMA and reinitialize over/under-used codes."""
    # avg_updates is (cluster_size, codebook_avg, codebook)
    avg_updates = jax.tree.map(lambda x: jnp.mean(x, axis=0), updates)

    # Normalize cluster_size to get probability distribution
    n_total = jnp.sum(avg_updates[0])
    h = avg_updates[0] / n_total

    # Reinitialize codes that deviate too much from uniform usage
    part_that_should_be = 1 / model.quantizer.K
    mask = (h > 2 * part_that_should_be) | (h < 0.5 * part_that_should_be)
    
    # Generate random embeddings for the dead codes
    rand_embed = (
        jax.random.normal(key, (model.quantizer.K, model.quantizer.D)) * mask[:, None]
    )

    # --- FIX START ---
    # 1. Reset cluster_size for dead codes to the mean size (giving them a "fresh start")
    # If we don't do this, they will be counted as "dead" again in the next step
    target_size = n_total / model.quantizer.K
    new_cluster_size = jnp.where(mask, target_size, avg_updates[0])

    # 2. Reset codebook_avg to match the new embedding * new size
    # Because: codebook = codebook_avg / cluster_size
    # Therefore: codebook_avg must equal codebook * cluster_size
    new_codebook_avg = jnp.where(
        mask[:, None],
        rand_embed * new_cluster_size[:, None],
        avg_updates[1]
    )

    # 3. Reset the codebook vector itself
    new_codebook = jnp.where(mask[:, None], rand_embed, avg_updates[2])

    avg_updates = (new_cluster_size, new_codebook_avg, new_codebook)
    # --- FIX END ---

    def where(q):
        return q.quantizer.cluster_size, q.quantizer.codebook_avg, q.quantizer.codebook

    model = eqx.tree_at(where, model, avg_updates)
    return model


@eqx.filter_value_and_grad(has_aux=True)
def calculate_losses_vqvae2d(model, x, commitment_weight=0.25):
    """Calculate VQ-VAE losses for 2D data."""
    z_e, z_q, codebook_updates, indices, y = jax.vmap(model)(x)

    reconstruct_loss = jnp.mean((x - y) ** 2)
    commit_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)
    total_loss = reconstruct_loss + commitment_weight * commit_loss

    return total_loss, (reconstruct_loss, commit_loss, codebook_updates, indices, y)


@eqx.filter_jit
def make_step(model, optimizer, opt_state, x, key, commitment_weight=0.25):
    """Single training step."""
    (total_loss, (reconstruct_loss, commit_loss, codebook_updates, indices, y)), grads = (
        calculate_losses_vqvae2d(model, x, commitment_weight)
    )

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    model = update_codebook_ema(model, codebook_updates, indices, key)

    return (
        model,
        opt_state,
        total_loss,
        reconstruct_loss,
        commit_loss,
        indices,
        y,
    )


def update_codebook_ema_multiscale(model, updates: tuple, indices_list: tp.List, key=None):
    """Update codebook using EMA for multi-scale VAR model."""
    avg_updates = jax.tree.map(lambda x: jnp.mean(x, axis=0), updates)

    # Normalize cluster_size to get probability distribution
    n_total = jnp.sum(avg_updates[0])
    h = avg_updates[0] / n_total

    # Reinitialize codes that deviate too much from uniform usage
    part_that_should_be = 1 / model.quantizer.K
    mask = (h < 0.25 * part_that_should_be) #"""(h > 2 * part_that_should_be) |""" 
    rand_embed = (
        jax.random.normal(key, (model.quantizer.K, model.quantizer.D)) * mask[:, None]
    )

    # --- FIX START (Identical logic to single scale) ---
    target_size = n_total / model.quantizer.K
    new_cluster_size = jnp.where(mask, target_size, avg_updates[0])

    new_codebook_avg = jnp.where(
        mask[:, None],
        rand_embed * new_cluster_size[:, None],
        avg_updates[1]
    )

    new_codebook = jnp.where(mask[:, None], rand_embed, avg_updates[2])

    avg_updates = (new_cluster_size, new_codebook_avg, new_codebook)
    # --- FIX END ---

    def where(q):
        return q.quantizer.cluster_size, q.quantizer.codebook_avg, q.quantizer.codebook

    model = eqx.tree_at(where, model, avg_updates)
    return model


@eqx.filter_value_and_grad(has_aux=True)
def calculate_losses_varvqvae2d(model, x, commitment_weight=0.1):
    """Calculate VAR VQ-VAE losses for 2D data."""
    z_e, z_q, codebook_updates, indices_list, commit_loss_per_sample, y = jax.vmap(model)(x)

    reconstruct_loss = jnp.mean((x - y) ** 2)
    commit_loss = jnp.mean(commit_loss_per_sample)
    total_loss = reconstruct_loss + commitment_weight * commit_loss

    return total_loss, (reconstruct_loss, commit_loss, codebook_updates, indices_list, y)


@eqx.filter_jit
def make_step_var(model, optimizer, opt_state, x, key, commitment_weight=0.1):
    """Single training step for VAR VQ-VAE."""
    (total_loss, (reconstruct_loss, commit_loss, codebook_updates, indices_list, y)), grads = (
        calculate_losses_varvqvae2d(model, x, commitment_weight)
    )

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    model = update_codebook_ema_multiscale(model, codebook_updates, indices_list, key)

    return (
        model,
        opt_state,
        total_loss,
        reconstruct_loss,
        commit_loss,
        indices_list,
        y,
    )