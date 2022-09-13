import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from chex import Array
from typing import Any, Dict, List, Tuple, Union

def action_mask_categorical_policies(
    distribution: tfd.Categorical, mask: Array
) -> tfd.Categorical:
    """TODO Add description"""
    masked_logits = jnp.where(
        mask.astype(bool),
        distribution.logits,
        jnp.finfo(distribution.logits.dtype).min,
    )
    return tfd.Categorical(logits=masked_logits)

def compute_running_mean_var_count(
    stats: Array,
    batch: Array
) -> Array:

    """
    Updates the running mean, variance and data counts for Normalisation
    during training
    TODO consider the case where we can normlaise multidimensional arrays on specified axes
    For now we will get the mean of the whole array
    Args:
        stats (array)
            mean, var, count
        batch (array)
            current batch of data
    Returns:
        stats (array)
    """

    batch_mean = jnp.mean(batch)
    batch_var = jnp.var(batch)
    batch_count = batch.size

    mean, var, count = stats

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    # convert to float before returning
    # using emptying style () indexing because the ouptut can be a 0-dimensional array
    return jnp.array([new_mean, new_var, new_count])

def normalize(
    stats: Array, 
    batch: Array
    ):
    """Normlaise batch of data using the running mean and variance 
    TODO consider the case where we can normlaise multidimensional arrays on specified axes
    Args
        stats (array)
            mean, var, count
        batch (array)
            current batch of data
    Returns:
        denormalize batch (array)
    """
    
    mean, var, _ = stats
    normalize_batch = (batch - mean) / (jnp.sqrt(jnp.clip(var, a_min=1e-2))) 
    
    return normalize_batch


def denormalize(
    stats: Array, 
    batch: Array
    ):
    """Transform normalized data back into original distribution 
    Args
        stats (array)
            mean, var, count
        batch (array)
            current batch of data
    Returns:
        denormalize batch (array)
    """
    
    mean, var, _ = stats
    denormalize_batch = batch * jnp.sqrt(var) + mean
    
    return denormalize_batch
