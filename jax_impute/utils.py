import jax
import jax.numpy as jnp
from jax import Array


def validate_input(X: Array) -> None:
    assert len(X.shape) == 2, "Expected float array as the input."
    assert jnp.issubdtype(X.dtype, jnp.floating), "Expected float array as the input."


@jax.vmap
def impute_by_mean(x: Array, inplace: bool = False) -> Array:
    mean = jnp.nanmean(x)
    return jnp.nan_to_num(x, copy=not inplace, nan=mean)
