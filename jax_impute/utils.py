import jax
import jax.numpy as jnp
from jax import Array


def validate_input(X: Array) -> None:
    assert len(X.shape) == 2, "Expected float array as the input."
    assert jnp.issubdtype(X.dtype, jnp.floating), "Expected float array as the input."
    assert not jnp.any(jnp.all(jnp.isnan(X), axis=0))


def center(X: Array) -> tuple[Array, Array]:
    mean = jnp.nanmean(X, axis=0)
    X = X - mean[None, :]
    return X, mean


def uncenter(X: Array, mean: Array) -> Array:
    X = X + mean[None, :]
    return X


@jax.vmap
def impute_by_mean(x: Array, inplace: bool = False) -> Array:
    mean = jnp.nanmean(x)
    return jnp.nan_to_num(x, copy=not inplace, nan=mean)
