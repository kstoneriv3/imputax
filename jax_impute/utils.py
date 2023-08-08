from typing import Optional
import jax
import jax.numpy as jnp
from jax import Array


def validate_input(X: Array, train: bool = False) -> None:
    assert len(X.shape) == 2, "Expected float array as the input."
    assert jnp.issubdtype(X.dtype, jnp.floating), "Expected float array as the input."
    if train:
        assert not jnp.any(jnp.all(jnp.isnan(X), axis=0))


def standardize(X: Array, mean: Optional[Array], std: Optional[Array]) -> tuple[Array, Array, Array]:
    if mean is None and std is None:
        mean = jnp.nanmean(X, axis=0)
        std = jnp.nanstd(X, axis=0)
    else:
        assert mean is not None
        assert std is not None
    X = (X - mean[None, :]) / std
    return X, mean, std


def unstandardize(X: Array, mean: Array, std: Array) -> Array:
    X = X * std + mean[None, :]
    return X


@jax.vmap
def impute_by_mean(x: Array, inplace: bool = False) -> Array:
    mean = jnp.nanmean(x)
    return jnp.nan_to_num(x, copy=not inplace, nan=mean)
