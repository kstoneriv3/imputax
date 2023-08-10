from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def validate_input(X: Float[Array, "n m"], train: bool = False) -> None:
    assert len(X.shape) == 2, "Expected float array as the input."
    assert jnp.issubdtype(X.dtype, jnp.floating), "Expected float array as the input."
    if train:
        assert not jnp.any(jnp.all(jnp.isnan(X), axis=0))


def standardize(
    X: Float[Array, "n m"],
    mean: Optional[Float[Array, " m"]] = None,
    std: Optional[Float[Array, " m"]] = None,
) -> tuple[Float[Array, "n m"], Float[Array, " m"], Float[Array, " m"]]:
    if mean is None and std is None:
        mean = jnp.nanmean(X, axis=0)
        std = jnp.nanstd(X, axis=0)
    else:
        assert mean is not None
        assert std is not None
    X = (X - mean[None, :]) / std
    return X, mean, std


def unstandardize(
    X: Float[Array, "n m"],
    mean: Float[Array, " m"],
    std: Float[Array, " m"],
) -> Float[Array, "n m"]:
    X = X * std + mean[None, :]
    return X


@jax.vmap
def impute_by_mean(x: Float[Array, "n m"], inplace: bool = False) -> Float[Array, "n m"]:
    mean = jnp.nanmean(x)
    return jnp.nan_to_num(x, copy=not inplace, nan=mean)


def orthonormalize(C: Float[Array, "d m"]) -> Float[Array, "d m"]:
    _, V = jnp.linalg.eigh(C @ C.T)
    return V.T @ C
