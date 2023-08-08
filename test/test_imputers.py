import jax
import jax.numpy as jnp
from jax import Array

from jax_impute import impute_by_mean, impute_by_pca, impute_by_ppca

jax.config.update("jax_platform_name", "cpu")  # type: ignore [no-untyped-call]


key = jax.random.PRNGKey(0)
n_components = 10
lag = 10


def sample_toy_data(n: int, d: int, key: jax.random.KeyArray) -> Array:
    keys = jax.random.split(key, num=4)
    W = jax.random.normal(keys[0], (d * n_components,)).reshape(d, n_components)
    F = jnp.repeat(jnp.exp(-jnp.arange(lag) / lag)[:, None], n_components, axis=1)
    z0 = jax.random.normal(keys[1], (n * n_components,)).reshape(n, n_components)
    # add serial correlation by exponentially decaying causal filter
    z = jax.vmap(jax.scipy.signal.convolve, in_axes=1, out_axes=1)(z0, F)[:n]
    eps = jax.random.normal(keys[2], (n * d,)).reshape(n, d)
    bias = jax.random.normal(keys[2], (d,))
    return z @ W.T + eps + bias[None, :]


def add_nans(x: Array) -> Array:
    shape = x.shape
    x = x.flatten()
    x = x.at[::3].set(jnp.nan)
    return x.reshape(shape)


X0 = sample_toy_data(10**3, 10**2, key)
X = add_nans(X0)


def score(x: Array) -> Array:
    SST = jnp.linalg.norm(X0 - impute_by_mean(X))
    SSR = jnp.linalg.norm(X0 - x)
    R2 = 1 - SSR / SST
    return R2


def test_impute_by_pca() -> None:
    assert score(impute_by_pca(X, 10)) > 0.5


def test_impute_by_ppca() -> None:
    assert score(impute_by_ppca(X, 10)) > 0.5
