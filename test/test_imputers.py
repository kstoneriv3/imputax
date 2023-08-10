import jax
import jax.numpy as jnp
import pytest
from jax import Array

from jax_impute import impute_by_factor_model, impute_by_mean, impute_by_pca, impute_by_ppca

jax.config.update("jax_platform_name", "cpu")  # type: ignore [no-untyped-call]
jax.disable_jit()

key = jax.random.PRNGKey(0)
n = 10**4
m = 10
n_components = 3
lag = 10


def sample_toy_data(n: int, m: int, key: jax.random.KeyArray) -> Array:
    keys = jax.random.split(key, num=5)
    W = jax.random.normal(keys[0], (m * n_components,)).reshape(m, n_components)
    F = jnp.repeat(jnp.exp(-jnp.arange(lag) / lag)[:, None], n_components, axis=1)
    z0 = jax.random.normal(keys[1], (n * n_components,)).reshape(n, n_components)
    # add serial correlation by exponentially decaying causal filter
    z = jax.vmap(jax.scipy.signal.convolve, in_axes=1, out_axes=1)(z0, F)[:n]
    eps = jax.random.normal(keys[2], (n * m,)).reshape(n, m)
    scale = jax.random.uniform(keys[3], (m,))
    bias = jax.random.normal(keys[4], (m,))
    return z @ W.T + scale * eps + bias[None, :]


def add_nans(x: Array) -> Array:
    shape = x.shape
    x = x.flatten()
    x = x.at[::3].set(jnp.nan)
    return x.reshape(shape)


X0 = sample_toy_data(n, m, key)
X = add_nans(X0)


def score(x: Array) -> Array:
    SST = jnp.linalg.norm(X0 - impute_by_mean(X))
    SSR = jnp.linalg.norm(X0 - x)
    R2 = 1 - SSR / SST
    breakpoint()
    return R2


def test_impute_by_pca() -> None:
    assert score(impute_by_pca(X, n_components)) > 0.1


@pytest.mark.parametrize("posterior_approximator", ["exact", "factorized"])
def test_impute_by_ppca(posterior_approximator: str) -> None:
    assert (
        score(impute_by_ppca(X, n_components, posterior_approximator=posterior_approximator)) > 0.1
    )


@pytest.mark.parametrize("posterior_approximator", ["exact", "factorized"])
def test_impute_by_factor_model(posterior_approximator: str) -> None:
    assert (
        score(
            impute_by_factor_model(X, n_components, posterior_approximator=posterior_approximator)
        )
        > 0.1
    )
