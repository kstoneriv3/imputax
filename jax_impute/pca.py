import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from jax_impute.utils import impute_by_mean, validate_input
from typing import Callable, cast

import jax.numpy as jnp
from jax import Array

from jax_impute.utils import center, uncenter, validate_input

jnp_diag = cast(Callable[[Array], Array], jnp.diag)


def impute_by_pca(X: Array, n_components: int, n_iter: int = 1) -> Array:
    """Missing value imputation by PCA.

    This corresponds maximum likelihood estimator of multivariate Gaussian model with
    low-rank mean matrix + ideosyncratic noise.
    """
    validate_input(X)
    X, mean = center(X)
    isnan = jnp.isnan(X)
    X = X.at[isnan].set(0)
    for i in range(n_iter):
        U, S, VT = jnp.linalg.svd(X, full_matrices=False)
        X_approx = U[:, :n_components] @ jnp_diag(S[:n_components]) @ VT[:n_components, :]
        X = jnp.where(isnan, X_approx, X)
    return uncenter(X, mean)
 

class PCAImputer:
    """Preliminary implementation of missing value imputation by PCA.

    The implementation of transform method is sub-optimal.
    """
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def fit(self, X: Array, n_iter: int = 3) -> Self:
        validate_input(X, train=True)
        X, mean = center(X)
        isnan = jnp.isnan(X)
        X = X.at[isnan].set(0)
        for i in range(n_iter):
            U, S, VT = jnp.linalg.svd(X, full_matrices=False)
            U = U[:, :self.n_components]
            S = S[:self.n_components]
            VT = VT[:self.n_components]
            X_approx = U @ jnp_diag(S) @ VT
            X = jnp.where(isnan, X_approx, X)
        self._mean = mean
        self._V = VT.T
        self._S = S
        return self

    def transform(self, X: Array) -> Array:
        validate_input(X)
        # vmap does not support non-concrete indexing and it memory inefficient...
        return jnp.stack([self._transform_row(x) for x in X])  

    def _transform_row(self, x: Array) -> Array:
        x -= self._mean
        isnan = jnp.isnan(x)
        VS = self._V @ jnp.diag(self._S)
        x = VS @ jnp.linalg.pinv(VS[~isnan, :]) @ x[~isnan]
        x += self._mean
        return x

    def fit_transform(self, X: Array) -> Array:
        self.fit(X)
        return self.transform(X)
