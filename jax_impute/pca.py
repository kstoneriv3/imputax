from typing import Callable, cast

import jax.numpy as jnp
from jax import Array

from jax_impute.utils import center, uncenter, validate_input

jnp_diag = cast(Callable[[Array], Array], jnp.diag)


def impute_by_pca(X: Array, n_components: int, n_iter: int = 1) -> Array:
    """Missing value imputation by PCA.

    This method can be extended to fit and transform API, by using low-rank update of SVD.
    """
    validate_input(X)
    X, mean = center(X)
    is_nan = jnp.isnan(X)
    X = X.at[is_nan].set(0)
    for i in range(n_iter):
        U, S, VT = jnp.linalg.svd(X, full_matrices=False)
        X_approx = U[:, :n_components] @ jnp_diag(S[:n_components]) @ VT[:n_components, :]
        X = jnp.where(is_nan, X_approx, X)
    return uncenter(X, mean)
