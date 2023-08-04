from typing import Callable, cast

import jax.numpy as jnp
from jax import Array

from jax_impute.utils import impute_by_mean, validate_input

jnp_diag = cast(Callable[[Array], Array], jnp.diag)


def impute_by_pca(X: Array, n_components: int) -> Array:
    """Missing value imputation by PCA.

    This method can be extended to fit and transform API, by using low-rank update of SVD.
    """
    validate_input(X)
    X = impute_by_mean(X)
    U, S, VT = jnp.linalg.svd(X, full_matrices=False)
    X_approx = U[:, :n_components] @ jnp_diag(S[:n_components]) @ VT[:n_components, :]
    return X_approx
    return jnp.where(jnp.isnan(X), X_approx, X)
