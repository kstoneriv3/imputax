import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from jax_impute.utils import impute_by_mean, validate_input


def impute_by_ppca(X: Array, n_components: int) -> Array:
    return PPCAImputer(n_components).fit_transform(X)


class PPCAImputer:
    """Missing value imputation by probabilistic PCA.

    The algorithm is as described in
    Verbeek, J. (2009). Notes on probabilistic PCA with missing values. Tech. report.
    """

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components

    def fit(self, X: Array) -> Self:
        validate_input(X)
        sigma, C = initialize_params(X)
        # E step
        pass

    def transform(self, X: Array) -> Array:
        pass

    def fit_transform(self, X: Array) -> Array:
        self.fit(X)
        return self.transform(X)


def initialize_params(X: Array) -> tuple[Array, Array]:
    # fill missing values by mean
    X = impute_by_mean(X)
