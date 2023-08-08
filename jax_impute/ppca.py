from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from typing_extensions import Self

from jax_impute import impute_by_pca
from jax_impute.utils import validate_input, standardize, unstandardize

PosteriorMoments = tuple[Array, Array]
Params = tuple[Array, Array]


def impute_by_ppca(X: Array, n_components: int, n_iter: int = 5) -> Array:
    return PPCAImputer(n_components).fit_transform(X, n_iter)


class PPCAImputer:
    """Missing value imputation by probabilistic PCA.

    We use similar notation to that of the following technical report [1] except that we use (X, Z)
    instead of (Y, X). However, they assume factorizable variational family q(z, x_h) = q(z)q(x_h),
    while we also implement full variational family that contains exact posterior.

    [1] Verbeek, J. (2009). Notes on probabilistic PCA with missing values. Tech. report.

    Args:
        n_components: The number of latent components of probabilistic PCA.
        posterior_approximator: The type of variational posteior approximator to be used.
            "full" or "factorizable".
    """

    def __init__(self, n_components: int, posterior_approximator: str = "full") -> None:
        self.n_components = n_components
        self.posterior_approximator = posterior_approximator

    def fit(self, X: Array, n_iter: int = 5) -> Self:
        if self.posterior_approximator == "full":
            return self.fit_with_full_posterior(X, n_iter)
        elif self.posterior_approximator == "factorizable":
            return self.fit_with_factorizable_posterior(X, n_iter)
        else:
            raise ValueError(
                f"posterior_approximator must be either \"full\" or \"factorizable\", "
                "but {self.posterior_approximator} is given."
            )

    def fit_exact(self, X: Array, n_iter: int = 5) -> Self:
        """For details, see jax-impute/doc/ppca_exact_em.md"""
        validate_input(X)
        X, mean, std = standardize(X)
        self._mean = mean
        self._std = std
        C, sigma2 = _get_initial_params(X, self.n_components)
        for i in range(n_iter):
            # E step
            EZ, CovZ = _e_step_exact(X, C, sigma2)
            # M step
            C, sigma2 = _m_step_exact(X, EZ, CovZ, C, sigma2)

        C = orthonormalize(C)
        self._C = C
        self._sigma2 = sigma2
        return self

    def fit_exact(self, X: Array, n_iter: int = 5) -> Self:
        raise NotImplementedError
        return self

    def transform(self, X: Array) -> Array:
        EZ, CovZ = _e_step_exact(X, self._C, self._sigma2)
        return jnp.where(jnp.isnan(X), EZ @ self._C, X)

    def fit_transform(self, X: Array, n_iter: int = 5) -> Array:
        self.fit(X, n_iter)
        return self.transform(X)


def _e_step_exact(X: Array, C: Array, sigma2: Array) -> PosteriorMoments:
    return _infer_posterior(X, C, sigma2)


def _m_step_exact(X: Array, EZ: Array, CovZ: Array, C_old: Array, sigma2_old: Array) -> Params:
    n_components = EZ.shape[1]
    n, d = X.shape
    n_obs = n * d - np.sum(np.isnan(X))
    EX = jnp.where(jnp.isnan(X), EZ @ C_old, X)
    C = jnp.linalg.solve(n * CovZ + EZ.T @ EZ, EZ.T @ EX)
    sigma2 = (
        n * jnp.trace(C @ C.T @ CovZ) + jnp.sum((EX - EZ @ C_old) ** 2) + n_components * sigma2_old
    ) / (n * d)
    return C, sigma2


def _get_initial_params(X: Array, n_components: int) -> Params:
    # fill missing values by mean
    X = impute_by_pca(X, n_components)
    U, S, VT = jnp.linalg.svd(X)
    U = U[:, :n_components]
    S = S[:n_components]
    V = VT[:n_components, :].T
    X_approx = U @ jnp.diag(S) @ V.T  # type: ignore [no-untyped-call]
    C = jnp.diag(S) @ V.T  # type: ignore [no-untyped-call]
    sigma2 = jnp.var(X - X_approx)
    return C, sigma2


def _infer_posterior_row(x: Array, C: Array, sigma2: Array) -> PosteriorMoments:
    n_components = C.shape[0]
    Co = jnp.where(jnp.isnan(x)[None, :], 0, C)
    xo = jnp.where(jnp.isnan(x), 0, x)
    Covz = jnp.linalg.inv(jnp.eye(n_components) + Co @ Co.T / sigma2)
    Ez = Covz @ (Co @ xo / sigma2)
    return Ez, Covz


def _infer_posterior(X: Array, C: Array, sigma2: Array) -> PosteriorMoments:
    return jax.vmap(partial(_infer_posterior_row, C=C, sigma2=sigma2))(X)


def orthonormalize(C: Array) -> Array:
    _, V = jnp.linalg.eigh(C @ C.T)
    return V.T @ C
        
