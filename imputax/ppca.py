from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import Self

from imputax import impute_by_pca
from imputax.utils import orthonormalize, standardize, validate_input

# TODO; maybe use namedtuple
PosteriorMoments = tuple[Float[Array, "n d"], Float[Array, "n d d"]]
FactorizedPosteriorMoments = tuple[
    Float[Array, "n m"],
    Float[Array, "n d"],
    Float[Array, ""],
    Float[Array, "d d"],
]
Parameters = tuple[Float[Array, "d m"], Float[Array, ""]]


def impute_by_ppca(
    X: Float[Array, "n m"],
    n_components: int,
    n_iter: int = 5,
    posterior_approximator: str = "exact",
) -> Array:
    return PPCAImputer(n_components, posterior_approximator).fit_transform(X, n_iter)


class PPCAImputer:
    """Missing value imputation by probabilistic PCA.

    We use similar notation to that of the following technical report [1] except that we use (X, Z)
    instead of (Y, X). In addition to the factorized posterior approximation described in the note,
    we implement the EM algorithm using the exact posterior.

    [1] Verbeek, J. (2009). Notes on probabilistic PCA with missing values. Tech. report.

    Args:
        n_components: The number of latent components of probabilistic PCA.
        posterior_approximator: The type of posteior approximator in the E-step to be used.
            It must be one of "exact" or "factorized", which correspond to exact and variational
            EM algorithm respectively.
    """

    def __init__(self, n_components: int, posterior_approximator: str = "exact") -> None:
        self.n_components = n_components
        self.posterior_approximator = posterior_approximator

    def fit(self, X: Float[Array, "n m"], n_iter: int = 5) -> Self:
        if self.posterior_approximator == "exact":
            return self.fit_with_exact_posterior(X, n_iter)
        elif self.posterior_approximator == "factorized":
            return self.fit_with_factorized_posterior(X, n_iter)
        else:
            raise ValueError(
                'posterior_approximator must be either "exact" or "factorized", '
                "but {self.posterior_approximator} is given."
            )

    def fit_with_exact_posterior(self, X: Float[Array, "n m"], n_iter: int = 5) -> Self:
        """For details, see imputax/doc/ppca_exact_em.md"""
        validate_input(X)
        X, mean, std = standardize(X)
        self._mean = mean
        self._std = std

        C, sigma2 = _get_initial_params(X, self.n_components)

        for i in range(n_iter):
            # E step
            EZ, CovZ = _e_step_exact(X, C, sigma2)
            # M step
            C, sigma2 = _m_step_exact(X, EZ, CovZ)

        C = orthonormalize(C)
        self._C = C
        self._sigma2 = sigma2
        return self

    def fit_with_factorized_posterior(self, X: Float[Array, "n m"], n_iter: int = 5) -> Self:
        """For details, see [1]."""
        validate_input(X)
        X, mean, std = standardize(X)
        self._mean = mean
        self._std = std

        C, sigma2 = _get_initial_params(X, self.n_components)
        EX = jnp.where(jnp.isnan(X), 0, X)
        EZ = jnp.linalg.lstsq(C.T, EX.T)[0].T

        for i in range(n_iter):
            # E step
            EX, EZ, varx, Covz = _e_step_factorized(X, C, sigma2, EX, EZ)
            # M step
            C, sigma2 = _m_step_factorized(X, EX, EZ, varx, Covz)

        C = orthonormalize(C)
        self._C = C
        self._sigma2 = sigma2
        return self

    def transform(self, X: Float[Array, "n m"]) -> Array:
        EZ, CovZ = _infer_posterior(X, self._C, self._sigma2)
        return jnp.where(jnp.isnan(X), EZ @ self._C, X)

    def fit_transform(self, X: Float[Array, "n m"], n_iter: int = 5) -> Array:
        self.fit(X, n_iter)
        return self.transform(X)


@jax.jit
def _e_step_factorized(
    X: Float[Array, "n m"],
    C: Float[Array, "d m"],
    sigma2: Float[Array, ""],
    EX_old: Float[Array, "n m"],
    EZ_old: Float[Array, "n d"],
) -> FactorizedPosteriorMoments:
    n_components = C.shape[0]
    EX = jnp.where(jnp.isnan(X), EZ_old @ C, X)
    varx = sigma2
    Covz = jnp.linalg.inv(jnp.eye(n_components) + C @ C.T / sigma2)
    # Instead of
    # EZ = EX_old @ C.T @ CovZ / sigma2
    # we can use up-to-date value of EX, when updating EZ.
    EZ = EX @ C.T @ Covz / sigma2
    return EX, EZ, varx, Covz


@jax.jit
def _m_step_factorized(
    X: Float[Array, "n m"],
    EX: Float[Array, "n m"],
    EZ: Float[Array, "n d"],
    varx: Float[Array, ""],
    Covz: Float[Array, "d d"],
) -> Parameters:
    n, m = X.shape
    n_missing = jnp.sum(jnp.isnan(X))
    C = jnp.linalg.inv(n * Covz + EZ.T @ EZ) @ (EZ.T @ EX)
    sigma2 = (
        1
        / (n * m)
        * (
            n * jnp.einsum("ij,im,jm->", Covz, C, C)
            + jnp.sum((EX - EZ @ C) ** 2)
            + n_missing * varx
        )
    )
    return C, sigma2


@jax.jit
def _e_step_exact(
    X: Float[Array, "n m"], C: Float[Array, "d m"], sigma2: Float[Array, ""]
) -> PosteriorMoments:
    return _infer_posterior(X, C, sigma2)


@jax.jit
def _m_step_exact(
    X: Float[Array, "n m"],
    EZ: Float[Array, "n d"],
    CovZ: Float[Array, "n d d"],
) -> Parameters:
    n, m = X.shape
    n_obs = n * m - jnp.sum(jnp.isnan(X))
    EZZ = CovZ + jax.vmap(jnp.outer)(EZ, EZ)
    C: Float[Array, "d m"] = jax.vmap(
        partial(_least_square_with_incomplete_y, batched_x=EZ, batched_xxT=EZZ),
        in_axes=1,
        out_axes=1,
        # )(batched_y=X)  # This is wrong, vmap always maps the first axis of a keyword argument
    )(X)
    sigma2 = (
        jnp.sum(jnp.where(jnp.isnan(X), 0, (X - EZ @ C) ** 2))
        + jnp.einsum("nkl,nm,km,lm->", CovZ, ~jnp.isnan(X), C, C)
    ) / n_obs
    return C, sigma2


def _least_square_with_incomplete_y(
    # The reason for extra space in " n" instead of "n" is described at
    # https://docs.kidger.site/jaxtyping/faq/#flake8-or-ruff-are-throwing-an-error
    batched_y: Float[Array, " n"],
    batched_x: Float[Array, "n d"],
    batched_xxT: Float[Array, "n d d"],
) -> Float[Array, " d"]:
    isnan = jnp.isnan(batched_y)
    XTX = jnp.where(isnan[:, None, None], 0, batched_xxT).sum(axis=0)
    XTy = jnp.where(isnan[:, None], 0, batched_x * batched_y[:, None]).sum(axis=0)
    return jnp.linalg.solve(XTX, XTy)


def _get_initial_params(X: Float[Array, "n m"], n_components: int) -> Parameters:
    # fill missing values by mean
    X = impute_by_pca(X, n_components)
    U, S, VT = jnp.linalg.svd(X, full_matrices=False)
    U = U[:, :n_components]
    S = S[:n_components]
    VT = VT[:n_components, :]
    X_approx = U @ jnp.diag(S) @ VT  # type: ignore [no-untyped-call]
    C = jnp.diag(S) @ VT  # type: ignore [no-untyped-call]
    sigma2 = jnp.var(X - X_approx)
    return C, sigma2


def _infer_posterior_row(
    x: Float[Array, " m"],
    C: Float[Array, "d m"],
    sigma2: Float[Array, ""],
) -> PosteriorMoments:
    n_components = C.shape[0]
    Co = jnp.where(jnp.isnan(x)[None, :], 0, C)
    xo = jnp.where(jnp.isnan(x), 0, x)
    Covz = jnp.linalg.inv(jnp.eye(n_components) + Co @ Co.T / sigma2)
    Ez = Covz @ (Co @ xo / sigma2)
    return Ez, Covz


def _infer_posterior(
    X: Float[Array, "n m"],
    C: Float[Array, "d m"],
    sigma2: Float[Array, ""],
) -> PosteriorMoments:
    return jax.vmap(partial(_infer_posterior_row, C=C, sigma2=sigma2))(X)
