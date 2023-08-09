from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing_extensions import Self

from jax_impute import impute_by_pca
from jax_impute.utils import standardize, validate_input

PosteriorMoments = tuple[Float[Array, "n d"], Float[Array, "n d d"]]
Params = tuple[Float[Array, "d m"], Float[Array, ""]]


def impute_by_ppca(
    X: Float[Array, "n m"],
    n_components: int,
    n_iter: int = 5,
    posterior_approximator: str = "full",
) -> Array:
    return PPCAImputer(n_components, posterior_approximator).fit_transform(X, n_iter)


class PPCAImputer:
    """Missing value imputation by probabilistic PCA.

    We use similar notation to that of the following technical report [1] except that we use (X, Z)
    instead of (Y, X). However, they assume factorizable variational family q(z, x_h) = q(z)q(x_h),
    while we also implement full variational family that contains exact posterior.

    [1] Verbeek, J. (2009). Notes on probabilistic PCA with missing values. Tech. report.

    Args:
        n_components: The number of latent components of probabilistic PCA.
        posterior_approximator: The type of posteior approximator in the E-step to be used.
            It must be one of "full" or "factorizable", which correspond to exact and variational
            EM algorithm respectively.
    """

    def __init__(self, n_components: int, posterior_approximator: str = "full") -> None:
        self.n_components = n_components
        self.posterior_approximator = posterior_approximator

    def fit(self, X: Float[Array, "n m"], n_iter: int = 5) -> Self:
        if self.posterior_approximator == "full":
            return self.fit_with_full_posterior(X, n_iter)
        elif self.posterior_approximator == "factorizable":
            return self.fit_with_factorizable_posterior(X, n_iter)
        else:
            raise ValueError(
                'posterior_approximator must be either "full" or "factorizable", '
                "but {self.posterior_approximator} is given."
            )

    def fit_with_full_posterior(self, X: Float[Array, "n m"], n_iter: int = 5) -> Self:
        """For details, see jax-impute/doc/ppca_exact_em.md"""
        validate_input(X)
        X, mean, std = standardize(X)
        self._mean = mean
        self._std = std
        C, sigma2 = _get_initial_params(X, self.n_components)
        for i in range(n_iter):
            # E step
            EZ, CovZ = _e_step_full(X, C, sigma2)
            # M step
            C, sigma2 = _m_step_full(X, EZ, CovZ, C, sigma2)

        C = orthonormalize(C)
        self._C = C
        self._sigma2 = sigma2
        return self

    def fit_with_factorizable_posterior(self, X: Float[Array, "n m"], n_iter: int = 5) -> Self:
        raise NotImplementedError
        return self

    def transform(self, X: Float[Array, "n m"]) -> Array:
        EZ, CovZ = _infer_posterior(X, self._C, self._sigma2)
        return jnp.where(jnp.isnan(X), EZ @ self._C, X)

    def fit_transform(self, X: Float[Array, "n m"], n_iter: int = 5) -> Array:
        self.fit(X, n_iter)
        return self.transform(X)


def _e_step_full(
    X: Float[Array, "n m"], C: Float[Array, "d m"], sigma2: Float[Array, ""]
) -> PosteriorMoments:
    return _infer_posterior(X, C, sigma2)


def _m_step_full(
    X: Float[Array, "n m"],
    EZ: Float[Array, "n d"],
    CovZ: Float[Array, "n d d"],
    C_old: Float[Array, "d m"],
    sigma2_old: Float[Array, ""],
) -> Params:
    EZ.shape[1]
    n, d = X.shape
    n_obs = n * d - jnp.sum(jnp.isnan(X))
    EZZ = CovZ + jax.vmap(jnp.outer)(EZ, EZ)
    C: Float[Array, "d m"] = jax.vmap(
        partial(_least_square_with_incomplete_y, batched_x=EZ, batched_xxT=EZZ),
        in_axes=1,
        out_axes=1,
    # )(batched_y=X)  # This is wrong, vmap always maps the first axis of a keyword argument 
    )(X)
    sigma2 = (
        1
        / n_obs
        * (
            jnp.sum(jnp.where(jnp.isnan(X), 0, (X - EZ @ C) ** 2))
            + jnp.einsum("nkl,nm,km,lm->", CovZ, (~jnp.isnan(X)).astype(float), C, C)
        )
    )
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


def _get_initial_params(X: Float[Array, "n m"], n_components: int) -> Params:
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


def orthonormalize(C: Float[Array, "d m"]) -> Float[Array, "d m"]:
    _, V = jnp.linalg.eigh(C @ C.T)
    return V.T @ C
