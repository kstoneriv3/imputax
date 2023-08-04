import jax
import jax.numpy as jnp
from jax.typing import Array
from typing_extensions import Self


class ProbabilisticPCA:
    def __init__(self) -> None:
        pass

    def fit(self, X: Array) -> Self:
        pass

    def transform(self, X: Array) -> Array:
        pass

    def fit_transform(self, X: Array) -> Array:
        self.fit(X)
        return self.transform(X)
