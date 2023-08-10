# Imputax
Missing value imputation with probabilistic models in Jax.


# Installation
```bash
$ pip install -e .
```


# Usage

```python
>>> import jax.numpy as jnp
>>> from imputax import impute_by_ppca
>>> X = jnp.array([[1, 2, 3], [2, None, 4], [None, 5, 6], [None, 6, None]], dtype=float)
>>> impute_by_ppca(X)
>>> impute_by_ppca(X, n_components=1)
Array([[1.       , 2.       , 3.       ],
       [2.       , 2.4348595, 4.       ],
       [5.086713 , 5.       , 6.       ],
       [5.0619698, 6.       , 5.031307 ]], dtype=float32)
>>> 
```
