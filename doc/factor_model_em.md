# EM algorithms for the factor model

This note describes the EM algorithm for the factor model with missing observations. First, we introduce the variational EM algorithm similar to the one for probabilistic PCA by Verbeek* [1], using the non-exact posterior approximation with factorizable variational family $\\{q(x_i^h, z_i): q(x_i^h, z_i) = q(x_i^h)q(z_i)\\}$. 
On top of the variation EM algorithm, we will discuss the exact EM algorithm in the latter half of this note.

Note that even though an exact E-step should lead to a better solution, his non-exact approximation is computationally more efficient we it does not require the matrix inversion for $\Sigma_i$ for all data points per each E-step.

*Note that I use $(X, Z)$ in place of $(Y, X)$ in his note.


## Model

The factor model is an i.i.d. data generation process of $\\{x_i, y_i\\}_{i=1}^n$ where (unobservable) latent variable $z_i$ and observable variable $x_i$ follows

$$p(z_i) \sim \mathcal{N}(0, I)$$

$$p(x_i|z_i) \sim \mathcal{N}(C^T z_i, \Psi)$$

where $\Psi$ is a diagonal matrix.


## Exact EM Algorithm

### E-step:

$$q(z_i) = p(z_i|x^o_i) = \mathcal{N}(z| \mu_i, \Sigma_i),$$
where
$\Sigma_i = (I + C^o_i {\Psi^o_i}^{-1} {C^o_i}^T)^{-1}$,
$\mu_i = \sigma^{-2} \Sigma_i^{-1}C_i^o x^o_i$. Note that we do not use $q(x^h_i|z_i)$ in the exact EM algorithm.

### M-step:

$$\hat C = 
\left(\sum_{i=1}^n \mathbb{E}\_q [x_iz_i^T] \right) 
\left(\sum_{i=1}^n \mathbb{E}_q [z_i z_i^T] \right)^{-1}
$$


$$\hat \Psi_{kk}
= \frac{1}{|\\{{i: x\_{i, k}\text{ observed}}\\}|} \sum\_{i: x\_{i, k}\text{ observed}} \mathbb{E}_q
\left[ |(x_i^o - (\hat C_i^o)^T \mu_i)_k |^2 + |((\hat C_i^o)^T \mu_i - z_i^o)_k |^2 \right]
$$

where we have

$\mathbb{E}_q[z_i z_i^T] = \Sigma_i + \mu_i \mu_i^T$, 
$\mathbb{E}_q[x_i^o z_i^T] = x_i^o \mu_i^T$,
$\mathbb{E}_q[x_i^h z_i^T] = (C^h_i)^T\left(\mathbb{E}_q[z_i z_i^T]\right)$,

and 

$\mathbb{E}_q |(x_i^h - (\hat C^h_i)^T z_i)_k|^2 = \mathbb{E}_q |(x_i^h - (C^h_i)^T z_i + (C^h_i)^T z_i - (\hat C^h_i)^T z_i)_k|^2 = \sigma^2_k + [(C^h_i - \hat C^h_i)^T \Sigma (C^h_i - \hat C^h_i)]\_{kk}$ when $x\_{i, k}$ is not observed, 

$\mathbb{E}_q |((\hat C_i^o)^T (\mu_i - z_i)_k|^2 
= [(\hat C_i^o)^T \Sigma_i \hat C_i^o ]\_{kk}$ when $x\_{i,k}$ is observed.



## Variational EM Algorithm


### E-step:

$$q(z_i) \propto \exp \int \log p(x_i, z_i) \mathrm{d}q(x_i^h) = \mathcal{N}(z| \mu_{z_i}, \Sigma_z),$$
where
$\Sigma_z = (I+ C \Psi^{-1} C^T)^{-1}$ (this does not depend on $i$!),
$\mu_{z_i} = \Sigma^{-1}C \Psi^{-1} \mu_{x_i,\text{old}}$,
and

$$q(x^h_i) \propto \exp \int \log p(x_i, z_i) \mathrm{d}q(z_i) = \mathcal{N}(z| \mu_{x_i}^h, \Psi^h),$$

where $\mu_{x_i}^h = (C^h)^T\mu_{z_i,\text{old}}$.
We also set observable part of $\mu_{x_i}$ as $\mu_{x_i}^o=x^o_i$ for our convenience.

Here, it is possible to find the solution of the above variational inference with the recurrent formula 
$\mu_{z_i} = \Sigma^{-1}C \Psi^{-1} \mu_{x_i} = \Sigma^{-1} [C^o (\Psi^o)^{-1} x_i^o + C^h (\Psi^h)^{-1} (C^h)^T \mu_{z_i}]$,
but it involves solving linear systems for all data points. If we can afford such an amount of computing, we should probably use the exact EM algorithm.


### M-step:

$$\hat C = 
\left(\sum_{i=1}^n \mathbb{E}\_q [x_iz_i^T] \right) 
\left(\sum_{i=1}^n \mathbb{E}_q [z_i z_i^T] \right)^{-1}
$$


$$\hat \Psi
= \mathrm{diag} \left[
\frac{1}{n} \sum_{i=1}^n \mathbb{E}\_q 
    (x_i - \hat C^T z_i) (x_i - \hat C^T z_i)^T
\right]
= \mathrm{diag} 
\frac{1}{n} \sum_{i=1}^n \left[
\mathrm{Cov}[x_i] + (\mu_{x_i} - \hat C^T \mu_{z_i}) (\mu_{x_i} - \hat C^T \mu_{z_i})^T + \hat C^T \mathrm{Cov}[z_i] \hat C
\right]
$$

where we have

$\mathbb{E}\_q[z_i z_i^T] = \Sigma_z + \mu_z \mu_z^T$, 
$\mathbb{E}\_q[x_i z_i^T] = \mu_{x_i} \mu_{z_i}^T$,
$\mathrm{Cov}[x_i^h] = \Psi^h_\text{old}$,
$\mathrm{Cov}[x_i^o] = 0$.



[1] Verbeek, J. (2009). Notes on probabilistic PCA with missing values. Tech. report.
