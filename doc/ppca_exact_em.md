# Exact EM algorithm for probabilistic PCA


Here, I will note the exact EM algorithm for probabilistic PCA with missing observations. This complements the note by Verbeek* [1], which derives the (variational) EM algorithm using the non-exact posterior approximation with factorizable variational family $\\{q(x_i^h, z_i): q(x_i^h, z_i) = q(x_i^h)q(z_i)\\}$. Though an exact E-step should lead to a better solution, his non-exact approximation is computationally more efficient we it does not require the matrix inversion for $\Sigma_i$ for all data points per each E-step.

*Note that I use $(X, Z)$ in place of $(Y, X)$ in his note.


## Model

The probabilistic PCA is an i.i.d. data generation process of $\\{x_i, y_i\\}_{i=1}^n$ where (unobservable) latent variable $z_i$ and observable variable $x_i$ follows

$$p(z_i) \sim \mathcal{N}(0, I)$$

$$p(x_i|z_i) \sim \mathcal{N}(C^T z_i, \sigma^2 I)$$


## Algorithm

### E-step:

$$q(z_i) = p(z_i|x^o_i) = \mathcal{N}(z| \mu_i, \Sigma_i),$$
where
$\Sigma_i = (I + \frac{1}{\sigma^2} C^o_i {C^o_i}^T)^{-1}$,
$\mu_i = \sigma^{-2} \Sigma_i^{-1}C_i^o x^o_i$,
and

$$q(x^h_i|z_i) = p(x^h_i|z_i, x^o_i) = p(x^h_i| z_i) = \mathcal{N}(x^h_i|{C^h_i}^T z_i, \sigma^2).$$

### M-step:

$$\hat C = 
\left(\sum_{i=1}^n \mathbb{E}\_q [x_iz_i^T] \right) 
\left(\sum_{i=1}^n \mathbb{E}_q [z_iz_i^T] \right)^{-1}
$$

$$\hat \sigma^2
= \frac{1}{nd} \sum_{i=1}^n \mathbb{E}\_q \\|x_i - \hat C^T z_i\\|^2 
= \frac{1}{nd} \sum_{i=1}^n \mathbb{E}_q \left[
    \\|x_i^h - (\hat C^h_i)^T z_i\\|^2
    + \\|x_i^o - (\hat C_i^o)^T \mu_i\\|^2
    + \\|(\hat C_i^o)^T (\mu_i^o -  z_i^o)\\|^2
\right]
$$

where we have

$\mathbb{E}_q[z_i(z_i)^T] = \Sigma_i + \mu_i(\mu_i)^T$, 
$\mathbb{E}_q[x_i^o (z_i)^T] = x_i^o(\mu_i)^T$,
$\mathbb{E}_q[x_i^h (z_i)^T] = (C^h_i)^T\left(\mathbb{E}_q[z_i(z_i)^T]\right)$,

and 

$\mathbb{E}_q \\|x_i^h - (\hat C^h_i)^T z_i\\|^2 =  \mathbb{E}_q \\|x_i^h - (C^h_i)^T z_i + (C^h_i)^T z_i - (\hat C^h_i)^T z_i\\|^2 = \sigma\_\text{old}^2 \cdot \mathrm{dim}(x^h_i) + \mathrm{tr}\left( (\hat C_i^h - C_i^h) (\hat C_i^h - C_i^h)^T (\Sigma_i + \mu_i \mu_i^T) \right)$, 

$\mathbb{E}_q \\|(\hat C_i^o)^T (\mu_i^o -  z_i^o) \\|^2 
= \mathrm{tr}\left( \hat C_i^o (\hat C_i^o)^T \Sigma_i \right)$.


[1] Verbeek, J. (2009). Notes on probabilistic PCA with missing values. Tech. report.
