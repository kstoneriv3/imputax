# Exact EM algorithm for probabilistic PCA

### E step:

$$p(z_i|x^o_i) = \mathcal{N}(z| \mu_i, \Sigma_i),$$
where
$\Sigma_i = (I + \frac{1}{\sigma^2} C^o_i {C^o_i}^T)^{-1}$,
$\mu_i = \sigma^{-2} \Sigma_i^{-1}C_i^o x^o_i$,
and

$$p(x^h_i|z_i, x^o_i) = p(x^h_i| z_i) = \mathcal{N}(x^h_i|{C^h_i}^T z_i, \sigma^2).$$

### M step:

$$\hat C = 
\left(\sum_{i=1}^n \mathbb{E}[x_iz_i^T]\right) 
\left(\sum_{i=1}^n\mathbb{E}[z_iz_i^T]\right)^{-1}
$$

$$\hat \sigma^2
= \frac{1}{nd} \sum_{i=1}^n \mathbb{E} \\|x_i - C z_i\\|^2 
= \frac{1}{nd} \sum_{i=1}^n \mathbb{E} \left[
    \\|x_i^h - (C^h_i)^T z_i\\|^2
    + \\|x_i^o - (C_i^o)^T \mu_i\\|^2
    + \\|(C_i^o)^T \mu_i - (C_i^o)^T z_i^o\\|^2
\right]
$$

where we have

$\mathbb{E}[z_i(z_i)^T] = \Sigma_i + \mu_i(\mu_i)^T$, 
$\mathbb{E}[x_i^o (z_i)^T] = x_i^o(\mu_i)^T$,
$\mathbb{E}[x_i^h (z_i)^T] = (C^h_i)^T\left(\mathbb{E}[z_i(z_i)^T]\right)$,

and 

$\mathbb{E}\\|x_i^h - (C^h_i)^T z_i\\|^2 = \sigma^2 \cdot \mathrm{dim}(x^h_i)$, 
$\mathbb{E}\\|(C_i^o)^T \mu_i - (C_i^o)^T z_i^o\\|^2 
= \mathrm{tr}\left( C_i^o (C_i^o)^T \Sigma_i \right)$.
