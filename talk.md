class: middle, center, title-slide

# Inverting scientific images with<br> score-based generative models

<br>

3rd Workshop on Uncertainty Quantification for Computer Vision, ECCV 2024<br>
September 30, 2024

<br>

Gilles Louppe<br>
[g.louppe@uliege.be](mailto:g.louppe@uliege.be)

???

Trim back to 15 slides max, to fit in 20 minutes + 5 minutes Q&A.

---

class: middle, black-slide, center
background-image: url(figures/y.png)
background-size: cover

.bold.larger[From a noisy observation $y$...]


---

class: middle, black-slide, center
background-image: url(figures/x.png)
background-size: cover

.bold.larger[... can we recover <br> all plausible images $x$?]


---

class: middle

.width-100[![](figures/setup.svg)]

## Problem statement

Given a noisy observation $y$, estimate the posterior distribution $p(x|y) = \frac{p(y|x) p(x)}{p(y)}$ of plausible latent states $x$.

???

Insist on the distributional aspect of the problem, we don't just care about a pretty point estimate, as for image inpainting, but about the full posterior distribution.

---

class: middle

.center.width-10[![](figures/icons/montagne.png)]

.center[
How do we estimate $p(x|y)$ when $x$ is high-dimensional?
]

---

class: middle

## Score-based generative models 101

Samples $x \sim p(x)$ are progressively perturbed through a diffusion process described by the forward SDE $$\text{d} x\_t = f\_t x\_t \text{d}t + g\_t \text{d}w\_t,$$
where $x\_t$ is the perturbed sample at time $t$, leading to a Gaussian diffusion kernel
$$p(x\_t | x) = \mathcal{N}(x\_t | \alpha\_t x, \Sigma\_t).$$

.center[
.width-100[![](figures/perturb_vp.gif)]
Forward diffusion process.
]

.footnote[Credits: [Song](https://yang-song.net/blog/2021/score/), 2021.]

---

class: middle

The reverse process satisfies a reverse-time SDE that can be derived analytically from the forward SDE as $$\text{d}x\_t = \left[ f\_t x\_t - \frac{1+\eta^2}{2} g\_t^2 \nabla\_{x\_t} \log p(x\_t) \right] \text{d}t + \eta g\_t \text{d}w\_t.$$

Therefore, to generate data samples $x\_0 \sim p(x\_0) \approx p(x)$, we can draw noise samples $x\_1 \sim p(x\_1) \approx \mathcal{N}(0, \Sigma\_1)$ and gradually remove the noise therein by simulating the reverse SDE from $t=1$ to $0$.

.center[
.width-100[![](figures/denoise_vp.gif)]
Reverse denoising process.
]

.footnote[Credits: [Song](https://yang-song.net/blog/2021/score/), 2021.]

---

class: middle 

.center.width-100[![](figures/architecture.svg)]

The score function $\nabla\_{x\_t} \log p(x\_t)$ is unknown, but can be approximated by a neural network $d\_\theta(x\_t, t)$ by minimizing the denoising score matching objective
$$\mathbb{E}\_{p(x)p(t)p(x\_t|x)} \left[ \lambda\_t || d\_\theta(x\_t, t) - x ||^2\_2 \right].$$
The optimal denoiser $d\_\theta$ is the mean $\mathbb{E}[x | x\_t]$ which, via Tweedie's formula, allows to use $s\_\theta(x\_t, t) = \Sigma\_t^{-1}(d\_\theta(x\_t, t) - x\_t)$ as a score estimate in the reverse SDE.

---

class: middle

.center.width-10[![](figures/icons/idee.png)]

## Inverting single observations

Because of the Bayes' rule, the posterior score $\nabla\_{x\_t} \log p(x\_t|y)$ to inject in the reverse SDE can be decomposed as
$$\nabla\_{x\_t} \log p(x\_t|y) = \nabla\_{x\_t} \log p(x\_t) + \nabla\_{x\_t} \log p(y|x\_t) - \sout{\nabla\_{x\_t} \log p(y)}.$$

This is particularly convenient as it enables .bold[zero-shot posterior sampling] from a diffusion prior $p(x\_0)$ without having to pre-wire the neural denoiser to the observation model $p(y|x)$.
---

class: middle

.avatars[![](figures/frozet.jpg)![](figures/gerome.jpg)![](figures/lanusse.jpg)![](figures/gilles.jpg)]

## Approximating $\nabla\_{x\_t} \log p(y | x\_t)$

We want to estimate the score $\nabla\_{x\_t} \log p(y | x\_t)$ of the noise-perturbed likelihood $$p(y | x\_t) = \int p(y | x) p(x | x\_t) \text{d}x.$$

Our approach:
- Assume a linear Gaussian observation model $p(y | x) = \mathcal{N}(y | Ax, \Sigma\_y)$.
- Assume the approximation $p(x | x\_t) \approx \mathcal{N}(x | \mathbb{E}[x | x\_t], \mathbb{V}[x | x\_t])$,  where $\mathbb{E}[x | x\_t]$ is estimated by the denoiser and $\mathbb{V}[x | x\_t]$ is estimated using Tweedie's covariance formula.
- Then $p(y | x\_t) \approx \mathcal{N}(y | A \mathbb{E}[x | x\_t], \Sigma\_y + A \mathbb{V}[x | x\_t] A^T)$.
- The score $\nabla\_{x\_t} \log p(y | x\_t)$ then approximates to 
$$\nabla\_{x\_t} \mathbb{E}[x | x\_t]^T A^T (\Sigma\_y + A \mathbb{V}[x | x\_t] A^T)^{-1} (y - A \mathbb{E}[x | x\_t]).$$

.footnote[See also [Daras et al (2024)](https://giannisdaras.github.io/publications/diffusion_survey.pdf) for a survey on diffusion models for inverse problems.]

---

class: middle

.center[
.width-65[![](figures/lensing.svg)]
.width-25[![](figures/lensing.webp)]
]

.center.width-100[![](figures/lensing.png)]

Plausible galaxy images $x$ can be recovered from lensed observations $y$ by zero-shot posterior sampling from a diffusion prior $p(x)$.

.footnote[Credits: [Adam et al.](https://arxiv.org/abs/2211.03812), 2022 (arXiv:2211.03812).]

---

class: middle

.avatars[![](figures/frozet.jpg)![](figures/gilles.jpg)]

## Score-based data assimilation in dynamical systems

.center.width-100[![](figures/dynamical.svg)]

The goal of .bold[data assimilation] is to estimate plausible trajectories $x\_{1:L}$ given one or more noisy observations $y$ (or $y\_{1:L})$ as the posterior $$p(x\_{1:L} | y) = \frac{p(y | x\_{1:L})}{p(y)} p(x\_0) \prod\_{i=1}^{L-1} p(x\_{i+1} | x\_i).$$

???

Assume the latent state $x$ evolves according to a transition model $p(x\_{i+1} | x\_i)$ and is observed through an observation model $p(y | x\_{1:L})$. (Typically, the observation model will be $p(y\_i | x\_i)$, but we consider the general case here.) 

---

class: middle

.center.width-100[![](figures/sda.svg)]

Our approach:
- Build a score-based generative model $p(x\_{1:L})$ of arbitrary-length trajectories$^\*$. 
- Use zero-shot posterior sampling to generate plausible trajectories from noisy observations $y$.

.footnote[*:The score of a (noise perturbed) trajectory can be approximated by a sum of scores. See paper for details.]

---

class: middle

.center.width-100[![](figures/sda1.png)]

.center[Sampling trajectories $x\_{1:L}$ from<br> noisy, incomplete and coarse-grained observations $y$.]

---

class: middle

.avatars[![](figures/frozet.jpg)![](figures/gerome.jpg)![](figures/lanusse.jpg)![](figures/gilles.jpg)]

.center.width-60[![](figures/eb.svg)]

## Learning priors from noisy observations

Assume only observations $y \sim p(y)$ and a known observation model $p(y | x)$.

The objective of .bold[Empirical Bayes] is find a prior model $q\_\theta(x)$ such that $$q\_\theta(y) = \int p(y | x) q\_\theta(x) \text{d}x$$ is closest to $p(y)$.

---

class: middle

Our approach:
- If we parameterize the latent state $x$ with a diffusion prior $q\_\theta(x)$, then Expectation-Maximization can be used to maximize $q\_\theta(y)$.

- It can be shown that the EM update
$$\theta\_{k+1} = \arg \max\_\theta \mathbb{E}\_{p(y)}\mathbb{E}\_{q\_{\theta\_k}(x|y)} \left[ \log q\_\theta(x) \right],$$
where $q\_{\theta\_k}(x|y)$ is obtained by posterior sampling from $q\_{\theta\_k}(x)$,
leads to a sequence of parameters $\theta\_k$ such that $\mathbb{E}\_{p(y)} \left[ \log q\_{\theta\_k} (y) \right]$ is monotonically increasing and converges to a local optimum.

---

class: middle

.center.width-75[![](figures/prior-cifar0.png)]

.center.width-75[![](figures/prior-cifar.png)]

.center[
    
Samples from the prior $q\_{\theta\_k}(x)$ (top) and the posterior $q\_{\theta\_k}(x|y)$ (bottom) along the EM iterations when training from corrupted CIFAR-10 images.

]

---

class: middle

.center.width-80[![](figures/prior-mri.png)]

.center[Posterior samples for accelerated MRI using a diffusion prior trained only from observations with subsampled frequencies.]

---

class: middle

.center.width-10[![](figures/icons/verifier.png)]

## Conclusions

Score-based generative models...
- can be used for high-dimensional inverse problems;
- enable zero-shot posterior sampling, without pre-wiring the network to observations;
- do not require paired data.

Next challenges:
- Rigorous diagnostics for the quality of the approximation;
- Scalability to even larger dimensions (Earth-scale weather models, videos);

---

class: middle

.avatars[![](figures/frozet.jpg)![](figures/gerome.jpg)![](figures/lanusse.jpg)![](figures/gilles.jpg)]

References:
- .bold[Score-based data assimilation]<br>François Rozet, Gilles Louppe.<br> NeurIPS 2023, [arXiv:2306.10574](https://arxiv.org/abs/2306.10574).
- .bold[Score-based Data Assimilation for a Two-Layer Quasi-Geostrophic Model]<br>François Rozet, Gilles Louppe.<br> ML4PS workshop NeurIPS 2023, [arXiv:2310.01853](https://arxiv.org/abs/2310.01853).
- .bold[Learning Diffusion Priors from Observations by Expectation Maximization]<br>François Rozet, Gérôme Andry, François Lanusse, Gilles Louppe.<br> NeurIPS 2024, [arXiv:2405.13712](https://arxiv.org/abs/2405.13712).

---

class: middle, center, end-slide
count: false

The end.

