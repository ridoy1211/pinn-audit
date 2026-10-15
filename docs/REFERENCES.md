# Related work

Five papers that shaped this repo. Notes are about what each one changed in
the code or the experimental design, not summaries of the papers.

Items marked *(planned)* refer to parts of the repo that are not written yet.

---

**1. Raissi, Perdikaris, Karniadakis (2019).**
*Physics-informed neural networks: A deep learning framework for solving
forward and inverse problems involving nonlinear partial differential
equations.* J. Comput. Phys. 378, 686-707.
[doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) ·
[arXiv:1711.10561](https://arxiv.org/abs/1711.10561)

Defines the method: a network trained on data misfit plus a PDE residual
computed by automatic differentiation. Its Burgers' equation example is the
standard forward benchmark in the field, so the Burgers problem here *(planned)*
uses the same viscosity (0.01/pi) and initial condition. That keeps numbers
comparable with published ones instead of only with each other.

**2. Karniadakis, Kevrekidis, Lu, Perdikaris, Wang, Yang (2021).**
*Physics-informed machine learning.* Nat. Rev. Phys. 3, 422-440.
[doi:10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5)

A survey. Source of the forward / inverse vocabulary used in the README and
in how experiments are labelled.

**3. Krishnapriyan, Gholami, Zhe, Kirby, Mahoney (2021).**
*Characterizing possible failure modes in physics-informed neural networks.*
NeurIPS 2021. [arXiv:2109.01050](https://arxiv.org/abs/2109.01050)

The paper this repo is a response to. PINNs fail on convection and
reaction-diffusion problems that ordinary finite differences handle, and the
cause is the optimisation landscape getting harder as the physics gets
stiffer, not a shortage of network capacity. Two consequences here: the
classical baselines are held to a measured convergence order before anything
is compared against them, and the PINN sweeps *(planned)* vary PDE stiffness
(diffusivity, viscosity) as an axis alongside network size.

**4. Wang, Yu, Perdikaris (2022).**
*When and why PINNs fail to train: A neural tangent kernel perspective.*
J. Comput. Phys. 449, 110768.
[arXiv:2007.14527](https://arxiv.org/abs/2007.14527)

Explains the mechanism: gradient descent fits low-frequency components of the
solution first and high-frequency ones far more slowly (spectral bias). This
is why the default initial condition is `sin(pi x) + 0.5 sin(3 pi x)` rather
than a single smooth mode. The wave problem keeps a high-frequency component
alive on purpose, so a PINN that struggles with it can be read against this
paper's prediction.

**5. Jekic, Natsaridou, Riemer-Sorensen, Langseth, Gundersen (2025).**
*Examining the robustness of Physics-Informed Neural Networks to noise for
inverse problems.* [arXiv:2509.20191](https://arxiv.org/abs/2509.20191)

The nearest precedent for the inverse-problem experiments. They identify
viscosity in 1D Burgers' equation (and in a 2D/3D Taylor-Green vortex) with
additive Gaussian noise, comparing PINNs against a finite-element model plus
a numerical optimiser. The classical baseline came out ahead in their tests.
The inverse experiments here *(planned)* use the same shape of comparison:
the classical side is a proper least-squares fit of the forward model, not a
grid search, so the comparison cannot be won by a weak baseline.
