# Related work

Eight papers that shaped this repo or bear directly on its claims. Each note says
what the paper changed here, or how this repo relates to it, and is explicit about
what was and was not done.

---

**1. Raissi, Perdikaris, Karniadakis (2019).**
*Physics-informed neural networks: A deep learning framework for solving
forward and inverse problems involving nonlinear partial differential
equations.* J. Comput. Phys. 378, 686-707.
[doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) ·
[arXiv:1711.10561](https://arxiv.org/abs/1711.10561)

Defines the method: a network trained on data misfit plus a PDE residual computed
by automatic differentiation. The PINN here follows that recipe (soft constraints,
residual at random collocation points). Its Burgers' equation example is the field's
standard forward benchmark; Burgers is not part of the current results and is listed
as future work.

**2. Karniadakis, Kevrekidis, Lu, Perdikaris, Wang, Yang (2021).**
*Physics-informed machine learning.* Nat. Rev. Phys. 3, 422-440.
[doi:10.1038/s42254-021-00314-5](https://doi.org/10.1038/s42254-021-00314-5)

A survey. Source of the forward / inverse vocabulary used in the README and in how
experiments are labelled.

**3. Krishnapriyan, Gholami, Zhe, Kirby, Mahoney (2021).**
*Characterizing possible failure modes in physics-informed neural networks.*
NeurIPS 2021. [arXiv:2109.01050](https://arxiv.org/abs/2109.01050)

PINNs fail on convection and reaction-diffusion problems that ordinary finite
differences handle, and the cause is the optimisation landscape getting harder as
the physics gets stiffer, not a shortage of network capacity. It motivated holding
the classical baselines to a measured convergence order before comparing against
them. This repo did *not* vary PDE stiffness (diffusivity or wave speed stay fixed),
so it does not test that paper's central axis; that is listed as future work.

**4. Wang, Yu, Perdikaris (2022).**
*When and why PINNs fail to train: A neural tangent kernel perspective.*
J. Comput. Phys. 449, 110768.
[arXiv:2007.14527](https://arxiv.org/abs/2007.14527)

Explains training failures through spectral bias and through loss terms that
converge at different rates. The default initial condition `sin(pi x) +
0.5 sin(3 pi x)` keeps a higher-frequency component alive so that this can be read
against the results. In the wave experiments, rebalancing the loss weights recovered
the third harmonic, which points to loss balance over spectral bias for that mode.
Only the third harmonic was tested, so this says nothing about higher frequencies.

**5. Jekic, Natsaridou, Riemer-Sorensen, Langseth, Gundersen (2025).**
*Examining the robustness of Physics-Informed Neural Networks to noise for
inverse problems.* [arXiv:2509.20191](https://arxiv.org/abs/2509.20191)

The nearest precedent for the inverse experiments. They identify viscosity in 1D
Burgers' equation (and in a 2D/3D Taylor-Green vortex) with additive Gaussian noise,
comparing PINNs with a finite-element model plus a numerical optimiser, and the
classical baseline came out ahead. The inverse experiments here use the same shape of
comparison: the classical side is a nonlinear least-squares fit of a numerical
forward model, not a grid search.

**6. Grossmann, Komorowska, Latz, Schoenlieb (2023).**
*Can physics-informed neural networks beat the finite element method?*
IMA Journal of Applied Mathematics. [arXiv:2302.04107](https://arxiv.org/abs/2302.04107)

The closest precedent for the forward comparison. They compare PINNs with the finite
element method on Poisson (1D, 2D, 3D), Allen-Cahn and Schroedinger problems and find
that PINNs did not outperform FEM in solution time or accuracy. The forward results
here are consistent with that on two further problems (heat and wave), against finite
differences, with controls and a mode analysis added. They are a confirmation, not a
new finding.

**7. Hao, Yao, Su, Su, Wang, Lu, Xia, Zhang, Liu, Lu, Zhu (2024).**
*PINNacle: A comprehensive benchmark of physics-informed neural networks for solving
PDEs.* NeurIPS 2024, Datasets and Benchmarks Track.
[arXiv:2306.08827](https://arxiv.org/abs/2306.08827)

A large benchmark: more than 20 PDEs and about ten PINN variants. It is far broader
than this repo. What this repo adds is narrow: an exact-solution protocol with
classical solvers, cost reported alongside error, and written predictions scored
afterwards.

**8. Rathore, Lei, Frangella, Lu, Udell (2024).**
*Challenges in training PINNs: a loss landscape perspective.* ICML 2024.
[arXiv:2402.01868](https://arxiv.org/abs/2402.01868)

Shows that the PINN loss is ill-conditioned because of the differential operators in
the residual, and that Adam followed by L-BFGS beats either optimiser alone. The
L-BFGS results here point the same way and replicate a known finding. Whether the
gain from L-BFGS depends on the loss weighting (in the wave experiments it grew much
larger once the weights were rebalanced) is something I have not checked against
their experiments.
