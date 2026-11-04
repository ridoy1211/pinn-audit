# Introduction draft

Working manuscript section, 5 November 2026. Read with `METHODS_DRAFT.md` and
`RESULTS_DRAFT.md`. The contribution statement is deliberately limited to the
completed study; publication novelty has not been established.

Estimating a physical transport coefficient from temperature observations is
more demanding when the starting temperature distribution is also unknown.
For the heat equation, each initial sine component decays at a rate proportional
to diffusivity and the square of its spatial frequency. Consequently, estimating
diffusivity and reconstructing the initial field are coupled tasks, and errors
in the recovered coefficient, initial state and space-time field need not move
together. A comparison based on only one of these outcomes can obscure important
differences between inverse procedures.

Neither joint heat inversion nor neural recovery of unobserved states is new.
Zhao, Banda and Guo developed a classical procedure for simultaneous recovery
of diffusivity, a spatial source and the initial state, using boundary control,
boundary observations and a distributed temperature snapshot. Their observation
and excitation conditions differ from the sparse interior observations examined
here. [Zhao et al. (2017)](https://doi.org/10.1002/mma.4245)
Oliveira and colleagues studied initial-temperature reconstruction with several
physics-informed learning formulations for a heat problem with specified
diffusivity. [Oliveira et al. (2026)](https://doi.org/10.1007/s40314-026-03818-x)

Comparative evidence also predates the present study. Grossmann and colleagues
compared PINNs with finite elements across forward PDE problems and distinguished
solution construction from subsequent evaluation. Their results motivate explicit
numerical references and cost definitions, but do not resolve the joint inverse
problem studied here. [Grossmann et al. (2024)](https://doi.org/10.1093/imamat/hxae011)
Zapf and colleagues compared PINN and finite-element estimation of apparent
diffusivity from brain tracer images, demonstrating the relevance of training
formulation in a noisy inverse application. Their treatment of initial and
boundary information differs from our unknown-initial-field formulation.
[Zapf et al. (2022)](https://doi.org/10.1038/s41598-022-19157-w)
Jekic and colleagues examined noisy inverse fluid problems using PINNs and a
finite-element/optimization reference, with prescribed initial-condition data.
Their work establishes a direct precedent for noisy inverse benchmarking, rather
than evidence that such comparisons remain unexplored.
[Jekic et al., preprint v2 (2026)](https://arxiv.org/html/2509.20191v2)

These precedents motivate a more specific question: **for joint constant-
diffusivity and unknown-initial-field heat inversion, how does the comparison
between specified neural and classical procedures change when initial-state
regularization, basis capacity and neural optimizer budget are made explicit?**
The purpose is to examine the sensitivity of comparative conclusions, rather
than to introduce a new solver or establish a ranking of entire method classes.

We address this question with an analytically generated one-dimensional heat
benchmark, two initial-state families and two observation counts. Four spectral
inverse procedures distinguish absent, fixed and observation-selected
regularization, and six- versus twelve-mode representations. Two hard-boundary
PINN procedures retain different L-BFGS iteration budgets after the same Adam
stage. Observations, realized noise and diffusivity bounds are shared across
procedures within each case. Diffusivity, initial-state and full-field errors
are evaluated separately, with dataset-level neural initialization summaries,
paired contrasts and a complete outcome archive.

The simulation, procedures and primary analysis were versioned before generating
the evaluation data. This prospective design reduces opportunities for selecting
favorable cases after inspection; it is not formal preregistration. The spectral
truth favors the classical representation, while classical cross-validation and
fixed neural settings provide different tuning opportunities. The study therefore
offers a controlled audit of specific pipelines, with explicit representation,
tuning, uncertainty and scope limits. It does not claim the first joint heat
inverse study or a universally preferable method.
