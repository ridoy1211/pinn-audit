# Publication scope and claims

Prepared 9 November 2026 after inspecting the completed results at commit
`1a4a790fc44991b47f8b815458215577e7aec259`.
This is a post-evaluation interpretation record, not a preregistration and not
a change to any frozen experiment, analysis, result or tuning decision.

## Study type and question

This is a controlled computational research study using synthetic observations
from a one-dimensional heat equation. It is not experimental validation on a
physical material, a new PINN architecture, or a new identifiability theorem.

The question is how the recovery errors of specified classical regularized and
PINN procedures vary with observation schedule, initial-state representation,
and optimization budget when diffusivity and initial temperature are unknown.
The procedures, their tuning rules, and their computational budgets must be
named explicitly. Their results do not establish a ranking of whole method classes.

## Relationship to prior work

Joint diffusivity/initial-temperature estimation and its practical motivation
predate this study. Scarpa and Tagliafico (2015) examine initial-temperature bias
and include the initial field among the unknowns:
https://doi.org/10.1155/2015/494051 . Zhao, Banda and Guo (2020) study simultaneous
identification through controlled boundary measurements:
https://doi.org/10.3934/dcdsb.2020021 . Their observation/control assumptions differ
from this study's finite interior samples and homogeneous Dirichlet boundaries.

PINN/classical comparisons also predate this study. Jekic et al., preprint v2,
compare noisy inverse fluid problems with a FEM/optimizer baseline:
https://arxiv.org/abs/2509.20191v2 . Zhang, Alla and Karniadakis, preprint v2,
explicitly study representation and other comparison controls:
https://arxiv.org/abs/2606.12337v2 . Oliveira et al. (2026) reconstruct initial
conditions in a backward heat problem with prescribed diffusivity:
https://doi.org/10.1007/s40314-026-03818-x .

The defensible contribution is a focused, reproducible empirical audit connecting
paired timing comparisons with cross-validated spectral inversion, initial-state
flexibility, finite-data identifiability diagnostics, an off-basis initial field,
and independent propagation of recovered PINN quantities. These components are
not individually new. A targeted search has not established an exact duplicate
of the entire design, but cannot prove that none exists. Avoid a priority claim
such as "the first" without stronger evidence.

## Claims and their evidence limits

| Topic | Supported statement | Statement to avoid |
|---|---|---|
| Primary timing interaction | For N=10, family B, the median within-block mixed-minus-late interaction is 3.7322 percentage points, with 95% percentile bootstrap interval [-0.0947, 56.2421]. This is inconclusive about the prespecified interaction. | Timing has no effect; or the primary interaction was demonstrated. |
| Other comparisons | Report paired effects and absolute errors together. Secondary intervals are pointwise and unadjusted. | Replacing the original primary result with a favorable secondary contrast. |
| Identifiability | With ten observations and twelve unrestricted coefficients, local full row rank permits interpolation across nearby diffusivities in exact arithmetic. Positive regularization selects among solutions. | Improved regularized estimates demonstrate restored unconstrained identifiability. |
| Representation | Recovery changes with the tested 6-, 12- and 24-mode procedures. The penalty and its selection are part of each procedure. | Basis dimension alone has been causally isolated from all prior and numerical effects. |
| Physical consistency | Independent propagation was checked for the 360 timing PINNs; the maximum recorded global discrepancy is about 11.65%. Numerical refinement passed after one separately recorded refinement. | All learned models are physically exact, or consistency demonstrates correct diffusivity. |
| Off-basis robustness | A localized initial field outside the twelve-mode span was tested on mixed/late schedules with independent truth verification. Initial projection error is about 17.86%. | All its performance changes are caused by basis mismatch, or this single field establishes general robustness. |
| Optimization | Budget sensitivity and recorded starts characterize specified procedures. Timing/off-basis evaluations use prespecified neural seed 101. | A completed budget proves optimizer convergence or initialization independence. |
| Cost | Report what each timer includes, separating fitting, selection, scoring and diagnostics where recorded. | Unequal total-job timers establish equal-budget superiority. |

## Additional off-basis interpretation check

A post-evaluation audit on all 120 saved off-basis observation datasets compared
the verified truth with the forward solution of its exact twelve-mode orthogonal
initial projection, at the true diffusivity. This is an approximation diagnostic,
not the fitted CV12 estimate and not an additional hypothesis test.

| N | Schedule | Median omitted-tail RMS / noise sigma | Maximum omitted-tail RMS / noise sigma |
|---|---|---:|---:|
| 10 | mixed | 0.159618 | 1.033807 |
| 10 | late | 2.1874e-9 | 7.1502e-8 |
| 50 | mixed | 0.291692 | 0.596833 |
| 50 | late | 1.2526e-8 | 4.0617e-8 |

The maximum absolute omitted-tail discrepancy over all late observation points
is approximately 2.49e-9, compared with noise sigma 0.01. Thus a substantial
initial-field projection error need not imply substantial forward approximation
error at delayed sensors. The late experiment tests recovery of an off-basis
initial field from heavily smoothed observations; it is not evidence that omitted
high modes directly dominate the late observation residuals. Energy, spectrum,
noise, regularization, sampling, and optimization remain relevant.

Archive this new descriptive diagnostic separately when assembling the evidence
release. Do not insert it into the historical prespecified analysis plan.

## Figure and reporting rules

1. Show the primary interaction and its interval directly, alongside absolute
   errors and paired method differences. A line joining marginal medians is not
   the median within-block interaction.
2. Show the localized initial field, its twelve-mode projection, and their
   forward differences relative to noise at the actual observation times.
3. Distinguish observations, collocation points and scoring grids.
4. Label the original study, timing follow-up, post-evaluation diagnostics,
   and off-basis evaluation separately. Do not pool their different estimator
   definitions, ridge grids or independent-block counts.
5. Keep numerical profiles, conditioning and physical consistency visible.
   Penalized objectives for different penalties are different quantities;
   their vertical ordering is not a rule for selecting regularization.
6. Label historical forward heat/wave and earlier inverse-pilot plots as
   historical. They are not the current paired timing study.

## References in the repository and paper

Keep one verified bibliography source in the repository and use it to build the
paper's citations. Cite sources in repository documentation where methods,
algorithms or external claims are explained. The manuscript must contain its
own citations and bibliography; a README reference list is not a substitute.
Use a separate CITATION.cff for how others should cite this project's software
and, once available, its paper/release. Do not invent a publication DOI or status.
Reference relevance and accuracy determine the bibliography size, not a quota.
