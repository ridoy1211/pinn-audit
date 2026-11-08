# Off-basis initial-temperature robustness study

Status: Development protocol; final evaluation has not begun.
Classification: A new robustness study motivated by the preceding audit.

## Question

Does the comparison between the frozen PINN procedure and CV12 remain similar
for a localized initial temperature not exactly representable by twelve sine
modes, under mixed and late observation schedules?

This is one additional initial field, not evidence over all possible fields.
The study does not test early-only sensing or estimate an optimal schedule.

## Initial field

Let w=0.04 and c=0.35. Define

g(x) = x*(1-x)*exp(-(x-c)^2/(2*w^2)),
f(x) = g(x)/max_{0<=z<=1} g(z).

The maximum is determined numerically from the unique interior stationary
point, and its location and normalization constant are saved.
This definition fixes the peak temperature at one and satisfies f(0)=f(1)=0.
It is smooth on the closed spatial interval and is not a finite sine sum.
Its sine extension need not satisfy every higher-order compatibility condition.

The width and location are proposed choices fixed before examining recovery.
Do not adjust them to make either estimator win.
Report the relative L2 error of the orthogonal twelve-mode projection and the
initial L2 norm. Nonzero projection error is distinct from statistical difficulty.

## Physical problem and observations

Retain the unit spatial interval and time horizon, zero Dirichlet boundaries,
diffusivity alpha=0.05, and additive Gaussian absolute noise sigma=0.01.
Use N=10 and N=50 observations.

Mixed: half the observations in [0.01,0.10], half in [0.20,0.80].
Late: all observations in [0.20,0.80].
Reuse spatial coordinates, normalized time draws, standardized noise, and
cross-validation fold labels within each mixed/late block.
Generate PDE collocation points separately using the established procedure.

Peak normalization does not match the L2 energy of initial fields A and B.
Report this difference and actual clean-signal RMS/noise ratios. Do not
attribute differences across truth families solely to basis mismatch.

## Forward truth verification before data generation

Primary generator: a converged sine-series solution of the heat equation,
with coefficients obtained by numerical integration of the specified f.
Check increasing projection quadrature resolution and retained mode count.
Evaluate the initial field directly at t=0 when scoring initial recovery.

Independent verification: a second-order spatial finite-difference heat
operator integrated with a tightly controlled stiff time integrator.
Use at least three spatial resolutions, inspect refinement, and compare
against the primary generator at positive times spanning both windows.
Check analytic sine solutions as software verification cases.

Proposed numerical acceptance target: maximum absolute forward difference
below 1e-4 on the declared verification points, with refinement evidence.
This target is one percent of the measurement-noise standard deviation.
It is a numerical accuracy target, not proof of an error bound everywhere.
Do not proceed to evaluation until the checks and tolerances are documented.

## Fitting procedures

Classical: twelve-mode inversion with existing five-fold cross-validation,
penalty sum(j^4*a_j^2), and candidates
1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2.

Neural: the unchanged bounded hard-boundary PINN, primary training seed 101,
existing Adam settings and PINN-1000 L-BFGS budget.
Both methods retain alpha bounds [1e-5,1].
No truth-based model selection or tuning to the new evaluation results.

## Development and evaluation

Development: two fresh blocks per observation count, both schedules.
This gives 8 datasets and 16 fitting jobs.

Evaluation planning target: 30 fresh blocks per observation count,
both schedules: 120 datasets and 240 fitting jobs.
These are 60 independent base blocks, not 120 independent replicates.
Thirty is a practical planning choice, not a power guarantee.

Record exact development/evaluation seeds, verify separation from preceding
studies, and freeze source, configuration and analysis rules before evaluation.
Keep development results separate. Preserve failures and capped estimates.

## Analysis

Main outcome: diffusivity relative absolute error in percent.
Also report initial-field error against the directly specified f, full-field
error against the verified forward truth, runtime, failures and bound proximity.

Within each block and schedule, D = PINN error minus CV12 error.
Report absolute errors and paired D for both observation counts and schedules.
Also report I = D_mixed - D_late, calculated within each block.
Use 10,000 whole-block bootstrap resamples and 95% percentile intervals.
All robustness comparisons are secondary, with pointwise rather than
simultaneous intervals. Do not replace the original primary test.

The goal is to quantify sensitivity to this one off-basis truth, not to certify
that the original conclusions are universal. A reversal is a valid result.
Known-versus-unknown initial temperature remains a separate optional control.
