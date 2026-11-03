# Diagnostic 1: investigate the sparse-data bound hit

Prepared 3 November 2026, after inspecting the archived exploratory pilot. This is an explicitly
post-hoc, exploratory diagnostic, not a preregistered or confirmatory experiment.

## Question

Why did the six-mode spectral estimator with lambda=1e-6 return alpha=0.001
(its lower search bound) for N=10, data seed 1?

Competing possibilities include a coarse alpha search, a preferred minimum beyond
the imposed search interval, or an unstable/weakly informative inverse objective.
These possibilities are not mutually exclusive. A bound hit is not itself an
optimizer crash, nor does it by itself prove structural non-identifiability.

## Fixed diagnostic design

- Reuse the saved N=10 observations for data seeds 0 and 1 without generating data.
- Seed 1 is the previously identified severe-error case; seed 0 is a comparison case.
  These are selected examples, not an unbiased estimate of failure prevalence.
- Keep six initial-condition modes, the original data, and upper alpha bound 1.0.
- Compare lambda=0 and lambda=1e-6.
- Compare lower alpha bounds 1e-3, 1e-4, and 1e-5.
- Use 401 profile grid points in each of the twelve fits.
- For seed 1 at lower bound 1e-5, repeat both lambdas with 1601 grid points.
- Report alpha, bound hits, data MSE, roughness penalty, and total objective.
- The true alpha is used only for reporting diagnostic error/marking a plot.
  It is not used to select the fitted estimate or tune a final method.

## Interpretation rules

If an estimate tracks the lower bound as it moves, a single reported boundary
estimate is substantially influenced by the imposed interval. Inspect the actual
profile before interpreting this as information about the physical coefficient.

If denser alpha grids materially change the objective/estimate, search resolution
needs further investigation before drawing a scientific conclusion. Agreement
across two grids is a useful check, not a proof of a global optimum.

A penalized objective is not automatically a likelihood or confidence interval.
Curves for different lambdas are different objectives; do not treat their absolute
values as a common likelihood comparison. Plot them in separate panels.

## Outputs and reproduction

Run from the project root using the existing environment:

    python scripts/diagnose_sparse_fit.py --results results/my_classical_pilot_v1 --out-dir results/sparse_inverse_diagnostic_v1

The script saves all fourteen estimates, raw profile arrays, objective decomposition,
a figure, a readable summary, the diagnostic configuration, and input/source hashes.
It requires a fresh output directory and preserves the archived pilot results.

This diagnostic will inform subsequent bounds, validation, and identifiability
checks. It cannot establish a PINN comparison, population failure rate, universally
optimal lambda, or a publication-level novelty claim.

## Implementation and source status

A reference implementation of this diagnostic was already exercised on the same
development observations during tool verification. This reproduction therefore
must not be described as a novel prediction or an unseen confirmation. The new
Git branch records a separate exploratory investigation. Commit and archive its
code, data, figure, and interpretation together after review.
