# Sparse inverse diagnostic review — 3 November 2026

## What the experiment means

We are trying to recover both heat diffusivity alpha and an unknown initial
temperature distribution from ten noisy temperature measurements. The simulated
truth has alpha=0.05. The estimator can adjust six initial sine-mode amplitudes.
This is a classical inverse-problem diagnostic; no PINNs were trained.

The diagnostic reuses two deliberately selected development datasets. There are
fourteen fits, but only two datasets. These fits are not fourteen independent
replications and cannot estimate the prevalence of a failure.

## Findings from the saved fits

1. For seed 0, estimates remain stable when the lower alpha bound moves from
   0.001 to 0.00001. Lambda=1e-6 reduces coefficient error from 12.90% to 0.91%.
2. For seed 1, widening the interval exposes lower-objective solutions near zero
   diffusivity. Both penalized and unpenalized estimates then track the lower bound.
   At the widest interval, coefficient error is 99.98% for both objectives.
3. Increasing the profile grid from 401 to 1601 points leaves these two widest-
   interval seed-1 estimates and objectives unchanged at the reported precision.
   A coarse search grid does not explain away these particular boundary results.
   This check is not a proof of global optimality.
4. The problem is not specific to adding the roughness penalty: it also appears
   without regularization. The original lower bound was concealing part of the
   unpenalized objective's behavior.

## Independent evaluation of the saved temperature fields

The assistant evaluated the saved amplitudes and alpha values without refitting,
using Python 3.12.14. The user's diagnostic fits were generated with Python 3.14.6.
Both environments are recorded in their respective manifests. The scoring script
uses the analytical two-mode truth, verified against the saved noiseless observation
values. The known alpha is used for scoring, not choosing a fitted model.

Relative L2 errors below use trapezoidal integration over the complete unit
space-time square for the temperature field, and over the unit interval at t=0
for the initial condition. Scores use a 401-by-401 field grid. All values are
percentages. This evaluates the whole simulated domain, rather than additional
independent noisy observations.

| dataset | lambda | lower alpha bound | alpha error | field error | initial-condition error |
|---|---|---|---|---|---|
| seed 0 | 0 | 0.001 | 12.90% | 21.42% | 73.15% |
| seed 0 | 1e-6 | 0.001 | 0.91% | 1.43% | 3.45% |
| seed 1 | 0 | 0.001 | 27.25% | 16.00% | 47.16% |
| seed 1 | 0 | 0.00001 | 99.98% | 39.48% | 32.84% |
| seed 1 | 1e-6 | 0.001 | 98.00% | 39.55% | 33.36% |
| seed 1 | 1e-6 | 0.00001 | 99.98% | 39.39% | 32.82% |

All fourteen rows, including repeated grid checks and intermediate bounds, are
preserved in `results/sparse_inverse_field_analysis_v1`. The largest field-error
change between 201- and 401-point quadrature grids is 0.009205 percentage points.
This is numerical consistency evidence, not a statistical uncertainty interval.

The clearest example is unpenalized seed 1: measurement MSE decreases from
4.556e-5 to 2.361e-5 when the lower bound is reduced from 0.001 to 0.00001, while
coefficient error increases from 27.25% to 99.98% and field error from 16.00% to
39.48%. Initial-condition error moves in the opposite direction, so it would also
be misleading to call every aspect of the recovered solution uniformly worse.
Each target must be scored separately.

## Physics and interpretation

For a sine mode, diffusion damps its amplitude as exp(-alpha*(j*pi)^2*t).
Unknown initial amplitudes and the diffusion rate both influence later measured
temperatures. Sparse noisy observations can make separating these effects
difficult. The present results motivate testing this explanation; they do not
establish structural non-identifiability or a unique causal mechanism.

Each fitted spectral field exactly satisfies the heat equation for its own fitted
alpha. Consequently, satisfying the PDE does not by itself identify the true
coefficient. A future PINN must be evaluated on recovered alpha, initial field,
and temperature field, alongside its optimization behavior and computational cost.

These are separate penalized objectives for separate lambdas. Their absolute
values should not be compared as if they were one likelihood. None of the plots
is a confidence interval. Lambda=1e-6 is not a validated tuning rule merely
because it works well for seed 0.

## Evidence and checks

- Saved diagnostic: `results/sparse_inverse_diagnostic_v1`, including all fits,
  profiles, plot, configuration, and original source/input hashes.
- Original diagnostic source and plan were left unchanged after the user run.
  Their hashes and the saved observation hashes were checked against the manifest.
- Separate scorer: `scripts/score_sparse_diagnostic.py`. It checks the analytical
  reference against the original clean observations and independently checks its
  field construction using the known two-mode amplitudes.
- Derived results: `results/sparse_inverse_field_analysis_v1`, including all
  scores, the scoring environment, and input/scorer SHA-256 hashes.
- Source lint and formatting checks pass. The existing checkout previously passed
  180 tests in the user's environment. This analysis adds no estimator changes.

Reproduce scoring in a fresh directory, keeping the saved analysis intact:

```cmd
python scripts/score_sparse_diagnostic.py --out-dir results\sparse_inverse_field_analysis_reproduction
```

That optional reproduction directory is ignored by Git by default.

## Next controlled investigation

First archive this exploratory checkpoint on the existing diagnostic branch.
The next useful control is to fit the saved clean observation values at exactly
the same coordinates and times, retaining the same model and search settings.
Compare clean and noisy profiles for both selected seeds and both lambdas. This
tests whether this failure persists after removing observation noise for these
particular designs. Penalization can still introduce bias in a clean-data fit.
If an unpenalized clean fit fails despite the truth being in the fitted model,
investigate rank, conditioning, numerical tolerances, and competing minima before
attributing the behavior to inadequate physical information.

Later, use independent development datasets to choose a data-only tuning rule and
shared parameter bounds, then freeze a protocol before collecting confirmation
results. Retain failures. A PINN comparison and a verified literature gap are still
needed before making a paper-level contribution claim. This checkpoint supports
a research direction; it does not establish novelty or publication readiness.
