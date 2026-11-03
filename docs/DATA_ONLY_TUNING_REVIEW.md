# Review of the 20-dataset data-only tuning development study

The user completed the planned 600 classical fits at source commit
4a9927a8915c1531b688b289fdca81612591198c using Python 3.14.6. There are 20 datasets,
100 full-data candidate fits and 500 cross-validation-fold fits. The experiment
is development, not held-out confirmation. All source/input/output evidence is
retained; original estimator and experiment sources have not been changed.

## Verification

The assistant checked all eight source hashes, all 64 output hashes, config hash,
and expected record counts. Recomputed held-out predictions and squared errors
for all 500 fold fits from saved amplitudes, alpha values, and fold memberships.
All candidate validation losses and all 20 selected lambdas reproduce within
numeric tolerances. See `audit/data_only_tuning_review.json` for the audit summary.
No fits were regenerated or excluded during this review.

## What the results support

At N=10, choosing lambda using noisy-observation cross-validation substantially
improves median full-field error (44.44% to 3.99%) and initial-field error (178%
to 11.4%) relative to unregularized fitting. Full-field error improves on all ten
paired datasets. Coefficient recovery is less consistent: median error decreases
from 7.57% to 3.98%, with seven paired improvements and three deteriorations.
Do not describe these as ten coefficient-recovery improvements.

The improvement is stronger for field reconstruction than for alpha recovery.
These targets need separate evaluation. Accurate alpha does not imply an accurate
initial temperature field when its amplitudes are also unknown.

The fixed lambda=1e-6 is better than CV selection on the N=10 aggregate medians:
2.75% coefficient, 3.21% field, and 8.21% initial-field error. It was specified
before this run, but motivated by earlier development experiments. This result
does not make it a universally optimal lambda or invalidate cross-validation.
Prediction-based validation and parameter recovery measure different outcomes;
small validation sets can also make selection unstable. Neither explanation has
been separately quantified by this experiment.

At N=50, median coefficient errors for CV and unregularized fitting are nearly
equal (1.085% and 1.088%). Median field errors are also close (0.8998% and 0.9027%).
There are eight paired improvements, one tie and one deterioration for both
metrics, but their magnitudes must be inspected rather than treating counts as
effect sizes. The fixed penalty also gives small errors. Do not claim a large
additional benefit in this condition from these medians.

## Failures and limits that must remain visible

The worst sparse CV fit, seed 1009, has 21.22% coefficient error, 26.84% field
error, and 95.02% initial-field error, despite being better in field error than
its unregularized counterpart. The median initial-field error therefore conceals
a substantial remaining tail. Zero observed bound hits does not establish that
boundary failures are absent in the data-generating population; the earlier
selected-example failure remains valid evidence.

There are only ten datasets per observation count, one true coefficient, one
initial shape, one noise convention and one observation-time distribution.
Datasets at N=10 and N=50 are different draws, so this is not a paired learning
curve. The true field lies exactly in the six-mode classical family. All results
remain specific to this favorable-model simulation setting. No significance,
global-optimization, uncertainty-calibration, or publication-novelty claim follows
from the descriptive summary. The tuning experiment also refits every candidate
for diagnostics, so its total runtime is not a fair method-cost comparison.

## Research decision

Retain both fixed-lambda and CV-selected classical baselines. Do not replace the
CV rule using the newly observed truth errors or hide its poorer cases. The
classical baseline is now implemented with explicit regularization and data-only
selection; further work should assess neural feasibility rather than repeatedly
expanding the same selected-example diagnostics.

Next run a small, explicitly untuned PINN feasibility pilot on the first saved
dataset of each observation-count group, with three independent initialization
seeds each. Share observations, alpha bounds, and exact boundary information,
keep initial truth out of training, and save checkpoints and field predictions.
See `docs/BOUNDED_PINN_PILOT_PLAN.md`. Representation/prior mismatch and fair
neural tuning remain unresolved before any final method comparison. A verified
literature gap and stronger independent evaluation are still required for a paper.
