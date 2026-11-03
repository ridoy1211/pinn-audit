# Review of the user's clean/noisy control — 3 November 2026

The user completed all 24 fits using Python 3.14.6. The run manifest records
2026-11-03T09:04:34.525601+00:00 through 09:04:39.596784+00:00, base commit
ce0146a1bcc0f1905a3f0736f68bf82fd64c5aa5, and exact source/input hashes.
The earlier assistant check used the same development data; this is reproduction,
not unseen confirmation. No estimator source changed between these diagnostics.
All 24 saved records and the manifest's source/input hashes were verified.

## What was established for these examples

- With noise removed at the same observation coordinates, all unregularized fits
  recover alpha=0.05 to numerical precision. The largest reported coefficient
  error among them is about 0.00000201%, not evidence of exact symbolic recovery.
- The fixed lambda=1e-6 introduces clean-data coefficient errors of approximately
  0.417% for seed 0 and 1.362% for seed 1. Penalization changes the target objective
  and can introduce bias even when the observations are clean.
- Noisy fits reproduce the previous diagnostic. In particular, seed 1 again
  tracks the smaller lower bounds and reaches 99.98% coefficient error.
- No search warning occurred: none of the selected fitted objectives was
  materially worse than the separately evaluated candidate at true alpha.
  This does not certify global optimality or uniqueness.

Holding locations, model, and fitting settings fixed while changing only the
target values shows that adding this particular realized noise is sufficient to
change the recovered solution markedly. This is consistent with noise-sensitive
inverse estimation. It does not establish structural non-identifiability, a
general noise threshold, or performance beyond these two selected examples.

## What we should do next

Stop expanding these two examples. Test data-only regularization selection over
more independently generated development datasets. The proposed next study
compares unregularized fitting, fixed lambda=1e-6, and five-fold cross-validation
over a declared candidate set, using identical data and bounds within each paired
comparison. It still cannot establish PINN superiority or publication novelty.

Keep the complete control evidence in `results/clean_noisy_control_v1`. Preserve
all outcomes, including the severe noisy failures. Do not use truth-based recovery
error to choose lambda or retrospectively narrow parameter bounds.
