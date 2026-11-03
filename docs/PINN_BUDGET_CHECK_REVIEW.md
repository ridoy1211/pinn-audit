# Budget-check review — 4 November 2026

The six-run larger-budget experiment completed successfully. Its manifest records
source commit d98bacb630a3f4190cdfb999a50f3dd1cd91e733 and complete status. The
later FileExistsError occurred before training when the user attempted the same
command again. The runner correctly refused to overwrite the completed results.
No repair, deletion, or repeat training is needed.

The earlier pilot and budget plan were already committed. Repeating their old
git-add commands did not stage the new `results/bounded_pinn_budget_check_v1`
folder. That folder and this review should be archived in a new checkpoint.

## Verified comparison

The assistant verified source/input/output hashes for both budgets (23 output
hashes each), reproduced all six Adam-phase endpoint losses and alpha estimates,
and checked that both runs used identical training and independent PDE points.
Reloaded the six larger-budget models to reconstruct complete predicted grids,
field and initial-trace errors, independent residual arrays, and training losses.
No fitting or model selection occurred in the review.

Reproducible analysis: `scripts/review_pinn_budget_check.py`.
Evidence and figure: `results/pinn_budget_review_v1`. The review records both
manifest hashes, its own source hash, all paired outcomes, optimizer counts and
post-training gradient norms. The original experiment sources remain unchanged.

All recovery errors below are percentages. Each median summarizes three
initializations on one fixed development dataset.

| N / data seed | L-BFGS limit | alpha error | field error | initial-field error |
|---|---|---|---|---|
| 10 / 1000 | 200 | 2.513 | 2.231 | 6.009 |
| 10 / 1000 | 1000 | 0.1296 | 0.8145 | 2.421 |
| 50 / 2000 | 200 | 1.195 | 0.5549 | 1.053 |
| 50 / 2000 | 1000 | 1.834 | 1.006 | 2.962 |

All three sparse-case runs improve on all three recovery metrics. All three
denser-case runs worsen on all three metrics, even though the final training
objective decreases in every run. Short-to-long objective ratios are approximately
0.734-0.834 in the sparse case and 0.944-0.954 in the denser case.

The optimizer is minimizing misfit to noisy observations plus a finite-point PDE
penalty. It is not minimizing the unknown true coefficient or initial-field error.
The observed divergence between objectives and recovery is compatible with noise
fitting or changed regularization from the training budget; it does not establish
a unique causal mechanism or a general population effect.

Relative to the classical CV baseline, the longer sparse-case PINN median alpha
error is slightly lower (0.1296% versus 0.1435%), while median field and IC errors
are slightly higher (0.8145% versus 0.8077%; 2.421% versus 2.308%). In the denser
case, the longer-budget PINN still has lower median errors than that classical
fit, but one initialization is worse on all three metrics. Two datasets cannot
support a universal ranking, an effect of observation count, or significance.

All longer runs also reach 1000 L-BFGS iterations; final gradient maxima remain
above the configured gradient tolerance. We therefore compare fixed computational
budgets, without describing either as a certified optimum. No further increasing-
budget search is planned on these examples. Retain both budgets without choosing
the favorable one for each dataset using truth.

## Next phase

The prototype/selected-case phase is finished. Draft the broader evaluation around
explicit classical regularization, function-family assumptions, and the two fixed
neural budgets. Keep fresh evaluation data untouched until implementation and
analysis rules are committed. `docs/BROADER_COMPARISON_PROTOCOL_DRAFT.md` specifies
a bounded proposed design, including an initial-field representation control.
It is a design draft, not a claim that confirmation has already been conducted.
