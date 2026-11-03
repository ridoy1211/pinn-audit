# Review of the first bounded PINN feasibility pilot

All six user runs completed on Python 3.14.6 and Torch 2.15.0+cpu, using source
commit c9e0eb77f376440ed9b774907e26eb3cb83f57ad. These are two fixed development
datasets, each with three network initializations. There are not six independent
data realizations, and this is not a held-out or tuned comparative experiment.

## Verification

The assistant checked all six source hashes, three input hashes, and 23 output
hashes against the manifest. All six saved networks were reloaded and used to
reconstruct their complete 401-by-401 predicted grids. Saved fields, coefficient
estimates, analytical-reference field/initial-field scores, independent residual
arrays, and training objectives reproduce within numeric tolerances. Evaluation
used Python 3.12.14, the same Torch version, and one CPU thread. No refitting or
model selection occurred during this review. Exact boundary errors are zero.

See `audit/bounded_pinn_pilot_review.json` for numerical verification and optimizer
diagnostics. The original experiment script, config, plan, and raw output remain
unchanged.

## Comparison on identical datasets

All errors below are percentages. PINN entries are medians across three
initializations for one dataset, not medians across independent datasets.
The fixed and CV-selected classical penalty coincide at lambda=1e-6 here.

| N / data seed | method | alpha error | full-field error | initial-field error |
|---|---|---|---|---|
| 10 / 1000 | classical, unregularized | 6.080 | 50.666 | 224.539 |
| 10 / 1000 | classical, regularized/CV | 0.144 | 0.808 | 2.308 |
| 10 / 1000 | PINN, median of 3 starts | 2.513 | 2.231 | 6.009 |
| 50 / 2000 | classical, unregularized | 2.539 | 2.324 | 7.897 |
| 50 / 2000 | classical, regularized/CV | 2.061 | 1.274 | 4.208 |
| 50 / 2000 | PINN, median of 3 starts | 1.195 | 0.555 | 1.053 |

For the sparse case, PINNs beat the unregularized classical fit on these targets,
but lose to the regularized fit in every initialization. For the denser case,
every PINN initialization beats both classical fits on these targets. Thus the
baseline's regularization changes the sparse-case ranking, while the two cases
do not support a universal method ranking or a population effect of N.

Training takes approximately 24-32 seconds per PINN on this implementation.
The earlier CV experiment's aggregate time includes candidate diagnostics; these
numbers must not be used as a fair tuning-plus-fitting cost comparison. The exact
six-mode classical model also has a representation advantage: the truth lies in
its family. Shared bounds and boundaries do not make all priors identical.

## Why check the neural optimizer budget next?

All six saved L-BFGS states report 200 iterations, the configured limit. Final
maximum absolute gradients of the full training objective are approximately
2.06e-4 to 5.49e-4, above the configured 1e-12 gradient tolerance. This does not
prove a useful further update exists, but it does not certify convergence either.

L-BFGS already changes the estimates materially after Adam. For example, sparse
seed 1000 / initialization 0 goes from alpha=0.022414 after Adam to 0.048972 after
L-BFGS, while loss decreases from 0.009464 to 0.000331. More observations alone
cannot explain optimizer-stage changes within the same fit.

Before expanding to many neural datasets, run one specified budget sensitivity
check, increasing L-BFGS to 1000 iterations / 1250 evaluations while retaining
the same Adam phase, data, initialization seeds, architecture, weights, points,
bounds, and training cap. Rerun from the same seeded initialization; do not call
this an exact continuation from a checkpoint lacking optimizer history. Report
both budgets; never select the budget using true recovery error.

See `docs/PINN_BUDGET_CHECK_PLAN.md`. Once that check is reviewed, specify a
broader comparison and its independent evaluation data before another large run.

## Research interpretation

This is evidence that a weak unregularized comparator can change the apparent
PINN advantage in a particular sparse inverse problem. The phenomenon needs
replication across datasets, initial-field families, and neural tuning choices.
It is a candidate audit question, not a new algorithm or established novelty.
Noisy inverse PINN/solver comparisons already exist ([Jekic et al., 2025, v1](https://arxiv.org/html/2509.20191v1));
unknown-initial-field heat reconstruction also has direct precedent ([Oliveira et
al., 2026](https://doi.org/10.1007/s40314-026-03818-x)). The existing five-question
literature notes in `audit/literature_review.md` should be used to position any
eventual narrow contribution.
