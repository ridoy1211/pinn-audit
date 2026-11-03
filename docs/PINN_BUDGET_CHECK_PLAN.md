# One L-BFGS budget sensitivity check

Prepared after reviewing the six completed development PINN fits. This is a
post-hoc development diagnostic, not confirmation or a truth-based budget search.
The trigger is that all six L-BFGS optimizers used their 200-iteration limit and
the final gradients remain above the configured gradient stopping tolerance.

## Fixed change

Use the existing `scripts/pilot_bounded_pinn.py` without modifying it. The new
configuration `configs/inverse/bounded_pinn_budget_check.json` retains every
training, sampling, initialization, data and evaluation setting from the original
pilot except for these linked computational budgets:

- L-BFGS max_iter: 200 -> 1000.
- L-BFGS max_eval: 250 -> 1250.

The stage label and provenance metadata also change. The 2000-step Adam phase,
three initialization seeds per saved dataset, all points and loss weights, alpha
initialization/bounds, architecture, and 600-second training cap remain fixed.
This repeats six fits from their original seeded initializations, with a longer
L-BFGS phase. It is not checkpoint continuation: optimizer histories were not saved.

The config includes the original config and result-manifest hashes and this
plan's SHA-256 digest, linking this diagnostic to the exact preceding evidence.
The generic pilot plan describes the original 200-iteration pilot; this separate
plan/config defines the larger-budget diagnostic.

## What to compare

For each initialization, compare the same training objective, final gradient
norm, optimizer counts, alpha, field and initial-trace changes, independent PDE
residual, and runtime. Recovery errors are diagnostic outcomes only, not stopping
rules or selectors. Keep both budgets even if additional optimization worsens
recovery. A lower training objective or PDE residual need not improve inverse
recovery from noisy observations.

No pass/fail threshold for scientific recovery is introduced. Similar results
would indicate stability across these two budgets, not proof of convergence or
global optimality. Material changes would show that budget matters for these
fixed cases. Neither outcome establishes general neural performance.

Do not add another automatic sequence of increasing budgets after this check.
Review this result, then define a bounded broader comparison with independent
data and explicit treatment of neural tuning and representation mismatch.

## Command

```cmd
python scripts/pilot_bounded_pinn.py --config configs\inverse\bounded_pinn_budget_check.json --out-dir results\bounded_pinn_budget_check_v1
```

The output directory must be fresh. All failures and timeout snapshots are kept
by the existing runner. This default diagnostic has not been run by the assistant.
