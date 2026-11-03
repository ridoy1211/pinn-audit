# Bounded PINN feasibility pilot

Stage: exploratory_lbfgs_budget_sensitivity
Fixed development cases; repeated initializations are not independent datasets.
All recovery errors are percentages; completed_budget does not certify convergence.

| N | data seed | training seed | status | alpha | alpha error | field error | IC error | training seconds |
|---|---|---|---|---|---|---|---|---|
| 10 | 1000 | 0 | completed_budget | 0.0500648 | 0.1296 | 0.8145 | 2.421 | 48.46 |
| 10 | 1000 | 1 | completed_budget | 0.050019 | 0.03801 | 0.8722 | 2.675 | 47.48 |
| 10 | 1000 | 2 | completed_budget | 0.0497538 | 0.4924 | 0.57 | 1.155 | 45.77 |
| 50 | 2000 | 0 | completed_budget | 0.0489429 | 2.114 | 1.386 | 4.395 | 44.23 |
| 50 | 2000 | 1 | completed_budget | 0.049083 | 1.834 | 1.006 | 2.962 | 47.41 |
| 50 | 2000 | 2 | completed_budget | 0.0491302 | 1.74 | 0.8704 | 2.401 | 49.19 |

Classical context for these exact datasets is saved separately.
This is an untuned feasibility pilot, not evidence of method superiority.
