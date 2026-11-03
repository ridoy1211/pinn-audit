# Bounded PINN feasibility pilot

Stage: exploratory_feasibility_pilot
Fixed development cases; repeated initializations are not independent datasets.
All recovery errors are percentages; completed_budget does not certify convergence.

| N | data seed | training seed | status | alpha | alpha error | field error | IC error | training seconds |
|---|---|---|---|---|---|---|---|---|
| 10 | 1000 | 0 | completed_budget | 0.048972 | 2.056 | 1.536 | 4.017 | 24.48 |
| 10 | 1000 | 1 | completed_budget | 0.0483731 | 3.254 | 2.231 | 6.177 | 28.64 |
| 10 | 1000 | 2 | completed_budget | 0.0487433 | 2.513 | 2.267 | 6.009 | 28.51 |
| 50 | 2000 | 0 | completed_budget | 0.049448 | 1.104 | 0.541 | 0.8844 | 31.52 |
| 50 | 2000 | 1 | completed_budget | 0.0493506 | 1.299 | 0.8175 | 2.213 | 31.92 |
| 50 | 2000 | 2 | completed_budget | 0.0494025 | 1.195 | 0.5549 | 1.053 | 26.16 |

Classical context for these exact datasets is saved separately.
This is an untuned feasibility pilot, not evidence of method superiority.
