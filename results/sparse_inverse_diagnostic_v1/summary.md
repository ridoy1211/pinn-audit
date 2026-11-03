# Sparse-fit diagnostic (exploratory)

Same saved development observations; selected examples; no PINNs trained.
Relative coefficient errors below are percentages. True alpha is used only for reporting.

| seed | lambda | lower alpha | grid points | alpha estimate | bound hit | error (%) | data MSE | penalty | total objective |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0.001 | 401 | 0.04355055 | False | 12.9 | 3.50001e-05 | 0 | 3.50001e-05 |
| 0 | 0 | 0.0001 | 401 | 0.04355055 | False | 12.9 | 3.50001e-05 | 0 | 3.50001e-05 |
| 0 | 0 | 1e-05 | 401 | 0.04355055 | False | 12.9 | 3.50001e-05 | 0 | 3.50001e-05 |
| 0 | 1e-06 | 0.001 | 401 | 0.05045689 | False | 0.9138 | 4.51095e-05 | 2.36903e-05 | 6.87999e-05 |
| 0 | 1e-06 | 0.0001 | 401 | 0.05045689 | False | 0.9138 | 4.51095e-05 | 2.36903e-05 | 6.87999e-05 |
| 0 | 1e-06 | 1e-05 | 401 | 0.05045689 | False | 0.9138 | 4.51095e-05 | 2.36903e-05 | 6.87999e-05 |
| 1 | 0 | 0.001 | 401 | 0.03637288 | False | 27.25 | 4.55597e-05 | 0 | 4.55597e-05 |
| 1 | 0 | 0.0001 | 401 | 0.0001 | True | 99.8 | 2.44121e-05 | 0 | 2.44121e-05 |
| 1 | 0 | 1e-05 | 401 | 1e-05 | True | 99.98 | 2.36075e-05 | 0 | 2.36075e-05 |
| 1 | 0 | 1e-05 | 1601 | 1e-05 | True | 99.98 | 2.36075e-05 | 0 | 2.36075e-05 |
| 1 | 1e-06 | 0.001 | 401 | 0.001 | True | 98 | 4.91203e-05 | 2.34871e-05 | 7.26074e-05 |
| 1 | 1e-06 | 0.0001 | 401 | 0.0001 | True | 99.8 | 2.44852e-05 | 1.82933e-05 | 4.27785e-05 |
| 1 | 1e-06 | 1e-05 | 401 | 1e-05 | True | 99.98 | 2.36759e-05 | 1.784e-05 | 4.15159e-05 |
| 1 | 1e-06 | 1e-05 | 1601 | 1e-05 | True | 99.98 | 2.36759e-05 | 1.784e-05 | 4.15159e-05 |

## Grid-resolution check

- lambda=0: 401 to 1601 points changes alpha by 0 relative; objective changes from 2.36075231e-05 to 2.36075231e-05.
- lambda=1e-06: 401 to 1601 points changes alpha by 0 relative; objective changes from 4.15159407e-05 to 4.15159407e-05.

Interpret using docs/SPARSE_INVERSE_DIAGNOSTIC_PLAN.md.
Bound sensitivity does not alone prove
structural non-identifiability; grid agreement does not prove global optimality.
Penalized profiles are not confidence intervals. Preserve all results.
