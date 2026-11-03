# Data-only tuning development study

Status: exploratory_development
All recovery errors below are percentages. No PINNs trained.
Datasets are development data, not held-out confirmation.

| N | method | datasets | median alpha error | mean alpha error | max alpha error | median field error | median IC error | bound hits |
|---|---|---|---|---|---|---|---|---|
| 10 | unregularized | 10 | 7.574 | 15.16 | 52.89 | 44.44 | 178 | 0 |
| 10 | fixed_1e-6 | 10 | 2.752 | 3.98 | 13.49 | 3.212 | 8.212 | 0 |
| 10 | CV_selected | 10 | 3.975 | 6.065 | 21.22 | 3.994 | 11.4 | 0 |
| 50 | unregularized | 10 | 1.088 | 1.426 | 4.358 | 0.9027 | 3.085 | 0 |
| 50 | fixed_1e-6 | 10 | 0.9703 | 1.165 | 2.898 | 0.844 | 2.313 | 0 |
| 50 | CV_selected | 10 | 1.085 | 1.146 | 2.898 | 0.8998 | 2.739 | 0 |

N=10 selected-lambda counts: {'0': 0, '1e-08': 3, '1e-06': 6, '0.0001': 1, '0.01': 0}
N=10, relative_alpha_error, CV vs unregularized paired wins/ties/losses: 7/0/3 (strict numerical comparisons, not a significance test).
N=10, field_relative_l2, CV vs unregularized paired wins/ties/losses: 10/0/0 (strict numerical comparisons, not a significance test).
N=50 selected-lambda counts: {'0': 1, '1e-08': 2, '1e-06': 7, '0.0001': 0, '0.01': 0}
N=50, relative_alpha_error, CV vs unregularized paired wins/ties/losses: 8/1/1 (strict numerical comparisons, not a significance test).
N=50, field_relative_l2, CV vs unregularized paired wins/ties/losses: 8/1/1 (strict numerical comparisons, not a significance test).

All candidates, fold fits, inputs, and selected fits are retained.
A lower validation error does not guarantee better coefficient recovery.
See docs/DATA_ONLY_TUNING_PLAN.md; this is not a fair timing benchmark.
