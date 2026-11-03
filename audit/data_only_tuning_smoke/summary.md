# Data-only tuning development study

Status: software_smoke_only
All recovery errors below are percentages. No PINNs trained.
Datasets are development data, not held-out confirmation.

| N | method | datasets | median alpha error | mean alpha error | max alpha error | median field error | median IC error | bound hits |
|---|---|---|---|---|---|---|---|---|
| 10 | unregularized | 1 | 1.833 | 1.833 | 1.833 | 51.91 | 211.7 | 0 |
| 10 | fixed_1e-6 | 1 | 3.798 | 3.798 | 3.798 | 4.38 | 9.248 | 0 |
| 10 | CV_selected | 1 | 3.798 | 3.798 | 3.798 | 4.38 | 9.248 | 0 |
| 50 | unregularized | 1 | 6.8 | 6.8 | 6.8 | 1.391 | 3.401 | 0 |
| 50 | fixed_1e-6 | 1 | 6.618 | 6.618 | 6.618 | 1.337 | 3.15 | 0 |
| 50 | CV_selected | 1 | 6.618 | 6.618 | 6.618 | 1.337 | 3.15 | 0 |

N=10 selected-lambda counts: {'0': 0, '1e-08': 0, '1e-06': 1, '0.0001': 0, '0.01': 0}
N=10, relative_alpha_error, CV vs unregularized paired wins/ties/losses: 0/0/1 (strict numerical comparisons, not a significance test).
N=10, field_relative_l2, CV vs unregularized paired wins/ties/losses: 1/0/0 (strict numerical comparisons, not a significance test).
N=50 selected-lambda counts: {'0': 0, '1e-08': 0, '1e-06': 1, '0.0001': 0, '0.01': 0}
N=50, relative_alpha_error, CV vs unregularized paired wins/ties/losses: 1/0/0 (strict numerical comparisons, not a significance test).
N=50, field_relative_l2, CV vs unregularized paired wins/ties/losses: 1/0/0 (strict numerical comparisons, not a significance test).

All candidates, fold fits, inputs, and selected fits are retained.
A lower validation error does not guarantee better coefficient recovery.
See docs/DATA_ONLY_TUNING_PLAN.md; this is not a fair timing benchmark.
