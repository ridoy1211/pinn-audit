# Independent field scoring of the saved diagnostic fits

Post-hoc evaluation, not tuning. All errors below are percentages.
Field/IC scores use 401-point trapezoidal quadrature; the 201-point scores are also saved.
Only two selected development datasets are represented, regardless of the number of fits.

| seed | lambda | lower alpha | search grid | alpha error | field L2 error | IC L2 error |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.001 | 401 | 12.8989% | 21.4249% | 73.1472% |
| 0 | 0 | 0.0001 | 401 | 12.8989% | 21.4249% | 73.1472% |
| 0 | 0 | 1e-05 | 401 | 12.8989% | 21.4249% | 73.1472% |
| 0 | 1e-06 | 0.001 | 401 | 0.9138% | 1.4325% | 3.4507% |
| 0 | 1e-06 | 0.0001 | 401 | 0.9138% | 1.4325% | 3.4507% |
| 0 | 1e-06 | 1e-05 | 401 | 0.9138% | 1.4325% | 3.4507% |
| 1 | 0 | 0.001 | 401 | 27.2542% | 15.9964% | 47.1582% |
| 1 | 0 | 0.0001 | 401 | 99.8000% | 39.4988% | 32.8825% |
| 1 | 0 | 1e-05 | 401 | 99.9800% | 39.4807% | 32.8415% |
| 1 | 0 | 1e-05 | 1601 | 99.9800% | 39.4807% | 32.8415% |
| 1 | 1e-06 | 0.001 | 401 | 98.0000% | 39.5509% | 33.3554% |
| 1 | 1e-06 | 0.0001 | 401 | 99.8000% | 39.4027% | 32.8578% |
| 1 | 1e-06 | 1e-05 | 401 | 99.9800% | 39.3885% | 32.8186% |
| 1 | 1e-06 | 1e-05 | 1601 | 99.9800% | 39.3885% | 32.8186% |

Largest field-error change between quadrature grids: 0.00920457 percentage points.
This is a quadrature consistency check, not a statistical confidence interval.
