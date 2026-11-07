# Controlled timing evaluation

720 completed fits; 30 independent blocks per observation count, shared across families and schedules.
Verified 5044 indexed artifact hashes.

## Prespecified primary result

N=10, family B; diffusivity error. D = PINN error minus CV12 error.
I = D(mixed) minus D(late), calculated within each block before taking the median.
Median I: 3.7322 percentage points; 95% percentile bootstrap interval [-0.0947, 56.2421].

A negative interaction means the PINN-relative-to-CV12 error is smaller under mixed observations; a positive interaction means it is larger. This does not by itself identify which method has lower absolute error.

## Median diffusivity errors (%)

| N | Family | Schedule | CV12 | PINN |
|---|---|---|---:|---:|
| 10 | A | early | 28.303 | 99.863 |
| 10 | A | mixed | 5.113 | 6.616 |
| 10 | A | late | 29.470 | 7.939 |
| 10 | B | early | 90.130 | 99.951 |
| 10 | B | mixed | 10.862 | 10.410 |
| 10 | B | late | 29.515 | 7.749 |
| 50 | A | early | 5.111 | 4.792 |
| 50 | A | mixed | 1.586 | 1.267 |
| 50 | A | late | 1.736 | 1.500 |
| 50 | B | early | 3.528 | 6.207 |
| 50 | B | mixed | 1.287 | 1.398 |
| 50 | B | late | 1.702 | 1.557 |

## Interpretation limits

All secondary intervals are pointwise, not simultaneous. Bootstrap intervals with 30 blocks may be sensitive to skewness and boundary estimates. Do not infer significance from a comparison of marginal medians.
The schedules change delay and temporal spread together. Fixed absolute noise means late-time signal-to-noise ratios differ.
The result concerns the frozen procedures, one PINN start, and two synthetic initial fields. It does not demonstrate restored unconstrained identifiability or universal method superiority.
Job runtimes include saved diagnostics and scoring; distinguish them from fitting-only costs.
