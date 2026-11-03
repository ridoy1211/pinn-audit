# Day 1 pilot review

Reviewed 3 November 2026. The saved run manifest records the start at 2026-11-02 19:13:10 UTC, corresponding to 3 November 2026 at 01:13:10 in Bangladesh. This is a local machine timestamp, not independent authentication.

Checked all 50 expected fits, ten saved observation datasets, and forty profile files. There are no duplicate or missing jobs; numerical estimates/errors are finite and internally consistent. All ten FD optimizer success flags are true. Source files match the hashes captured in the run manifest.

The observation arrays are identical to the reference pilot in 10/10 files. Maximum absolute difference in relative coefficient errors across all 50 fits is 0. This is reproducibility evidence for the same code/data, not new independent datasets.

## Results in percent

| N | FD, no penalty | Spectral, no penalty | Spectral lambda 1e-6 | Spectral lambda 1e-4 | Spectral lambda 1e-2 |
|---|---|---|---|---|---|
| 10 | 25.4019% | 25.4006% | 12.1030% | 21.0209% | 54.2955% |
| 50 | 0.3978% | 0.3994% | 0.2005% | 11.2573% | 50.2526% |

## Interpretation

The numerical and exact spectral unregularized estimators give closely agreeing medians in these cells. This makes forward discretization an unlikely main explanation for this particular sparse-data error. Regularization changes recovery substantially: a small tested penalty improves the median here, whereas a large penalty worsens it. This does not identify a universally useful penalty.

The penalty is applied to unknown initial-condition amplitudes, with larger weights for higher modes. It can stabilize initial-field estimates, while an excessive penalty can bias both the inferred field and diffusivity. This is a mechanism to investigate; the medians alone do not establish a full causal explanation.

The lambda 1e-6 comparison used the same five development datasets as the original pilot. No held-out confirmation or PINN run is present. Do not choose a final lambda using the true alpha. Future selection requires the declared data-only validation procedure and matched priors.

The extracted ZIP has no Git repository, so base_git_revision is recorded as unknown. Preserve the recorded manifest unchanged. The export has source hashes and a SOURCE_COMMIT file; establish a proper Git checkout before freezing the confirmation protocol.

Day 1 software execution is complete. Remaining Day 1 work is to explain the model, the role of regularization, and the five-question reading notes.

## Individual sparse-data outcomes: a lower median hides a severe error

| Data seed | Spectral, no penalty | Spectral, lambda=1e-6 |
|---|---|---|
| 0 | 12.899% | 0.914% |
| 1 | 27.254% | 98.000% |
| 2 | 25.401% | 21.178% |
| 3 | 2.270% | 3.620% |
| 4 | 40.713% | 12.103% |

At lambda=1e-6, recovery improves on 3/5 sparse datasets and worsens on 2/5. Data seed 1 returns alpha=0.001, the search lower bound, against truth alpha=0.05: a 98% relative error. This is a bound-limited parameter estimate, not an optimizer crash.
For lambda=0: mean error 21.7074%, median 25.4006%, maximum 40.7133%.
For lambda=1e-06: mean error 27.1630%, median 12.1030%, maximum 98.0000%.

Therefore the small penalty has a lower median but a higher mean and worse maximum error in this five-dataset pilot. It is not defensible to say the regularized method is consistently more reliable. This motivates the planned analysis of full error distributions, parameter bounds, alpha profiles, and data-only regularization selection. No confidence or general superiority claim follows from five development datasets.
