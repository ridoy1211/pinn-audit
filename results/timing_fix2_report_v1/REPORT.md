# Fix 2: explanatory diagnostic report

These analyses were specified after evaluation and explain the frozen procedures; they do not replace them.
All 720 added basis fits completed. All 360 rollout numerical screens pass after replacing one result with its separately archived refinement.

## Identifiability

At alpha=0.05, all 30 distinct N=10 designs per schedule have numerical row rank 10 with 12 coefficients. A and B share these designs and are not independent replications of this rank check. For a full-row-rank matrix B(alpha), a=B^T(BB^T)^(-1)y gives an exact interpolant in exact arithmetic. Continuity preserves full row rank locally, so nearby diffusivities can also interpolate using different coefficients. This is established linear algebra, not a new theorem. Numerical rank uses the recorded floating-point cutoff; ill-conditioning and large coefficient norms can limit numerical interpolation.
At N=50, full column rank 12 at the true diffusivity is not by itself proof of joint identification of alpha and the coefficients.
Positive regularization can select a solution without establishing data-only identifiability. Penalized profile curvature includes the prior penalty.

## Initial-state flexibility

| N | Field | Timing | 6-mode alpha error (%) | 12-mode | 24-mode |
|---|---|---|---:|---:|---:|
| 10 | A | early | 16.658 | 28.303 | 99.980 |
| 10 | A | mixed | 3.830 | 5.113 | 4.936 |
| 10 | A | late | 8.693 | 29.470 | 42.361 |
| 10 | B | early | 69.122 | 90.130 | 99.980 |
| 10 | B | mixed | 11.289 | 10.862 | 11.509 |
| 10 | B | late | 8.681 | 29.515 | 42.338 |
| 50 | A | early | 4.296 | 5.111 | 5.275 |
| 50 | A | mixed | 1.431 | 1.586 | 1.576 |
| 50 | A | late | 1.736 | 1.736 | 1.736 |
| 50 | B | early | 56.376 | 3.528 | 5.666 |
| 50 | B | mixed | 5.880 | 1.287 | 1.245 |
| 50 | B | late | 1.702 | 1.702 | 1.702 |

Six modes omit field B's seventh mode. More modes increase flexibility but do not guarantee better recovery. Each basis uses the same data-only CV rule and candidate penalties; the prior is not representation-neutral. The tables are descriptive post-evaluation sensitivities, not truth-based model selection.

## Independent physical consistency

| N | Field | Timing | Median discrepancy (%) | Maximum (%) |
|---|---|---|---:|---:|
| 10 | A | early | 0.0298 | 1.3538 |
| 10 | A | mixed | 0.2038 | 1.0295 |
| 10 | A | late | 0.1501 | 0.5246 |
| 10 | B | early | 0.0425 | 1.3924 |
| 10 | B | mixed | 0.4533 | 3.4683 |
| 10 | B | late | 0.1577 | 0.5376 |
| 50 | A | early | 0.0768 | 0.2679 |
| 50 | A | mixed | 0.1910 | 0.5665 |
| 50 | A | late | 0.1425 | 0.3724 |
| 50 | B | early | 1.0193 | 11.6525 |
| 50 | B | mixed | 1.3491 | 5.8135 |
| 50 | B | late | 0.1488 | 0.3689 |

Discrepancy is the space-time L2 norm of PINN minus independent rollout, divided by the rollout norm. The rollout uses the learned initial field and learned diffusivity, not their true values. Refinement screens are numerical checks, not thresholds defining physical acceptability. Comparisons are global in space-time and can hide local errors. Small consistency error does not establish accurate parameter or initial-state recovery.

## Figure and reporting notes

- basis_sensitivity: marginal median diffusivity errors; no inferential intervals are claimed.
- physical_consistency: every model is shown; black marks are medians.
- profile_examples: field B, first evaluation seed per N, all schedules. The selection uses seed order, not outcome. Profiles re-fit coefficients at each diffusivity. Each positive-penalty curve is a different objective; their absolute levels must not be used to select lambda. Zero plotting values are floored at 1e-30.
- The regularization-range table includes poorly validating candidates. It describes sensitivity across the grid, not statistical uncertainty.
- These diagnostics do not uniquely separate optimization error from identifiability and prior effects.

Fix 2 calculations and reporting are complete. Manuscript integration and external archiving remain separate tasks.
