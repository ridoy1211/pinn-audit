# Claims and evidence ledger

Prepared 3 November 2026. Use this ledger while writing; update status only when the
named evidence exists. A plausible explanation is not an experimental finding.

| Prospective claim | Current status | Evidence required or correct boundary |
|---|---|---|
| The repository implements exact heat/wave references and convergence-tested FD schemes | Source inspected; runtime verification reported separately in the handoff | Relevant analytic/residual/convergence tests and saved runtime verification log |
| Historical reported PINN/FD error and time ratios are correct | Not independently verified | Original row-level results or a newly labeled reproduction on a recorded environment |
| Historical predictions were committed before the relevant runs started | Not established by the old checker | Actual start records plus the exact prediction content/revision; completion stamps alone are insufficient |
| Increasing IC/BC weights and adding L-BFGS helped the tested wave PINN | Reported historical observation | Original paired rows; bounded to that equation, mode content, configurations, and budget |
| This intervention disproves spectral bias | Unsupported | A training improvement does not uniquely identify or exclude a frequency-dependent optimization mechanism |
| The PINN is better at joint diffusivity/unknown-IC estimation with ten noisy observations | Exploratory historical comparison against an unregularized baseline | Independent paired datasets, explicit regularized comparator, shared priors, tuning controls, and uncertainty |
| Regularization changes classical sparse-data recovery | Supported in a new small development pilot | N=10 median error moved from 25.4% to 12.1% at lambda=1e-6, but rose to 54.3% at lambda=1e-2; report the full grid and five datasets |
| Regularization removes the apparent PINN advantage | Hypothesis, not established | Final held-out paired study; the classical-only development pilot cannot answer it |
| Observation timing changes the relative recovery performance | Proposed hypothesis | Frozen timing/noise convention and the new primary/secondary cell results |
| A single-temperature snapshot identifies diffusivity with an arbitrary unknown initial field | False without additional assumptions | State the coefficient-decay confounding argument and the exact data/prior assumptions |
| Low data loss or PDE residual implies accurate physical parameters | Unsupported in general | Measure parameter error separately and inspect sensitivity/identifiability |
| A narrow negative benchmark is automatically journal-publishable | Unsupported | A useful original contribution, adequate evidence, journal scope, and external peer review |
| This generic heat project is a fusion, quantum, or particle-physics application | Not demonstrated | Actual domain model/data or a clearly described future application |

## Oral-defense checks

Before drafting conclusions, explain these in your own words and reproduce at least
one small calculation for each relevant item:

1. Derive the sine-mode heat solution and explain the alpha–time–frequency decay.
2. Explain why CN needs convergence testing even when it is stable.
3. Explain what automatic differentiation measures and how the exact residual test works.
4. Derive the linear amplitude fit conditional on alpha and the regularized augmented system.
5. Explain why ten observations and seven coefficients may still be poorly conditioned.
6. Distinguish structural identifiability, practical identifiability, regularization bias,
   numerical approximation error, and neural optimization error.
7. Explain why data seeds and neural restarts are different uncertainty sources.
8. Explain why test-time access to exact alpha would invalidate a practical tuning claim.
9. Explain why a penalized profile is not automatically a parameter confidence interval.
10. Trace one figure from raw observation file through estimator and metric to the plotted point.
