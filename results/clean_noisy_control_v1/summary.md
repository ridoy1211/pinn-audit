# Clean/noisy control (exploratory)

Same coordinates; two selected development datasets; 24 classical fits; no PINNs.
All recovery errors below are percentages. True alpha is not used for selection.

| seed | target | lambda | lower alpha | alpha estimate | bound hit | alpha error (%) | field error (%) | IC error (%) | search warning |
|---|---|---|---|---|---|---|---|---|---|
| 0 | clean | 0 | 0.001 | 0.05 | False | 2.1848e-08 | 3.6533e-08 | 1.3803e-07 | False |
| 0 | clean | 0 | 0.0001 | 0.05 | False | 2.0301e-07 | 3.3945e-07 | 1.2826e-06 | False |
| 0 | clean | 0 | 1e-05 | 0.05 | False | 4.3828e-07 | 7.3285e-07 | 2.769e-06 | False |
| 0 | clean | 1e-06 | 0.001 | 0.049791561 | False | 0.41688 | 0.28322 | 0.74059 | False |
| 0 | clean | 1e-06 | 0.0001 | 0.049791562 | False | 0.41688 | 0.28322 | 0.74059 | False |
| 0 | clean | 1e-06 | 1e-05 | 0.049791561 | False | 0.41688 | 0.28322 | 0.74059 | False |
| 0 | noisy | 0 | 0.001 | 0.043550551 | False | 12.899 | 21.425 | 73.147 | False |
| 0 | noisy | 0 | 0.0001 | 0.043550552 | False | 12.899 | 21.425 | 73.147 | False |
| 0 | noisy | 0 | 1e-05 | 0.043550551 | False | 12.899 | 21.425 | 73.147 | False |
| 0 | noisy | 1e-06 | 0.001 | 0.050456889 | False | 0.91378 | 1.4325 | 3.4507 | False |
| 0 | noisy | 1e-06 | 0.0001 | 0.050456889 | False | 0.91378 | 1.4325 | 3.4507 | False |
| 0 | noisy | 1e-06 | 1e-05 | 0.050456888 | False | 0.91378 | 1.4325 | 3.4507 | False |
| 1 | clean | 0 | 0.001 | 0.05 | False | 1.5596e-07 | 1.6712e-07 | 5.8988e-07 | False |
| 1 | clean | 0 | 0.0001 | 0.050000001 | False | 2.0129e-06 | 2.1569e-06 | 7.6133e-06 | False |
| 1 | clean | 0 | 1e-05 | 0.05 | False | 7.3928e-08 | 7.9217e-08 | 2.7962e-07 | False |
| 1 | clean | 1e-06 | 0.001 | 0.049318805 | False | 1.3624 | 0.58575 | 1.772 | False |
| 1 | clean | 1e-06 | 0.0001 | 0.049318806 | False | 1.3624 | 0.58575 | 1.772 | False |
| 1 | clean | 1e-06 | 1e-05 | 0.049318806 | False | 1.3624 | 0.58575 | 1.772 | False |
| 1 | noisy | 0 | 0.001 | 0.036372876 | False | 27.254 | 15.996 | 47.158 | False |
| 1 | noisy | 0 | 0.0001 | 0.0001 | True | 99.8 | 39.499 | 32.882 | False |
| 1 | noisy | 0 | 1e-05 | 1e-05 | True | 99.98 | 39.481 | 32.842 | False |
| 1 | noisy | 1e-06 | 0.001 | 0.001 | True | 98 | 39.551 | 33.355 | False |
| 1 | noisy | 1e-06 | 0.0001 | 0.0001 | True | 99.8 | 39.403 | 32.858 | False |
| 1 | noisy | 1e-06 | 1e-05 | 1e-05 | True | 99.98 | 39.388 | 32.819 | False |

Noisy estimates reproduce the archived diagnostic within declared tolerances.
Search warning means a fitted objective is materially worse than a separately
evaluated feasible candidate at true alpha; it is not a confidence statement.
Rank/conditioning in records.json concerns amplitudes conditional on fitted alpha.
See docs/CLEAN_NOISY_CONTROL_PLAN.md for interpretation and limitations.
