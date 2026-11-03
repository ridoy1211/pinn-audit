# Exploratory classical-only pilot

Five development data seeds; no PINNs were trained by this script. All lambda values
are shown. The smallest error is not a justified selection rule for confirmation.

| N | method | ridge | median relative alpha error | median seconds |
|---|---|---|---|---|
| 10 | fd_unregularized | None | 0.254019 | 1.17929 |
| 10 | spectral | 0.0 | 0.254006 | 0.0230077 |
| 10 | spectral | 1e-06 | 0.12103 | 0.0243789 |
| 10 | spectral | 0.0001 | 0.210209 | 0.0228878 |
| 10 | spectral | 0.01 | 0.542955 | 0.0252646 |
| 50 | fd_unregularized | None | 0.00397788 | 0.526691 |
| 50 | spectral | 0.0 | 0.00399417 | 0.0283463 |
| 50 | spectral | 1e-06 | 0.00200549 | 0.0280683 |
| 50 | spectral | 0.0001 | 0.112573 | 0.0301726 |
| 50 | spectral | 0.01 | 0.502526 | 0.0245592 |
