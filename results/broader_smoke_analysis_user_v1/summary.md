# Broader comparison analysis

Accuracy summaries are conditional on completion; failures remain in the completeness ledger.

Dataset cases: 4; independent designs: 2.

Mode: **software_smoke_only**. Collection status: complete.
Median errors below are percentages. Neural case scores require all three starts.

| N | family | method | complete / expected datasets | median alpha error | median field error | median IC error |
|---|---|---|---|---|---|---|
| 10 | A | unreg6 | 1 / 1 | 23.459 | 61.593 | 208.2 |
| 10 | A | fixed6 | 1 / 1 | 4.1173 | 7.6324 | 18.114 |
| 10 | A | cv6 | 1 / 1 | 4.1173 | 7.6324 | 18.114 |
| 10 | A | cv12 | 1 / 1 | 4.1208 | 7.6338 | 18.117 |
| 10 | A | pinn_200 | 1 / 1 | 176.61 | 55.882 | 78.557 |
| 10 | A | pinn_1000 | 1 / 1 | 16.147 | 23.555 | 47.418 |
| 10 | B | unreg6 | 1 / 1 | 23.486 | 62.277 | 206.31 |
| 10 | B | fixed6 | 1 / 1 | 3.7723 | 9.1904 | 28.012 |
| 10 | B | cv6 | 1 / 1 | 3.7723 | 9.1904 | 28.012 |
| 10 | B | cv12 | 1 / 1 | 3.7758 | 9.1751 | 27.923 |
| 10 | B | pinn_200 | 1 / 1 | 176.6 | 56.04 | 79.661 |
| 10 | B | pinn_1000 | 1 / 1 | 16.121 | 24.094 | 51.121 |
| 50 | A | unreg6 | 1 / 1 | 1.471 | 0.69477 | 1.4686 |
| 50 | A | fixed6 | 1 / 1 | 1.8118 | 0.72331 | 1.488 |
| 50 | A | cv6 | 1 / 1 | 1.8118 | 0.72331 | 1.488 |
| 50 | A | cv12 | 1 / 1 | 1.8064 | 0.71776 | 1.4742 |
| 50 | A | pinn_200 | 1 / 1 | 186.6 | 44.835 | 55.144 |
| 50 | A | pinn_1000 | 1 / 1 | 35.125 | 21.186 | 44.184 |
| 50 | B | unreg6 | 1 / 1 | 0.46337 | 6.4286 | 24.83 |
| 50 | B | fixed6 | 1 / 1 | 0.21433 | 6.4256 | 24.729 |
| 50 | B | cv6 | 1 / 1 | 0.21433 | 6.4256 | 24.729 |
| 50 | B | cv12 | 1 / 1 | 1.2757 | 3.6 | 13.885 |
| 50 | B | pinn_200 | 1 / 1 | 186 | 44.596 | 57.356 |
| 50 | B | pinn_1000 | 1 / 1 | 36.284 | 21.933 | 48.089 |

## Paired contrasts against CV12

Differences are neural minus CV12 in percentage points; negative values favor the neural pipeline.

| N | Family | Neural method | Metric | Complete pairs | Median pp | Mean pp | Wins / ties / losses | 95% percentile CI for median pp |
|---:|---|---|---|---:|---:|---:|---:|---|
| 10 | A | pinn_200 | relative_alpha_error | 1 | 172.4847749578678 | 172.4847749578678 | 0 / 0 / 1 | not estimated |
| 10 | A | pinn_200 | field_relative_l2 | 1 | 48.24820095064923 | 48.24820095064923 | 0 / 0 / 1 | not estimated |
| 10 | A | pinn_200 | ic_relative_l2 | 1 | 60.43973065781919 | 60.43973065781919 | 0 / 0 / 1 | not estimated |
| 10 | A | pinn_1000 | relative_alpha_error | 1 | 12.026526122805553 | 12.026526122805553 | 0 / 0 / 1 | not estimated |
| 10 | A | pinn_1000 | field_relative_l2 | 1 | 15.921227997938892 | 15.921227997938892 | 0 / 0 / 1 | not estimated |
| 10 | A | pinn_1000 | ic_relative_l2 | 1 | 29.301125051564537 | 29.301125051564537 | 0 / 0 / 1 | not estimated |
| 10 | B | pinn_200 | relative_alpha_error | 1 | 172.8213568104992 | 172.8213568104992 | 0 / 0 / 1 | not estimated |
| 10 | B | pinn_200 | field_relative_l2 | 1 | 46.86464596347042 | 46.86464596347042 | 0 / 0 / 1 | not estimated |
| 10 | B | pinn_200 | ic_relative_l2 | 1 | 51.73799731139969 | 51.73799731139969 | 0 / 0 / 1 | not estimated |
| 10 | B | pinn_1000 | relative_alpha_error | 1 | 12.344822564855765 | 12.344822564855765 | 0 / 0 / 1 | not estimated |
| 10 | B | pinn_1000 | field_relative_l2 | 1 | 14.918757693980222 | 14.918757693980222 | 0 / 0 / 1 | not estimated |
| 10 | B | pinn_1000 | ic_relative_l2 | 1 | 23.197498004911953 | 23.197498004911953 | 0 / 0 / 1 | not estimated |
| 50 | A | pinn_200 | relative_alpha_error | 1 | 184.79330666258275 | 184.79330666258275 | 0 / 0 / 1 | not estimated |
| 50 | A | pinn_200 | field_relative_l2 | 1 | 44.11690424168978 | 44.11690424168978 | 0 / 0 / 1 | not estimated |
| 50 | A | pinn_200 | ic_relative_l2 | 1 | 53.669740519594946 | 53.669740519594946 | 0 / 0 / 1 | not estimated |
| 50 | A | pinn_1000 | relative_alpha_error | 1 | 33.31877282909278 | 33.31877282909278 | 0 / 0 / 1 | not estimated |
| 50 | A | pinn_1000 | field_relative_l2 | 1 | 20.468362763979115 | 20.468362763979115 | 0 / 0 / 1 | not estimated |
| 50 | A | pinn_1000 | ic_relative_l2 | 1 | 42.709453613644044 | 42.709453613644044 | 0 / 0 / 1 | not estimated |
| 50 | B | pinn_200 | relative_alpha_error | 1 | 184.72023510160918 | 184.72023510160918 | 0 / 0 / 1 | not estimated |
| 50 | B | pinn_200 | field_relative_l2 | 1 | 40.99570909163499 | 40.99570909163499 | 0 / 0 / 1 | not estimated |
| 50 | B | pinn_200 | ic_relative_l2 | 1 | 43.47082855405277 | 43.47082855405277 | 0 / 0 / 1 | not estimated |
| 50 | B | pinn_1000 | relative_alpha_error | 1 | 35.00787029525328 | 35.00787029525328 | 0 / 0 / 1 | not estimated |
| 50 | B | pinn_1000 | field_relative_l2 | 1 | 18.332776532606204 | 18.332776532606204 | 0 / 0 / 1 | not estimated |
| 50 | B | pinn_1000 | ic_relative_l2 | 1 | 34.204336516218056 | 34.204336516218056 | 0 / 0 / 1 | not estimated |
