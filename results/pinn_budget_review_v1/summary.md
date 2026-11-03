# Paired neural budget review

Two fixed development datasets; all recovery errors are percentages.
PINN medians summarize three initializations of each dataset, not three datasets.

| N | L-BFGS limit | median alpha error | median field error | median IC error | median training seconds |
|---|---|---|---|---|---|
| 10 | 200 | 2.513 | 2.231 | 6.009 | 28.51 |
| 10 | 1000 | 0.1296 | 0.8145 | 2.421 | 47.48 |
| 50 | 200 | 1.195 | 0.5549 | 1.053 | 31.52 |
| 50 | 1000 | 1.834 | 1.006 | 2.962 | 47.41 |

All six Adam phases reproduce. Longer-budget training objectives and recovery metrics are retained together.
No budget is selected by truth error, and no new fit was run by this review.
