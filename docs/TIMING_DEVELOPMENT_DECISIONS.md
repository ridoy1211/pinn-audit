\# Timing-study development decisions



\## Regularization candidate range



The original 36-dataset development pilot used:

1e-10, 1e-8, 1e-6, 1e-4, 1e-2.



One dataset selected the smallest candidate:

N=10, initial field B, early observations, data seed 806000.



A separate classical-only sensitivity check used:

1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2.



All 36 fits completed. Validation scores for all shared candidates

reproduced within the check's declared numerical tolerance.

Only the identified dataset changed its selected penalty.



For that dataset:

\- At 1e-10: validation MSE 0.00304791931579;

&#x20; fitted diffusivity 0.03464022413178398.

\- At 1e-12: validation MSE 0.00304245764810;

&#x20; fitted diffusivity 0.03463718814761951.

\- At 1e-14: validation MSE 0.00304240297775;

&#x20; fitted diffusivity 0.03463715738955979.



Validation scores approach a plateau, and the fitted diffusivity

changes little across these three candidates. This check does not

establish stability of every initial-state coefficient.



Decision: use the expanded seven-candidate grid for the forthcoming

evaluation. Do not extend it further on the basis of this pilot.

Retain the existing cross-validation selection and tie-breaking rule.



The selected penalty remains at the lower endpoint in one development

case. No claim is made that an interior regularization optimum was found.



The decision uses development validation evidence. Evaluation data

have not been used to select the grid.



Both completed development runs remain unchanged and are retained.

The evaluation configuration and seeds will be frozen separately.

