# Off-basis development decisions

All 16 development fits completed. All eight PINNs completed their optimization
budget. No classical fit selected an endpoint of the seven-candidate ridge grid.

Across the checked first blocks, the maximum absolute change in reported field
or initial-field relative errors between 401- and 801-point scoring grids was
0.01789171 percentage points. This is a numerical refinement check,
not a universal error bound. Retain the existing scoring and refinement settings.

Recovery was difficult in several pilot cases. Preserve these outcomes; do not
change the initial field, noise, schedules, model settings, or optimization budget
to improve truth-based performance.

Decision: retain the verified initial field and all pilot fitting settings.
Proceed with 30 fresh blocks per observation count and two schedules:
120 evaluation datasets, 240 fits, 60 independent design/noise blocks.
This is a planning sample size, not a power guarantee.

Data seeds: N=10, 1106000 through 1106029; N=50, 1107000 through 1107029.
Bootstrap seed: 1108000. PINN initialization remains 101.
Comparisons are secondary robustness analyses with pointwise intervals.
The evaluation source/configuration freeze must precede data generation.

Development manifest SHA256: 02b57c32dc9f86136a2975ccd07ae4909c7a42757ca855090d07131cfa4a9cef
