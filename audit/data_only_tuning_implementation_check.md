# Data-only tuning implementation verification — 3 November 2026

The assistant ran four focused tests: validation observations must be excluded
from training, candidate selection follows held-out squared prediction errors,
exact loss ties favor smaller lambda, fold assignment is balanced/reproducible,
and missing folds are rejected. These checks are grouped into four tests; all
passed using Python 3.12.14. No estimator source was changed.

The final script completed its separate two-dataset smoke configuration, involving
60 fits and seeds 987000/987001. Full evidence is in `audit/data_only_tuning_smoke`,
copied byte-for-byte from the separate workspace validation run. All source and
output hashes were checked before archiving. An earlier software-check run used
the same smoke seeds; these are not independent scientific replications.

The default 20-dataset study (seeds 1000-1009/2000-2009) had not been executed or
inspected by the assistant at handover. The smoke cases must not be pooled into
that study. Both the study and smoke mode are labeled development/software checks,
not a preregistered confirmation. Source lint and formatting checks pass.

Separately, all 24 user clean/noisy control records were present, with complete
run status, no feasible-candidate search warnings, and matching source/input
hashes. The original control source and its plan remain unchanged.
