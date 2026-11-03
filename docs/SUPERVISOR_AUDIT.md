# Supervision repair and evidence boundary (3 November 2026)

This working copy extends source commit
`c0c6413e174102878dfa168603593bcbde4c504c` from
https://github.com/ridoy1211/pinn-audit.

The historical numerical results in README and LOG were not regenerated in this
audit. Their raw result files were absent from GitHub and the supplied Downloads
archives. Preserve the historical entries; do not treat this repair as independent
verification of those numbers or of old prediction chronology.

## Implemented

- Actual run-start and completion timestamps, with the historical timestamp field
  retained as completion. Legacy/mixed rows without starts cannot verify chronology.
- Code/source-aware execution identities, resolved config snapshots, and unique
  forward solution filenames. A frozen, fresh output directory remains essential.
- Summary guards against duplicate replicates and conflicting new-record identities.
- An exact spectral variable-projection baseline for unknown-IC heat inversion,
  with an explicit coefficient smoothness penalty and alpha-objective profile.
- An exploratory classical-only pilot that saves all observations and estimates,
  all declared ridge values, code hashes, profiles, config, and environment information.
- Regression tests for the scientific baseline and provenance safeguards.

## Not implemented by this repair

The proposed confirmation study still requires independently crossed dataset and
training seeds, shared parameter bounds, a declared tuning/validation procedure,
common field/IC scoring, noise/timing controls, and a frozen final run plan. The
spectral baseline does not currently plug into the existing `pinnaudit inverse`
CLI; use the supplied Python API or pilot script. It is specialized to constant
diffusivity with homogeneous Dirichlet boundaries and is not a new general method.

Summary guards do not establish the provenance of legacy data. Commit subjects and
local clocks do not independently authenticate prediction content. Bind a new
prediction to its actual file/content and immutable revision before execution.

## Run in a fresh checkout/output directory

```console
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev,nn]"
pytest
python scripts/pilot_regularization.py --out-dir results/spectral_pilot_v1
```

The pilot uses only development seeds 0 through 4, N=10/50, 5% noise under the old
sample-standard-deviation convention, six IC modes, and four predeclared ridge
values. It does not train PINNs, choose a final ridge by ground-truth error, or
establish a comparative research conclusion. Repeated invocations require a new
output directory to protect the existing evidence.

The ZIP handoff contains a source snapshot, not the original Git history. For
full provenance, clone the public repository at the base commit and apply the
provided repair patch, or preserve the base revision and all file hashes together.
