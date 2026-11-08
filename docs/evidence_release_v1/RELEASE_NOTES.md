# Controlled heat-inversion evidence release

**Draft release notes. No public download has yet been verified.**

This package contains saved synthetic observations, fitted models, diagnostics and
analysis outputs for the completed controlled-timing and off-basis extensions.
It is an evidence release, not a claim that the portable reproduction workflow is finished.

Documentation/inventory source commit: `2f0bb2f65fcfd8b2260ae9605308f9438d15e151`.
Completed results branch commit: `1a4a790fc44991b47f8b815458215577e7aec259`.
The eventual release tag should identify the commit containing these release documents
and the completed restore instructions; it has not been assigned by this command.

## Required evidence assets

| Archive | Bytes | Files |
|---|---:|---:|
| fix1-results-b6906ae.zip | 328000107 | 5856 |
| fix2-evidence-v1.zip | 1261508106 | 5076 |
| fix3-evidence-v1.zip | 97979869 | 1799 |

Download all three archives and SHA256SUMS.txt. The automatically generated
GitHub source-code ZIP is not a substitute for these ignored result archives.

## Contents

### fix1-results-b6906ae.zip

- `results/timing_development_v1/`: 508 files
- `results/timing_evaluation_analysis_v1/`: 9 files
- `results/timing_evaluation_v1/`: 5045 files
- `results/timing_ridge_check_v1/`: 294 files

### fix2-evidence-v1.zip

- `results/timing_basis_flexibility_v1/`: 4324 files
- `results/timing_fix2_report_v1/`: 13 files
- `results/timing_identifiability_v1/`: 365 files
- `results/timing_rollout_pilot_v1/`: 6 files
- `results/timing_rollout_refinement_v1/`: 4 files
- `results/timing_rollout_v1/`: 364 files

### fix3-evidence-v1.zip

- `results/offbasis_analysis_v1/`: 7 files
- `results/offbasis_development_v1/`: 101 files
- `results/offbasis_evaluation_v1/`: 1685 files
- `results/offbasis_truth_verification_v1/`: 6 files

## Integrity and restoration

SHA256SUMS.txt records the expected archive bytes. Verify downloads before extraction.
Each archive stores paths beginning with results/. Use a new empty staging directory;
do not extract over a working research checkout or its committed report folders.
About 2.39 billion bytes are needed for extracted evidence, in addition to about
1.69 billion bytes for the three downloads. Allow extra space for replay outputs.
A tested restore command and a clean-checkout verification record remain required
before publication. Do not treat this draft as an operational reproduction guide.

## Provenance and interpretation

The original freezes and their hashes remain unchanged. See the repository source
provenance addendum for the demonstrated Git LF versus historical CRLF mapping.
The evidence archives retain their original contents and study-stage classifications.
The primary timing interaction is inconclusive; secondary results must retain their
pointwise interval and multiplicity qualifications. The off-basis study is one
localized initial field, not a universal robustness claim.

## Scope exclusions

Local Git bundles are redundant history backups and are not required evidence assets.
Code history remains in the repository; preserve the bundles locally.
The later publication audit and off-basis observation-tail diagnostic are not silently
inserted into these historical archives. Archive those separately when finalized.
No paper DOI, acceptance status, or permanent archive identifier is claimed.
