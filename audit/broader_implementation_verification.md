# Broader workflow verification — 4 November 2026

The assistant implemented the workflow without generating the reserved evaluation
designs (6000-6009 or 7000-7009). The complete test suite passed: **224 passed in
33.35 seconds**, using the separate workspace Python 3.12.14 environment.
New tests cover analytical A/B solutions and paired noise, the six-mode IC
truncation error, observation-only CV with both basis widths, partial CV evidence,
paired statistics, missing/failed/invalid jobs, committed freeze enforcement,
source changes, missing-freeze blocking, and preservation of interrupted attempts.

A complete software run used only design seeds 987100/987101 and training seeds
901/902/903. It completed all 40 jobs (16 classical pipeline jobs and 24 tiny neural
fits) with status complete. Its separate verified analysis reports four dataset
cases from two independent designs, all methods complete, and no underpowered
bootstrap intervals. The source/config/output hashes were checked by the analysis.
These are implementation checks, not scientific performance measurements.

Full assistant-run software evidence remains in the workspace's separate work
directories `broader_smoke_validation_v1` and `broader_smoke_analysis_v1`. This
document is a validation report, not a substitute for those raw artifacts. The
user's analogous smoke folders are explicitly allowed for versioning and will
record the actual user interpreter and software environment before the freeze.

Separate subagents implemented and tested physics/CV helpers and reviewed analysis
logic. The lead agent integrated them, corrected the toy failure-denominator test
(two incomplete short-budget cases, not three), ran the complete tests, and
exercised collection and analysis end to end. A failure in the long-budget arm
does not censor an otherwise valid short-budget case.

The historical estimators, pilot script/configs, and original raw evidence remain
unchanged. The new workflow imports the established bounded neural trainer using
only noisy data and physical settings. Source/config hashes are checked during
collection, paths are portable relative names, and raw output bytes are retained.
Prospective evaluation remains locked until the implementation and protocol are
committed and a matching freeze record is itself committed. The main study has
not been run, and no publication or superiority claim is made by this verification.
