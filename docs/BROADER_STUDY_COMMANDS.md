# Broader study commands

**Status update, 5 November 2026:** the main collection and original analysis are
complete and archived. The instructions below record how that work was performed;
they are not a new task list. The original user-verification count of 224 was
correct at that checkpoint; three subsequent post-evaluation checks bring the
current committed suite to 227. Do not repeat a completed collection or overwrite
its output directory.

To analyze the saved outcomes again, use a fresh output location:

```cmd
python scripts/analyze_broader_study.py --results results\broader_study_v1 --out-dir results\broader_analysis_replay_01
```

This fits no evaluation models. Current artifact allowlists in `.gitignore`
include the completed study; statements below about ignored outputs describe
collection-time policy. Preserve the frozen protocol's original text and hashes.

The implementation lives on `broader-comparison-implementation`. Run commands
from the existing project root with its environment active. No package reinstall
is required. The final evaluation is protected by a committed source/config freeze.

## First: user-environment software verification

```cmd
python -m pytest --basetemp work_test_temp_broader_user_1
python scripts/run_broader_study.py --smoke --out-dir results\broader_smoke_user_v1
python scripts/analyze_broader_study.py --results results\broader_smoke_user_v1 --out-dir results\broader_smoke_analysis_user_v1
```

Run later commands only after the preceding command succeeds. Expected test
count: 224. Expected collection line: `Finished 40 jobs. Mode: software_smoke_only.
Status: complete.` Expected analysis counts: four dataset cases, two independent
designs, and no ten-design bootstrap intervals. These tiny-budget error values
are software checks and must not enter the paper's performance results.

If any command fails, retain its output directory and report the error. The smoke
directories must be fresh. An already complete run should be analyzed, not rerun.
`--resume` is for a stopped job schedule with unchanged source/config, not for
replacing failed trials or re-running completed fits.

## Then: checkpoint verified implementation and smoke evidence

After checking the user-environment verification:

```cmd
git add .gitignore .gitattributes configs/inverse/broader_comparison.json scripts tests docs audit
git add results/broader_smoke_user_v1 results/broader_smoke_analysis_user_v1
git commit -m "Implement and verify prospective joint heat comparison"
git push
```

The working tree must be clean. Use `git status --short` to check; do not delete
or ignore unexpected scientific evidence merely to make the gate pass.

## Final prospective freeze

```cmd
python scripts/freeze_broader_study.py
git add docs/BROADER_COMPARISON_FREEZE.json
git commit -m "Freeze broader comparison before evaluation data generation"
git push
```

The record identifies the exact implementation commit, source/config/protocol
hashes, and UTC creation time. It is a versioned prospective protocol, not a
formal preregistration. Editing frozen sources/configuration afterward blocks the
evaluation. The freeze file itself must be committed before the main command.

## Main evaluation, after the freeze is verified

```cmd
python scripts/run_broader_study.py --out-dir results\broader_study_v1
python scripts/analyze_broader_study.py --results results\broader_study_v1 --out-dir results\broader_analysis_v1
```

The default study has 40 datasets from 20 independent observation/noise designs,
160 classical pipeline jobs and 240 neural fits. Count paired IC families and
network starts correctly. Allow approximately five hours including tuning and
evaluation; timing is an estimate. Keep all failures. The runner continues after
individual job failures and exits nonzero after preserving a complete schedule
with failures; its analysis can then report completeness and conditional accuracy.
An aborted or interrupted schedule must be handled before planned analysis.

For a stopped schedule, use the same command/config/freeze and output directory
with `--resume`. Previously sealed jobs are checked and skipped. Unsealed attempts
are retained as interrupted, with original start times; only pending jobs run.
No optimizer-state continuation is claimed. If a stale run lock remains after a
process crash, preserve the folder and verify that the process has ended before
recovering the lock. Do not start concurrent writers.

Main-study outputs remain ignored by Git during collection. After completion,
inspect storage and archive the complete artifact set with its hashes before
publishing results or changing the ignore policy. Individual model states and
amplitudes allow dense fields to be reproduced. The protocol specifies which
dense fields and 801-point numerical checks are retained directly.
