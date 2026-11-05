# Reproducibility audit and regularized inverse pilot - 2026-11-03

Recorded after execution on 3 November 2026. This is a post-hoc exploratory checkpoint,
not a preregistration or an independent confirmation study.

## What is preserved

- Repairs to start-time chronology, code/config provenance, summaries, and the
  regularized exact-spectral heat inverse baseline.
- User CMD installation/test transcript: Python 3.14.6; 180 tests passed in 55.22 s.
- Fifty classical fits, ten observation datasets, forty profile arrays, all declared
  regularization values, run metadata, and reading notes.
- No PINNs were trained in this pilot. Historical neural raw results remain absent.

## Findings and limits

For N=10, spectral lambda=1e-6 reduces median relative diffusivity error from 25.40%
to 12.10%, but raises mean error from 21.71% to 27.16%. The worst observed error
rises from 40.71% to 98.00%; one estimate reaches the lower alpha bound. The lower
median therefore does not demonstrate consistently better recovery. These five
development datasets support exploration, not a population superiority claim.

The numerical and unregularized exact-spectral estimators give closely agreeing
medians here, suggesting numerical forward discretization is not the dominant
source of this specific sparse-data error. Identifiability, bounds, noise, and
regularization need further investigation before a matched PINN comparison.

## Files

- [Full pilot summary](../results/my_classical_pilot_v1/summary.md)
- [Raw fit records](../results/my_classical_pilot_v1/records.json)
- [Reading and research notes](history/initial_supervision/RESEARCH_NOTES_2026-11-03.md)
- [Study guide](history/initial_supervision/STUDY_GUIDE_2026-11-03.md)
- [Original-source audit](../audit/code_audit.md)
- [Run review](../audit/user_run_evidence/day1_pilot_review.md)
- [Input/output hashes](../audit/user_run_evidence/day1_pilot_review.json)
- [Checkpoint import manifest](audit/regularized_inverse_checkpoint_2026-11-03.json)

## Provenance

The repaired export was based on source commit
c0c6413e174102878dfa168603593bcbde4c504c. It was originally extracted without Git
metadata, so its pilot manifest correctly records base_git_revision as unknown.
Those original records have not been backfilled or altered. The checkpoint import
manifest records the Git parent used for this import and the imported file hashes.

Scoped Git attributes preserve original byte hashes for the saved pilot and
user-run evidence. The raw environment snapshot includes the earlier editable
installation path; requirements-audit.txt removes that machine-specific editable
entry and is a software-version record, not a cross-platform guarantee.

## Reproduction

Use a fresh environment with a compatible Python version. The user's validation
was on 3.14.6. Then run:

```bat
python -m pip install -r requirements-audit.txt
python -m pip install -e ".[dev,nn]"
python -m pytest --basetemp work_test_temp_reproduce_1
python scripts/pilot_regularization.py --out-dir results/my_classical_pilot_reproduce_1
```

The temporary test directory must be dedicated and disposable. Use a fresh pilot
output directory. Do not overwrite archived records. Repeated timings will vary.

## Next milestone

Investigate the bound-limited estimate, then freeze matched priors, separate data
and training replication, validation-only tuning, evaluation, and the final protocol.
The separate bounds/search diagnostic script and its selected-example results are
not included in this checkpoint. The Git commit saves an already completed exploratory stage.

## Relationship to the earlier project history

This is a continuation of the existing project, not a restart of project Day 1.
The original lab log begins in October 2026 and already identifies regularized
unknown-initial-condition inversion as an open task. A dated 2026-11-03 audit
entry is appended to docs/LOG.md; earlier entries are preserved.

Some working files contain day1 or DAY1 in their names. Those are retained
supervision-session/capture labels, not a claim that the project began in November.
Original experimental records and hashes are preserved. Formal branch, commit,
report, and log labels use the work description and date.
