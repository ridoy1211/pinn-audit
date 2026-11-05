# CI test-launch correction — 5 November 2026

The user's documentation checkpoint 2816923 pushed successfully. Its first
research-branch CI run, [41903474469](https://github.com/ridoy1211/pinn-audit/actions/runs/41903474469),
failed in both Python jobs before test execution. Installation, lint/formatting
and mypy succeeded; collection of three broader-study test modules failed with
`ModuleNotFoundError: No module named 'scripts'`.

Local verification used `python -m pytest`, but the workflow used the console
launcher `pytest`. Tests import repository-level `scripts` helpers, which are
not installed as part of the src-layout library. The module invocation puts the
working project root on the import path. The earlier audit did not test both
entry points, so it missed this CI discrepancy.

## Repair and verification

The Test step now runs:

```text
python -m pytest --cov=pinnaudit --cov-report=term-missing
```

The console-launcher collection failure was reproduced locally. With the
corrected module invocation, **227 tests passed in 63.76 seconds**, with 92%
coverage of the `pinnaudit` source package (not all repository scripts).
Scratch cache/coverage/temp paths were placed in the workspace. No evaluation
model was retrained. The 15 frozen source/config/protocol hashes remain unchanged.

CI failure excerpts are retained in
[`audit/ci2816923_import_failure.txt`](../audit/ci2816923_import_failure.txt).
The repair is local until committed and pushed. New GitHub results are not
claimed as passed before the new head has actually run.

## Unrelated line-ending warning

Git normalized the archived SciSpace text from CRLF to LF. Comparing it with the
supplied attachment confirms the text is identical after only that conversion.
This is not a commit failure, a numerical-result change or the cause of CI's
import error. No change to the archived experiment is needed for that warning.
