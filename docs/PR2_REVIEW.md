# PR 2 review — 4 November 2026

## Verified remote state

[PR 2](https://github.com/ridoy1211/pinn-audit/pull/2) is already merged into main
at 99586c4347e66fcb36aae3123c3794f161ec26aa. Its five commits end at
857e730, matching the inspected local checkpoint. The merge's parents are
d1de65e and 857e730. Both PR jobs succeeded on that head:

- [Python 3.11](https://github.com/ridoy1211/pinn-audit/actions/runs/41713804974/job/126188432738?pr=2)
- [Python 3.12](https://github.com/ridoy1211/pinn-audit/actions/runs/41713804974/job/126188433140?pr=2)

The public page was inspected after the connected check-diagnostics tool reported
that its GitHub connection was unavailable. The attached PR, public merge record,
head commit, and job pages agree. No merge action was performed by the assistant.

## Scope and scientific interpretation

The checkpoint adds six experiment/analysis scripts, two test files, saved
development evidence and a prospective design draft. Existing src implementations
were not changed by this PR. Source/input/output verification and reconstruction
of saved predictions were documented during the preceding reviews.

This is a completed exploratory checkpoint. It supports further controlled
investigation of baseline regularization and neural optimizer budgets, with
separate scalar-diffusivity, initial-field and temperature-field outcomes. The
broader study remains a draft: fresh evaluation data, generic two-family scoring,
both classical basis widths, complete failure accounting, and its prospective
analysis implementation have not been run as a final study.

## Follow-up finding and repair

The retrospective reviewer joined recorded manifest names directly to native
paths. Archived Windows names such as `scripts\pilot_bounded_pinn.py` contain
backslashes; on Linux those are ordinary filename characters. Thus the archived
Windows manifest could not be verified there using that implementation, despite
the original Windows run being reproducible.

Prepared a follow-up on local branch `broader-comparison-implementation`:
`evidence_path` parses relative names with either separator style and rejects
absolute/drive paths and parent traversal. Applied it to source, input, output,
protocol, and observation-directory lookups. Eleven focused tests cover POSIX
path semantics, native-file hash identity, and invalid relative paths. They pass;
the complete archived budget review also reproduces in a new workspace folder
without refitting or modifying raw evidence. Original recorded hashes are retained.
Actual Colab execution is still a separate environment check for the next phase.

## Next tasks

1. Bring the new local branch up to the merged origin/main without rewriting
   history, then checkpoint the portability repair.
2. Implement the broader design on this new branch: paired A/B data generation,
   generic exact references, CV with 6/12 modes, both neural budgets, and independent
   data/training seed bookkeeping. Keep the proposed evaluation seeds untouched.
3. Test the workflow using separate smoke seeds, including leakage checks,
   classical basis truncation, model reload, numerical scoring and failures.
4. Verify storage/runtime, finalize the analysis rules, and commit the complete
   code/config/protocol before generating evaluation data.
5. Run the declared study once, retain all outcomes, then perform the prespecified
   analysis and prepare the paper draft around the evidence actually obtained.
