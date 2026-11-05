# GitHub repository review — 5 November 2026

> Historical checkpoint review. Its outstanding tasks and branch status describe
> the checkpoint below. For current status use [Research progress](RESEARCH_PROGRESS.md)
> and [Next steps](NEXT_STEPS.md); for the later SIAM/JOSS assessment use
> [the venue review](SIAM_JOSS_AND_COAUTHOR_REVIEW.md).

## Scope and result

Read-only GitHub API checks and a focused code/document review were performed
for `ridoy1211/pinn-audit`, followed by local documentation/CI repairs. No remote
commit, pull request, review comment or merge was created by the assistant.
The findings below describe the remote snapshot at review time; local repairs
are pending the user's Git checkpoint.

The saved primary analysis reproduced from an isolated export of the exact
remote research-branch commit. No numerical discrepancy was found in that
replay. This does not certify every historical result or establish publication
novelty.

## Findings and local repairs

| Finding | Evidence | Resolution/state |
|---|---|---|
| Default branch omits the completed study | Remote main=99586c4; research branch=0977887, five commits ahead and zero behind | Main is still an earlier checkpoint; no remote merge performed |
| No pull request covers the research branch | The API returned only PRs 1 and 2, both merged | A new PR is needed for review/integration after the checkpoint |
| README's current headline/status describes older experiments | Main and research-head README blobs are identical: dcff0223c65708ffd9169b98a82ae42f933ccd64 | Local README now introduces the fixed study, links all outcomes and identifies legacy evidence explicitly |
| New branch head has no hosted CI check | API reports no commit-associated PR runs and zero check runs; workflow push filter was main only | Added research-branch push trigger locally. New hosted results will exist only after push; not claimed as passed |
| Environment page applies an old record too broadly | October document says Torch 2.14.1; main run records 2.15.0+cpu | Historical record preserved, with a per-study manifest note added locally |
| Collection instructions appear current despite completed outputs | Commands describe pre-freeze steps, expected count 224 and collection-time ignore policy | Added completion/replay guidance and explanation of the later 227-test suite |
| Newer writing files are local, not on GitHub | Combined manuscript, abstract/conclusion, venue review, SciSpace assessment and progress map were uncommitted | Must be included in the next checkpoint; the remote section drafts at 0977887 remain intact |
| Local main tracking is stale | Local main=d1de65e while fetched/remote main=99586c4 | Do not switch/pull over unfinished work; current research branch matches the verified remote head |

The archived protocol retains pre-run wording because its exact bytes are part
of the freeze. The current README explains that it is historical protocol text;
it was not edited to look retrospectively current. Likewise, no old result or
raw evidence file was rewritten to resolve a presentation discrepancy.

## Independent replay and software checks

Source: isolated `git archive` export of
`09778874da8c14f4103d6b41dc31aff4b154aba8`, the exact GitHub research head.
This is an isolated committed tree, not a fresh network clone or an independent
laboratory replication. The environment was Python 3.12.14, NumPy 2.5.4,
SciPy 1.18.1 and Torch 2.15.0+cpu; collection used Python 3.14.6.

- **227 tests passed in 52.38 seconds** in the exported tree. Small software
  training tests are part of this suite; the 240 evaluation models were not retrained.
- The analysis verified all **15 frozen source/config/protocol hashes** and
  **2365 run-output hashes** before summarization.
- All four derived CSVs and `summary.md` match the archived analysis byte for byte.
- `analysis.json` matches after excluding only its new generation timestamp.
- The review retains the main-study counts: 400 completed jobs, 40 cases from
  20 independent designs, and ten designs per observation-count/family stratum.

Machine-readable comparisons and hashes:
[`audit/github_repository_check_2026-11-05.json`](../audit/github_repository_check_2026-11-05.json).
The compact remote state is saved in
[`audit/github_remote_snapshot_2026-11-05.json`](../audit/github_remote_snapshot_2026-11-05.json).

Local repair checks also passed: Ruff lint and formatting for 74 files, mypy
for 23 source files, README relative-link existence, workflow branch-trigger
parsing, and a second check that all 15 frozen source hashes are unchanged.
These checks do not substitute for the new remote-head CI that will run on push.

The manifest contains absolute DLL paths as descriptive threadpool metadata.
These are not artifact lookup paths; replay does not require those original
locations. The successful replay uses portable artifact references. Do not
normalize or alter the original manifest to remove historical environment facts.

## GitHub sources

- [Repository/default branch](https://github.com/ridoy1211/pinn-audit)
- [Research branch at reviewed head](https://github.com/ridoy1211/pinn-audit/tree/09778874da8c14f4103d6b41dc31aff4b154aba8)
- [Compare main and research branch](https://github.com/ridoy1211/pinn-audit/compare/main...broader-comparison-implementation)
- [Pull requests](https://github.com/ridoy1211/pinn-audit/pulls)
- [Workflow at the reviewed head](https://github.com/ridoy1211/pinn-audit/blob/09778874da8c14f4103d6b41dc31aff4b154aba8/.github/workflows/ci.yml)

## Next checkpoint

Commit the local README/CI/document repairs, audit records and pending manuscript
work, then push the same research branch. This should trigger the newly configured
CI. Review those exact-head checks and open a PR to main; no merge or deletion of
historical branches is necessary to preserve the evidence. Author metadata and
final manuscript/declaration review remain Day 6 work. The isolated analysis
verification completes one of Day 7's checks, not all submission preparation.
