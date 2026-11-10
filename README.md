# pinn-audit

A computational study of joint heat inversion. We compare how observation
timing and initial-state representation affect recovery of thermal diffusivity
and an unknown initial temperature from sparse, noisy synthetic observations
of a one-dimensional heat equation.

## Controlled timing and robustness studies

The follow-up evaluations compare cross-validated twelve-mode spectral
inversion (CV12) with a physics-informed neural network (PINN-1000). The PINN
uses one prespecified training start, 2,000 Adam steps and an L-BFGS budget of
1,000 iterations. Each evaluation uses 30 independent design and noise blocks
per observation count. Schedules share spatial locations and noise draws within
each block. The timing study also shares blocks across initial fields A and B.

| Stage | Design | Evidence |
|---|---|---|
| Controlled timing | 360 datasets and 720 fits. Early, mixed and late observations at N=10 and N=50 for initial fields A and B. | [Protocol](docs/TIMING_PROTOCOL.md), [analysis](results/timing_evaluation_analysis_v1/summary.md), [Main timing figure and explanation](figures/publication_timing_v1/REPORT.md) |
| Explanatory diagnostics | Post-evaluation diffusivity profiles, conditioning and 6/12/24-mode sensitivity. An independent heat-equation solution uses the initial field and diffusivity recovered by each of the 360 timing PINNs. | [Diagnostic report](results/timing_fix2_report_v1/REPORT.md), [rollout figure](results/timing_fix2_report_v1/physical_consistency.png) |
| Off-basis evaluation | 120 datasets and 240 fits. One localized initial field outside the twelve-mode span, with mixed and late schedules at both observation counts. | [Protocol](docs/OFFBASIS_PROTOCOL.md), [analysis](results/offbasis_analysis_v1/REPORT.md) |

For N=10 and field B, the prespecified primary contrast was the within-block
PINN-minus-CV12 diffusivity error gap under mixed observations minus the gap
under late observations. The median contrast was 3.7322 percentage points (95%
percentile bootstrap interval [−0.0947, 56.2421]), leaving the interaction
inconclusive. The [publication claims record](docs/PUBLICATION_CLAIMS.md)
describes the interpretation limits and relationship to prior work.

## Replay the follow-up analyses

Download the three ZIP archives and `SHA256SUMS.txt` from the
[public evidence release](https://github.com/ridoy1211/pinn-audit/releases/tag/heat-inversion-evidence-v1.0.0).
The [evidence manifest](docs/evidence_release_v1/release_manifest.json) lists their sizes and SHA-256 digests.
Use the source at tag `heat-inversion-evidence-v1.0.0` for archive verification
and analysis replay.

Follow the [restoration instructions](docs/evidence_release_v1/RELEASE_NOTES.md)
to verify the archives and extract them into `..\heat-evidence-restored`. The
[source provenance addendum](docs/source_provenance_addendum_v1/README.md)
relates the frozen source hashes to the corresponding Git files.

After restoration, run these commands from the checkout root in Windows CMD
with an active Python environment. The verified replay used Python 3.14.6, NumPy 2.5.4 and
Matplotlib 3.11.2. Install the replay dependencies if needed.

```cmd
python -m pip install numpy==2.5.4 matplotlib==3.11.2
```

Use a new output directory for each analysis. These commands regenerate the
reports from saved fits.

```cmd
set "EVIDENCE_RESULTS=..\heat-evidence-restored\results"
python scripts\analyze_timing_evaluation.py --results "%EVIDENCE_RESULTS%\timing_evaluation_v1" --out-dir "..\heat-timing-replay"
python scripts\report_timing_fix2.py --results "%EVIDENCE_RESULTS%" --out-dir "..\heat-diagnostic-replay"
python scripts\analyze_offbasis_evaluation.py --results "%EVIDENCE_RESULTS%\offbasis_evaluation_v1" --out-dir "..\heat-offbasis-replay"
```

The five CSV tables and Markdown report from each analysis matched the archived
copies byte for byte in the verified replay. For example, compare the primary
timing table with the restored copy.

```cmd
fc /b "%EVIDENCE_RESULTS%\timing_evaluation_analysis_v1\timing_interactions.csv" "..\heat-timing-replay\timing_interactions.csv"
```

## Earlier broader comparison

The earlier 400-job study compared four spectral procedures and two PINN
budgets across 40 cases. Its neural summaries use outcome-wise medians over
three starts. Its [protocol](docs/BROADER_COMPARISON_PROTOCOL.md),
[paired results](results/broader_analysis_v1/summary.md) and
[reproduction commands](docs/BROADER_STUDY_COMMANDS.md) cover that study.
The [author-review manuscript](docs/MANUSCRIPT_REVIEW_DRAFT.md) describes this
earlier comparison. The [lab log](docs/LOG.md) and
[history archive](docs/history/README.md) record the preceding pilots.

## Citation, assistance and license

Use [CITATION.cff](CITATION.cff) for the repository's citation metadata.

AI tools assisted with developing and debugging the Python code used for
experiments and statistical analysis. Numerical calculations were performed by
Python scripts. See the draft [Declarations](docs/DECLARATIONS_DRAFT.md) for
the research-assistance disclosure.

Licensed under the [MIT License](LICENSE).
