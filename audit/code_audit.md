# Source-based scientific audit: `pinn-audit` at `c0c6413e174102878dfa168603593bcbde4c504c`

## Scope and evidence boundary

I read the pinned source, tests, configs, README, `docs/PROTOCOL.md`, `docs/LOG.md`, and the repository history. The audit findings below refer to the requested commit; no experiments or tests were run. After the audit, a separate working-tree patch added explicit start-time and source/config provenance fields, so those changes are not part of the findings about the pinned commit. `git ls-files results` contains only `results/.gitkeep`, so the Parquet run rows and saved solution arrays are not available here. Accordingly, the published numerical values, timing exclusions, and claimed chronology cannot be independently reproduced or checked against row-level records in this checkout. The code findings are source-verifiable; data-dependent conclusions are labeled separately.

## Findings

### 1. The chronology check compares prediction commits with run completion, not run start

**Classification: source bug in the audit trail; high priority for a pre-registration claim.**

`build_record()` writes `timestamp_utc` when the result row is built (`src/pinnaudit/io.py:23-49`). In the forward sweep, this happens after `run_once()` returns (`src/pinnaudit/sweep.py:155-173`); inverse rows are timestamped after `_fit()` returns (`src/pinnaudit/inverse.py:159-175`). The chronology script then selects the earliest of those timestamps and calls the comparison “before the first run” (`scripts/check_chronology.py:102-131`). A prediction commit made during a long training job, after observing partial output but before the row was saved, can therefore be labeled verified. The result timestamp is a completion timestamp, not a start timestamp.

The checker also selects the earliest commit with a matching subject (`scripts/check_chronology.py:85-99, 123-130`), rather than proving that the prediction text itself first appeared in that commit. This is a weaker audit trail than the README/Protocol language implies; it does not demonstrate that the tracked prediction content was frozen before execution.

**Minimal correction:** record an explicit UTC start timestamp immediately before each computation, keep the existing timestamp as completion, and compare the prediction commit only with start. If any selected legacy row lacks a start field, report `NOT VERIFIED` rather than falling back to completion or inferring a start from elapsed time. Bind each prediction to an explicit commit hash or file-content history rather than just a subject string.

### 2. Resume identity and aggregation can silently mix incompatible runs

**Classification: resumability/summary bug; high priority for long-running sweeps.**

The forward resume set contains only the semantic config fingerprint (`src/pinnaudit/sweep.py:121-124, 146-151`). That fingerprint hashes the config but excludes the source-code revision (`src/pinnaudit/config.py:138-147`). Consequently, after code changes, an unchanged config may be treated as already complete and old results silently reused. Conversely, if a setting changes under the same sweep name and variant label, the new fingerprint causes reruns, but the old rows remain appended (`src/pinnaudit/sweep.py:167-173`; `src/pinnaudit/io.py:53-64`). `summarise()` then groups only by `variant` and counts unique seeds (`src/pinnaudit/summary.py:8-17`), so old and new configurations are pooled and duplicate seed rows are hidden in `n_seeds`. The inverse path similarly appends config-sensitive jobs to one file and summarizes only by observation count, noise, and method (`src/pinnaudit/inverse.py:81-103, 171-189`; `src/pinnaudit/summary.py:35-46`).

**Scientific implication:** the advertised resume behavior is safe only if code and all settings remain unchanged; reusing a results directory after edits can either reuse stale results or produce a misleading mixed-design summary. This is a capability/risk of the implementation, not evidence that the pinned README results were mixed.

**Minimal correction:** include the code revision in each run identity, persist a resolved config snapshot for inverse runs, and make summaries reject multiple fingerprints or duplicate `(variant, seed)` / `(method, n_obs, noise, seed)` rows unless the user explicitly filters one immutable experiment revision.

### 3. The sparse unknown-initial-condition inverse comparison is ill-conditioned and not complexity-matched

**Classification: design limitation, openly acknowledged in the docs; high priority before making a PINN advantage claim.**

The unknown-IC least-squares model estimates diffusivity plus six sine amplitudes (`configs/inverse/heat_unknown_ic.yaml:1-10`; `src/pinnaudit/identify.py:73-83, 92-105`). At 10 observations this is **10 observations for 7 fitted parameters**, so the system is overdetermined in count; it should not be called mathematically underdetermined. However, high heat modes decay rapidly, so their amplitudes can have little sensitivity at later observation times and can be strongly correlated with diffusivity for a sparse random design. The code has no regularization or parameter bounds and starts all six amplitudes at zero (`src/pinnaudit/identify.py:92-105`).

The PINN uses the same observations but has a 4-layer, width-32 unconstrained network (about 3,297 weights, plus log-diffusivity) and drops the initial-condition loss entirely in this setting (`src/pinnaudit/inverse_pinn.py:76-90, 150-155`; `src/pinnaudit/solvers/neural.py:42-66`; config at `configs/inverse/heat_unknown_ic.yaml:21-35`). The two methods therefore have different function classes and implicit regularization. The docs correctly disclose the unregularized seven-parameter LSQ and call the sparse-data result suggestive; the source does not quantify either estimator’s identifiability or sensitivity.

**Scientific implication:** the PINN’s lower median in the 10-observation cells does not isolate a benefit of physics-informed estimation. It may reflect instability of the six-mode LSQ parameterization, PINN implicit regularization, random design, or optimizer variance. The stated ranges from five seeds do not resolve those alternatives.

**Minimal correction:** for each observation design, inspect singular values/condition numbers of the residual Jacobian or profile likelihood for alpha; report convergence and sensitivity across seeds/designs. Compare against regularized or lower-dimensional least squares chosen using a disclosed validation protocol, and treat the result as exploratory until repeated on independent observation designs. Preserve the nuance that 10 > 7 in raw count while effective sensitivity can still be poor.

### 4. Classical inverse optimizer failures are retained as estimates without an analysis gate

**Classification: robustness/reporting gap.**

`fit_lsq()` records `fit.success` but returns `fit.x[0]` regardless of that flag (`src/pinnaudit/identify.py:103-115`). The inverse summary uses `rel_err` and time but ignores convergence metadata (`src/pinnaudit/summary.py:35-46`). Thus a non-converged fit would silently contribute to the reported median. I cannot tell whether any pinned result row had this status because raw rows are absent.

**Minimal correction:** fail or explicitly classify unsuccessful/non-finite fits, report convergence counts and termination messages, and include a model-grid refinement check for inverse estimates. The fit uses a 161×161 Crank–Nicolson forward solve with linear interpolation (`src/pinnaudit/identify.py:65-77, 88-100`), but the inverse configs/tests do not establish that alpha estimates are insensitive to this surrogate resolution.

### 5. Accuracy is scored on each method’s own output grid

**Classification: acknowledged evaluation limitation; moderate priority.**

`run_once()` evaluates the exact solution on the returned solver grid and computes the relative L2 there (`src/pinnaudit/experiment.py:37-49`). Neural grids are fixed by config while classical grids change with resolution (`src/pinnaudit/config.py:90-95`; `configs/sweeps/forward_heat.yaml:10-35`, `configs/sweeps/forward_wave.yaml:6-30`). The protocol notes that scoring grids differ (`docs/PROTOCOL.md:73-77`). For smooth fields and uniform grids, the normalized unweighted sample norm is a reasonable approximation, and this likely does not overturn the very large forward gaps. It still means the compared error is a grid-dependent discrete norm, rather than one common quadrature approximation to the continuous space-time error.

**Minimal correction:** interpolate each numerical solution to a shared dense evaluation grid and use explicit space-time quadrature weights; retain the original grid for solver cost and resolution metadata.

### 6. Timing comparisons are order-of-magnitude evidence, not a stable microbenchmark

**Classification: design limitation, partly disclosed.**

Forward neural timing starts after collocation/IC/BC/observation generation, model construction, and loss closure construction (`src/pinnaudit/solvers/neural.py:163-183`); the heat FD timer starts after grid and matrix construction (`src/pinnaudit/solvers/fd_heat.py:31-59`). The inverse PINN timer starts after its data tensors, network, and loss are built (`src/pinnaudit/inverse_pinn.py:56-96`), while LSQ starts before constructing the solver and begins its optimization (`src/pinnaudit/identify.py:88-104`). These boundaries are not identical. The FD values are sub-millisecond, while the docs report up to roughly 2× variation across sessions; confirmation times are medians across seeds rather than repeated controlled timing trials (`docs/PROTOCOL.md:78-79`; `scripts/confirm_report.py:129-135`).

**Scientific implication:** the reported several-orders forward and roughly thousand-fold inverse gaps are useful only as scale comparisons. Setup asymmetry could shift small ratios, and single-session CPU timings should not be presented as portable performance constants.

**Minimal correction:** record setup, optimization, and inference separately; define and apply one end-to-end timing boundary to both methods; repeat short classical solves enough times for a stable median and report hardware/software context. The current docs appropriately tell readers to interpret ratios as orders of magnitude.

### 7. Numerical claims are not auditable from the pinned repository snapshot

**Classification: unverifiable evidence limitation, not proof of incorrect results.**

The tracked `results/` content is only `.gitkeep`; Parquet measurements and saved solutions are absent. The README/LOG contain pasted tables and figures, while the protocol says figures and tables should come from saved results (`docs/PROTOCOL.md:90-97`). Environment details are recorded in a separate Markdown file, but individual forward result rows store Python and NumPy versions only (`src/pinnaudit/io.py:29-49`); they do not store the source commit. The chronology script itself notes that results must exist on the machine that generated them (`scripts/check_chronology.py:4-6`).

**Scientific implication:** this checkout supports a source-level methodology review but cannot independently verify medians, outlier handling, p-values, timing exclusions, or preregistration chronology. The docs are candid that result files are unversioned; a paper should archive machine-readable run rows and config/code provenance in a DOI-backed artifact or supplement.

## Statistical interpretation

The confirmation report pairs tuned and plain PINN errors by the same held-out seeds and applies a two-sided Wilcoxon signed-rank test (`scripts/confirm_report.py:99-110`). Ten all-direction pairs produce the reported exact p≈0.002; that supports the narrow paired comparison for the selected variants, not a general PINN superiority claim. The two confirmatory comparisons are few, but adjusted p-values should still be reported if the paper frames both as simultaneous tests. Percentile bootstrap intervals for a median are implemented in `src/pinnaudit/stats.py:18-41`; with n=10 they are coarse and should accompany (not replace) raw seed-level results. The inverse comparison has five seeds and reports medians/ranges without a paired uncertainty analysis; the LOG explicitly treats the sparse-unknown-IC result as suggestive.

## Overall assessment

The repository does several things well: it fixes a small exact-solution scope, verifies second-order classical schemes, uses identical inverse observations for both methods, separates development from held-out forward seeds, discloses post-hoc tuning, and limits conclusions to the tested setting. The most important fix before paper use is to make chronology genuinely start-time based, make resume/summary identity immutable and provenance-aware, and strengthen the inverse sparse-data comparison with regularized baselines and identifiability diagnostics. The existing evidence supports a careful benchmark note, not a general claim that PINNs beat classical solvers.
