# Broader comparison: design draft — 4 November 2026

**Status:** proposed design, awaiting implementation verification and a final
versioned freeze before generating evaluation data. This is not preregistration,
completed confirmation, or an established publication-novelty claim. The preceding
development results motivate these choices and must remain labeled development.

## Scientific question

How does the apparent PINN advantage in joint diffusivity/initial-state heat
inversion change when classical IC regularization, IC basis capacity, and neural
optimizer budget are made explicit? Assess diffusivity, initial state, and the
temperature field separately rather than declaring a winner from one score.

This is a controlled comparison of specified pipelines. It is not a new solver,
a comparison of optimally tuned method classes, or an equal-compute study. The
narrow comparative gap remains conditional on the literature assessment in
`audit/literature_review.md`; a simple heat-equation benchmark is not automatically
publishable merely because it is reproducible.

## Proposed simulation matrix

Use the unit domain, zero Dirichlet boundaries, and alpha=0.05 throughout.

| factor | proposed values |
|---|---|
| observation count | 10, 50 |
| IC family A | sin(pi*x) + 0.5*sin(3*pi*x) |
| IC family B | family A + 0.25*sin(7*pi*x) |
| independent observation/noise designs | 10 per observation count |
| measurement noise SD | fixed absolute sigma=0.01 in benchmark temperature units |
| proposed data seeds | 6000-6009 for N=10; 7000-7009 for N=50 |
| neural initialization seeds | 101, 102, 103 |

For each N/design seed, generate one set of uniform space-time coordinates and
one independent standard-normal noise vector; reuse both across IC families,
scaling noise by the same absolute sigma. This produces 40 datasets from 20
independent observation/noise designs. Family pairs share randomness and must
not be counted as independent replications. There are ten designs per stratum.
Use all datasets, with no selection by difficulty or recovery error.

The fixed absolute noise is an intentional change from development's 5% of each
sampled clean-signal standard deviation. It isolates the IC change without also
changing the measurement-noise SD. Do not pool the new study with the earlier
development data or describe the two noise conventions as equivalent.

Family B exposes six-mode classical truncation, while a richer classical basis
can represent it. This is a representation control, not a guarantee that the
seventh mode is identifiable from sparse noisy data. The richer classical family
still contains both truths, so it retains favorable model knowledge relative to
an unrestricted neural field. No IC truth or clean samples may enter a trainer.

## Prespecified pipelines

1. Six-mode spectral, unregularized reference.
2. Six-mode spectral, fixed lambda=1e-6 reference.
3. Six-mode spectral, observation-based five-fold CV regularization.
4. Twelve-mode spectral, the same CV regularization rule.
5. Hard-boundary PINN, 2000 Adam + 200 L-BFGS iterations.
6. The same PINN, 2000 Adam + 1000 L-BFGS iterations.

All methods share observations, alpha bounds [1e-5,1], unit-domain knowledge,
and exact zero boundary information. Neural architecture, loss weights, alpha0,
sampling and optimizer tolerances remain as in the pilot; both budgets are
retained, not selected retrospectively. Use the same fixed PDE training/evaluation
point sets across cases and initializations. Known truth is for simulation and
post-training evaluation only.

Proposed CV candidates are strictly positive: 1e-10, 1e-8, 1e-6, 1e-4, 1e-2.
Positive ridge makes conditional amplitude fitting unique even when a twelve-mode
training subset has fewer observations than amplitudes. Lambda=0 remains a
separate six-mode control. This changes the earlier development CV procedure and
must be documented as a new specified rule, not a retrospective correction of
the old results. No penalty is chosen by true coefficient/field error.

Use identical five balanced folds for every candidate, basis and IC family of
each design; fold seed 30000+data_seed. Select minimum pooled validation MSE,
with exact ties favoring smaller lambda, then refit on all observations. Use a
fixed 1601-point alpha grid plus the existing local refinement for all classical
fits. Increased numerical resolution does not prove global optimality.

Neural hyperparameters are fixed reference settings, not selected with equivalent
CV budgets. State this limitation explicitly. A result may support conclusions
about these pipelines, not a claim that PINNs or classical inversion in general
are superior. Report all candidates and all neural initializations.

## Analysis rules to freeze with implementation

- Coefficient and initial-field relative L2 errors are co-primary reported
  outcomes; here coefficient means the single diffusivity alpha, not Fourier
  amplitudes or neural weights. Its error is abs(alpha_hat-alpha_true)/alpha_true.
  Initial-field error is ||u_hat(x,0)-u_true(x,0)||_L2(x) / ||u_true(x,0)||_L2(x).
  Full-field error uses the same relative L2 definition over space and time and
  is also mandatory. No comparison of individual Fourier amplitudes is proposed.
  Do not collapse tradeoffs
  into an unspecified overall winner. Use the same 401-point trapezoidal scoring
  grid and exact analytical reference for every method.
  Check 801-point scoring on the first declared design in each N/IC stratum,
  retaining both grids' scores and reporting their discrepancies without refitting
  or selecting a preferred grid using truth error.
- For each neural procedure/dataset, summarize performance by the median error
  over the three declared initializations, while retaining all three raw scores.
  This is a typical-initialization performance summary, not an ensemble predictor
  or truth-based selection of a deployed model. Do not report the best seed alone.
- Compare each neural procedure with the richer CV classical baseline within
  each N/IC stratum. Report per-design error differences in percentage points,
  medians, means, maxima, and paired win/tie/loss counts. Other classical
  comparisons are secondary context and must also remain visible.
- Report descriptive 95% percentile bootstrap intervals for paired median
  differences, resampling design seeds (10000 replicates; seed 42000+N), with all
  initialization scores kept together. Strata contain only ten independent
  designs, so intervals may be imprecise. They are pointwise, not simultaneous
  inference across all outcomes. No unplanned p-value threshold or significance
  search. Any pooled across-family calculation must resample whole family pairs.
- Keep all failures/timeouts and never substitute seeds. Bound hits with finite
  outputs are scored, not discarded as optimizer crashes. Report completeness
  before accuracy. Accuracy summaries requiring three neural starts use complete
  paired designs, explicitly labeled conditional on completion, alongside every
  failure count and reason. Do not infer unconditional superiority from filtered
  successful cases. If fewer than ten complete paired designs remain in a stratum,
  report descriptive scores without bootstrap intervals and flag the planned
  inference as incomplete.
- Report fitting time, CV-selection time, and evaluation time separately; include
  all three neural starts when reporting initialization-sweep cost. Use consistent
  CPU thread controls and record actual software/BLAS settings. Do not compare
  classical diagnostic sweep totals with single neural training times.

## Resources, evidence, and freeze

There are 240 neural fits: 40 datasets x 3 initializations x 2 budgets. Using the
observed median short/long times (28.51/47.48 seconds at N=10; 31.52/47.41 at N=50),
the training-only proxy is 20*3*(28.51+47.48)+20*3*(31.52+47.41)=9295.2 seconds,
or about 2.58 hours. Allocate approximately five hours including classical tuning,
evaluation, I/O and a margin for slower cases. This is a planning estimate, not
a runtime guarantee; the declared 600-second per-fit cap remains in force and
actual failures/slow runs are retained. Both pilot budgets reached their iteration
limits, so these are two capped optimization procedures, not converged baselines.
Inspect expected artifact size before deciding
which complete model/field archives belong in Git versus a linked release.

Implement generic analytical-reference scoring for both IC families and CV for
both basis widths. Validate on separate smoke seeds, including no truth in
training/selection, paired noise reuse, basis-truncation behavior, checkpoint
reload, and failure accounting. Final code/config/analysis rules must be committed
before any proposed evaluation-seed data are generated. Record run starts and
source/config hashes; never backdate a prediction or relabel inspected data.

After the freeze, run the declared matrix once and analyze every outcome. If a
bug or resource limit requires changes, preserve the failed run, document an
amendment, and distinguish any later data used for development from untouched
evaluation. This draft itself does not authorize a scientific conclusion.
