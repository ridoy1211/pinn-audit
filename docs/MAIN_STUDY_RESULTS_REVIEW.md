# Completed main-study review — 4 November 2026

## Completion, independence and chronology

The declared evaluation completed all 400 pipeline jobs: 160 selected classical
pipeline outputs and 240 neural fits. These cover 40 dataset cases from 20
independent observation/noise designs. Each N/IC stratum contains ten designs;
family A/B share design randomness, and each neural case has three initializations.
No job failed, timed out, or was omitted. Do not count 240 neural starts as 240
independent datasets.

The implementation checkpoint is b9d2b3e24446827127b67ce41f1c1c12c1f7dfdf.
The freeze was created at 2026-11-04 05:46:00 UTC and committed as
dad8cd5bb005f0c6c619620b78969994d8024d80 at 05:46:12 UTC. Collection began at
05:46:59 UTC; first data generation was recorded at 05:47:17 UTC. Collection
finished at 08:26:44 UTC (about 2 hours 40 minutes). Source hashes and Git ancestry
match this prospective ordering. This is a versioned prospective protocol, not
formal preregistration or proof of publication novelty.

All 15 frozen source/config/protocol hashes, 2365 collected-output hashes, and six
analysis-output hashes match. The raw run is approximately 121.6 MiB; its largest
individual file is approximately 1.15 MiB. Version-control rules preserve original
evidence bytes. Keep both budgets and every condition together.

## What was compared

Family A is sin(pi*x)+0.5*sin(3*pi*x). Family B adds 0.25*sin(7*pi*x).
Both use alpha=0.05 on the unit space-time square, zero Dirichlet boundaries,
uniform observations and absolute Gaussian noise SD=0.01. This noise convention
differs from the earlier sampled-signal-relative development pilot.

The six pipelines are unregularized six-mode fitting, fixed-ridge six-mode fitting,
observation-CV six- and twelve-mode fitting, and the same hard-boundary PINN with
200 or 1000 L-BFGS iterations after 2000 Adam steps. Neural settings are fixed
reference settings; classical regularization is chosen by observation-based CV.
This is neither equally tuned method-class optimization nor equal computation.

## Main descriptive results

All values below are median relative errors in percent. A neural dataset score
first takes the median over its three starts; the displayed median then spans
ten dataset scores. The complete six-method table is in the original analysis.

| N | family | pipeline | diffusivity | temperature field | initial field |
|---|---|---|---|---|---|
| 10 | A | CV12 | 4.7545 | 3.9421 | 10.900 |
| 10 | A | PINN-200 | 3.9193 | 3.7173 | 8.8028 |
| 10 | A | PINN-1000 | 2.5796 | 3.8419 | 8.9633 |
| 10 | B | CV12 | 11.752 | 7.4025 | 26.111 |
| 10 | B | PINN-200 | 9.2047 | 6.0398 | 23.810 |
| 10 | B | PINN-1000 | 3.4001 | 6.0706 | 23.368 |
| 50 | A | CV12 | 1.6066 | 0.8893 | 2.9189 |
| 50 | A | PINN-200 | 1.4022 | 0.7596 | 1.5732 |
| 50 | A | PINN-1000 | 1.4940 | 0.9086 | 2.3665 |
| 50 | B | CV12 | 1.9024 | 2.1911 | 10.882 |
| 50 | B | PINN-200 | 5.3564 | 4.9326 | 22.352 |
| 50 | B | PINN-1000 | 2.3343 | 3.7025 | 16.482 |

These marginal medians are not the primary paired effect estimates. For example,
subtracting 3.4001-11.752 does not produce the paired median -4.445 below. The
prespecified analysis subtracts CV12 from the neural score within each design,
then summarizes the ten differences.

## Prespecified paired results

Differences are PINN minus CV12 in percentage points. Negative favors PINN.
Intervals are pointwise 95% percentile bootstrap intervals for the median paired
difference, with design-level resampling and no multiplicity adjustment.

| condition | budget | outcome | median paired difference | interval | PINN wins / 10 |
|---|---|---|---|---|---|
| N=10, B | 1000 | diffusivity | -4.445 | [-22.17, -1.109] | 9 |
| N=10, B | 1000 | field | -1.117 | [-3.099, -0.136] | 8 |
| N=10, B | 1000 | initial field | -1.508 | [-5.148, -0.467] | 9 |
| N=50, B | 200 | diffusivity | +2.031 | [1.209, 4.901] | 1 |
| N=50, B | 200 | field | +2.862 | [2.021, 3.562] | 0 |
| N=50, B | 200 | initial field | +11.866 | [7.869, 15.37] | 0 |
| N=50, B | 1000 | diffusivity | +0.410 | [-0.386, 1.616] | 4 |
| N=50, B | 1000 | field | +0.803 | [0.032, 2.142] | 2 |
| N=50, B | 1000 | initial field | +2.577 | [-0.303, 11.08] | 3 |

The clearest favorable contrast for the longer PINN is sparse family B. The
denser family-B field comparison favors CV12 for both neural budgets, especially
the short one. Family A's diffusivity contrasts all have intervals spanning zero;
avoid declaring a diffusivity winner from its marginal medians. N=50/A has one
favorable short-budget IC contrast (-0.694 pp, interval [-1.543,-0.216], 9/10
wins), while its other outcomes are less decisive. All 24 contrasts are retained
in the original table and figure; selected examples here do not replace it.

Ten designs per stratum and many pointwise contrasts limit precision. Intervals
spanning zero do not establish equality, and intervals excluding zero are not
familywise evidence that a method class is generally superior. No new p-values,
budget selection, or best-initialization selection were introduced after results.

## Representation and tails

Six-mode fields cannot represent family B's seventh sine component. Orthogonality
gives the IC-error floor sqrt(0.25^2/(1^2+0.5^2+0.25^2))=sqrt(1/21)=21.82%.
All sixty family-B six-mode outputs respect that floor. This is a standard
projection calculation, not a new theorem or a proof that the seventh mode is
identifiable from these observations.

For N=50/B, CV6's median IC error is 24.98%, versus 10.88% for CV12 and 16.48%
for the longer PINN. The richer classical representation therefore changes the
context of an apparent neural advantage over the restricted comparator. In the
sparse setting, the richer basis does not ensure stronger recovery: regularization
and limited information still matter. These results motivate a bias/variance and
information interpretation, but do not uniquely establish that mechanism.

Completion is not accuracy. Unregularized sparse classical fits reach 645.5%
field error and 2622.4% IC error, despite finite outputs. Short-budget neural
initializations reach 107.8% diffusivity error. Keep tail outcomes visible in
plots/tables rather than reporting medians alone. No observed fit failed and no
bound hits occurred; those facts do not establish an absence of population risk.

## Numerical and statistical verification

- A separate read-only statistical audit reproduced every dataset score, paired
  difference and 10000-replicate bootstrap endpoint from the raw records exactly.
- The reproducible post-evaluation report checked all 2000 held-out fold
  predictions and selected CV penalties, reloaded all 240 neural checkpoints,
  checked their saved observation predictions, and reconstructed all 24 saved
  dense neural fields. No models were retrained.
- All 160 reported classical model states were independently checked with exact
  spatial sine orthogonality and analytic time integration. IC errors agree with
  the grid calculation to floating-point precision. Maximum field-score
  discrepancy is 0.1632 pp for an unregularized fit with approximately 645.5%
  field error; the CV12 maximum is only 0.00321 pp. This supplemental audit does
  not replace the prespecified 401-point primary scores.
- The planned first-design 401/801-point check covers forty job outputs. Its
  largest field difference is 0.001868 pp; largest IC difference is approximately
  7.04e-8 pp. These previously saved checks are now explicitly reported, closing
  a presentation gap in the original automatic summary.
- Every neural arm reaches its configured 200/1000 iteration cap. These are
  fixed-budget procedures; convergence or a global optimum is not certified.

Collection lasted about 2.66 hours. Median CV12 selection-plus-selected-fit costs
are approximately 7-11 seconds per design, excluding nonselected full-fit
diagnostics and evaluation. Median single-start neural training is approximately
24-41 seconds; three-start totals are approximately 74-124 seconds per design.
The detailed timing table retains both cost definitions. Do not compare one
neural start with a three-start performance summary without stating the distinction.

## Reporting decision and next work

The declared study is finished. Preserve the freeze, full data, every outcome,
and both budgets before writing. Further tuning of these evaluation cases would
make them development data for that new question. No extra experiment is needed
to report the prespecified comparison accurately.

The supported message is conditional: apparent method rankings depend on the
reference regularization/representation, observation regime, chosen outcome and
optimization budget. It is not a new PINN architecture, universal ranking, or
novelty certification. One alpha, one geometry, one noise level, two IC families,
fixed neural hyperparameters, a favorable spectral family, and small strata
limit scope. Journal fit and the precise contribution still need critical
positioning against the verified related work.

Figures, model checks and timings are in `results/broader_results_review_v1`.
The next deliverable is an evidence-led Methods/Results draft and a claims map,
using the full result table and its limitations. The study's collection and
analysis sources remain frozen and unchanged.
