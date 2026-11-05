# Results draft — completed fixed evaluation

Draft based on archived evidence commit `4a64ed4cae0991bb957c18dd2bcea26162bd4e22`.
Tables are derived from the original analysis CSV files; no models were refitted,
budgets selected or primary scores changed for this draft. Read alongside
`METHODS_DRAFT.md`. Publication novelty and the related-work argument remain
to be assessed separately.

## Completion and numerical verification

All 400 scheduled pipeline jobs completed: 160 reported classical pipeline
outputs and 240 neural fits across 40 dataset cases. Those cases originated
from 20 independent observation/noise designs; family A/B pairs shared a design.
Every observation-count/family stratum contained ten designs and every neural
case retained all three starts. There were no failed, timed-out or interrupted
jobs and no recorded bound hits. All neural fits reached their configured
L-BFGS iteration caps. Completion therefore describes a fixed-budget procedure,
not certified convergence or a global optimum.

The archived source, run-output and analysis-output hashes matched the integrity
records. The planned 401/801-point checks covered forty procedure/start outputs.
Their maximum absolute changes were 0.001868 percentage points for field error
and approximately 7.04e-8 percentage points for initial-field error. The primary
401-point scores were retained.

A supplementary post-evaluation audit recomputed 2000 held-out CV predictions
and their selection decisions, reloaded all 240 neural checkpoints, and verified
the 24 saved dense neural fields without retraining. Analytic sine-orthogonality
and time-integration checks of all 160 reported classical states agreed with
initial-field scores to floating-point precision. The largest field-score
discrepancy was 0.1632 percentage points for an unregularized fit whose field
error was about 645.5%; the CV12 maximum discrepancy was 0.00321 percentage
points. These supplementary checks do not replace the primary analysis.

## Recovery across observation counts and initial fields

Table 1 reports all procedures. Each value is a median percentage error over ten
designs. A neural design score first takes the median of its three initialization
errors. These are marginal medians; paired effects are reported separately.

Table 1. Marginal median relative errors (%).

| N | family | procedure | diffusivity | field | initial field |
|---|---|---|---:|---:|---:|
| 10 | A | unreg6 | 9.259 | 10.242 | 37.274 |
| 10 | A | fixed6 | 4.030 | 3.439 | 8.508 |
| 10 | A | CV6 | 4.030 | 3.439 | 8.508 |
| 10 | A | CV12 | 4.755 | 3.942 | 10.900 |
| 10 | A | PINN-200 | 3.919 | 3.717 | 8.803 |
| 10 | A | PINN-1000 | 2.580 | 3.842 | 8.963 |
| 10 | B | unreg6 | 9.268 | 12.976 | 51.270 |
| 10 | B | fixed6 | 6.343 | 6.187 | 24.206 |
| 10 | B | CV6 | 7.550 | 6.278 | 24.206 |
| 10 | B | CV12 | 11.752 | 7.402 | 26.111 |
| 10 | B | PINN-200 | 9.205 | 6.040 | 23.810 |
| 10 | B | PINN-1000 | 3.400 | 6.071 | 23.368 |
| 50 | A | unreg6 | 1.325 | 0.862 | 2.473 |
| 50 | A | fixed6 | 1.326 | 0.834 | 2.064 |
| 50 | A | CV6 | 1.420 | 0.834 | 2.064 |
| 50 | A | CV12 | 1.607 | 0.889 | 2.919 |
| 50 | A | PINN-200 | 1.402 | 0.760 | 1.573 |
| 50 | A | PINN-1000 | 1.494 | 0.909 | 2.366 |
| 50 | B | unreg6 | 3.625 | 5.899 | 26.250 |
| 50 | B | fixed6 | 3.761 | 5.218 | 23.362 |
| 50 | B | CV6 | 5.526 | 6.233 | 24.982 |
| 50 | B | CV12 | 1.902 | 2.191 | 10.882 |
| 50 | B | PINN-200 | 5.356 | 4.933 | 22.352 |
| 50 | B | PINN-1000 | 2.334 | 3.703 | 16.482 |

Family A contains modes 1 and 3; family B additionally contains mode 7.
Unregularized sparse fitting showed substantial initial-field error, while
regularization reduced that error in these simulations. At N=10/A, median
initial-field error was 37.27% for unreg6, 8.51% for fixed6, 8.51% for CV6,
and 10.90% for CV12. The corresponding PINN-200 and PINN-1000 errors were
8.80% and 8.96%. These descriptive comparisons show why the choice of classical
regularization matters to an apparent neural advantage.

The denser family-B setting illustrates the importance of basis capacity.
CV6 had median initial-field error 24.98%, compared with 10.88% for CV12,
22.35% for PINN-200 and 16.48% for PINN-1000. The richer classical reference
therefore changes the comparison suggested by the restricted six-mode baseline.
Increasing basis capacity was not uniformly beneficial in the sparse case:
at N=10/B, CV12 had median initial-field error 26.11%, compared with 24.21%
for CV6. Representation alone did not ensure accurate recovery from limited
noisy observations.

## Prespecified paired contrasts against CV12

Table 2 contains all 24 planned contrasts. Differences are neural minus CV12
in percentage points, so negative values favor the neural procedure. Each
contrast contains ten complete design pairs. Intervals are pointwise 95%
percentile bootstrap intervals for the median paired difference, not simultaneous
intervals across outcomes or procedures. Wins are strict error comparisons and
are not hypothesis tests.

Table 2. Paired differences against CV12 (percentage points). Wins/ties/losses are for the neural procedure.

| N | family | procedure | outcome | median difference | pointwise 95% interval | wins/ties/losses |
|---|---|---|---|---:|---|---|
| 10 | A | PINN-200 | diffusivity | +0.078 | [-12.210, 2.788] | 5/0/5 |
| 10 | A | PINN-200 | field | +0.072 | [-3.303, 1.391] | 4/0/6 |
| 10 | A | PINN-200 | initial field | +0.420 | [-8.141, 3.834] | 4/0/6 |
| 10 | A | PINN-1000 | diffusivity | -0.135 | [-11.756, 0.895] | 6/0/4 |
| 10 | A | PINN-1000 | field | -0.382 | [-2.983, 0.661] | 6/0/4 |
| 10 | A | PINN-1000 | initial field | -0.766 | [-7.597, 2.538] | 6/0/4 |
| 10 | B | PINN-200 | diffusivity | -1.333 | [-14.067, 3.610] | 6/0/4 |
| 10 | B | PINN-200 | field | -0.527 | [-2.774, 1.585] | 6/0/4 |
| 10 | B | PINN-200 | initial field | -0.743 | [-4.046, 2.201] | 6/0/4 |
| 10 | B | PINN-1000 | diffusivity | -4.445 | [-22.168, -1.109] | 9/0/1 |
| 10 | B | PINN-1000 | field | -1.117 | [-3.099, -0.136] | 8/0/2 |
| 10 | B | PINN-1000 | initial field | -1.508 | [-5.148, -0.467] | 9/0/1 |
| 50 | A | PINN-200 | diffusivity | +0.059 | [-0.597, 1.016] | 4/0/6 |
| 50 | A | PINN-200 | field | -0.058 | [-0.261, 0.019] | 6/0/4 |
| 50 | A | PINN-200 | initial field | -0.694 | [-1.543, -0.216] | 9/0/1 |
| 50 | A | PINN-1000 | diffusivity | +0.042 | [-0.643, 0.878] | 5/0/5 |
| 50 | A | PINN-1000 | field | +0.055 | [-0.049, 0.269] | 3/0/7 |
| 50 | A | PINN-1000 | initial field | -0.024 | [-0.516, 0.884] | 5/0/5 |
| 50 | B | PINN-200 | diffusivity | +2.031 | [1.209, 4.901] | 1/0/9 |
| 50 | B | PINN-200 | field | +2.862 | [2.021, 3.562] | 0/0/10 |
| 50 | B | PINN-200 | initial field | +11.866 | [7.869, 15.369] | 0/0/10 |
| 50 | B | PINN-1000 | diffusivity | +0.410 | [-0.386, 1.616] | 4/0/6 |
| 50 | B | PINN-1000 | field | +0.803 | [0.032, 2.142] | 2/0/8 |
| 50 | B | PINN-1000 | initial field | +2.577 | [-0.303, 11.081] | 3/0/7 |

The strongest favorable co-primary pattern for PINN-1000 occurred at N=10/B.
Its diffusivity error was lower than CV12 in nine of ten designs, with median
paired difference -4.445 percentage points (interval [-22.168,-1.109]).
Initial-field error was lower in nine designs, with median difference -1.508
percentage points (interval [-5.148,-0.467]). The mandatory field outcome also
favored PINN-1000 in eight designs, with median difference -1.117 percentage
points (interval [-3.099,-0.136]). The wide diffusivity interval and ten-design
sample limit the precision of the estimated advantage.

At N=50/B, CV12 had lower field error than PINN-200 in every design and lower
diffusivity error in nine designs. For PINN-200 minus CV12, median differences
were +2.862 percentage points for field error, +2.031 for diffusivity and
+11.866 for initial-field error; all three pointwise intervals were positive.
Against PINN-1000, CV12 had lower field error in eight designs, with median
paired difference +0.803 percentage points (interval [0.032,2.142]). The longer
budget's diffusivity and initial-field intervals spanned zero, so those outcomes
did not resolve a direction as clearly as the field comparison.

For family A, all four diffusivity contrasts had intervals spanning zero. At
N=50/A, PINN-200 had lower initial-field error in nine designs, with median
paired difference -0.694 percentage points (interval [-1.543,-0.216]); other
family-A field/initial-field intervals spanned zero. The study therefore supports
outcome- and regime-specific comparisons rather than an overall method ranking.
Intervals spanning zero do not demonstrate equal performance, and intervals
excluding zero should be interpreted in the context of small strata and multiple
pointwise comparisons.

## Representation floor, optimizer budget and tail outcomes

For family B, orthogonality of the sine modes implies that every field restricted
to modes 1–6 has initial-field relative L2 error at least

\[
\frac{0.25}{\sqrt{1^2+0.5^2+0.25^2}}
=\sqrt{1/21}\simeq21.82\%.
\]

All sixty family-B six-mode outputs respected this floor. It describes a known
function-space restriction, not a new theorem or evidence that the seventh
mode is identifiable from the available observations. The twelve-mode reference
contains the truth and hence benefits from a favorable representation assumption.

More L-BFGS iterations did not uniformly reduce all physical recovery errors.
For example, at N=50/A the marginal median initial-field error increased from
1.57% for PINN-200 to 2.37% for PINN-1000, while field error increased from
0.76% to 0.91%. At N=50/B, the corresponding medians decreased from 22.35%
to 16.48% for the initial field and from 4.93% to 3.70% for the field. These
are descriptive budget patterns, not a newly introduced paired budget test,
proof of overfitting or a basis for choosing a budget retrospectively using truth.

Median summaries did not capture all large errors. Across individual raw fits,
unreg6 reached approximately 645.5% field error and 2622.4% initial-field error;
PINN-200 reached 107.8% diffusivity error. All outcomes were retained. The lack
of numerical failures does not imply uniformly reliable physical reconstruction
or zero failure risk beyond this sample.

## Computational cost

Collection took about 2.66 hours. Table 3 reports median per-design costs for
CV12 selection plus the selected fit, and the summed training costs of all three
neural starts. These costs correspond to different procedures and are not an
equal-compute comparison. Neural performance summaries use three starts; a
single-start time should not be presented as their full computation cost.

Table 3. Median per-design computation costs (seconds), excluding evaluation.

| N | family | CV12 selection + selected fit | PINN-200 three-start training | PINN-1000 three-start training |
|---|---|---:|---:|---:|
| 10 | A | 7.25 | 73.91 | 115.24 |
| 10 | B | 9.56 | 83.70 | 121.64 |
| 50 | A | 11.05 | 88.67 | 124.17 |
| 50 | B | 10.40 | 95.01 | 121.62 |

Single-start neural training medians ranged from approximately 24–41 seconds
across strata and budgets. CV12 costs in Table 3 exclude nonselected full-data
candidate diagnostics and evaluation; separate timing components are retained
in the archive. These machine- and implementation-specific measurements do not
establish general timing superiority across hardware or tuned solver families.

## Scope of the findings

The observed rankings depend on regularization, initial-state representation,
observation count, recovery target and neural optimization budget. The experiment
supports a controlled audit of these specified pipelines. It covers one domain,
one true diffusivity, one noise level and two finite spectral initial fields,
with ten designs per stratum, fixed collocation points and three neural starts.
It does not establish structural identifiability, generalize to experimental
data, compare equally tuned method classes or demonstrate a new PINN algorithm.

The figures `method_errors.pdf` and `paired_contrasts.pdf` in
`results/broader_results_review_v1` display all dataset scores and all planned
contrasts, respectively. Raw initialization-level scores remain available in
the experiment archive. Literature positioning must determine whether this
narrow audit provides a sufficient contribution for a particular journal.
