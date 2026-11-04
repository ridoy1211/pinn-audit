# Methods draft — joint heat inversion

Draft prepared from archived evidence commit `4a64ed4cae0991bb957c18dd2bcea26162bd4e22`.
This is a manuscript section for author review, not a complete paper. The
Introduction, related-work positioning and publication-novelty assessment remain
separate tasks. All numerical choices below describe the completed fixed study.

## Problem and simulation

We considered joint recovery of a constant diffusivity and an unknown initial
temperature field in the one-dimensional heat equation,

\[
u_t=\alpha u_{xx},\qquad (x,t)\in(0,1)\times(0,1],
\qquad u(0,t)=u(1,t)=0,\qquad u(x,0)=f(x).
\]

The domain and coefficients define a normalized synthetic benchmark; no material
or experimental measurement units were assigned. Both the scalar diffusivity
\(\alpha\) and the initial field \(f\) were unknown to the estimators. Boundary
values and the domain were known. The true diffusivity was 0.05. Two initial
fields were used:

\[
f_A(x)=\sin(\pi x)+0.5\sin(3\pi x),\qquad
f_B(x)=f_A(x)+0.25\sin(7\pi x).
\]

For mode amplitudes \(a_j^*\), the analytical reference was

\[
u^*(x,t)=\sum_j a_j^*\sin(j\pi x)
                 \exp[-0.05(j\pi)^2t].
\]

We sampled \(N=10\) or 50 paired space-time coordinates uniformly on the unit
square. Observations were \(y_i=u^*(x_i,t_i)+0.01z_i\), with independent standard
normal \(z_i\). The noise standard deviation was fixed in benchmark temperature
units rather than scaled by each dataset's signal. There were ten independent
observation/noise designs per observation count. Within each design, coordinates,
realized noise and validation folds were reused for families A and B. Thus the
study contained 40 dataset cases from 20 independent designs, with ten designs
in each observation-count/family stratum. The two family cases within a design
were not independent replications.

Data seeds were 6000–6009 for N=10 and 7000–7009 for N=50. No initial-field
measurements, true mode amplitudes or clean observation values were supplied to
the fitting or regularization-selection procedures. Truth was used to generate
data and to score fitted outputs.

## Spectral reference procedures

We represented the estimated temperature field by

\[
\widehat u_M(x,t)=\sum_{j=1}^M a_j\sin(j\pi x)
                          \exp[-\alpha(j\pi)^2t].
\]

This representation satisfies the boundary conditions and heat equation exactly
for its estimated diffusivity. For each candidate diffusivity we solved

\[
\min_{\boldsymbol a}\;
 \frac{1}{N}\lVert B(\alpha)\boldsymbol a-\boldsymbol y\rVert_2^2
 +\lambda\sum_{j=1}^{M}j^4a_j^2,
\qquad B_{ij}(\alpha)=\sin(j\pi x_i)e^{-\alpha(j\pi)^2t_i}.
\]

Conditional amplitudes were computed by augmented least squares, avoiding
normal-equation formation. Diffusivity was constrained to [1e-5,1]. We evaluated
the profiled objective at 1601 points equally spaced in log diffusivity, refined
every detected interior grid-local minimum with bounded scalar minimization
(log-coordinate tolerance 1e-11), and retained the lowest objective among the
refined, grid and endpoint candidates. This search does not certify a global
minimum. Its objective profile is not a confidence interval.

Four spectral procedures were evaluated: unregularized six-mode fitting
(unreg6); six-mode fitting with fixed lambda=1e-6 (fixed6); and six- or twelve-mode
fitting with observation-based regularization selection (CV6 and CV12).
For CV6/CV12, candidates were {1e-10,1e-8,1e-6,1e-4,1e-2}. We used five balanced
folds, generated with seed 30000 plus the data seed. Every candidate was fitted
using only its training folds. Selection minimized pooled held-out squared error
divided by the total observation count, with exact ties favoring smaller lambda.
The selected candidate was refitted on all observations. Positive CV penalties
ensure unique conditional amplitude estimates, including when the training set
has fewer points than basis functions. They do not ensure joint identification
of diffusivity and amplitudes.

The twelve-mode family can represent both simulated truths, while the six-mode
family excludes the seventh component of family B. This is a deliberate
representation control. The exact spectral representation and finite spectral
truth favor the classical reference; the comparison does not make its prior
information equivalent to that of the neural field.

## Physics-informed neural procedures

We fitted a fully connected neural field

\[
\widehat u_\theta(x,t)=x(1-x)g_\theta(x,t),
\]

where \(g_\theta\) had three hidden layers of 32 tanh units and a linear scalar
output. Inputs were scaled to [-1,1]; linear layers used Xavier-normal weights
with tanh gain and zero biases. The prefactor enforces zero boundary values
exactly. No initial-condition loss was imposed. A learnable scalar \(r\)
parameterized diffusivity by

\[
\alpha(r)=\exp\{\log(10^{-5})+
 [\log(1)-\log(10^{-5})]\operatorname{sigmoid}(r)\},
\]

initialized to alpha=0.2. We minimized

\[
\mathcal L=\frac{10}{N}\sum_i(\widehat u_\theta(x_i,t_i)-y_i)^2
 +\frac{1}{512}\sum_{k=1}^{512}
 (\partial_t\widehat u_\theta-\alpha\partial_{xx}\widehat u_\theta)^2(x_k^c,t_k^c).
\]

Derivatives were computed by automatic differentiation. The 512 uniform
collocation points, drawn with seed 314159, were fixed across all cases and
initializations. An independent set of 2048 points, seed 271828, was retained
for residual evaluation. This residual diagnostic was not a selection rule or
a substitute for reference-based recovery errors.

Both neural procedures used 2000 full-batch Adam steps, with learning rates
0.001 for network weights and 0.01 for the diffusivity parameter. These were
followed by one L-BFGS stage with either a 200-iteration cap (PINN-200) or a
1000-iteration cap (PINN-1000). L-BFGS used learning rate 1, strong-Wolfe line
search, history size 50, gradient tolerance 1e-12, change tolerance 1e-14, and
maximum evaluation settings 250 or 1250, respectively. A 600-second training
limit was declared per fit. Initializations used seeds 101, 102 and 103 for
every case and both budgets. Both budgets were retained without selecting one
using recovery error. Network settings were fixed following development work;
they were not subjected to a tuning search matching classical cross-validation.

## Evaluation and statistical summaries

Diffusivity and initial-field errors were the co-primary reported outcomes.
Full-field error was also mandatory. We calculated

\[
E_\alpha=\frac{|\widehat\alpha-0.05|}{0.05},\qquad
E_f=\frac{\lVert\widehat u(\cdot,0)-f\rVert_{L^2(0,1)}}
                 {\lVert f\rVert_{L^2(0,1)}},\qquad
E_u=\frac{\lVert\widehat u-u^*\rVert_{L^2((0,1)^2)}}
                 {\lVert u^*\rVert_{L^2((0,1)^2)}}.
\]

Errors were evaluated by trapezoidal quadrature using 401 spatial and 401
temporal grid points and reported as percentages. A planned 801-point check
was performed for every procedure/start on the first design in each stratum,
without refitting or replacing the primary scores.

For each dataset and neural procedure, we first took the median error across
the three initializations, separately for each outcome. This describes typical
initialization performance. It is not an ensemble predictor, and its three
outcome medians need not correspond to one fitted network. Stratum summaries
then describe the ten dataset scores. All individual initialization scores were
retained.

The prespecified paired contrasts subtracted CV12 error from the corresponding
neural dataset score within each design. Differences were expressed in percentage
points; negative differences favor the neural procedure. We reported means,
medians, maxima and strict numerical win/tie/loss counts. Pointwise 95% percentile
bootstrap intervals for median differences used 10000 design-level resamples,
with seed 42000+N. Families were analyzed separately. These are descriptive
small-sample intervals, with no adjustment for the 24 contrasts and no unplanned
significance tests. A median paired difference is not the difference between
two marginal medians. Intervals spanning zero do not establish equivalence.

Completion was assessed before accuracy. Failures, timeouts, interruptions and
bound hits were to be retained without replacement. Three-start summaries
required complete neural triads, and intervals required all ten complete pairs
in a stratum. The completed study met these requirements throughout.

## Prospective freeze, computation and reproducibility

The implementation and protocol were committed as b9d2b3e. The freeze record
was created on 4 November 2026 at 05:46:00 UTC and committed as dad8cd5 at
05:46:12 UTC, before collection began at 05:46:59 UTC and before the first data
generation timestamp at 05:47:17 UTC. This was a prospective versioned protocol,
not formal preregistration. Earlier pilots were development work and were not
pooled into evaluation.

The run used CPU float64 computation on Windows, Python 3.14.6, NumPy 2.5.4 and
PyTorch 2.15.0+cpu. Recorded Torch/BLAS thread controls were one thread. Exact
environment details and threadpool metadata are retained in the run manifest.
Timings are measurements of these implementations on this machine, rather than
hardware-independent complexity estimates. Fitting, CV selection and evaluation
costs were recorded separately; three-start neural cost was also summarized.
CV selection-plus-selected-fit costs exclude additional nonselected full-fit
diagnostics and evaluation.

The archived evidence includes observations and folds, candidate fits and
profiles, selected amplitudes, neural checkpoints and traces, complete status
records, residual diagnostics, scoring checks and integrity manifests. Dense
fields were retained for the first design in each stratum; other fields can be
reconstructed from amplitudes or checkpoints. The complete archive is in
commit 4a64ed4cae0991bb957c18dd2bcea26162bd4e22 of
[pinn-audit](https://github.com/ridoy1211/pinn-audit/tree/4a64ed4cae0991bb957c18dd2bcea26162bd4e22).

## Items to resolve before submission

Confirm the exact CPU model and add a concise environment table using recorded
versions, rather than guessing hardware specifications. Add verified citations
for methodological precedents in the eventual related-work section. Preserve
the distinction between the prespecified analysis and supplementary numerical
verification. Neither the study design nor reproducibility alone establishes
the contribution's publication novelty.
