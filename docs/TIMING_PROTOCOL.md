# Controlled observation-timing study

Status: Development protocol. Evaluation has not started.

Starting commit: 75cd304.

Branch: experiment/controlled-timing.

## Research question

At fixed observation count and absolute measurement-noise level,

how do early, mixed, and late observation schedules affect joint

recovery of thermal diffusivity and unknown initial temperature?

Does the relative accuracy of cross-validated regularized spectral

inversion and a specified PINN depend on the observation schedule?

This follow-up is motivated by the original exploratory results.

It is not a retrospective preregistration of those results.

## Physical problem

Retain the original one-dimensional heat equation on x,t in \[0,1],

homogeneous Dirichlet boundary conditions, and true diffusivity 0.05.

Initial fields:

A: sin(pi\*x) + 0.5\*sin(3\*pi\*x).

B: sin(pi\*x) + 0.5\*sin(3\*pi\*x) + 0.25\*sin(7\*pi\*x).

Observation counts: 10 and 50.

Additive Gaussian noise: absolute standard deviation 0.01.

Both diffusivity and initial temperature remain unknown to fitting.

Truth is available only for data generation and evaluation.

## Observation schedules

Early: all observations in \[0.01, 0.10].

Mixed: exactly half in \[0.01, 0.10], half in \[0.20, 0.80].

Late: all observations in \[0.20, 0.80].

These are prescribed schedules, not claimed optimal schedules.

They differ in both delay and temporal spread. Comparisons concern

the complete schedule, not the isolated effect of observation delay.

## Paired generation

For each observation count and independent replicate, generate:

\- One vector of uniform spatial coordinates.

\- One vector q of uniform draws in \[0,1].

\- One vector of independent standard-normal noise.

\- A random assignment selecting exactly half the observation indices.

\- One set of five-fold cross-validation labels.

Map times as:

early\_time = 0.01 + 0.09\*q

late\_time = 0.20 + 0.60\*q

The mixed schedule uses early\_time for the selected half and

late\_time for the remaining observations.

Reuse spatial coordinates, q, standardized noise, the mixed-group

assignment, and fold labels across schedules and initial fields.

Each observation equals the exact temperature at its coordinates

plus 0.01 times its standardized noise.

Both methods receive identical observations for each dataset.

PDE collocation points remain separate from observation points.

## Methods

Primary classical method:

Existing twelve-mode spectral inversion with five-fold

cross-validation for the regularization strength.

Primary neural method:

Existing bounded, hard-boundary PINN using the original Adam

budget and the PINN-1000 L-BFGS budget.

Use training seed 101 for the primary neural result.

Do not select a model using true diffusivity or true field error.

Retain existing architecture, loss weights, parameter bounds,

collocation settings, stopping rules, and scoring procedures.

Any development-stage changes must be documented and frozen

before evaluation.

## Development and evaluation separation

Development pilot:

Three independent blocks per observation count.

Both initial fields and all three schedules: 36 datasets.

Planning target for evaluation:

Thirty fresh independent blocks per observation count.

Both initial fields and all three schedules: 360 datasets.

Thirty is a practical planning target, not a power guarantee.

Finalize the count after checking pilot runtime and precision,

before generating evaluation data.

Development and evaluation seeds must be disjoint from each

other and from the original study's data seeds.

The exact seed lists and final configuration will be recorded

before evaluation. Development results will not be pooled with

evaluation results.

## Outcomes and analysis

Primary outcome:

Diffusivity relative absolute error, in percent:

100 \* abs(alpha\_hat - alpha\_true) / alpha\_true.

Primary follow-up stratum:

N=10, initial field B, chosen because the original exploratory

comparison motivated this follow-up.

Within each replicate and schedule, calculate:

D = PINN diffusivity error - classical diffusivity error.

Primary timing contrast:

I = D\_mixed - D\_late, calculated within each replicate.

Report the median I and a 95% paired bootstrap interval using

10,000 resamples of complete independent blocks.

Negative I means the PINN's error relative to the classical

method is smaller under mixed observations than late observations.

Report absolute method errors and within-schedule differences.

Other strata and early-schedule contrasts are secondary analyses;

their pointwise intervals are not simultaneous evidence.

Secondary outcomes:

Initial-field error, full-field error, runtime, bound hits,

training-cap events, and numerical failures.

Do not treat shared initial-field cases or neural starts as

independent design replicates.

## Failure handling

Keep every scheduled job in the run manifest.

Record exceptions, nonfinite outputs, and training-cap events.

Score finite outputs from capped runs and flag them.

Do not silently omit failed fits or replace their seeds.

If a method produces no finite estimate, report that failure

explicitly. Error contrasts requiring that estimate are unavailable;

report their denominator and label them conditional on successful

paired fits. Report failure rates alongside accuracy.

## Interpretation limits

Late observations have lower signal-to-noise ratio under fixed

absolute noise because diffusion attenuates temperature signals.

With ten observations and twelve unrestricted initial-state

coefficients, observation timing alone need not restore data-only

identifiability. Recovery depends on modelling and regularization.

This experiment tests specified procedures on this heat problem.

It does not establish universal PINN superiority or inferiority.

A small or absent timing interaction is a valid result.

Do not redesign evaluation to obtain a preferred method ranking.
