# Discussion draft

Working manuscript section, 5 November 2026. Numerical statements refer to the
fixed evaluation archived at `4a64ed4`; this discussion introduces no new fits,
selection rule or hypothesis test.

## Interpretation of the comparison

The completed study does not yield a single method ranking across observation
counts, initial fields and outcomes. The longer-budget PINN's clearest favorable
co-primary pattern against CV12 occurs for the sparse family-B case: diffusivity
and initial-field errors are lower in nine of ten designs. In the denser
family-B case, CV12 has lower full-field error than the longer-budget PINN in
eight of ten designs. Several other contrasts remain uncertain. These results
support a conditional comparison of the declared procedures.

This interpretation is narrower than a claim that neural inversion succeeds
where numerical inversion cannot. Classical joint recovery already has
precedents under different measurement conditions.
[Zhao et al. (2017)](https://doi.org/10.1002/mma.4245)
Existing noisy inverse PINN–numerical comparisons also make the broader
benchmarking question established territory.
[Zapf et al. (2022)](https://doi.org/10.1038/s41598-022-19157-w),
[Jekic et al., v2 (2026)](https://arxiv.org/html/2509.20191v2)
Our contribution, if judged sufficient, is the documented sensitivity of the
comparison to explicit choices within this small joint heat-inversion problem.

## Representation and regularization

The family-B representation control is essential to interpreting any apparent
neural advantage over six-mode inversion. That reference cannot express the
seventh sine component, and its initial-field relative error has a 21.82% floor
even with ideal coefficients for the retained modes. CV12 removes this particular
restriction. Its stronger denser-family-B recovery changes the impression given
by CV6 alone. An inadequately expressive comparator can therefore support a
different conclusion from a richer one within this benchmark.

The richer representation does not guarantee stronger sparse-data recovery.
The contrast between N=10 and N=50 is consistent with a tradeoff between allowed
complexity and the information available to constrain it, but our experiment
does not uniquely identify that mechanism. Both representation width and the
regularization selected from noisy observations affect the resulting fit.
Cross-validation targets held-out observation prediction, whereas our physical
outcomes include an unobserved initial state and a scalar coefficient.

Heat smoothing provides relevant physical context. In our analytical solution,
mode j is attenuated by exp[-alpha*(j*pi)^2*t]; higher-frequency initial detail
therefore decays faster. This explains why accurate later temperatures alone
need not ensure accurate initial-field reconstruction. It does not establish
structural non-identifiability for the chosen data or explain every numerical
failure. No uniqueness theorem or complete information analysis was established
by these comparisons.

## Optimization budget and the recovery target

Longer training did not uniformly improve physical recovery. The family-A
N=50 median initial-field error increased from 1.57% to 2.37% between the two
budgets, while the corresponding family-B median decreased from 22.35% to
16.48%. These are descriptive patterns, not a separately tested budget effect
or proof of overfitting. Both budgets reached their iteration caps.

The neural loss combines observation discrepancy and PDE residual; it is not
the hidden-truth recovery error. A completed optimization run, small residual or
favorable single outcome should consequently not replace direct reporting of
diffusivity, initial-state and full-field scores. Retaining all starts also
reveals large errors that a median alone would conceal. Our three-start median
summarizes an outcome distribution; it is not a deployable ensemble or a model
selected without access to truth.

## Limits of inference and fairness

There are ten independent designs in each stratum. The reported bootstrap
intervals are pointwise and unadjusted for the 24 contrasts. They should not be
read as simultaneous evidence of method-class superiority, and intervals
spanning zero do not establish equivalence. Family A/B cases share randomness;
the study contains twenty independent designs rather than forty independent
replications. Neural initializations further reuse observations and do not
increase the number of independent data designs.

The twelve-mode representation contains the generating truth, and the classical
time dependence satisfies the specified PDE exactly. This is favorable knowledge
relative to an unrestricted neural field. Conversely, classical regularization
was selected by cross-validation while neural architecture and weights were
fixed from development. These differences prevent a claim of matched priors,
equally tuned method classes or equal computation. Timing comparisons also
require distinct accounting for CV selection, selected fitting, additional
diagnostics, evaluation and all three neural starts.

Only one diffusivity, one noise level, one spatial domain and two finite
spectral initial states were investigated. Noise was independent additive
Gaussian error, PDE and boundary assumptions were correct, and the data were
synthetic. Fixed collocation and three initialization seeds do not capture all
neural training variability. Application to real measurements, model mismatch,
other noise distributions or more complex domains remains untested. Numerical
verification supports the recorded scores, not generalization beyond this setup.

## What follows from this study

A defensible report should present the full result matrix and state exactly
which conclusions change when comparator representation, regularization or
neural budget changes. Its value would lie in a transparent, inspectable example
of comparative sensitivity, rather than a new architecture or a broad performance
claim. Whether that example adds enough to the existing literature remains an
editorial and scientific judgment.

Future investigation could examine non-spectral initial fields, multiple noise
levels and diffusivities, model mismatch, or explicitly balanced tuning budgets.
Those questions require a new protocol and fresh evaluation data. Retuning the
current cases would make them development evidence for a new procedure; their
present outcomes should remain unchanged and fully archived.
