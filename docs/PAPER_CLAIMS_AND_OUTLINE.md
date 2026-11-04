# Paper claims and outline after the fixed evaluation

Working title: **Regularization, Initial-State Representation, and Optimization
Budget in a Controlled Audit of Physics-Informed Heat Inversion**.

This is a computational benchmark/audit using exact synthetic data, not an
experimental measurement paper or a new neural architecture. The manuscript
must describe the specified pipelines and simulation conditions precisely. A
completed reproducible benchmark is not by itself a publication-novelty guarantee.

## Evidence-to-claim map

| candidate statement | evidence | wording limit |
|---|---|---|
| Regularization changes the sparse classical reference substantially | all-method distributions, unreg6 versus fixed6/CV6 | specify this benchmark; do not claim the ingredients are new |
| A restricted IC basis can change apparent method rankings | family-B six-mode IC floor; CV6/CV12 context | richer family contains the truth and is favorable to classical fitting |
| The longer PINN has a favorable sparse-family-B primary contrast against CV12 | paired diffusivity and IC intervals below zero, 9/10 wins each | ten designs; pointwise intervals; conditional pipeline claim |
| CV12 is stronger for the denser-family-B field outcome | short PINN loses 10/10, long loses 8/10; field intervals positive | do not generalize to all PINNs, PDEs or noise levels |
| More optimizer iterations do not uniformly improve all physical targets | retained short/long outcomes, especially family A | keep both budgets; do not select a winning budget per case using truth |
| Field, initial state and scalar diffusivity require separate scoring | discordant errors and paired contrasts | small PDE residual alone does not establish inverse correctness |
| No failures occurred in the declared schedule | all 400 statuses and complete ledger | not a claim that failure probability is zero |

Do not claim first PINN inverse heat solution, first unknown-IC recovery, first
regularized solver or variable projection. The existing five-question literature
review documents precedents. The possible contribution is the narrow joint audit
of how these choices affect the conclusions under a prospectively fixed design.

## Manuscript structure

1. **Introduction.** Joint recovery of diffusivity and an unobserved initial field
   couples physical parameter identification with state reconstruction. Explain
   why a comparator's function space, explicit regularization and optimizer budget
   matter to a reader assessing a PINN advantage. State the narrow question.
2. **Related work.** Distinguish forward comparisons (Grossmann), sparse inverse
   coefficient studies with different IC/BC information (Zapf), classical joint
   identification (Zhao), noisy inverse comparisons (Jekic), and backward IC
   reconstruction (Oliveira). Use the verified notes and primary sources in
   `audit/literature_review.md`. Recheck final bibliographic versions when drafting.
3. **Problem and methods.** Define the unit-domain heat equation, unknown scalar
   alpha and initial state, exact data generator, shared boundary/bound information,
   six/twelve-mode variable projection and j^4 ridge, observation-only CV, and
   hard-boundary PINN with bounded alpha. Explain the different representation
   priors and tuning procedures rather than calling them perfectly matched.
4. **Prospective evaluation design.** Give all four N/family strata, sigma=0.01,
   paired family noise/coordinates, twenty independent designs, three fixed
   initialization seeds per neural case, both budgets, complete failure rules,
   and versioned freeze chronology. Separate earlier development from evaluation.
5. **Outcomes and analysis.** Define scalar relative alpha error, initial-field
   and space-time relative L2 errors, the dataset-level three-start median,
   paired differences against CV12, pointwise percentile intervals and their
   small-stratum/multiplicity limits. A paired median difference is not a
   difference of marginal medians. Explain timing components and numerical checks.
6. **Results.** Start with all outcomes/complete ledger, then all-method
   distributions, prespecified paired contrasts, representation control,
   budget-dependent behavior and tails, followed by costs and numerical checks.
   Keep favorable, unfavorable and uncertain results together.
7. **Discussion and limitations.** Interpret the conditional patterns with heat
   smoothing and finite-data regularization in mind. Do not assert a unique
   mechanism or structural non-identifiability. Discuss favorable finite spectral
   truth, fixed untuned neural settings, one alpha/noise/domain, only ten designs
   per stratum, fixed collocation and limited initialization sampling.
8. **Conclusion and availability.** State the supported audit findings, source
   and artifact commits/hashes, and reproducibility commands. New scientific
   questions require an amended protocol and fresh evaluation data.

## Figures and tables ready for drafting

- `results/broader_results_review_v1/method_errors.pdf`: every design score and
  all six pipelines; neural points summarize starts; six-mode IC floor shown.
- `results/broader_results_review_v1/paired_contrasts.pdf`: every prespecified
  paired contrast and pointwise interval; no significance stars.
- `results/broader_analysis_v1/method_summary.csv` and `paired_contrasts.csv`:
  original locked analysis values, including complete denominators.
- `results/broader_results_review_v1/timing.csv`: per-fit and three-start costs.
- The supplementary numerical audit and the original 401/801-point checks must
  be labeled as verification; original primary values remain unchanged.

No model retuning or selective removal of evaluation outcomes is needed for the
write-up. Publication venue and zero-mandatory-fee policy should be verified
against the eventual manuscript's scope before submission.
