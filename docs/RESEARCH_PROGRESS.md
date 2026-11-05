# Research progress — 5 November 2026

This is the current progress guide. The project now uses research stages rather
than day numbers. Earlier plans are in [the history archive](history/README.md).
This does not claim seven days or 56 hours of actual work were measured.

| Stage | Evidence and outcome | Status |
|---|---|---|
| Audit and mathematical foundations | Original evidence boundary, setup/tests, classical pilot, Grossmann/Zapf reading notes | Completed checkpoint; author understanding remains an ongoing responsibility |
| Classical development | Bounds/profile diagnostics, clean/noisy controls and data-only regularization tuning | Completed exploratory work |
| Neural feasibility and fixed design | Bounded PINN and budget pilots; broader implementation b9d2b3e and freeze dad8cd5 | Completed before evaluation |
| Fixed evaluation | 40 cases from 20 independent designs; 160 classical outputs plus 240 neural fits; archive 4a64ed4 | All 400 scheduled jobs completed; this does not certify convergence |
| Analysis and numerical verification | 24 paired contrasts, pointwise bootstrap intervals, figures, grid and exact-norm checks | Completed for the frozen design |
| Manuscript and author preparation | Main text, abstract, conclusion, title-page preparation and declarations integrated in the review draft | Draft available; final author review, coauthors, affiliation, citations and submission formatting unresolved |
| Reproducibility and release | Isolated analysis replay at 0977887; 227 tests passed; PR #3 merged as 5000363 and its main CI passed | Analysis replay completed; final submission package and human approval pending |

The [review manuscript](MANUSCRIPT_REVIEW_DRAFT.md) is the complete reading copy.
It is not an approved submission. The component sections remain editable sources.
See [Next steps](NEXT_STEPS.md) for the current action.

## Decisions and changes in scope

The original objective remains a defensible physics/ML project and academic draft.
The design was narrowed before the final evaluation; these differences must be
reported honestly:

- Observation count, initial-field complexity, classical basis size and neural
  training budget were studied. Observation timing was not varied systematically.
- The main classical reference is spectral variable projection, not a broad
  FEM/FDM comparison. Six- and twelve-mode representations are distinct references.
- The final study uses 40 cases from 20 shared designs and 240 neural fits, rather
  than the original illustrative 160-neural-fit schedule.
- One absolute noise SD (0.01) and one true diffusivity (0.05) were used. This is
  not a noise sweep or a result about all physical diffusion problems.
- Classical cross-validation and fixed neural settings offer different tuning
  opportunities. Matched tuning and equal computation were not implemented.

These changes preceded main evaluation and are reflected in the frozen protocol.
The fixed results support conditional pipeline comparisons, not a universal PINN
or classical-method winner. No new architecture or identifiability theorem is claimed.

## Publication and collaboration decisions

SN Computer Science remains a provisional preparation candidate, not a promised
acceptance or finalized venue. SISC/SIMODS are weak immediate targets given the
present contribution. JOSS is a different software-paper route and its current
development-history requirement is not met. See [the venue and collaboration
assessment](SIAM_JOSS_AND_COAUTHOR_REVIEW.md).

Independent academic review would be valuable. No professor has been contacted
or added as a coauthor. Coauthorship must reflect actual intellectual work and
approval; a professor's affiliation would not automatically become Ridoy's.
Ridoy confirmed no external funding and no competing interests; any later
coauthor must supply their own declarations. Human final approval is pending.

## What remains

1. Obtain critical feedback on the physics, comparator fairness and contribution.
2. Complete the author's own review and resolve substantive comments.
3. Finalize authorship/affiliation and the target journal, then check its current
   citation, AI-disclosure and submission-file requirements.
4. Prepare and visually verify Word/LaTeX/PDF submission files. The earlier Word
   rendering attempt lacked the required renderer; no verified Word file is claimed.
5. Package the reviewed manuscript and reproducibility references for submission.

No training command is required now. Any scientifically justified new experiment
needs a separate documented plan and fresh evaluation data.
