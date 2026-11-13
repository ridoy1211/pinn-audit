# Manuscript integration and evidence map

## One paper, with connected stages

The research question is how regularization, initial-state representation,
optimization budget and observation timing affect joint recovery of heat
diffusivity and an unknown initial temperature from sparse, noisy measurements.
The [integrated draft](../manuscript/integrated/README.md) presents the completed
work as one article. Integration makes the scientific argument easier to assess;
it does not add independent data or turn established mathematics into a new result.

| Stage | Role in the paper | Evidence and sampling unit |
|---|---|---|
| Original comparison | Establish the baseline and show the effects of regularization, representation and neural budget | 20 independent design/noise blocks; 40 dataset cases; 400 jobs. Neural scores are outcome-specific medians over three starts. |
| Controlled timing | Test whether the comparison changes with the measurement schedule | 60 fresh blocks, 30 per observation count; 360 datasets; 720 fits. Shared spatial locations and noise draws preserve pairing across schedules and initial fields. One declared neural start. |
| Explanatory diagnostics | Examine what the observations identify and whether learned quantities generate the network trajectory | Profiles, ranks, regularization sensitivity, 720 added basis-sensitivity fits and independent propagation of 360 timing PINNs. These reuse the timing datasets and are post-evaluation analyses. |
| Localized initial field | Examine dependence on truths within the twelve-mode representation | 60 fresh blocks; 120 datasets; 240 fits. One smooth field outside that span, with independently verified forward accuracy. Mixed and late schedules only. |

The original comparison has its own results subsection and complete appendix
tables. It is the foundation of the paper, not discarded development data.
The timing result has a separate subsection and retains the prespecified primary
interaction: **3.7322 percentage points, 95% percentile interval
[-0.0947, 56.2421]**. Its direction remains unresolved. Paired effects are
calculated within blocks before taking medians.

The original three-start neural summary and five-candidate ridge grid differ
from the follow-ups' one-start estimator and expanded ridge grid. Stages must
not be pooled as one unchanged procedure with a larger independent sample.
The protocols were locally frozen prospectively, not formally preregistered;
later diagnostics retain their post-evaluation status.

## What the explanatory checks add

Ten observations can permit interpolation by twelve unrestricted initial
coefficients across nearby diffusivities when the observation matrix has full
row rank. Better regularized recovery does not establish unconstrained
identifiability. Independent propagation discrepancies reach about 11.65%;
passing a numerical refinement screen does not establish physical exactness.
The localized field's initial projection error is about 17.86%, but omitted modes
are negligible at the delayed sensors. Its energy differs from the original
fields, preventing an attribution solely to basis mismatch.

Joint heat inversion and initial-condition-dependent identifiability have prior
literature. The draft now discusses Salmaniw and Browning (2025) and BiLO (2026)
alongside the existing joint-inversion and PINN-comparison references. The
contribution is the controlled empirical comparison and its explanatory checks,
not a new inverse algorithm, identifiability theorem or priority claim.

## Provenance and review status

This documentation update is based on main at
bdc52599cd9941c7a4b9c257e2cc3f94280d2c1a and research reporting at
02af8a24de9d536f5f6b68352ab8fc980a37819b. The earlier Markdown manuscript had
the same Git blob, 42d2941137595cb61df76f8dea88e8265781ca2f, on both branches.
The integrated source prepared on 12 November was checked against the archived
tables and updated here with the two close references. All 168 marginal-median
table entries were checked against the existing reports. The five existing
figures and all four table-source files are copied without changing their bytes.

Earlier manuscript files, freezes, experimental sources and released evidence
are preserved. No experiments were run or results selected for this integration.
The optional known-versus-unknown initial-temperature control remains future
work. The author must review the argument, cited sources and declarations before
submission; a journal-ready submission package is a later formatting step.
