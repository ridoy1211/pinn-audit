# Abstract and conclusion draft

Prepared 5 November 2026 from the completed study archived at `4a64ed4`.
These are review drafts, not author-approved submission text. The four abstract
labels follow the provisional SN Computer Science format; journal choice remains
conditional. No new analysis or experiment was introduced.

## Structured abstract

**Purpose:** We examine how classical regularization, initial-state representation
and neural optimization budget affect comparisons in joint recovery of constant
diffusivity and an unknown initial temperature field.

**Methods:** A prospectively versioned one-dimensional heat benchmark used two
initial fields, 10 or 50 noisy observations, and 20 independent observation/noise
designs, shared across field families to form 40 cases. Four spectral inverse
procedures varied regularization and six- or twelve-mode representations. Two
hard-boundary physics-informed neural network procedures used fixed architecture,
three initializations per case, and 200 or 1000 L-BFGS iterations after Adam.
We separately assessed diffusivity, initial-field and full-field errors, with
paired comparisons against cross-validated twelve-mode inversion (CV12).

**Results:** All 400 scheduled jobs completed. For the richer initial field with
10 observations, the longer-budget neural procedure had lower diffusivity and
initial-field errors than CV12 in nine of ten designs each. With 50 observations,
CV12 had lower full-field error in eight of ten designs. Several contrasts
remained uncertain under pointwise bootstrap intervals. The six-mode
representation imposed a 21.82% initial-field error floor for the richer pattern.

**Conclusion:** Comparative conclusions varied with the specified procedure,
observation regime and recovery target. The results support a limited pipeline
audit rather than a universal ranking. Spectral truth, unequal tuning opportunities
and ten designs per condition limit generalization.

## Keywords

Physics-informed neural networks; inverse heat conduction; joint parameter and
state estimation; regularization; spectral methods; computational benchmarking.

## Conclusion

This controlled study compared specified neural and spectral procedures for
joint diffusivity and unknown-initial-field heat inversion. Explicit
regularization and representation controls changed the context of apparent
neural advantages, while retaining both neural budgets revealed that longer
optimization did not uniformly improve physical recovery. The strongest
favorable co-primary pattern for the longer neural budget occurred with sparse
observations of the richer initial field; denser observations favored CV12 for
the full-field comparison. Other contrasts remained uncertain.

These findings concern one synthetic heat problem, two finite spectral initial
states and fixed procedural choices. They establish neither a new solver nor
general superiority of a method class. The versioned protocol and complete
artifact archive support inspection of the conditional findings, including
unfavorable and uncertain outcomes. Broader conclusions require fresh evaluation
under additional physical, data and tuning conditions.

## Declaration preparation note

Substantive LLM assistance in implementation, debugging, experiment automation,
literature triage, analysis scripting and manuscript drafting should be described
accurately in the final manuscript. The human author must review the work and
accept responsibility. Do not insert an assertion that all human verification
is complete merely because automated checks passed. Actual author contributions,
affiliation, funding and conflicts remain to be confirmed.
