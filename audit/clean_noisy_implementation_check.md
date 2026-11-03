# Clean/noisy control implementation check — 3 November 2026

Before asking the user to run the control, the assistant exercised all 24 cases
in a separate workspace directory, using Python 3.12.14 and the same saved
development observations. This is implementation verification on already-seen
development data, not a preregistered prediction or independent confirmation.
The subsequent user run will be a reproduction in the user's environment.

The complete script finished successfully. Every noisy estimate and objective
matched the saved diagnostic within the script's declared tolerances. No feasible-
candidate search warnings occurred. Unpenalized clean fits recovered alpha=0.05
to numerical precision for both selected seeds and all three lower bounds.
The fixed penalty introduced clean-data coefficient errors of approximately
0.417% and 1.362% for seeds 0 and 1, respectively. These observations do not prove
uniqueness, stability, or general performance. They must not be presented as
unseen confirmatory results.

Independent analytic-reference checks, objective decomposition checks, and
source lint/format checks passed. The script changes no estimator code. It writes
its own source/input hashes and environment details in each run's manifest,
refuses to overwrite a previous output directory, and marks the run complete
only after all cases and reproduction checks finish.
