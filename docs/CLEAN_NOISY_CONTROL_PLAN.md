# Clean-versus-noisy control — 3 November 2026

## Purpose and fixed design

This is a further exploratory diagnostic of the same two selected development
datasets, not a new independent confirmation or a prevalence estimate. Remove
the realized measurement noise while holding the observation coordinates fixed.
Can the unpenalized inverse fit recover the simulated coefficient from the clean
values at those coordinates? Does adding the same penalty introduce clean-data
bias? Do the previously observed noisy boundary estimates reproduce?

- Read archived N=10 observations for seeds 0 and 1. Use saved `clean` and `y`
  arrays as separate fitting targets, with identical `x` and `t` arrays.
- Retain six unknown sine amplitudes, length 1, upper alpha bound 1,
  lower bounds 0.001, 0.0001, 0.00001, and lambda=0 or 1e-6.
- Use 401 alpha-profile points and the existing local refinement in every fit.
  This gives 24 fits: 2 datasets x 2 targets x 2 lambdas x 3 lower bounds.
- Fit all conditions. Do not select the best lambda or bounds using true alpha.
- Preserve amplitudes and profiles, objective components, coefficient error,
  full-field and initial-field error on the established 401-point quadrature.
- Check noisy estimates against the archived diagnostic to detect accidental
  changes in inputs, fitting settings, or implementation.
- Record singular values/rank/condition number of the *unaugmented amplitude
  design matrix* at the fitted alpha. These describe conditional amplitude
  estimation; they do not establish joint identifiability of alpha and amplitudes.
- Separately project amplitudes at true alpha as a diagnostic feasible candidate.
  This candidate must not be used to initialize, repair, or select the fitted
  estimate. A materially worse fitted objective flags a search issue.

The exact reference is alpha=0.05 and u(x,0)=sin(pi*x)+0.5*sin(3*pi*x).
These facts are used to construct/check the simulation control and score recovery;
the fitting routine receives only x, t, the selected target, and declared settings.
No PINNs are trained. No new random data are generated.

## Interpretation before inspecting the output

If the unpenalized clean fit recovers the truth but the noisy one does not, noise
is sufficient to change the preferred fitted solution in that particular setup.
This does not establish stability under other perturbations, uniqueness, a general
noise threshold, or the best observation design.

If the unpenalized clean fit fails despite the truth being representable, inspect
objective gaps, rank/conditioning, profiles, and tolerances before making a
physical identifiability claim. A fixed penalty can bias even clean-data fits.
Do not require penalized estimates to equal the truth simply because data are clean.

## Run and evidence

From the existing checkout with its environment active:

```cmd
python scripts/compare_clean_noisy.py --out-dir results\clean_noisy_control_v1
```

The output directory must be new. Existing experiments are never overwritten.
Input/source hashes and environment details are saved with configuration, all 24
records, profiles, and a readable summary. The assistant may exercise this same
development-data control in a separate work directory to verify implementation;
the user's subsequent run is then reproduction, not an unseen prediction.
