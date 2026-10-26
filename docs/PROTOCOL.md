# Protocol

This file records how the experiments were designed and, just as important, which
choices were made *after* seeing data. A reader should be able to tell the
difference between a prediction that was tested and a result that was explained
afterwards. Dated entries and the evidence behind every statement here are in
[`docs/LOG.md`](LOG.md) and in the git history.

## 1. Question and scope

**Question.** When does a physics-informed neural network (PINN) beat a classical
solver, in accuracy per unit of compute, on forward problems, on inverse problems,
and as the dimension grows?

**Scope of the results so far.** Smooth, linear PDEs in one space dimension with
exact solutions (heat and wave), plus one inverse parameter (the heat diffusivity).
Nothing yet covers nonlinear PDEs, shocks, complex geometry or high dimension. The
dimension study is planned next, and conclusions must not be read beyond the scope
of what was run.

## 2. Fixed before the data of each experiment

| Item | Choice |
|---|---|
| Problems | Heat (alpha = 0.05) and wave (c = 1) on [0, 1] x [0, 1], zero Dirichlet boundaries, initial condition `sin(pi x) + 0.5 sin(3 pi x)`. Exact solutions by separation of variables |
| Classical solvers | Crank-Nicolson (heat) and leapfrog (wave), each verified against its theoretical second-order convergence before use |
| Metrics | Relative L2 error (headline), maximum error, wall-clock time |
| Hardware settings | CPU, float64, one thread, for every method |
| Baseline PINN | tanh network, 4 hidden layers of 32, soft initial and boundary terms, Adam at 1e-3 for 5000 steps, 2000 fixed random collocation points, all loss weights 1 |
| Scoring grids | Neural solvers on 41 x 41 (heat) and 41 x 81 (wave). Classical solvers at five resolutions |
| Seeds | 0 to 4 |
| Inverse design | 10, 50 or 200 observations; noise 0, 5 or 10% of the signal's standard deviation; both methods start from alpha = 0.2 and see identical data; PINN settings fixed in advance (Adam then L-BFGS, 3000 steps plus 300, data weight 10, learning rate for alpha 10 times larger) |

## 3. Decided after seeing data

These are the choices that were *not* fixed in advance. Each is listed with what
triggered it.

| # | Choice | What triggered it |
|---|---|---|
| P1 | Added an L-BFGS stage | A single-seed run showed a loss jump in the heat PINN and a slowly falling loss in the wave PINN |
| P2 | Added the "200 observations" PINN and the no-physics network | To separate the effect of physics from the effect of data, after the first single-seed comparison |
| P3 | Loss weights 10, 100, 1000 on the initial and boundary terms | The five-seed mode analysis showed the wave PINN missing the third harmonic while the no-physics network learned it. The hypothesis and prediction were written in the log before the runs (see section 4). The weight was selected by exact error on seeds 0 to 4 |
| P4 | Weight 100 combined with L-BFGS, on both problems | The result of P3. The weight tuned on the wave problem was applied to heat without tuning it there |
| P5 | An unknown-initial-condition inverse setting | The classical fit in the known setting was so strong that a fair test needed a case with a partly unspecified model. Added after seeing classical results, before any PINN inverse result |
| P6 | Scope changes | The oscillator inverse problem was replaced by a heat inverse problem, to keep one equation throughout. Poisson in 2D and a Gaussian-process baseline were dropped. Burgers' equation became optional. A dimension study was added after reading Grossmann et al., PINNacle and Rathore et al. (see `docs/REFERENCES.md`) |
| P7 | One timing run excluded | A run of the weight-10 variant took far longer than the others because the laptop went to sleep. It was dropped and re-run; the error for the same seed was identical |
| P8 | Held-out confirmation on ten seeds, 5 to 14 | The development results were selected and reported on the same five seeds. With only five pairs the smallest possible p-value of a paired test is 0.0625, so ten seeds are used. The variants, settings and predictions are fixed in the log and committed before the run. It covers the headline variants only (plain PINN and the best variant per problem, plus three classical grids timed in the same session) |

## 4. Pre-registration record

A prediction counts as pre-registered only if a commit recording it exists and is
earlier than the first run it concerns. `python scripts/check_chronology.py`
compares the commit times with the run timestamps and prints one verdict per
experiment. Its output is the evidence. Where it reports NOT VERIFIED, the entry in
the lab notes says so, and the result is reported as exploratory.

| Experiment | Commit message that recorded the prediction |
|---|---|
| Loss-weight test (wave) | "Write down the Day 3 hypothesis and prediction before running it" |
| Best shot (wave and heat) | none: the prediction was drafted in the notes but committed together with the results |
| Inverse problem | "Write down the inverse-problem prediction before running the PINN" |
| Confirmation on held-out seeds | "Write down the confirmation-run predictions before running them" |

## 5. Known threats to validity

- **Selection on the evaluation seeds.** Settings were chosen by looking at seeds 0
  to 4 and reported on the same seeds, which is optimistic for the PINN. The fix is a
  confirmation run on ten held-out seeds (5 to 14), whose design and predictions are
  committed before it starts.
- **Tuning used the exact error,** which a real user would not have.
- **Unequal tuning effort.** The PINN received several rounds of tuning and the
  classical solvers none. The heat PINN was not tuned separately from the wave one.
- **Scoring grids differ.** Each method is scored against the exact solution on its
  own output grid.
- **Timing noise.** One machine; times vary by up to about two-fold between sessions
  for identical work. Time ratios are reported as orders of magnitude.
- **The unknown-initial-condition classical fit is unregularised.** A regularised or
  lower-dimensional fit might do better at 10 observations. A regularised baseline is
  planned.
- **Five seeds** support medians and ranges, not significance claims. The confirmation
  run uses ten, and bootstrap intervals and paired tests are planned for it.
- **Scope.** See section 1.

## 6. Conventions from now on

1. Each new experiment gets its prediction committed first. Check it with
   `python scripts/check_chronology.py`.
2. Seeds 0 to 4 are for development and tuning. Seeds 5 to 14 are held out and used
   only to report. `tests/test_confirm_sweeps.py` fails if a confirmation sweep uses a
   development seed.
3. Every change to the plan is entered in section 3 on the day it is made, with the
   reason.
4. Tables in the README and the report are generated from the saved results.
