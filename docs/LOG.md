# Lab notes

Dated and append-only. Includes the things that went wrong.

---

## 2026-10-14

**Aim.** Build the referee before any neural network exists. If the classical
baselines are weak, "PINNs lose to finite differences" means nothing.

**Setup.** Heat and wave equations on [0, 1], zero Dirichlet boundaries. Initial
conditions are *defined* as finite sine series, default `sin(pi x) + 0.5 sin(3 pi x)`,
so the separated solution is exact rather than a truncated Fourier series.
The n = 3 mode is there on purpose: it is the component a PINN should find hardest.

**Choices.**
- Crank-Nicolson for heat, not explicit Euler. Explicit needs `alpha k / h^2 <= 1/2`,
  which forces tiny time steps and would make the baseline look slow. I'm testing
  claims about cost, so the baseline has to be the strong one. The matrix is
  constant in time, so it is factorised once (`splu`) and reused.
- Leapfrog for wave, with a second-order Taylor start for the first step.
- Analytic solutions checked by substituting them back into their own PDEs on a
  2001 x 2001 grid: max residual 6.0e-6 (heat) and 8.4e-9 (wave). Both are
  finite-difference truncation error, not error in the solutions.

**Measured.** Heat, Crank-Nicolson, k = h, alpha = 0.05:

| nx | rel. L2 | order |
|---|---|---|
| 21 | 2.115e-3 | |
| 41 | 5.307e-4 | 1.995 |
| 81 | 1.333e-4 | 1.994 |
| 161 | 3.341e-5 | 1.996 |
| 321 | 8.366e-6 | 1.998 |

Wave, leapfrog, Courant number held at 0.5:

| nx | rel. L2 | order |
|---|---|---|
| 21 | 1.652e-2 | |
| 41 | 4.153e-3 | 1.992 |
| 81 | 1.043e-3 | 1.993 |
| 161 | 2.615e-4 | 1.996 |
| 321 | 6.546e-5 | 1.998 |

Crank-Nicolson at nx = 201, nt = 11 (mesh ratio 200) stays stable, rel. L2 2.3e-3.

**The lambda = 1 result.** I copied `wave_fd.yaml` to a scratch file and set
`nt: 41`, so that lambda = c*k/h = 1, then ran it with `--no-save`. The error
was 5.3e-16, which is rounding noise. I expected a small error, similar to the
4e-3 I got at lambda = 0.5, and instead it was about 10^13 times smaller. At
lambda = 1 the update formula reduces to `u[j,n+1] = u[j+1,n] + u[j-1,n] - u[j,n-1]`,
which is d'Alembert's exact solution evaluated on the grid, so the scheme is
exact up to rounding. `test_leapfrog_is_exact_at_unit_courant` checks the same
thing. Because of this, convergence studies run at lambda = 0.5: at lambda = 1
the error is pure rounding noise, so a measured "order" would mean nothing.

I also tried `nt: 21` (lambda = 2). The solver stopped with "CFL violated"
instead of returning numbers, which is what I want from a baseline.

Experiments I ran myself (scratch configs, `--no-save`):
- wave, nx=41, nt=21 (lambda=2): solver refused, "CFL violated"
- wave, nx=41, nt=41 (lambda=1): rel. L2 5.323e-16
- heat, nx=201, nt=11 (mesh ratio 200): stable, rel. L2 2.262e-03

**Open.** Score the PINN on the same grid as the FD solvers, or on a dense
independent grid? Same grid is fairer for accuracy per second; a dense grid
reflects the mesh-free claim. Probably report both.


## 2026-10-18

**Aim.** First neural solvers, behind the same interface and scored on the same grid as the classical ones. Vanilla PINN: soft initial/boundary constraints, Adam only, 5000 epochs, width 32, depth 4, 2000 random collocation points, float64, one thread. One seed (0) only, so everything below is a first look and not a conclusion.

**Measured.**

| Problem | Method | rel. L2 | max error | time |
|---|---|---|---|---|
| heat | Crank-Nicolson | 5.307e-4 | 6.879e-4 | 0.80 ms |
| heat | PINN | 2.892e-3 | 1.442e-2 | 96.5 s |
| heat | MLP control (200 observations, no physics) | 2.113e-2 | 4.737e-2 | 50.5 s |
| wave | leapfrog | 4.153e-3 | 6.978e-3 | 0.52 ms |
| wave | PINN | 4.998e-1 | 7.569e-1 | 205.7 s |

The finite-difference times are single runs under a millisecond, so the time ratios are order-of-magnitude only.

**What the numbers say.**
- The PINN lost to the classical solver on both problems, on accuracy and on time. Heat: about 5x larger error and about 120,000x slower. Wave: about 120x larger error and about 400,000x slower. This is one configuration and one seed, so it says nothing yet about PINNs in general.
- The heat PINN loss went from 8.6 to 2.0e-4 and mostly fell steadily. At epoch 4000 it jumped from 3.9e-4 to 1.05e-3 and then recovered. I haven't tested whether L-BFGS or a decaying learning rate prevents that.
- Wave: if the PINN had learned the n = 1 mode perfectly and missed the n = 3 mode entirely, the relative L2 error on this grid would be 0.447 (checked numerically, it is sqrt(0.2)). I got 0.50. That fits the spectral-bias explanation from Wang et al., but 0.50 is above 0.447, so mode 1 is imperfect too. "It missed the high-frequency mode" is a hypothesis, not a result. To test it I need to project the PINN solution onto each mode.
- The wave loss was still creeping down at epoch 5000 (9.2e-2), so the network may be undertrained and not stuck. I can't tell which yet.
- Heat: the PINN (no interior data) reached 2.9e-3 and the MLP (200 interior observations) reached 2.1e-2, about 7x worse. So physics alone beat 200 labelled points on this seed. This is not a clean test of the physics term, because the two methods get different information and collocation points cost nothing while observations don't. The clean test is a PINN given the same 200 observations.

**What surprised me.** The wave PINN was 120x worse while the heat PINN was only 5x worse. I did not expect the gap between the two problems to be that big.

**Open.** Five seeds with mean and spread. L-BFGS after Adam. A PINN given the same 200 observations as the MLP control. Projecting the wave solution onto its modes.

## 2026-10-19

**Aim.** Replace the single-seed first look with five seeds per variant, and find out what the PINN's optimiser and its data are worth. New this round: a resumable sweep runner (`pinnaudit sweep`), an across-seed summary, and a Fourier-mode decomposition of the error. 90 runs: 2 problems x 9 variants x 5 seeds (0-4).

**Variants.** `fd_N`: finite differences on an N-point grid, which gives the classical cost/accuracy curve. `pinn`: the vanilla PINN from before. `pinn_lbfgs`: the same, plus 500 L-BFGS iterations after Adam. `pinn_data200`: the same PINN, also shown 200 exact observations. `mlp_data200`: no physics term, same 200 observations (the control). Neural solvers are scored on the 41-point grid (heat 41 x 41, wave 41 x 81).

**Heat.**

| variant | rel. L2 mean | std | min to max | mean time |
|---|---|---|---|---|
| fd_21 | 2.12e-3 | 0 | | 0.64 ms |
| fd_41 | 5.31e-4 | 0 | | 1.0 ms |
| fd_81 | 1.33e-4 | 0 | | 1.6 ms |
| fd_161 | 3.34e-5 | 0 | | 3.5 ms |
| fd_321 | 8.37e-6 | 0 | | 8.6 ms |
| pinn | 3.14e-3 | 1.56e-3 | 1.40e-3 to 5.01e-3 | 147 s |
| pinn_lbfgs | 4.12e-4 | 1.19e-4 | 2.80e-4 to 5.97e-4 | 175 s |
| pinn_data200 | 3.84e-3 | 1.52e-3 | 1.82e-3 to 5.36e-3 | 180 s |
| mlp_data200 | 2.28e-2 | 9.09e-3 | 1.08e-2 to 3.35e-2 | 27 s |

**Wave.**

| variant | rel. L2 mean | std | min to max | mean time |
|---|---|---|---|---|
| fd_21 | 1.65e-2 | 0 | | 0.28 ms |
| fd_41 | 4.15e-3 | 0 | | 0.72 ms |
| fd_81 | 1.04e-3 | 0 | | 1.2 ms |
| fd_161 | 2.61e-4 | 0 | | 2.3 ms |
| fd_321 | 6.55e-5 | 0 | | 5.1 ms |
| pinn | 5.10e-1 | 3.92e-2 | 4.84e-1 to 5.80e-1 | 230 s |
| pinn_lbfgs | 3.63e-1 | 5.83e-2 | 2.87e-1 to 4.40e-1 | 268 s |
| pinn_data200 | 4.44e-1 | 2.05e-2 | 4.24e-1 to 4.74e-1 | 234 s |
| mlp_data200 | 8.38e-2 | 3.20e-2 | 5.29e-2 to 1.35e-1 | 36 s |

**Heat findings.**
- Finite differences win. The coarsest grid (21 points, 0.64 ms) has a lower mean error than the vanilla PINN (2.1e-3 vs 3.1e-3) in over 200,000x less time. The best neural variant (PINN + L-BFGS, 4.1e-4) lands between the 41-point and 81-point grids, which take 1.0 to 1.6 ms, against 175 s.
- L-BFGS cut the mean error 7.6x (3.1e-3 to 4.1e-4) for roughly 20% more wall-clock, which is within the timing noise noted below. It was better in all five seeds and the ranges do not overlap. So much of the vanilla PINN's error comes from the optimiser, not from the architecture.
- 200 exact observations did not help the heat PINN: 3.1e-3 vs 3.8e-3, better in 2 of 5 seeds and worse in 3, well inside the seed spread. Hypothesis, not tested: the PDE plus initial and boundary conditions already pin down the forward solution, so exact data adds little and optimisation is the bottleneck.
- At equal data, physics helps on heat: `pinn_data200` (3.8e-3) beats `mlp_data200` (2.3e-2) by about 6x, and the ranges do not overlap.

**Wave findings.**
- Finite differences win by more. The best neural mean (PINN + L-BFGS, 3.6e-1) is about 22x worse than the coarsest FD grid (1.65e-2) and takes about 10^6 times longer (268 s vs 0.28 ms). The best single run was 2.9e-1.
- L-BFGS lowered the mean error 29% (5.1e-1 to 3.6e-1) and 200 observations lowered it 13% (to 4.4e-1). Both were better in 5 of 5 seeds with non-overlapping ranges. Both are modest.
- The physics term hurts here. At equal data, `pinn_data200` (4.4e-1) is about 5x worse than `mlp_data200` (8.4e-2), and all five PINN runs are worse than all five MLP runs. This is the opposite sign to heat. It is a statement about this configuration only: loss weights all 1, 5000 Adam steps, width 32 x depth 4. I have not tried loss weighting or longer training. One guess, untested: the wave residual loss starts about 30x larger than the heat one (about 249 vs 8.6 at epoch 0), so with equal weights it may dominate the optimisation.
- Arithmetic from the mode amplitudes: a solver that missed mode 3 completely would have relative L2 error 0.447, because mode 3 holds 20% of the energy. So any run below 0.447 must have learned part of mode 3. Vanilla PINN: 0 of 5 runs below (0.484 to 0.580). PINN + L-BFGS: 5 of 5 below. `pinn_data200`: 3 of 5 below. This is consistent with the spectral-bias explanation for the vanilla PINN, but it does not prove it.

- Mode analysis (from the mode figure): in the vanilla PINN mode 3 holds about three-quarters of the squared error (its share of the relative error is 0.44, almost exactly the 0.447 you get by dropping mode 3 entirely) and mode 1 about a quarter (0.24: its oscillation lags the exact one). In the seed-0 example run the mode-3 amplitude is nearly flat instead of oscillating, and at t = 0 it starts near 0.1 (read off the figure) instead of 0.5, so the network did not even fit that mode of the initial condition. L-BFGS lowers both (mode 1: 0.10, mode 3: 0.34) but mode 3 still holds about 90% of the squared error. Giving the PINN the 200 observations barely changed mode 3 (0.43). The data-only MLP, same network and same observations, has mode 3 at 0.04 with its small error spread over many modes, so this network can learn that mode in this training budget. That weakens a pure spectral-bias explanation. What fits better is that, at equal weights, the physics residual overpowers the initial-condition and data terms (Wang et al. discuss this kind of imbalance between loss terms), but I have not tested that. Next: raise the data and initial-condition weights and see whether mode 3 comes back.

**Reproducibility.** FD error is identical across seeds (std 0), so seeds only change its timing. Seed 0 reproduces the same error to four digits in separate invocations: heat PINN 2.892e-3, wave PINN 4.998e-1 and heat MLP 2.113e-2 all matched between the single runs and the sweep.

**Caveats.**
- Five seeds is small. I report spread and ranges and run no significance tests.
- Wall-clock is noisy. The heat PINN at seed 0 took 96 s in the single run and 184 s in the pilot with identical work, and the 41-point FD solve took 0.6 to 2.6 ms across invocations. Treat all time ratios as order-of-magnitude.
- Neural solvers are scored on the 41-point grid and the FD solvers on their own grids, each against the exact solution on its own grid.
- One architecture, fixed random collocation points, equal loss weights, no learning-rate schedule. This is the plain PINN, not a tuned one.

**Open.** A tuned PINN (loss weighting, learning-rate decay, longer training, resampled points) before drawing any general conclusion. Burgers' equation, where the solution develops a shock. Inverse problems under noise and sparse data. A Gaussian-process baseline.

## 2026-10-20

**Day 3 plan: give the PINN a fair shot on the wave problem.** Written before running anything.

**What I know.** On the wave problem the vanilla PINN loses mode 3 (share of the error 0.44, against 0.24 for mode 1), while a no-physics network with the same architecture learns it (0.04). In the seed-0 example the PINN's mode-3 amplitude even starts near 0.1 at t = 0 instead of 0.5, so it does not fit that part of the initial condition.

**Two explanations.** H1, loss imbalance: with every weight at 1 the PDE residual dominates the loss and drags the solution away from the initial condition. H2, spectral bias: the network learns the fast mode too slowly whatever the weights. The data-only network learning mode 3 in the same budget argues against H2 being the whole story, but does not rule it out.

**Test.** Raise the initial and boundary weights (w_ic = w_bc) to 10, 100 and 1000 with the PDE weight fixed at 1. Five seeds each. Nothing else changes.

**Prediction.** If H1 is right, at the best weight the mean relative L2 error falls below 0.25 (it is 0.51 now) and mode 3's share falls below 0.25. I expect a U shape: weight 1000 should be worse than 100, because the PDE term then barely matters. If mode 3's share stays above 0.4 at every weight, H1 is rejected as the main cause and the next test is Fourier features, a direct test of H2. Anything in between is a partial effect and I will say so.

## 2026-10-20 (result)

**Day 3 result.** The prediction above is unchanged.

| variant | w_ic = w_bc | rel. L2 mean | std | min to max | mode 1 | mode 3 |
|---|---|---|---|---|---|---|
| pinn | 1 | 5.10e-1 | 3.92e-2 | 4.84e-1 to 5.80e-1 | 0.240 | 0.442 |
| pinn_w10 | 10 | 2.83e-1 | 8.76e-2 | 2.04e-1 to 4.20e-1 | 0.069 | 0.260 |
| pinn_w100 | 100 | 1.77e-1 | 1.72e-2 | 1.49e-1 to 1.97e-1 | 0.025 | 0.156 |
| pinn_w1000 | 1000 | 1.84e-1 | 2.08e-2 | 1.67e-1 to 2.17e-1 | 0.020 | 0.170 |

(Mode columns are each mode's share of the relative L2 error, mean over five seeds. Wave problem.)

**Against the prediction.**
- Supported. At the best weight (100) the mean error is 0.177 (below 0.25) and mode 3's share is 0.156 (below 0.25). Every run at weight 10, 100 or 1000 beat every vanilla run, since the ranges do not overlap (the worst weight-10 run, 0.420, is below the best vanilla run, 0.484). So loss imbalance is the main reason the vanilla wave PINN fails.
- Wrong. I expected a U shape, with weight 1000 worse than 100. It is 0.184 against 0.177, a difference far smaller than the seed spread (about 0.02), so this is a plateau from 100 to 1000.

**What is still open.**
- H2 (spectral bias) is not ruled out. Mode 1 is nearly fixed (0.240 to 0.025), but mode 3 still holds about 85% of the remaining squared error in modes 1 to 6, and at 0.156 it is about 3.6 times the data-only network's 0.043. Leakage into mode 5 also went up (0.037 to 0.055).
- The classical solver still wins. The tuned PINN (0.177, about 310 s) is about 11 times less accurate than the coarsest finite-difference grid (1.65e-2) and about a million times slower (0.28 ms).

**Caveats.**
- I chose the weight using the exact error, which a real user would not have, so this flatters the PINN.
- Only three weights, one architecture, no learning-rate schedule. The tuning effort for the PINN was a lot more than the classical solver needed, which needed none.
- The `pinn_w10` mean time (6291 s) is not real. At least one run paused, most likely the laptop sleeping, so I am not using that time until I re-run it. The errors are unaffected.

**Next.** Combine the best weight with L-BFGS on both problems to get a final best-effort PINN number. Then the inverse problem, which is the second half of my bet and the PINN's best chance.