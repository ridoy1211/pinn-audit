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

**What surprised me.** The wave PINN was 120x worse while the heat PINN was only 5x worse. I didn’t expect the gap between the two problems to be that big.

**Open.** Five seeds with mean and spread. L-BFGS after Adam. A PINN given the same 200 observations as the MLP control. Projecting the wave solution onto its modes.