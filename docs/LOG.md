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
