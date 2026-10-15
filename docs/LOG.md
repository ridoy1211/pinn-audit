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

**The lambda = 1 result.** My first wave convergence run gave errors near 1e-16
and a nonsense "order". Not a bug. At `lambda = c k / h = 1` the stencil reduces to
`u[j,n+1] = u[j+1,n] + u[j-1,n] - u[j,n-1]`, which is d'Alembert's solution on the
grid, so it is exact up to rounding (4.1e-16 at nx = 21, 6.2e-16 at nx = 81).
Convergence studies therefore use lambda = 0.5. Kept as a test so nobody
"fixes" the config back.

TODO: write what I expected when I ran the Courant experiment myself, and what actually happened.

**Open.** Score the PINN on the same grid as the FD solvers, or on a dense
independent grid? Same grid is fairer for accuracy per second; a dense grid
reflects the mesh-free claim. Probably report both.
