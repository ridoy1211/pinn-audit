"""Explicit leapfrog (CTCS) finite differences for the 1D wave equation.

Centred second differences in both space and time give

    u_j^{n+1} = 2 u_j^n - u_j^{n-1} + lam^2 (u_{j+1}^n - 2 u_j^n + u_{j-1}^n)

with Courant number ``lam = c k / h``. The scheme is second-order accurate and
stable only for ``lam <= 1`` (the CFL condition), which we enforce rather than
silently producing an exponentially divergent solution.

The first time level cannot use the three-level recursion, so it is taken from
a second-order Taylor expansion that substitutes the PDE for u_tt:

    u_j^1 = u_j^0 + k g_j + (lam^2 / 2) (u_{j+1}^0 - 2 u_j^0 + u_{j-1}^0)

Using the naive first-order start instead silently degrades the whole run to
first order -- see docs/LOG.md.
"""

from __future__ import annotations

import time

import numpy as np

from pinnaudit.problems.base import Problem
from pinnaudit.problems.wave import Wave1D
from pinnaudit.solvers.base import Solution, Solver

CFL_TOL = 1e-12


class LeapfrogWave(Solver):
    name = "fd_wave_leapfrog"

    def solve(self, problem: Problem, nx: int, nt: int) -> Solution:
        if not isinstance(problem, Wave1D):
            raise TypeError(f"{self.name} solves wave1d, got {type(problem).__name__}")
        c = problem.cfg.c
        x, t, h, k = self._grid(problem, nx, nt)
        lam = c * k / h

        if lam > 1.0 + CFL_TOL:
            raise ValueError(
                f"CFL violated: lam = c*k/h = {lam:.4f} > 1 "
                f"(c={c}, k={k:.3e}, h={h:.3e}). Increase nt or decrease nx."
            )

        u = np.zeros((nt, nx), dtype=float)
        u[0] = problem.initial(x)
        g = problem.initial_velocity(x)
        lam2 = lam * lam

        start = time.perf_counter()
        # Second-order accurate first step.
        curvature = u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2]
        u[1, 1:-1] = u[0, 1:-1] + k * g[1:-1] + 0.5 * lam2 * curvature

        for n in range(1, nt - 1):
            curvature = u[n, 2:] - 2.0 * u[n, 1:-1] + u[n, :-2]
            u[n + 1, 1:-1] = 2.0 * u[n, 1:-1] - u[n - 1, 1:-1] + lam2 * curvature
        wall = time.perf_counter() - start

        return Solution(
            x=x,
            t=t,
            u=u,
            wall_time_s=wall,
            solver_name=self.name,
            meta={
                "scheme": "leapfrog-ctcs",
                "h": h,
                "k": k,
                "courant_lambda": lam,
                "n_dof": nx - 2,
                "unconditionally_stable": False,
            },
        )
