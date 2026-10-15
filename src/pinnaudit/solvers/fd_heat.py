"""Crank-Nicolson finite differences for the 1D heat equation.

Discretising ``u_t = alpha u_xx`` with centred second differences in space and
the trapezoidal rule in time gives, for the interior unknowns,

    (I - (r/2) A) u^{n+1} = (I + (r/2) A) u^n,     r = alpha * k / h^2

where ``A`` is the tridiagonal second-difference operator. The scheme is
second-order accurate in both h and k and unconditionally stable, so unlike an
explicit scheme it imposes no relationship between k and h. The LHS matrix is
constant in time, so we factorise it once and reuse the factorisation -- that
is the main reason this baseline is so hard for a PINN to beat on cost.

Homogeneous Dirichlet boundaries are assumed, which is consistent with the
sine-series problems in ``pinnaudit.problems``.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pinnaudit.problems.base import Problem
from pinnaudit.problems.heat import Heat1D
from pinnaudit.solvers.base import Solution, Solver


class CrankNicolsonHeat(Solver):
    name = "fd_heat_cn"

    def solve(self, problem: Problem, nx: int, nt: int) -> Solution:
        if not isinstance(problem, Heat1D):
            raise TypeError(f"{self.name} solves heat1d, got {type(problem).__name__}")
        alpha = problem.cfg.alpha
        x, t, h, k = self._grid(problem, nx, nt)
        r = alpha * k / (h * h)

        n_int = nx - 2
        if n_int < 1:
            raise ValueError(f"nx={nx} leaves no interior points")

        off = np.ones(n_int - 1)
        main = np.full(n_int, -2.0)
        A = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csc")
        identity = sp.identity(n_int, format="csc")

        lhs = (identity - 0.5 * r * A).tocsc()
        rhs = (identity + 0.5 * r * A).tocsc()

        start = time.perf_counter()
        lu = spla.splu(lhs)

        u = np.zeros((nt, nx), dtype=float)
        u[0] = problem.initial(x)
        interior = u[0, 1:-1].copy()
        for n in range(1, nt):
            interior = lu.solve(rhs @ interior)
            u[n, 1:-1] = interior
        wall = time.perf_counter() - start

        # Boundary rows stay at zero, matching the homogeneous Dirichlet data.
        return Solution(
            x=x,
            t=t,
            u=u,
            wall_time_s=wall,
            solver_name=self.name,
            meta={
                "scheme": "crank-nicolson",
                "h": h,
                "k": k,
                "mesh_ratio_r": r,
                "n_dof": n_int,
                "unconditionally_stable": True,
            },
        )
