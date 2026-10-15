"""1D heat equation with homogeneous Dirichlet boundaries.

    du/dt = alpha * d2u/dx2,      x in [0, L],  t in [0, T]
    u(0, t) = u(L, t) = 0
    u(x, 0) = sum_n a_n sin(n*pi*x/L)

Because the initial condition is given directly as a finite sine series, the
Fourier solution is *exact* rather than truncated:

    u(x, t) = sum_n a_n sin(k_n x) exp(-alpha k_n^2 t),   k_n = n*pi/L

That matters for this audit: any error we measure belongs to the solver, not
to a truncated reference series.
"""

from __future__ import annotations

import numpy as np

from pinnaudit.config import ProblemConfig
from pinnaudit.problems.base import Array, Domain, Problem


class Heat1D(Problem):
    name = "heat1d"

    def __init__(self, cfg: ProblemConfig) -> None:
        if cfg.name != "heat1d":
            raise ValueError(f"Heat1D received config for problem {cfg.name!r}")
        self.cfg = cfg

    @property
    def domain(self) -> Domain:
        return Domain(x_min=0.0, x_max=self.cfg.x_max, t_min=0.0, t_max=self.cfg.t_max)

    def pde_residual_symbolic(self) -> str:
        return "u_t - alpha * u_xx"

    def exact(self, x: Array, t: Array) -> Array:
        x = np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)
        L = self.cfg.x_max
        alpha = self.cfg.alpha

        u = np.zeros((t.size, x.size), dtype=float)
        for mode in self.cfg.displacement_modes:
            k = mode.n * np.pi / L
            decay = np.exp(-alpha * k * k * t)  # (nt,)
            shape = np.sin(k * x)  # (nx,)
            u += mode.amplitude * np.outer(decay, shape)
        return u
