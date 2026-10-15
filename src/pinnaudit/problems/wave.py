"""1D wave equation with homogeneous Dirichlet boundaries.

    d2u/dt2 = c^2 * d2u/dx2,     x in [0, L],  t in [0, T]
    u(0, t) = u(L, t) = 0
    u(x, 0)      = sum_n a_n sin(k_n x)
    du/dt(x, 0)  = sum_n b_n sin(k_n x)

Separation of variables gives the exact standing-wave solution

    u(x, t) = sum_n [ a_n cos(k_n c t) + (b_n / (k_n c)) sin(k_n c t) ] sin(k_n x)

with k_n = n*pi/L. This is the oscillatory counterpart to the heat problem and
is where PINNs are expected to struggle: spectral bias makes high-frequency
modes disproportionately hard for a smooth network to represent.
"""

from __future__ import annotations

import numpy as np

from pinnaudit.config import ProblemConfig
from pinnaudit.problems.base import Array, Domain, Problem


class Wave1D(Problem):
    name = "wave1d"

    def __init__(self, cfg: ProblemConfig) -> None:
        if cfg.name != "wave1d":
            raise ValueError(f"Wave1D received config for problem {cfg.name!r}")
        self.cfg = cfg

    @property
    def domain(self) -> Domain:
        return Domain(x_min=0.0, x_max=self.cfg.x_max, t_min=0.0, t_max=self.cfg.t_max)

    def pde_residual_symbolic(self) -> str:
        return "u_tt - c^2 * u_xx"

    def exact(self, x: Array, t: Array) -> Array:
        x = np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)
        L = self.cfg.x_max
        c = self.cfg.c

        u = np.zeros((t.size, x.size), dtype=float)
        for mode in self.cfg.displacement_modes:
            k = mode.n * np.pi / L
            u += mode.amplitude * np.outer(np.cos(k * c * t), np.sin(k * x))
        for mode in self.cfg.velocity_modes:
            k = mode.n * np.pi / L
            u += (mode.amplitude / (k * c)) * np.outer(np.sin(k * c * t), np.sin(k * x))
        return u

    def initial_velocity(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        L = self.cfg.x_max
        g = np.zeros_like(x)
        for mode in self.cfg.velocity_modes:
            k = mode.n * np.pi / L
            g += mode.amplitude * np.sin(k * x)
        return g
