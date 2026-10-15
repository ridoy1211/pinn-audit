"""Abstract interface shared by every solver, classical or neural.

Day 2's PINN implements exactly this interface, which is what makes the
comparison fair: both return a solution sampled on the same grid, and both
report the wall-clock time it took to get there.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinnaudit.problems.base import Array, Problem


@dataclass
class Solution:
    """A solution sampled on a space-time grid, plus its cost."""

    x: Array
    t: Array
    u: Array
    wall_time_s: float
    solver_name: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = (self.t.size, self.x.size)
        if self.u.shape != expected:
            raise ValueError(f"u has shape {self.u.shape}, expected {expected} (nt, nx)")

    @property
    def n_dof(self) -> int:
        """Degrees of freedom the method carried. Grid points for FD, weights for a PINN."""
        return int(self.meta.get("n_dof", self.x.size))


class Solver(ABC):
    """Maps a problem plus a discretisation onto a sampled solution."""

    name: str

    @abstractmethod
    def solve(self, problem: Problem, nx: int, nt: int) -> Solution:
        """Solve ``problem`` on an ``nt`` x ``nx`` grid."""

    @staticmethod
    def _grid(problem: Problem, nx: int, nt: int) -> tuple[Array, Array, float, float]:
        """Uniform space-time grid, returned with its spacings."""
        d = problem.domain
        x = np.linspace(d.x_min, d.x_max, nx, dtype=float)
        t = np.linspace(d.t_min, d.t_max, nt, dtype=float)
        return x, t, float(x[1] - x[0]), float(t[1] - t[0])
