"""Abstract interface shared by every PDE benchmark problem.

A ``Problem`` owns the *mathematics*: the domain, the initial and boundary
conditions, and -- crucially for this audit -- a closed-form reference
solution. Solvers never define their own notion of truth; they are always
scored against ``Problem.exact``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Domain:
    """Space-time box on which a problem is posed."""

    x_min: float
    x_max: float
    t_min: float
    t_max: float

    @property
    def length(self) -> float:
        return self.x_max - self.x_min

    @property
    def duration(self) -> float:
        return self.t_max - self.t_min


class Problem(ABC):
    """A PDE benchmark with a known analytic solution."""

    name: str

    @property
    @abstractmethod
    def domain(self) -> Domain:
        """Space-time box the problem is posed on."""

    @abstractmethod
    def exact(self, x: Array, t: Array) -> Array:
        """Analytic solution evaluated on the tensor grid ``t`` x ``x``.

        Parameters
        ----------
        x
            Spatial coordinates, shape ``(nx,)``.
        t
            Times, shape ``(nt,)``.

        Returns
        -------
        Array
            Shape ``(nt, nx)``, so ``u[n, j] == u(x[j], t[n])``.
        """

    @abstractmethod
    def pde_residual_symbolic(self) -> str:
        """Human-readable residual, echoed into run records for traceability."""

    def initial(self, x: Array) -> Array:
        """Initial displacement, derived from ``exact`` to avoid duplication."""
        return self.exact(x, np.array([self.domain.t_min]))[0]

    def initial_velocity(self, x: Array) -> Array:
        """du/dt at t_min. Zero unless a problem is second order in time."""
        return np.zeros_like(np.asarray(x, dtype=float))

    def boundary_values(self, t: Array) -> tuple[Array, Array]:
        """Dirichlet data at ``x_min`` and ``x_max``. Homogeneous by default."""
        t = np.asarray(t, dtype=float)
        return np.zeros_like(t), np.zeros_like(t)
