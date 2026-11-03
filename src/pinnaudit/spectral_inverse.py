"""Exact-eigenbasis heat inverse baseline with explicit IC regularization.

This is a diagnostic baseline for a constant coefficient, zero Dirichlet
boundary heat equation, not a general PDE solver or a new numerical method.
For each alpha the unknown amplitudes enter linearly (variable projection).
No ground-truth coefficient or initial condition is supplied to the estimator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from pinnaudit.problems.base import Array


def heat_design(x: Array, t: Array, alpha: float, length: float, modes: int) -> Array:
    """Paired-point design matrix: B_ij = sin(k_j*x_i)*exp(-alpha*k_j²*t_i)."""
    k = np.arange(1, modes + 1, dtype=float) * np.pi / length
    return np.sin(np.asarray(x)[:, None] * k) * np.exp(-alpha * np.asarray(t)[:, None] * k**2)


def project_amplitudes(design: Array, y: Array, ridge: float) -> tuple[Array, float]:
    """Minimize mean((Ba-y)^2) + ridge * sum(j^4*a_j^2).

    The dimensionless j^4 penalty favors a smooth initial field. Lambda must be
    selected using development data or a declared data-only rule, never alpha truth.
    Augmented least squares avoids squaring the design matrix condition number.
    """
    n, m = design.shape
    penalty = np.diag(np.arange(1, m + 1, dtype=float) ** 2)
    a, *_ = np.linalg.lstsq(
        np.vstack((design / np.sqrt(n), np.sqrt(ridge) * penalty)),
        np.concatenate((y / np.sqrt(n), np.zeros(m))),
        rcond=None,
    )
    objective = np.mean((design @ a - y) ** 2) + ridge * np.sum((penalty @ a) ** 2)
    return a, float(objective)


@dataclass(frozen=True)
class SpectralFit:
    alpha_hat: float
    amplitudes: Array
    objective: float
    wall_time_s: float
    profile_alpha: Array
    profile_objective: Array
    at_bound: bool
    modes: int
    ridge: float
    length: float

    def predict(self, x: Array, t: Array) -> Array:
        """Predictions at paired, rather than tensor-product, coordinates."""
        return heat_design(x, t, self.alpha_hat, self.length, self.modes) @ self.amplitudes


def fit_spectral(
    x: Array,
    t: Array,
    y: Array,
    *,
    length: float = 1.0,
    modes: int = 6,
    ridge: float = 0.0,
    alpha_bounds: tuple[float, float] = (0.001, 1.0),
    profile_points: int = 101,
) -> SpectralFit:
    """Profile alpha over a log grid, refining every grid-local minimum.

    The returned objective profile is a diagnostic, not a confidence interval.
    Bounds and the basis are explicit prior assumptions; test their sensitivity.
    Grid plus local refinement is not a proof of global optimality.
    """
    start = time.perf_counter()
    x, t, y = (np.asarray(v, dtype=float) for v in (x, t, y))
    if x.ndim != 1 or x.shape != t.shape or x.shape != y.shape or x.size == 0:
        raise ValueError("x, t, y must be nonempty paired 1D arrays")
    if not all(np.isfinite(v).all() for v in (x, t, y)):
        raise ValueError("observations must be finite")
    if not np.isfinite(length) or length <= 0 or np.any((x < 0) | (x > length)):
        raise ValueError("length must be positive and x must lie within the domain")
    if np.any(t < 0):
        raise ValueError("observation times must be nonnegative")
    if not isinstance(modes, int) or modes < 1:
        raise ValueError("modes must be a positive integer")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    lo, hi = alpha_bounds
    if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
        raise ValueError("alpha_bounds must be finite, ordered, and positive")
    if not isinstance(profile_points, int) or profile_points < 5:
        raise ValueError("profile_points must be an integer of at least five")

    def profile(log_alpha: float) -> tuple[Array, float]:
        design = heat_design(x, t, float(np.exp(log_alpha)), length, modes)
        return project_amplitudes(design, y, ridge)

    grid = np.linspace(np.log(lo), np.log(hi), profile_points)
    values = np.array([profile(z)[1] for z in grid])
    candidates = [(float(values[i]), float(grid[i])) for i in (0, len(grid) - 1)]
    # Refine all local minima instead of assuming the whole profile is unimodal.
    for i in range(1, len(grid) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            result = minimize_scalar(
                lambda z: profile(z)[1],
                bounds=(grid[i - 1], grid[i + 1]),
                method="bounded",
                options={"xatol": 1e-11},
            )
            if result.success and np.isfinite(result.fun):
                candidates.append((float(result.fun), float(result.x)))
    # Retain every coarse candidate in case no interior minimum was detected.
    best_grid = int(np.argmin(values))
    candidates.append((float(values[best_grid]), float(grid[best_grid])))
    objective, log_alpha = min(candidates)
    amplitudes, _ = profile(log_alpha)
    alpha_hat = float(np.exp(log_alpha))
    return SpectralFit(
        alpha_hat=alpha_hat,
        amplitudes=amplitudes,
        objective=objective,
        wall_time_s=time.perf_counter() - start,
        profile_alpha=np.exp(grid),
        profile_objective=values,
        at_bound=bool(np.isclose(log_alpha, grid[0]) or np.isclose(log_alpha, grid[-1])),
        modes=modes,
        ridge=ridge,
        length=length,
    )
