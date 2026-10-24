"""Inverse problem for the heat equation: recover alpha from noisy observations.

Observations are samples of the exact solution at random space-time points with
Gaussian noise added. The noise level is relative: ``noise = 0.05`` means the noise
standard deviation is 5% of the standard deviation of the clean observations.

Positions and noise are drawn from one seeded generator in a fixed order, so for a
given seed and number of observations the positions never change with the noise
level and the noise is the same draw, only rescaled. Noise levels are therefore
directly comparable, and the classical and neural fits always see identical data.

The classical estimator is a nonlinear least-squares fit of a Crank-Nicolson
forward model (scipy), the same recipe as the baseline in Jekic et al. (2025).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

from pinnaudit.config import SineMode
from pinnaudit.problems.base import Array
from pinnaudit.problems.heat import Heat1D
from pinnaudit.solvers.fd_heat import CrankNicolsonHeat


@dataclass(frozen=True)
class Observations:
    x: Array
    t: Array
    u: Array  # noisy values
    clean: Array
    sigma: float  # noise standard deviation actually used


@dataclass
class IdentResult:
    alpha_hat: float
    wall_time_s: float
    meta: dict[str, Any] = field(default_factory=dict)


def make_observations(problem: Heat1D, n_obs: int, noise: float, seed: int) -> Observations:
    """Draw ``n_obs`` noisy observations of the exact solution."""
    if n_obs < 1:
        raise ValueError("n_obs must be at least 1")
    if noise < 0.0:
        raise ValueError("noise must be non-negative")
    rng = np.random.default_rng(seed)
    d = problem.domain
    x = rng.uniform(d.x_min, d.x_max, n_obs)
    t = rng.uniform(d.t_min, d.t_max, n_obs)
    unit_noise = rng.standard_normal(n_obs)

    clean = np.diag(problem.exact(x, t))  # exact() is a tensor grid; take u(x_i, t_i)
    sigma = noise * float(np.std(clean))
    return Observations(x=x, t=t, u=clean + sigma * unit_noise, clean=clean, sigma=sigma)


def fit_lsq(
    problem: Heat1D,
    obs: Observations,
    alpha0: float,
    ic_modes: int | None = None,
    nx: int = 161,
    nt: int = 161,
) -> IdentResult:
    """Fit alpha by nonlinear least squares against a Crank-Nicolson forward model.

    The parameter is log(alpha), which keeps alpha positive and makes the problem
    well scaled. The forward model is solved on an ``nt`` x ``nx`` grid and
    interpolated linearly to the observation points.

    With ``ic_modes=None`` the initial condition is taken from ``problem``. With
    ``ic_modes=M`` it is treated as unknown: the amplitudes of the first M sine
    modes become extra parameters, starting from zero. The true initial condition
    lies inside that family, which favours the classical method.
    """
    if alpha0 <= 0.0:
        raise ValueError("alpha0 must be positive")
    if ic_modes is not None and ic_modes < 1:
        raise ValueError("ic_modes must be at least 1")
    start = time.perf_counter()
    solver = CrankNicolsonHeat()
    points = np.column_stack([obs.t, obs.x])

    def residuals(theta: Array) -> Array:
        update: dict[str, Any] = {"alpha": float(np.exp(theta[0]))}
        if ic_modes is not None:
            update["displacement_modes"] = [
                SineMode(n=i + 1, amplitude=float(a)) for i, a in enumerate(theta[1:])
            ]
        trial = Heat1D(problem.cfg.model_copy(update=update))
        sol = solver.solve(trial, nx=nx, nt=nt)
        interp = RegularGridInterpolator((sol.t, sol.x), sol.u)
        return np.asarray(interp(points) - obs.u)

    x0 = np.concatenate([[np.log(alpha0)], np.zeros(ic_modes or 0)])
    fit = least_squares(residuals, x0=x0, xtol=1e-10, ftol=1e-10)
    wall = time.perf_counter() - start
    return IdentResult(
        alpha_hat=float(np.exp(fit.x[0])),
        wall_time_s=wall,
        meta={
            "n_params": int(x0.size),
            "n_evals": int(fit.nfev),
            "final_cost": float(fit.cost),
            "success": bool(fit.success),
        },
    )
