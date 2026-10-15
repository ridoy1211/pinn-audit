"""Solver correctness: convergence order, stability, and boundary handling.

A solver that merely "looks about right" is useless as a baseline. Each scheme
is required to hit its theoretical order of accuracy, because that is what
justifies calling it a strong baseline when a PINN loses to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinnaudit.config import ProblemConfig
from pinnaudit.metrics import observed_order, relative_l2
from pinnaudit.problems import Heat1D, Wave1D
from pinnaudit.solvers import CrankNicolsonHeat, LeapfrogWave

# ----------------------------------------------------------------- heat -----


def _heat_errors(cfg: ProblemConfig, nxs: list[int]) -> list[float]:
    p = Heat1D(cfg)
    errors = []
    for nx in nxs:
        sol = CrankNicolsonHeat().solve(p, nx=nx, nt=nx)  # k = h when T = L = 1
        errors.append(relative_l2(sol.u, p.exact(sol.x, sol.t)))
    return errors


def test_crank_nicolson_is_second_order(heat_cfg: ProblemConfig) -> None:
    nxs = [21, 41, 81, 161]
    orders = observed_order(_heat_errors(heat_cfg, nxs))
    # Measured: 1.995, 1.994, 1.996.
    assert orders[-1] == pytest.approx(2.0, abs=0.1)
    assert all(o > 1.85 for o in orders)


def test_crank_nicolson_accuracy_at_reference_resolution(heat_cfg: ProblemConfig) -> None:
    """Locks in the number so a future refactor cannot quietly degrade it."""
    p = Heat1D(heat_cfg)
    sol = CrankNicolsonHeat().solve(p, nx=41, nt=41)
    assert relative_l2(sol.u, p.exact(sol.x, sol.t)) < 1e-3


def test_crank_nicolson_stable_at_large_mesh_ratio(heat_cfg: ProblemConfig) -> None:
    """r >> 1/2 would blow up an explicit scheme; Crank-Nicolson must not."""
    p = Heat1D(heat_cfg)
    sol = CrankNicolsonHeat().solve(p, nx=201, nt=11)
    assert sol.meta["mesh_ratio_r"] > 100.0
    assert np.all(np.isfinite(sol.u))
    assert relative_l2(sol.u, p.exact(sol.x, sol.t)) < 0.5


def test_solver_preserves_dirichlet_boundaries(heat_cfg: ProblemConfig) -> None:
    sol = CrankNicolsonHeat().solve(Heat1D(heat_cfg), nx=41, nt=41)
    assert np.allclose(sol.u[:, 0], 0.0)
    assert np.allclose(sol.u[:, -1], 0.0)


def test_solution_shape_contract(heat_cfg: ProblemConfig) -> None:
    sol = CrankNicolsonHeat().solve(Heat1D(heat_cfg), nx=31, nt=17)
    assert sol.u.shape == (17, 31)
    assert sol.wall_time_s > 0.0


# ----------------------------------------------------------------- wave -----


def _nt_for_courant(nx: int, lam: float, c: float, t_max: float, x_max: float) -> int:
    h = x_max / (nx - 1)
    k = lam * h / c
    return int(round(t_max / k)) + 1


def test_leapfrog_is_second_order(wave_cfg: ProblemConfig) -> None:
    """Courant number held fixed at 0.5 so the measured order is genuine."""
    p = Wave1D(wave_cfg)
    errors = []
    for nx in [21, 41, 81, 161]:
        nt = _nt_for_courant(nx, 0.5, wave_cfg.c, wave_cfg.t_max, wave_cfg.x_max)
        sol = LeapfrogWave().solve(p, nx=nx, nt=nt)
        assert sol.meta["courant_lambda"] == pytest.approx(0.5, abs=1e-9)
        errors.append(relative_l2(sol.u, p.exact(sol.x, sol.t)))
    orders = observed_order(errors)
    # Measured: 1.992, 1.993, 1.996.
    assert orders[-1] == pytest.approx(2.0, abs=0.1)


def test_leapfrog_is_exact_at_unit_courant(wave_cfg: ProblemConfig) -> None:
    """At lam = 1 the stencil reproduces d'Alembert exactly on the grid.

    This is a real property of the scheme, not a bug, and it is the reason the
    convergence study above pins lam to 0.5. Documenting it as a test stops a
    future reader from "fixing" the config.
    """
    p = Wave1D(wave_cfg)
    nt = _nt_for_courant(41, 1.0, wave_cfg.c, wave_cfg.t_max, wave_cfg.x_max)
    sol = LeapfrogWave().solve(p, nx=41, nt=nt)
    # Measured: 4.1e-16.
    assert relative_l2(sol.u, p.exact(sol.x, sol.t)) < 1e-12


def test_leapfrog_rejects_cfl_violation(wave_cfg: ProblemConfig) -> None:
    with pytest.raises(ValueError, match="CFL violated"):
        LeapfrogWave().solve(Wave1D(wave_cfg), nx=81, nt=41)


def test_leapfrog_conserves_energy_approximately(wave_cfg: ProblemConfig) -> None:
    """The wave equation has no dissipation; a stable scheme must not damp much."""
    p = Wave1D(wave_cfg)
    nt = _nt_for_courant(81, 0.5, wave_cfg.c, wave_cfg.t_max, wave_cfg.x_max)
    sol = LeapfrogWave().solve(p, nx=81, nt=nt)
    amplitude = np.max(np.abs(sol.u), axis=1)
    assert amplitude[-1] > 0.9 * amplitude[0]


def test_solvers_reject_the_wrong_problem(heat_cfg: ProblemConfig, wave_cfg: ProblemConfig) -> None:
    with pytest.raises(TypeError, match="solves heat1d"):
        CrankNicolsonHeat().solve(Wave1D(wave_cfg), nx=21, nt=21)
    with pytest.raises(TypeError, match="solves wave1d"):
        LeapfrogWave().solve(Heat1D(heat_cfg), nx=21, nt=41)
