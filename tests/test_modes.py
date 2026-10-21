"""Mode projection: the decomposition must be a faithful breakdown of the L2 error."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pinnaudit.config import ProblemConfig, SineMode
from pinnaudit.metrics import relative_l2
from pinnaudit.modes import load_mode_contributions, mode_amplitudes, mode_error_contributions
from pinnaudit.problems import Heat1D, Wave1D
from pinnaudit.solvers import CrankNicolsonHeat, LeapfrogWave

X = np.linspace(0.0, 1.0, 41)
T = np.linspace(0.0, 1.0, 81)


def test_amplitudes_of_the_exact_wave_solution(wave_cfg: ProblemConfig) -> None:
    """Modes 1 and 3 must oscillate as cos(pi t) and 0.5 cos(3 pi t)."""
    amps = mode_amplitudes(Wave1D(wave_cfg).exact(X, T), X, 8)
    assert np.allclose(amps[:, 0], np.cos(np.pi * T), atol=1e-12)
    assert np.allclose(amps[:, 2], 0.5 * np.cos(3 * np.pi * T), atol=1e-12)


def test_modes_absent_from_the_solution_are_empty(wave_cfg: ProblemConfig) -> None:
    amps = mode_amplitudes(Wave1D(wave_cfg).exact(X, T), X, 8)
    assert np.max(np.abs(np.delete(amps, [0, 2], axis=1))) < 1e-12


def test_wave_errors_recombine_into_relative_l2(wave_cfg: ProblemConfig) -> None:
    problem = Wave1D(wave_cfg)
    sol = LeapfrogWave().solve(problem, nx=41, nt=81)
    ref = problem.exact(sol.x, sol.t)
    parts = mode_error_contributions(sol.u, ref, sol.x, n_modes=39)
    assert np.sqrt(np.sum(parts**2)) == pytest.approx(relative_l2(sol.u, ref), rel=1e-9)


def test_heat_errors_recombine_into_relative_l2(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    sol = CrankNicolsonHeat().solve(problem, nx=41, nt=41)
    ref = problem.exact(sol.x, sol.t)
    parts = mode_error_contributions(sol.u, ref, sol.x, n_modes=39)
    assert np.sqrt(np.sum(parts**2)) == pytest.approx(relative_l2(sol.u, ref), rel=1e-9)


def test_a_missing_high_mode_is_blamed_on_that_mode(wave_cfg: ProblemConfig) -> None:
    """Keep mode 1, drop mode 3: all the error belongs to mode 3, and it is sqrt(0.2)."""
    ref = Wave1D(wave_cfg).exact(X, T)
    only_mode_1 = Wave1D(
        ProblemConfig(name="wave1d", displacement_modes=[SineMode(n=1, amplitude=1.0)])
    ).exact(X, T)
    parts = mode_error_contributions(only_mode_1, ref, X, n_modes=8)
    assert parts[2] == pytest.approx(np.sqrt(0.2), abs=1e-6)
    assert np.max(np.delete(parts, 2)) < 1e-12


def test_leakage_into_an_empty_mode_is_detected(wave_cfg: ProblemConfig) -> None:
    ref = Wave1D(wave_cfg).exact(X, T)
    leaky = ref + 0.1 * np.outer(np.ones_like(T), np.sin(2 * np.pi * X))
    parts = mode_error_contributions(leaky, ref, X, n_modes=8)
    scale = np.sqrt(np.sum(mode_amplitudes(ref, X, 8) ** 2))
    assert parts[1] == pytest.approx(0.1 * np.sqrt(T.size) / scale, rel=1e-9)
    assert np.max(np.delete(parts, 1)) < 1e-12


def test_too_many_modes_are_rejected(wave_cfg: ProblemConfig) -> None:
    with pytest.raises(ValueError, match="cannot be resolved"):
        mode_amplitudes(Wave1D(wave_cfg).exact(X, T), X, n_modes=40)


def test_a_reference_with_no_energy_is_rejected() -> None:
    zeros = np.zeros((T.size, X.size))
    with pytest.raises(ValueError, match="no energy"):
        mode_error_contributions(zeros, zeros, X, n_modes=4)


def test_contributions_can_be_loaded_from_saved_solutions(
    tmp_path: Path, wave_cfg: ProblemConfig
) -> None:
    only_mode_1 = Wave1D(
        ProblemConfig(name="wave1d", displacement_modes=[SineMode(n=1, amplitude=1.0)])
    ).exact(X, T)
    paths = []
    for i in range(2):
        path = tmp_path / f"run{i}.npz"
        np.savez_compressed(path, x=X, t=T, u=only_mode_1)
        paths.append(path)

    parts = load_mode_contributions(paths, Wave1D(wave_cfg), n_modes=8)
    assert parts.shape == (2, 8)
    assert np.allclose(parts[0], parts[1])
    assert parts[0, 2] == pytest.approx(np.sqrt(0.2), abs=1e-6)
