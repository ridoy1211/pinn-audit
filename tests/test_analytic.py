"""The analytic solutions are this project's ground truth.

If they are wrong, every number in the audit is wrong, so they are checked
independently: by substituting them back into their own PDE with high-order
finite differences, and by verifying initial and boundary conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from pinnaudit.config import ProblemConfig, SineMode
from pinnaudit.problems import Heat1D, Wave1D, build_problem


def test_heat_exact_satisfies_its_pde(heat_cfg: ProblemConfig) -> None:
    p = Heat1D(heat_cfg)
    x = np.linspace(0.0, 1.0, 2001)
    t = np.linspace(0.0, 1.0, 2001)
    u = p.exact(x, t)
    h, k = x[1] - x[0], t[1] - t[0]

    u_t = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * k)
    u_xx = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / h**2
    residual = np.max(np.abs(u_t - heat_cfg.alpha * u_xx))

    # Verified at 5.98e-06; the bound is the finite-difference truncation error,
    # not the analytic solution's error, so it is deliberately loose.
    assert residual < 1e-4


def test_wave_exact_satisfies_its_pde(wave_cfg: ProblemConfig) -> None:
    p = Wave1D(wave_cfg)
    x = np.linspace(0.0, 1.0, 2001)
    t = np.linspace(0.0, 1.0, 2001)
    u = p.exact(x, t)
    h, k = x[1] - x[0], t[1] - t[0]

    u_tt = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / k**2
    u_xx = (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / h**2
    assert np.max(np.abs(u_tt - wave_cfg.c**2 * u_xx)) < 1e-6


@pytest.mark.parametrize("name", ["heat1d", "wave1d"])
def test_boundaries_are_homogeneous(name: str) -> None:
    p = build_problem(ProblemConfig(name=name))  # type: ignore[arg-type]
    x = np.linspace(0.0, 1.0, 101)
    t = np.linspace(0.0, 1.0, 101)
    u = p.exact(x, t)
    assert np.allclose(u[:, 0], 0.0, atol=1e-14)
    assert np.allclose(u[:, -1], 0.0, atol=1e-14)


def test_heat_initial_condition_matches_requested_modes(heat_cfg: ProblemConfig) -> None:
    p = Heat1D(heat_cfg)
    x = np.linspace(0.0, 1.0, 257)
    expected = np.sin(np.pi * x) + 0.5 * np.sin(3 * np.pi * x)
    assert np.allclose(p.initial(x), expected, atol=1e-14)


def test_heat_decays_monotonically_in_energy(heat_cfg: ProblemConfig) -> None:
    """Diffusion can only destroy L2 energy; a sign error would show up here."""
    p = Heat1D(heat_cfg)
    x = np.linspace(0.0, 1.0, 401)
    t = np.linspace(0.0, 1.0, 101)
    energy = np.sum(p.exact(x, t) ** 2, axis=1)
    assert np.all(np.diff(energy) < 0.0)


def test_wave_initial_velocity_matches_centred_difference() -> None:
    """The series is valid for t < 0, so a centred difference gives O(eps^2)."""
    cfg = ProblemConfig(name="wave1d", velocity_modes=[SineMode(n=2, amplitude=0.3)])
    p = Wave1D(cfg)
    x = np.linspace(0.0, 1.0, 401)

    eps = 1e-4
    u = p.exact(x, np.array([-eps, eps]))
    numeric = (u[1] - u[0]) / (2 * eps)
    # Verified at 1.97e-08.
    assert np.allclose(numeric, p.initial_velocity(x), atol=1e-6)


def test_wave_is_time_periodic() -> None:
    """A standing wave of speed c on [0, L] has period 2L/c."""
    cfg = ProblemConfig(name="wave1d", x_max=1.0, c=1.0, t_max=2.0)
    p = Wave1D(cfg)
    x = np.linspace(0.0, 1.0, 201)
    u = p.exact(x, np.array([0.0, 2.0]))
    assert np.allclose(u[0], u[1], atol=1e-12)


def test_problem_rejects_mismatched_config(wave_cfg: ProblemConfig) -> None:
    with pytest.raises(ValueError, match="received config for problem"):
        Heat1D(wave_cfg)
