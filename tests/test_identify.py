"""The inverse data model and the classical estimator (numpy and scipy only)."""

from __future__ import annotations

import numpy as np
import pytest

from pinnaudit.config import ProblemConfig
from pinnaudit.identify import fit_lsq, make_observations
from pinnaudit.problems import Heat1D

TRUE_ALPHA = 0.05


def _median_error(
    problem: Heat1D, n_obs: int, noise: float, seeds: range, modes: int | None
) -> float:
    errors = []
    for seed in seeds:
        obs = make_observations(problem, n_obs, noise, seed)
        alpha_hat = fit_lsq(problem, obs, 0.2, ic_modes=modes).alpha_hat
        errors.append(abs(alpha_hat - TRUE_ALPHA) / TRUE_ALPHA)
    return float(np.median(errors))


# ------------------------------------------------------------- data model ---


def test_noise_free_observations_equal_the_exact_solution(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    obs = make_observations(problem, 30, 0.0, seed=1)
    assert np.array_equal(obs.u, obs.clean)
    for x, t, u in zip(obs.x[:5], obs.t[:5], obs.u[:5], strict=True):
        assert u == pytest.approx(problem.exact(np.array([x]), np.array([t]))[0, 0], abs=1e-14)


def test_noise_is_relative_to_the_signal(heat_cfg: ProblemConfig) -> None:
    obs = make_observations(Heat1D(heat_cfg), 50, 0.05, seed=3)
    assert obs.sigma == pytest.approx(0.05 * np.std(obs.clean))


def test_noise_levels_share_positions_and_the_same_noise_draw(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    low, high = make_observations(problem, 50, 0.05, 3), make_observations(problem, 50, 0.10, 3)
    assert np.array_equal(low.x, high.x)
    assert np.array_equal(low.t, high.t)
    assert np.allclose((high.u - high.clean) / high.sigma, (low.u - low.clean) / low.sigma)


def test_observations_are_deterministic_and_depend_on_the_seed(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    a = make_observations(problem, 20, 0.05, seed=7)
    assert np.array_equal(a.u, make_observations(problem, 20, 0.05, seed=7).u)
    assert not np.array_equal(a.u, make_observations(problem, 20, 0.05, seed=8).u)


def test_observations_reject_nonsense(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    with pytest.raises(ValueError, match="n_obs"):
        make_observations(problem, 0, 0.0, 0)
    with pytest.raises(ValueError, match="noise"):
        make_observations(problem, 5, -0.1, 0)


# ---------------------------------------------------- classical estimator ---


def test_noise_free_fit_recovers_alpha(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    result = fit_lsq(problem, make_observations(problem, 50, 0.0, 0), alpha0=0.2)
    assert result.meta["success"]
    assert abs(result.alpha_hat - TRUE_ALPHA) / TRUE_ALPHA < 1e-3  # measured 6e-5


def test_fit_does_not_depend_on_the_starting_guess(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    obs = make_observations(problem, 50, 0.05, 0)
    far = fit_lsq(problem, obs, alpha0=0.2).alpha_hat
    near = fit_lsq(problem, obs, alpha0=0.01).alpha_hat
    assert far == pytest.approx(near, rel=1e-4)


def test_fit_leaves_the_problem_untouched(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    fit_lsq(problem, make_observations(problem, 20, 0.0, 0), alpha0=0.2, ic_modes=4)
    assert problem.cfg.alpha == TRUE_ALPHA
    assert [m.n for m in problem.cfg.displacement_modes] == [1, 3]


def test_error_grows_with_noise(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    low = _median_error(problem, 50, 0.05, range(5), None)  # measured 0.008
    high = _median_error(problem, 50, 0.10, range(5), None)  # measured 0.016
    assert 0.0 < low < high


def test_more_observations_help(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    few = _median_error(problem, 10, 0.05, range(5), None)  # measured 0.011
    many = _median_error(problem, 200, 0.05, range(5), None)  # measured 0.004
    assert many < few


def test_unknown_initial_condition_is_recovered_from_clean_data(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    result = fit_lsq(problem, make_observations(problem, 50, 0.0, 0), 0.2, ic_modes=6)
    assert result.meta["n_params"] == 7
    assert abs(result.alpha_hat - TRUE_ALPHA) / TRUE_ALPHA < 1e-3  # measured 4e-5


def test_sparse_noisy_data_breaks_the_unknown_ic_fit(heat_cfg: ProblemConfig) -> None:
    """Seven parameters against ten noisy points is badly underdetermined."""
    problem = Heat1D(heat_cfg)
    sparse = _median_error(problem, 10, 0.05, range(3), 6)  # measured 0.25
    dense = _median_error(problem, 200, 0.05, range(3), 6)  # measured 0.008
    assert sparse > 5 * dense


def test_fit_rejects_bad_settings(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    obs = make_observations(problem, 10, 0.0, 0)
    with pytest.raises(ValueError, match="alpha0"):
        fit_lsq(problem, obs, alpha0=0.0)
    with pytest.raises(ValueError, match="ic_modes"):
        fit_lsq(problem, obs, alpha0=0.2, ic_modes=0)
