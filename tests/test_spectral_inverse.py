"""Independent analytic and identifiability checks for the new baseline."""

import numpy as np
import pytest

from pinnaudit.spectral_inverse import fit_spectral, heat_design, project_amplitudes


def test_recovers_diffusivity_and_unknown_initial_modes_from_clean_irregular_data():
    rng = np.random.default_rng(413)
    x, t = rng.uniform(0, 1, (2, 70))
    # Build data independently from the estimator's design-matrix implementation.
    y = np.sin(np.pi * x) * np.exp(-0.05 * np.pi**2 * t)
    y += 0.5 * np.sin(3 * np.pi * x) * np.exp(-0.05 * (3 * np.pi) ** 2 * t)
    fit = fit_spectral(x, t, y, modes=6)
    assert fit.alpha_hat == pytest.approx(0.05, rel=1e-6)
    assert fit.amplitudes == pytest.approx([1, 0, 0.5, 0, 0, 0], abs=1e-6)
    assert np.max(np.abs(fit.predict(x, t) - y)) < 1e-8
    assert not fit.at_bound


def test_one_snapshot_cannot_separate_alpha_from_free_amplitudes():
    x = np.linspace(0.02, 0.98, 30)
    t = np.full_like(x, 0.5)
    y = np.sin(np.pi * x) * np.exp(-0.05 * np.pi**2 * t)
    # For every alpha, free initial amplitude can absorb decay at a single time.
    for alpha in (0.01, 0.05, 0.2):
        _, cost = project_amplitudes(heat_design(x, t, alpha, 1.0, 1), y, ridge=0)
        assert cost < 1e-25


def test_smoothing_reduces_weighted_initial_mode_norm():
    rng = np.random.default_rng(41)
    x, t = rng.uniform(0, 1, (2, 15))
    design = heat_design(x, t, 0.05, 1.0, 6)
    y = design[:, 0] + rng.normal(0, 0.05, 15)
    a, _ = project_amplitudes(design, y, 0)
    b, _ = project_amplitudes(design, y, 0.01)
    weights = np.arange(1, 7) ** 2
    assert np.linalg.norm(weights * b) < np.linalg.norm(weights * a)


@pytest.mark.parametrize("kwargs", [{"ridge": -1}, {"modes": 0}, {"alpha_bounds": (0.1, 0.01)}])
def test_invalid_prior_settings_fail(kwargs):
    with pytest.raises(ValueError):
        fit_spectral(np.array([0.5]), np.array([0.1]), np.array([1.0]), **kwargs)


def test_nonfinite_measurements_fail():
    with pytest.raises(ValueError, match="finite"):
        fit_spectral(np.array([0.5]), np.array([0.1]), np.array([np.nan]))
