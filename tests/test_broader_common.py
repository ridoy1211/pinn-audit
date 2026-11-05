from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scripts import broader_common as common

from pinnaudit.spectral_inverse import SpectralFit, fit_spectral

FAMILY_A = [{"n": 1, "amplitude": 1.0}, {"n": 3, "amplitude": 0.5}]
FAMILY_B = [*FAMILY_A, {"n": 7, "amplitude": 0.25}]


def test_exact_paired_and_grid_agree_and_preserve_two_ic_families() -> None:
    x = np.linspace(0, 1, 31)
    t = np.linspace(0, 1, 17)
    for modes in (FAMILY_A, FAMILY_B):
        grid = common.exact_grid(x, t, 0.07, modes)
        xx, tt = np.meshgrid(x, t)
        paired = common.exact_paired(xx.ravel(), tt.ravel(), 0.07, modes)
        np.testing.assert_allclose(grid, paired.reshape(len(t), len(x)), atol=1e-14)
    low_modes = FAMILY_B[:2]
    full_ic = common.exact_grid(x, t[:2] * 0, 0.07, FAMILY_B)[0]
    truncated_ic = common.exact_grid(x, t[:2] * 0, 0.07, low_modes)[0]
    scored = common.score_field(np.tile(truncated_ic, (len(t), 1)), x, t, 0.07, FAMILY_B)
    assert scored["ic_relative_l2"] == pytest.approx(np.sqrt(1 / 21), rel=2e-3)
    assert np.linalg.norm(full_ic - truncated_ic) > 0


def test_score_field_rejects_zero_reference_and_nonfinite_prediction() -> None:
    x = np.linspace(0, 1, 7)
    t = np.linspace(0, 1, 5)
    zero_modes = [{"n": 1, "amplitude": 0.0}]
    with pytest.raises(ValueError, match="zero or nonfinite reference norm"):
        common.score_field(np.zeros((len(t), len(x))), x, t, 0.05, zero_modes)
    prediction = common.exact_grid(x, t, 0.05, FAMILY_A)
    prediction[1, 1] = np.nan
    with pytest.raises(ValueError, match="only finite values"):
        common.score_field(prediction, x, t, 0.05, FAMILY_A)


def test_noise_is_reused_without_new_random_draws() -> None:
    design = common.generate_design(30, 987100)
    low = common.make_dataset(design, 0.05, FAMILY_A, 0.1)
    high = common.make_dataset(design, 0.05, FAMILY_A, 0.3)
    np.testing.assert_array_equal(low["unit_noise"], high["unit_noise"])
    np.testing.assert_allclose((high["y"] - high["clean"]) / 3, low["y"] - low["clean"])
    np.testing.assert_array_equal(low["fold_labels"], design["fold_labels"])
    assert np.bincount(design["fold_labels"]).max() - np.bincount(design["fold_labels"]).min() <= 1


def test_atomic_strict_json_and_sha256(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "evidence.json"
    common.write_json(path, {"value": [1, 2]})
    assert path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in path.read_bytes()
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": [1, 2]}
    assert common.sha256(path) == __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        common.write_json(path, {"invalid": float("nan")})


def test_positive_ridge_handles_underdetermined_amplitudes() -> None:
    x = np.array([0.13, 0.47, 0.82])
    t = np.array([0.01, 0.2, 0.6])
    y = common.exact_paired(x, t, 0.04, FAMILY_A)
    fit = fit_spectral(x, t, y, modes=8, ridge=1e-4, profile_points=5)
    assert fit.amplitudes.shape == (8,)
    assert np.isfinite(fit.amplitudes).all()
    assert np.isfinite(fit.objective)


def _mock_fit(alpha: float, ridge: float) -> SpectralFit:
    return SpectralFit(
        alpha_hat=alpha,
        amplitudes=np.array([1.0]),
        objective=0.0,
        wall_time_s=0.01,
        profile_alpha=np.array([alpha]),
        profile_objective=np.array([0.0]),
        at_bound=False,
        modes=1,
        ridge=ridge,
        length=1.0,
    )


def test_cv_excludes_validation_observations_and_uses_pooled_weighted_sse(monkeypatch) -> None:
    x = np.arange(6, dtype=float) / 6
    t = np.zeros(6)
    y = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    folds = np.array([0, 0, 1, 1, 1, 1])
    seen: list[np.ndarray] = []

    def fake_fit(train_x, train_t, train_y, *, ridge, **kwargs):
        seen.append(np.asarray(train_y).copy())
        # Both folds are trained independently; prediction below supplies varied fold errors.
        alpha = float(ridge)
        return _mock_fit(alpha, float(ridge))

    def prediction(self, px, pt):
        if self.ridge == 0:
            return np.asarray(pt) * 0 + 2  # pooled SSE is 19
        return np.asarray(pt) * 0 + (1 if len(px) == 2 else 3)  # pooled SSE is 7

    monkeypatch.setattr(common, "fit_spectral", fake_fit)
    monkeypatch.setattr(SpectralFit, "predict", prediction)
    result = common.cv_spectral(
        x, t, y, folds, modes=1, ridges=[0.0, 1.0], alpha_bounds=(0.01, 1), profile_points=5
    )
    assert result.selected_index == 1
    assert result.losses == pytest.approx([19 / 6, 7 / 6])
    np.testing.assert_array_equal(seen[0], y[2:])
    np.testing.assert_array_equal(seen[1], y[:2])
    assert len(result.fold_records) == 4


def test_cv_exact_tie_selects_smaller_ridge(monkeypatch) -> None:
    x = np.linspace(0, 1, 10)
    t = np.zeros(10)
    y = np.ones(10)
    folds = np.tile(np.arange(5), 2)
    monkeypatch.setattr(common, "fit_spectral", lambda *a, ridge, **k: _mock_fit(0.1, ridge))
    monkeypatch.setattr(SpectralFit, "predict", lambda self, x, t: np.zeros_like(x))
    result = common.cv_spectral(
        x, t, y, folds, modes=1, ridges=[0.5, 0.1], alpha_bounds=(0.01, 1), profile_points=5
    )
    assert result.losses == [1.0, 1.0]
    assert result.selected_index == 1


def test_cv_failure_retains_completed_and_failed_fold_evidence(monkeypatch) -> None:
    x = np.linspace(0, 1, 10)
    t = np.zeros(10)
    y = np.ones(10)
    folds = np.tile(np.arange(5), 2)
    calls = 0

    def fake_fit(*args, ridge, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected failure")
        return _mock_fit(0.1, ridge)

    monkeypatch.setattr(common, "fit_spectral", fake_fit)
    monkeypatch.setattr(SpectralFit, "predict", lambda self, x, t: np.zeros_like(x))
    with pytest.raises(common.CVFailure) as captured:
        common.cv_spectral(
            x, t, y, folds, modes=1, ridges=[0.1, 1.0], alpha_bounds=(0.01, 1), profile_points=5
        )
    error = captured.value
    assert len(error.fold_fits[0]) == 2
    assert len(error.fold_records) == 3
    assert error.fold_records[-1]["failure_type"] == "RuntimeError"
    assert error.fold_records[-1]["candidate_index"] == 0
