"""Independent analytic checks for the post-evaluation spectral norm audit."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def review(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("report_broader_results")


def test_exact_solution_has_zero_norm_errors(review):
    modes = [{"n": 1, "amplitude": 1.0}, {"n": 3, "amplitude": 0.5}]
    result = review.exact_spectral_errors(0.05, np.array([1, 0, 0.5]), 0.05, modes)
    assert result["ic_relative_l2"] == 0
    assert result["field_relative_l2"] == pytest.approx(0, abs=1e-8)


def test_missing_seventh_mode_has_analytic_representation_floor(review):
    modes = [{"n": 1, "amplitude": 1.0}, {"n": 3, "amplitude": 0.5}, {"n": 7, "amplitude": 0.25}]
    result = review.exact_spectral_errors(0.05, np.array([1, 0, 0.5, 0, 0, 0]), 0.05, modes)
    assert result["ic_relative_l2"] == pytest.approx(np.sqrt(1 / 21))


def test_space_time_norm_matches_independent_gauss_quadrature(review):
    nodes, weights = np.polynomial.legendre.leggauss(120)
    x, t = (nodes + 1) / 2, (nodes + 1) / 2
    w = weights / 2
    estimate = np.exp(-0.08 * np.pi**2 * t[:, None]) * 1.1 * np.sin(np.pi * x)
    truth = np.exp(-0.05 * np.pi**2 * t[:, None]) * np.sin(np.pi * x)
    expected = np.sqrt(
        np.sum(w[:, None] * w * (estimate - truth) ** 2) / np.sum(w[:, None] * w * truth**2)
    )
    result = review.exact_spectral_errors(0.08, np.array([1.1]), 0.05, [{"n": 1, "amplitude": 1.0}])
    assert result["field_relative_l2"] == pytest.approx(expected, rel=1e-11)
