"""Guard validation exclusion and data-only loss selection in the study script."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def cv_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("development_cv")


def install_spy(cv_module, monkeypatch):
    calls = []

    def fake_fit(x, t, y, *, ridge, **kwargs):
        train_ids = set(x.tolist())
        np.testing.assert_array_equal(t, x + 100)
        calls.append(train_ids)

        def predict(held_x, held_t):
            assert train_ids.isdisjoint(held_x.tolist())
            assert len(train_ids) + len(held_x) == 10
            np.testing.assert_array_equal(held_t, held_x + 100)
            return np.full(held_x.shape, ridge)

        return SimpleNamespace(predict=predict, alpha_hat=0.1, at_bound=False, wall_time_s=0.0)

    monkeypatch.setattr(cv_module, "fit_spectral", fake_fit)
    return calls


def test_selection_excludes_validation_and_uses_observed_errors(cv_module, monkeypatch):
    calls = install_spy(cv_module, monkeypatch)
    x = np.arange(10, dtype=float)
    selected, losses, _, details = cv_module.cross_validate(
        x, x + 100, np.full(10, 2.0), cv_module.balanced_folds(10, 17), [0.0, 2.0, 4.0]
    )
    assert selected == 1
    np.testing.assert_allclose(losses, [4, 0, 4])
    assert len(calls) == len(details) == 15
    assert all(len(ids) == 8 for ids in calls)


def test_exact_tie_selects_smaller_lambda_regardless_of_order(cv_module, monkeypatch):
    install_spy(cv_module, monkeypatch)
    x = np.arange(10, dtype=float)
    selected, losses, _, _ = cv_module.cross_validate(
        x, x + 100, np.full(10, 2.0), cv_module.balanced_folds(10, 17), [4.0, 0.0]
    )
    np.testing.assert_allclose(losses, [4, 4])
    assert selected == 1


def test_balanced_folds_partition_all_observations_reproducibly(cv_module):
    first = cv_module.balanced_folds(13, 17)
    np.testing.assert_array_equal(first, cv_module.balanced_folds(13, 17))
    counts = np.bincount(first)
    assert len(counts) == 5
    assert counts.sum() == 13
    assert counts.max() - counts.min() == 1


def test_missing_validation_fold_is_rejected(cv_module):
    x = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="All five"):
        cv_module.cross_validate(x, x, x, np.zeros(10, dtype=int), [0.0])
