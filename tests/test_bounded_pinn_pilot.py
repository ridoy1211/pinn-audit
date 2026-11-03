"""Physical constraints, parameter learning, and evidence retention for the pilot."""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("pilot_bounded_pinn")


@pytest.fixture
def settings():
    values = json.loads(Path("configs/inverse/bounded_pinn_pilot.json").read_text())
    values.update(
        width=8,
        depth=2,
        adam_steps=2,
        lbfgs_iters=2,
        lbfgs_max_eval=5,
        trace_every=1,
        max_train_seconds=60,
    )
    return values


def test_bounds_initialization_and_exact_boundary(pilot, settings):
    model = pilot.BoundedHeatField(settings)
    assert float(model.alpha().detach()) == pytest.approx(0.2)
    t = pilot.column(np.linspace(0, 1, 9))
    for edge in (0.0, 1.0):
        assert torch.count_nonzero(model(torch.full_like(t, edge), t)) == 0
    for raw in (-1000.0, 1000.0):
        with torch.no_grad():
            model.raw_alpha.fill_(raw)
        assert 1e-5 * (1 - 1e-12) <= float(model.alpha().detach()) <= 1.0


def test_residual_loss_backpropagates_to_bounded_alpha(pilot, settings):
    torch.manual_seed(7)
    model = pilot.BoundedHeatField(settings)
    x = pilot.column([0.2, 0.5, 0.8])
    t = pilot.column([0.1, 0.4, 0.7])
    loss = pilot.heat_residual(model, x, t, model.alpha()).square().mean()
    loss.backward()
    assert torch.isfinite(model.raw_alpha.grad)
    assert model.raw_alpha.grad.abs().item() > 1e-12


def test_training_preserves_reloadable_state_and_settings(pilot, settings, tmp_path):
    x, t, y = np.array([0.2, 0.5, 0.8]), np.array([0.1, 0.4, 0.7]), np.array([0.4, 0.3, 0.2])
    points = np.random.default_rng(7).uniform(size=(12, 2))
    previous = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    model, trace, elapsed = pilot.train_one(
        x, t, y, points, settings, 3, tmp_path / "job", echo=lambda _: None
    )
    assert torch.get_num_threads() == previous
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert trace[-1]["stage"] == "completed_budget"
    assert elapsed > 0
    state = torch.load(tmp_path / "job/checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["adam_steps_completed"] == 2
    assert state["lbfgs_closure_calls"] > 0
    restored = pilot.BoundedHeatField(state["settings"])
    restored.load_state_dict(state["model_state"])
    np.testing.assert_allclose(
        pilot.predict_paired(model, x, t), pilot.predict_paired(restored, x, t)
    )


def test_timeout_preserves_periodic_and_failed_checkpoints(pilot, settings, tmp_path):
    settings["max_train_seconds"] = -1
    with pytest.raises(TimeoutError):
        pilot.train_one(
            np.array([0.5]),
            np.array([0.5]),
            np.array([0.2]),
            np.array([[0.3, 0.4]]),
            settings,
            3,
            tmp_path / "job",
            echo=lambda _: None,
        )
    assert (tmp_path / "job/checkpoint.pt").is_file()
    assert (tmp_path / "job/failed_checkpoint.pt").is_file()
    state = torch.load(tmp_path / "job/failed_checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["status"] == "timeout"


def test_timeout_during_lbfgs_records_partial_iterations(pilot, settings, tmp_path, monkeypatch):
    class InterruptedLBFGS:
        def __init__(self, parameters, **kwargs):
            self.state = {parameters[0]: {"n_iter": 3}}

        def step(self, closure):
            raise TimeoutError("Interrupted during L-BFGS")

    monkeypatch.setattr(torch.optim, "LBFGS", InterruptedLBFGS)
    with pytest.raises(TimeoutError):
        pilot.train_one(
            np.array([0.5]),
            np.array([0.5]),
            np.array([0.2]),
            np.array([[0.3, 0.4]]),
            settings,
            3,
            tmp_path / "job",
            echo=lambda _: None,
        )
    state = torch.load(tmp_path / "job/failed_checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["status"] == "timeout"
    assert state["lbfgs_iterations"] == 3
