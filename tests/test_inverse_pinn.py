"""The PINN estimator. These tests need PyTorch and are skipped without it."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinnaudit.config import NeuralConfig, ProblemConfig  # noqa: E402
from pinnaudit.identify import make_observations  # noqa: E402
from pinnaudit.inverse_pinn import fit_pinn  # noqa: E402
from pinnaudit.problems import Heat1D  # noqa: E402
from pinnaudit.solvers.neural import DTYPE, heat_residual  # noqa: E402


def _tiny(**overrides: float) -> NeuralConfig:
    settings: dict[str, float] = {
        "width": 8,
        "depth": 2,
        "epochs": 60,
        "lr": 0.005,
        "n_collocation": 64,
        "n_initial": 20,
        "n_boundary": 20,
        "w_data": 10.0,
        "log_every": 10,
    }
    settings.update(overrides)
    return NeuralConfig(**settings)  # type: ignore[arg-type]


def _exact_heat(cfg: ProblemConfig):  # type: ignore[no-untyped-def]
    def exact(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for mode in cfg.displacement_modes:
            k = mode.n * math.pi / cfg.x_max
            out = out + mode.amplitude * torch.sin(k * x) * torch.exp(-cfg.alpha * k * k * t)
        return out

    return exact


def _points(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    return (
        torch.rand(n, 1, generator=gen, dtype=DTYPE),
        torch.rand(n, 1, generator=gen, dtype=DTYPE),
    )


# ------------------------------------------------- learnable diffusivity -----


def test_residual_accepts_a_tensor_alpha_and_still_vanishes(heat_cfg: ProblemConfig) -> None:
    x, t = _points(100)
    alpha = torch.tensor(heat_cfg.alpha, dtype=DTYPE)
    residual = heat_residual(_exact_heat(heat_cfg), x, t, alpha)
    assert residual.abs().max().item() < 1e-10


def test_residual_loss_has_a_gradient_with_respect_to_alpha(heat_cfg: ProblemConfig) -> None:
    x, t = _points(100)
    alpha = torch.tensor(0.06, dtype=DTYPE, requires_grad=True)  # deliberately wrong
    loss = (heat_residual(_exact_heat(heat_cfg), x, t, alpha) ** 2).mean()
    (grad,) = torch.autograd.grad(loss, alpha)
    assert torch.isfinite(grad)
    assert grad.abs().item() > 0.0


# ---------------------------------------------------------- the estimator ---


def test_fit_returns_a_finite_positive_alpha_and_lowers_the_loss(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    result = fit_pinn(problem, make_observations(problem, 20, 0.05, 0), 0.2, _tiny(), seed=0)
    assert math.isfinite(result.alpha_hat)
    assert result.alpha_hat > 0.0
    assert result.meta["final_loss"] < result.meta["initial_loss"]
    assert "final_loss_ic" in result.meta


def test_fit_is_deterministic(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    obs = make_observations(problem, 20, 0.05, 0)
    a = fit_pinn(problem, obs, 0.2, _tiny(epochs=30), seed=2).alpha_hat
    b = fit_pinn(problem, obs, 0.2, _tiny(epochs=30), seed=2).alpha_hat
    assert a == pytest.approx(b, abs=1e-12)


def test_unknown_ic_drops_the_initial_condition_term(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    obs = make_observations(problem, 20, 0.0, 0)
    result = fit_pinn(problem, obs, 0.2, _tiny(w_ic=0.0), seed=0, ic_known=False)
    assert "final_loss_ic" not in result.meta
    assert np.isfinite(result.alpha_hat)


def test_lbfgs_phase_never_makes_the_loss_worse(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    obs = make_observations(problem, 20, 0.05, 0)
    adam_only = fit_pinn(problem, obs, 0.2, _tiny(epochs=30), seed=4)
    with_lbfgs = fit_pinn(problem, obs, 0.2, _tiny(epochs=30, lbfgs_iters=10), seed=4)
    assert with_lbfgs.meta["final_loss"] <= adam_only.meta["final_loss"] * (1 + 1e-9)


def test_fit_restores_the_thread_count(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    before = torch.get_num_threads()
    fit_pinn(problem, make_observations(problem, 10, 0.0, 0), 0.2, _tiny(epochs=3), seed=0)
    assert torch.get_num_threads() == before


def test_fit_rejects_a_nonpositive_starting_guess(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    with pytest.raises(ValueError, match="alpha0"):
        fit_pinn(problem, make_observations(problem, 10, 0.0, 0), 0.0, _tiny(), seed=0)
