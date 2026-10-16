"""Neural solvers: autograd residuals, training behaviour, determinism.

The residual functions are checked first and on their own. If the autograd
derivatives are wrong, every PINN result is wrong, and a training run cannot
tell you that. Substituting a known exact solution makes the residual zero by
construction, which isolates the derivative code from everything else.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinnaudit.config import (  # noqa: E402
    NeuralConfig,
    ProblemConfig,
    SineMode,
    SolverConfig,
    load_run_config,
)
from pinnaudit.experiment import run_once  # noqa: E402
from pinnaudit.problems import Heat1D, Wave1D  # noqa: E402
from pinnaudit.solvers import build_solver  # noqa: E402
from pinnaudit.solvers.neural import (  # noqa: E402
    DTYPE,
    MLP,
    MLPSolver,
    PINNSolver,
    heat_residual,
    wave_residual,
)


def _points(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return (
        torch.rand(n, 1, generator=gen, dtype=DTYPE),
        torch.rand(n, 1, generator=gen, dtype=DTYPE),
    )


def _tiny(**overrides: float) -> NeuralConfig:
    settings: dict[str, float] = {
        "width": 8,
        "depth": 2,
        "epochs": 150,
        "lr": 0.005,
        "n_collocation": 64,
        "n_initial": 20,
        "n_boundary": 20,
        "log_every": 10,
    }
    settings.update(overrides)
    return NeuralConfig(**settings)  # type: ignore[arg-type]


# ------------------------------------------------------------ residuals -----


def test_heat_residual_vanishes_on_the_exact_solution(heat_cfg: ProblemConfig) -> None:
    def exact(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for mode in heat_cfg.displacement_modes:
            k = mode.n * math.pi / heat_cfg.x_max
            out = out + mode.amplitude * torch.sin(k * x) * torch.exp(-heat_cfg.alpha * k * k * t)
        return out

    x, t = _points(200)
    residual = heat_residual(exact, x, t, heat_cfg.alpha)
    assert residual.abs().max().item() < 1e-10


def test_wave_residual_vanishes_on_the_exact_solution() -> None:
    cfg = ProblemConfig(name="wave1d", velocity_modes=[SineMode(n=2, amplitude=0.3)])

    def exact(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for mode in cfg.displacement_modes:
            k = mode.n * math.pi / cfg.x_max
            out = out + mode.amplitude * torch.cos(k * cfg.c * t) * torch.sin(k * x)
        for mode in cfg.velocity_modes:
            k = mode.n * math.pi / cfg.x_max
            out = out + mode.amplitude / (k * cfg.c) * torch.sin(k * cfg.c * t) * torch.sin(k * x)
        return out

    x, t = _points(200)
    residual = wave_residual(exact, x, t, cfg.c)
    assert residual.abs().max().item() < 1e-9


def test_heat_residual_has_the_right_sign_and_size() -> None:
    """u = sin(pi x), constant in time, so u_t = 0 and u_xx = -pi^2 sin(pi x).

    At x = 0.5 the residual u_t - alpha u_xx must therefore be alpha * pi^2.
    The 0 * t * t term only makes u depend on t, which autograd requires.
    """

    def wrong(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * x) + 0.0 * t * t

    x = torch.tensor([[0.5]], dtype=DTYPE)
    t = torch.tensor([[0.3]], dtype=DTYPE)
    residual = heat_residual(wrong, x, t, alpha=0.05)
    assert residual.item() == pytest.approx(0.05 * math.pi**2, abs=1e-10)


def test_wave_residual_has_the_right_sign_and_size() -> None:
    def wrong(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(math.pi * x) + 0.0 * t * t

    x = torch.tensor([[0.5]], dtype=DTYPE)
    t = torch.tensor([[0.3]], dtype=DTYPE)
    residual = wave_residual(wrong, x, t, c=2.0)
    # u_tt = 0 and u_xx = -pi^2, so u_tt - c^2 u_xx = c^2 pi^2
    assert residual.item() == pytest.approx(4.0 * math.pi**2, abs=1e-9)


def test_residual_loss_backpropagates_to_the_weights() -> None:
    torch.manual_seed(0)
    model = MLP(width=8, depth=2, x_range=(0.0, 1.0), t_range=(0.0, 1.0)).to(DTYPE)
    x, t = _points(32)
    loss = (heat_residual(model, x, t, 0.05) ** 2).mean()
    loss.backward()
    grad = model.net[0].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


# -------------------------------------------------------------- network -----


def test_network_parameter_count() -> None:
    model = MLP(width=8, depth=2, x_range=(0.0, 1.0), t_range=(0.0, 1.0))
    # (2*8 + 8) + (8*8 + 8) + (8*1 + 1)
    assert sum(p.numel() for p in model.parameters()) == 105


def test_network_maps_points_to_a_column() -> None:
    model = MLP(width=8, depth=2, x_range=(0.0, 1.0), t_range=(0.0, 1.0)).to(DTYPE)
    x, t = _points(17)
    assert model(x, t).shape == (17, 1)


# ------------------------------------------------------------- training -----


def test_pinn_training_reduces_the_loss(heat_cfg: ProblemConfig) -> None:
    sol = PINNSolver(_tiny(), seed=0).solve(Heat1D(heat_cfg), nx=11, nt=11)
    assert sol.meta["final_loss"] < sol.meta["initial_loss"]
    assert "final_loss_pde" in sol.meta


def test_pinn_runs_on_the_wave_problem(wave_cfg: ProblemConfig) -> None:
    """Exercises the second time derivative and the initial-velocity term."""
    sol = PINNSolver(_tiny(), seed=0).solve(Wave1D(wave_cfg), nx=11, nt=21)
    assert sol.meta["final_loss"] < sol.meta["initial_loss"]
    assert sol.u.shape == (21, 11)


def test_mlp_control_has_no_physics_term(heat_cfg: ProblemConfig) -> None:
    sol = MLPSolver(_tiny(n_obs=50), seed=0).solve(Heat1D(heat_cfg), nx=11, nt=11)
    assert "final_loss_pde" not in sol.meta
    assert "final_loss_data" in sol.meta
    assert sol.meta["final_loss"] < sol.meta["initial_loss"]


def test_pinn_without_observations_has_no_data_term(heat_cfg: ProblemConfig) -> None:
    sol = PINNSolver(_tiny(), seed=0).solve(Heat1D(heat_cfg), nx=11, nt=11)
    assert "final_loss_data" not in sol.meta


def test_same_seed_gives_the_same_solution(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    a = PINNSolver(_tiny(epochs=40), seed=3).solve(problem, nx=11, nt=11)
    b = PINNSolver(_tiny(epochs=40), seed=3).solve(problem, nx=11, nt=11)
    assert np.allclose(a.u, b.u, atol=1e-12)


def test_different_seeds_give_different_solutions(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    a = PINNSolver(_tiny(epochs=40), seed=1).solve(problem, nx=11, nt=11)
    b = PINNSolver(_tiny(epochs=40), seed=2).solve(problem, nx=11, nt=11)
    assert not np.allclose(a.u, b.u, atol=1e-6)


def test_pinn_and_mlp_see_the_same_observations(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    cfg = _tiny(n_obs=30)
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    a = PINNSolver(cfg, seed=5)._build_data(problem, rng_a)
    b = MLPSolver(cfg, seed=5)._build_data(problem, rng_b)
    assert a.u_obs is not None
    assert b.u_obs is not None
    assert torch.equal(a.u_obs, b.u_obs)


def test_observations_come_from_the_exact_solution(heat_cfg: ProblemConfig) -> None:
    problem = Heat1D(heat_cfg)
    data = PINNSolver(_tiny(n_obs=25), seed=0)._build_data(problem, np.random.default_rng(0))
    assert data.x_obs is not None
    assert data.t_obs is not None
    assert data.u_obs is not None
    for i in range(25):
        x_i, t_i = data.x_obs[i, 0].item(), data.t_obs[i, 0].item()
        expected = problem.exact(np.array([x_i]), np.array([t_i]))[0, 0]
        assert data.u_obs[i, 0].item() == pytest.approx(expected, abs=1e-12)


# ------------------------------------------------------------ interface -----


def test_solution_follows_the_solver_contract(heat_cfg: ProblemConfig) -> None:
    sol = PINNSolver(_tiny(), seed=0).solve(Heat1D(heat_cfg), nx=13, nt=9)
    assert sol.u.shape == (9, 13)
    assert sol.n_dof == 105
    assert np.all(np.isfinite(sol.u))
    assert sol.wall_time_s == pytest.approx(sol.meta["train_time_s"] + sol.meta["inference_time_s"])


def test_solve_restores_the_torch_thread_count(heat_cfg: ProblemConfig) -> None:
    before = torch.get_num_threads()
    PINNSolver(_tiny(epochs=5), seed=0).solve(Heat1D(heat_cfg), nx=5, nt=5)
    assert torch.get_num_threads() == before


def test_build_solver_returns_the_right_class() -> None:
    pinn = build_solver(SolverConfig(name="pinn", nx=21, nt=21), seed=0)
    mlp_cfg = SolverConfig(name="mlp", nx=21, nt=21, neural=NeuralConfig(n_obs=10))
    mlp = build_solver(mlp_cfg, seed=0)
    assert isinstance(pinn, PINNSolver)
    assert isinstance(mlp, MLPSolver)


def test_pinn_smoke_config_runs_end_to_end() -> None:
    result = run_once(load_run_config("configs/pinn_smoke.yaml"))
    assert math.isfinite(result.metrics["relative_l2"])
    assert result.solution.solver_name == "pinn"
