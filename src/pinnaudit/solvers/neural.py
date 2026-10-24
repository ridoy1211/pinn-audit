"""Neural solvers: a physics-informed network and a data-only control.

Both use the same network, the same training loop, the same initial and
boundary terms, and the same ``Solver`` interface as the classical schemes, so
they are scored by the same metrics on the same grid. The only difference is
the loss:

* ``pinn`` adds the PDE residual, computed with automatic differentiation.
* ``mlp`` drops the residual and fits ``n_obs`` observations of the exact
  solution instead. It is the control that shows what the physics term buys.

Everything runs in float64 on the CPU. Small PINNs are sensitive to precision,
and the autograd residual tests need it to be checked tightly.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from pinnaudit.config import NeuralConfig
from pinnaudit.problems.base import Array, Problem
from pinnaudit.problems.heat import Heat1D
from pinnaudit.problems.wave import Wave1D
from pinnaudit.solvers.base import Solution, Solver

log = logging.getLogger(__name__)

DTYPE = torch.float64

UFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
LossFn = Callable[[], tuple[torch.Tensor, dict[str, torch.Tensor]]]


# ---------------------------------------------------------------- network ---


class MLP(nn.Module):
    """Fully connected tanh network u(x, t), with inputs scaled to [-1, 1]."""

    def __init__(
        self,
        width: int,
        depth: int,
        x_range: tuple[float, float],
        t_range: tuple[float, float],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.Tanh())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.x_range = x_range
        self.t_range = t_range

        gain = nn.init.calculate_gain("tanh")
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0, x1 = self.x_range
        t0, t1 = self.t_range
        xs = 2.0 * (x - x0) / (x1 - x0) - 1.0
        ts = 2.0 * (t - t0) / (t1 - t0) - 1.0
        return self.net(torch.cat([xs, ts], dim=1))


# -------------------------------------------------------------- residuals ---


def _d(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx by autograd, keeping the graph so it can be differentiated again."""
    (grad,) = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)
    return grad


def heat_residual(
    u_fn: UFn, x: torch.Tensor, t: torch.Tensor, alpha: float | torch.Tensor
) -> torch.Tensor:
    """u_t - alpha * u_xx at the points (x, t); shapes are (N, 1).

    ``alpha`` may be a tensor, so it can be a learnable parameter (inverse problems).
    """
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    u = u_fn(x, t)
    u_t = _d(u, t)
    u_xx = _d(_d(u, x), x)
    return u_t - alpha * u_xx


def wave_residual(u_fn: UFn, x: torch.Tensor, t: torch.Tensor, c: float) -> torch.Tensor:
    """u_tt - c^2 * u_xx at the points (x, t); shapes are (N, 1)."""
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    u = u_fn(x, t)
    u_tt = _d(_d(u, t), t)
    u_xx = _d(_d(u, x), x)
    return u_tt - c * c * u_xx


def pde_residual(problem: Problem, u_fn: UFn, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if isinstance(problem, Heat1D):
        return heat_residual(u_fn, x, t, problem.cfg.alpha)
    if isinstance(problem, Wave1D):
        return wave_residual(u_fn, x, t, problem.cfg.c)
    raise TypeError(f"no PDE residual for {type(problem).__name__}")


# ------------------------------------------------------------------- data ---


def _col(values: Array) -> torch.Tensor:
    """numpy vector -> (N, 1) float64 tensor."""
    return torch.tensor(np.asarray(values, dtype=float).reshape(-1, 1), dtype=DTYPE)


@dataclass(frozen=True)
class TrainingData:
    x_ic: torch.Tensor
    u_ic: torch.Tensor
    v_ic: torch.Tensor | None  # initial velocity, wave only
    x_bc: torch.Tensor
    t_bc: torch.Tensor
    u_bc: torch.Tensor
    x_c: torch.Tensor | None  # collocation points, pinn only
    t_c: torch.Tensor | None
    x_obs: torch.Tensor | None
    t_obs: torch.Tensor | None
    u_obs: torch.Tensor | None


# ----------------------------------------------------------------- solver ---


class NeuralSolver(Solver):
    """Shared machinery. Subclasses only decide whether the physics term is on."""

    physics: bool

    def __init__(self, cfg: NeuralConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.seed = seed

    def solve(self, problem: Problem, nx: int, nt: int) -> Solution:
        previous = torch.get_num_threads()
        torch.set_num_threads(self.cfg.threads)
        try:
            return self._solve(problem, nx, nt)
        finally:
            torch.set_num_threads(previous)

    def _solve(self, problem: Problem, nx: int, nt: int) -> Solution:
        cfg = self.cfg
        d = problem.domain
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        data = self._build_data(problem, rng)
        model = MLP(cfg.width, cfg.depth, (d.x_min, d.x_max), (d.t_min, d.t_max)).to(DTYPE)
        loss_fn = self._make_loss(problem, model, data)

        start = time.perf_counter()
        history, final = self._train(model, loss_fn)
        train_time = time.perf_counter() - start

        x, t, _, _ = self._grid(problem, nx, nt)
        start = time.perf_counter()
        u = self._predict(model, x, t)
        infer_time = time.perf_counter() - start

        n_params = sum(p.numel() for p in model.parameters())
        meta: dict[str, Any] = {
            "scheme": self.name,
            "n_dof": n_params,
            "width": cfg.width,
            "depth": cfg.depth,
            "epochs": cfg.epochs,
            "lbfgs_iters": cfg.lbfgs_iters,
            "n_collocation": cfg.n_collocation if self.physics else 0,
            "n_obs": cfg.n_obs,
            "train_time_s": train_time,
            "inference_time_s": infer_time,
            "initial_loss": history[0],
            "final_loss": final["total"],
            "loss_history": history,
            "torch_threads": cfg.threads,
            "torch_version": str(torch.__version__),
            "dtype": "float64",
        }
        meta.update({f"final_loss_{k}": v for k, v in final.items() if k != "total"})

        return Solution(
            x=x,
            t=t,
            u=u,
            wall_time_s=train_time + infer_time,
            solver_name=self.name,
            meta=meta,
        )

    # -- data ---------------------------------------------------------------

    def _build_data(self, problem: Problem, rng: np.random.Generator) -> TrainingData:
        cfg = self.cfg
        d = problem.domain

        # Observations are drawn first, so `pinn` and `mlp` with the same seed
        # are shown exactly the same measurements.
        x_obs = t_obs = u_obs = None
        if cfg.n_obs > 0:
            xo = rng.uniform(d.x_min, d.x_max, cfg.n_obs)
            to = rng.uniform(d.t_min, d.t_max, cfg.n_obs)
            uo = np.diag(problem.exact(xo, to))  # exact() is a tensor grid; take u(x_i, t_i)
            x_obs, t_obs, u_obs = _col(xo), _col(to), _col(uo)

        x_c = t_c = None
        if self.physics:
            x_c = _col(rng.uniform(d.x_min, d.x_max, cfg.n_collocation))
            t_c = _col(rng.uniform(d.t_min, d.t_max, cfg.n_collocation))

        x_ic = np.linspace(d.x_min, d.x_max, cfg.n_initial)
        v_ic = _col(problem.initial_velocity(x_ic)) if isinstance(problem, Wave1D) else None

        t_b = np.linspace(d.t_min, d.t_max, cfg.n_boundary)
        left, right = problem.boundary_values(t_b)

        return TrainingData(
            x_ic=_col(x_ic),
            u_ic=_col(problem.initial(x_ic)),
            v_ic=v_ic,
            x_bc=_col(np.concatenate([np.full_like(t_b, d.x_min), np.full_like(t_b, d.x_max)])),
            t_bc=_col(np.concatenate([t_b, t_b])),
            u_bc=_col(np.concatenate([left, right])),
            x_c=x_c,
            t_c=t_c,
            x_obs=x_obs,
            t_obs=t_obs,
            u_obs=u_obs,
        )

    # -- loss ---------------------------------------------------------------

    def _make_loss(self, problem: Problem, model: MLP, data: TrainingData) -> LossFn:
        cfg = self.cfg
        weights = {"pde": cfg.w_pde, "ic": cfg.w_ic, "bc": cfg.w_bc, "data": cfg.w_data}

        def loss_fn() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            parts: dict[str, torch.Tensor] = {}

            if data.x_c is not None and data.t_c is not None:
                parts["pde"] = torch.mean(pde_residual(problem, model, data.x_c, data.t_c) ** 2)

            t0 = torch.zeros_like(data.x_ic).requires_grad_(True)
            u0 = model(data.x_ic, t0)
            parts["ic"] = torch.mean((u0 - data.u_ic) ** 2)
            if data.v_ic is not None:
                parts["ic"] = parts["ic"] + torch.mean((_d(u0, t0) - data.v_ic) ** 2)

            parts["bc"] = torch.mean((model(data.x_bc, data.t_bc) - data.u_bc) ** 2)

            if data.x_obs is not None and data.t_obs is not None and data.u_obs is not None:
                parts["data"] = torch.mean((model(data.x_obs, data.t_obs) - data.u_obs) ** 2)

            total = torch.zeros((), dtype=DTYPE)
            for name, value in parts.items():
                total = total + weights[name] * value
            return total, parts

        return loss_fn

    # -- training -----------------------------------------------------------

    def _train(self, model: MLP, loss_fn: LossFn) -> tuple[list[float], dict[str, float]]:
        cfg = self.cfg
        history: list[float] = []

        optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        for epoch in range(cfg.epochs):
            optimiser.zero_grad(set_to_none=True)
            total, _ = loss_fn()
            total.backward()
            optimiser.step()

            if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
                value = float(total.detach())
                if not math.isfinite(value):
                    raise RuntimeError(f"training diverged at epoch {epoch} (loss = {value})")
                history.append(value)
                log.info("epoch %6d   loss %.3e", epoch, value)

        if cfg.lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=1.0,
                max_iter=cfg.lbfgs_iters,
                history_size=50,
                tolerance_grad=1e-12,
                tolerance_change=1e-14,
                line_search_fn="strong_wolfe",
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                loss, _ = loss_fn()
                loss.backward()
                return loss

            lbfgs.step(closure)  # type: ignore

        total, parts = loss_fn()
        final = {"total": float(total.detach())}
        final.update({name: float(value.detach()) for name, value in parts.items()})
        if not all(math.isfinite(v) for v in final.values()):
            raise RuntimeError(f"training diverged (final losses = {final})")
        history.append(final["total"])
        return history, final

    # -- evaluation ---------------------------------------------------------

    @staticmethod
    def _predict(model: MLP, x: Array, t: Array) -> Array:
        xx, tt = np.meshgrid(x, t)  # both (nt, nx), so u[n, j] = u(x[j], t[n])
        with torch.no_grad():
            u = model(_col(xx.ravel()), _col(tt.ravel()))
        return u.numpy().reshape(t.size, x.size)


class PINNSolver(NeuralSolver):
    name = "pinn"
    physics = True


class MLPSolver(NeuralSolver):
    name = "mlp"
    physics = False
