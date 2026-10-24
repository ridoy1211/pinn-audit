"""PINN estimator for the inverse problem: a network and a learnable diffusivity.

The network learns u(x, t) and, at the same time, log(alpha). The loss is the
usual one (PDE residual, boundary, initial condition) plus a misfit term on the
noisy observations. alpha enters only through the PDE residual, so the data pull
it towards the value that makes the learned solution consistent with them.

If the initial condition is unknown, set ``w_ic = 0`` in the settings (the
config validator enforces this) and ``ic_known=False``: the initial-condition
term is then dropped and the network must infer the initial shape from the data.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import torch

from pinnaudit.config import NeuralConfig
from pinnaudit.identify import IdentResult, Observations
from pinnaudit.problems.heat import Heat1D
from pinnaudit.solvers.neural import DTYPE, MLP, _col, heat_residual

log = logging.getLogger(__name__)

# log(alpha) has to travel a long way from a poor starting guess, so it gets a
# larger Adam step than the network weights.
ALPHA_LR_FACTOR = 10.0


def fit_pinn(
    problem: Heat1D,
    obs: Observations,
    alpha0: float,
    cfg: NeuralConfig,
    seed: int,
    ic_known: bool = True,
) -> IdentResult:
    """Estimate alpha with a PINN trained on ``obs``. Runs on the CPU in float64."""
    if alpha0 <= 0.0:
        raise ValueError("alpha0 must be positive")
    previous = torch.get_num_threads()
    torch.set_num_threads(cfg.threads)
    try:
        return _fit(problem, obs, alpha0, cfg, seed, ic_known)
    finally:
        torch.set_num_threads(previous)


def _fit(
    problem: Heat1D,
    obs: Observations,
    alpha0: float,
    cfg: NeuralConfig,
    seed: int,
    ic_known: bool,
) -> IdentResult:
    d = problem.domain
    torch.manual_seed(seed)
    rng = np.random.default_rng([seed, 1])  # independent of the stream that drew the data

    x_c = _col(rng.uniform(d.x_min, d.x_max, cfg.n_collocation))
    t_c = _col(rng.uniform(d.t_min, d.t_max, cfg.n_collocation))

    x_ic_np = np.linspace(d.x_min, d.x_max, cfg.n_initial)
    x_ic, u_ic = _col(x_ic_np), _col(problem.initial(x_ic_np))

    t_b = np.linspace(d.t_min, d.t_max, cfg.n_boundary)
    left, right = problem.boundary_values(t_b)
    x_bc = _col(np.concatenate([np.full_like(t_b, d.x_min), np.full_like(t_b, d.x_max)]))
    t_bc = _col(np.concatenate([t_b, t_b]))
    u_bc = _col(np.concatenate([left, right]))

    x_obs, t_obs, u_obs = _col(obs.x), _col(obs.t), _col(obs.u)

    model = MLP(cfg.width, cfg.depth, (d.x_min, d.x_max), (d.t_min, d.t_max)).to(DTYPE)
    log_alpha = torch.nn.Parameter(torch.tensor(math.log(alpha0), dtype=DTYPE))
    weights = {"pde": cfg.w_pde, "ic": cfg.w_ic, "bc": cfg.w_bc, "data": cfg.w_data}

    def loss_fn() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        alpha = torch.exp(log_alpha)
        parts = {"pde": torch.mean(heat_residual(model, x_c, t_c, alpha) ** 2)}
        if ic_known:
            parts["ic"] = torch.mean((model(x_ic, torch.zeros_like(x_ic)) - u_ic) ** 2)
        parts["bc"] = torch.mean((model(x_bc, t_bc) - u_bc) ** 2)
        parts["data"] = torch.mean((model(x_obs, t_obs) - u_obs) ** 2)
        total = torch.zeros((), dtype=DTYPE)
        for name, value in parts.items():
            total = total + weights[name] * value
        return total, parts

    start = time.perf_counter()
    first_loss = float("nan")

    groups: list[dict[str, Any]] = [
        {"params": list(model.parameters())},
        {"params": [log_alpha], "lr": cfg.lr * ALPHA_LR_FACTOR},
    ]
    adam = torch.optim.Adam(groups, lr=cfg.lr)
    for epoch in range(cfg.epochs):
        adam.zero_grad(set_to_none=True)
        total, _ = loss_fn()
        total.backward()
        adam.step()
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            value = float(total.detach())
            if not math.isfinite(value):
                raise RuntimeError(f"training diverged at epoch {epoch} (loss = {value})")
            if epoch == 0:
                first_loss = value
            log.info(
                "epoch %6d   loss %.3e   alpha %.5f",
                epoch,
                value,
                float(torch.exp(log_alpha).detach()),
            )

    if cfg.lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            [*model.parameters(), log_alpha],
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
    wall = time.perf_counter() - start
    alpha_hat = float(torch.exp(log_alpha).detach())
    final = {"total": float(total.detach())}
    final.update({name: float(value.detach()) for name, value in parts.items()})
    if not (math.isfinite(alpha_hat) and all(math.isfinite(v) for v in final.values())):
        raise RuntimeError(f"training diverged (alpha = {alpha_hat}, losses = {final})")

    meta: dict[str, float | int | str] = {
        "initial_loss": first_loss,
        "final_loss": final["total"],
        "epochs": cfg.epochs,
        "lbfgs_iters": cfg.lbfgs_iters,
        "n_params": sum(p.numel() for p in model.parameters()) + 1,
        "torch_threads": cfg.threads,
        "torch_version": str(torch.__version__),
    }
    meta.update({f"final_loss_{k}": v for k, v in final.items() if k != "total"})
    return IdentResult(alpha_hat=alpha_hat, wall_time_s=wall, meta=meta)
