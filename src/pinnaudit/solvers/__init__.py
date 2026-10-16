"""Solver registry."""

from __future__ import annotations

from pinnaudit.config import NEURAL_SOLVERS, SolverConfig
from pinnaudit.solvers.base import Solution, Solver
from pinnaudit.solvers.fd_heat import CrankNicolsonHeat
from pinnaudit.solvers.fd_wave import LeapfrogWave

_REGISTRY: dict[str, type[Solver]] = {
    CrankNicolsonHeat.name: CrankNicolsonHeat,
    LeapfrogWave.name: LeapfrogWave,
}


def build_solver(cfg: SolverConfig, seed: int = 0) -> Solver:
    """Instantiate the solver named in ``cfg``. ``seed`` only matters for neural solvers."""
    if cfg.name in NEURAL_SOLVERS:
        # PyTorch is an optional extra, so it is imported only when a neural solver is built.
        try:
            from pinnaudit.solvers.neural import MLPSolver, PINNSolver
        except ImportError as exc:
            raise ImportError('neural solvers need PyTorch: pip install -e ".[nn]"') from exc
        if cfg.neural is None:  # the config validator fills this in
            raise ValueError(f"solver {cfg.name!r} has no neural settings")
        neural_cls = PINNSolver if cfg.name == "pinn" else MLPSolver
        return neural_cls(cfg.neural, seed=seed)

    try:
        cls = _REGISTRY[cfg.name]
    except KeyError as exc:  # pragma: no cover - unreachable while config is a Literal
        raise KeyError(f"unknown solver {cfg.name!r}; known: {available_solvers()}") from exc
    return cls()


def available_solvers() -> list[str]:
    return sorted([*_REGISTRY, *NEURAL_SOLVERS])


__all__ = [
    "CrankNicolsonHeat",
    "LeapfrogWave",
    "Solution",
    "Solver",
    "available_solvers",
    "build_solver",
]
