"""Solver registry."""

from __future__ import annotations

from pinnaudit.config import SolverConfig
from pinnaudit.solvers.base import Solution, Solver
from pinnaudit.solvers.fd_heat import CrankNicolsonHeat
from pinnaudit.solvers.fd_wave import LeapfrogWave

_REGISTRY: dict[str, type[Solver]] = {
    CrankNicolsonHeat.name: CrankNicolsonHeat,
    LeapfrogWave.name: LeapfrogWave,
}


def build_solver(cfg: SolverConfig) -> Solver:
    """Instantiate the solver named in ``cfg``."""
    try:
        cls = _REGISTRY[cfg.name]
    except KeyError as exc:  # pragma: no cover - unreachable while config is a Literal
        raise KeyError(f"unknown solver {cfg.name!r}; known: {sorted(_REGISTRY)}") from exc
    return cls()


def available_solvers() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "CrankNicolsonHeat",
    "LeapfrogWave",
    "Solution",
    "Solver",
    "available_solvers",
    "build_solver",
]
