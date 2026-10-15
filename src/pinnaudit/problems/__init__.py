"""Problem registry."""

from __future__ import annotations

from pinnaudit.config import ProblemConfig
from pinnaudit.problems.base import Domain, Problem
from pinnaudit.problems.heat import Heat1D
from pinnaudit.problems.wave import Wave1D

_REGISTRY: dict[str, type[Problem]] = {
    Heat1D.name: Heat1D,
    Wave1D.name: Wave1D,
}


def build_problem(cfg: ProblemConfig) -> Problem:
    """Instantiate the problem named in ``cfg``."""
    try:
        cls = _REGISTRY[cfg.name]
    except KeyError as exc:  # pragma: no cover - unreachable while config is a Literal
        raise KeyError(f"unknown problem {cfg.name!r}; known: {sorted(_REGISTRY)}") from exc
    return cls(cfg)  # type: ignore[call-arg]


def available_problems() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["Domain", "Heat1D", "Problem", "Wave1D", "available_problems", "build_problem"]
