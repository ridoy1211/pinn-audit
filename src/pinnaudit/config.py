"""Typed, validated run configuration.

Every number that can change an experiment's outcome lives in a YAML file that
maps onto these models. Nothing is passed as an ad-hoc keyword argument, so a
run is fully described by its config plus its seed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROBLEM_NAMES = Literal["heat1d", "wave1d"]
SOLVER_NAMES = Literal["fd_heat_cn", "fd_wave_leapfrog", "pinn", "mlp"]

# Solvers that need PyTorch. Everything else runs on numpy/scipy alone.
NEURAL_SOLVERS = frozenset({"pinn", "mlp"})

# Which solvers are mathematically applicable to which problems. Catching a
# mismatch here turns a silent garbage result into a loud startup error.
COMPATIBLE: dict[str, set[str]] = {
    "heat1d": {"fd_heat_cn", "pinn", "mlp"},
    "wave1d": {"fd_wave_leapfrog", "pinn", "mlp"},
}


class SineMode(BaseModel):
    """One term ``amplitude * sin(n*pi*x/L)`` of a Fourier sine series."""

    model_config = ConfigDict(extra="forbid")

    n: int = Field(ge=1, description="Mode number; must be >= 1.")
    amplitude: float


class ProblemConfig(BaseModel):
    """Physical specification of a benchmark problem."""

    model_config = ConfigDict(extra="forbid")

    name: PROBLEM_NAMES
    x_max: float = Field(default=1.0, gt=0.0, description="Domain length L.")
    t_max: float = Field(default=1.0, gt=0.0, description="Final time T.")
    alpha: float = Field(default=0.05, gt=0.0, description="Heat diffusivity (heat1d).")
    c: float = Field(default=1.0, gt=0.0, description="Wave speed (wave1d).")
    displacement_modes: list[SineMode] = Field(
        default_factory=lambda: [SineMode(n=1, amplitude=1.0), SineMode(n=3, amplitude=0.5)],
        description="Sine modes of u(x, 0).",
    )
    velocity_modes: list[SineMode] = Field(
        default_factory=list,
        description="Sine modes of du/dt(x, 0). Ignored for first-order-in-time problems.",
    )

    @model_validator(mode="after")
    def _require_nonempty_initial_condition(self) -> ProblemConfig:
        if not self.displacement_modes and not self.velocity_modes:
            raise ValueError("initial condition is identically zero; specify at least one mode")
        return self


class NeuralConfig(BaseModel):
    """Architecture, data and optimisation settings for ``pinn`` and ``mlp``."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=32, ge=2, description="Neurons per hidden layer.")
    depth: int = Field(default=4, ge=1, description="Number of hidden layers.")
    epochs: int = Field(default=5000, ge=1, description="Adam steps.")
    lr: float = Field(default=1e-3, gt=0.0, description="Adam learning rate.")
    lbfgs_iters: int = Field(default=0, ge=0, description="L-BFGS iterations after Adam.")
    n_collocation: int = Field(default=2000, ge=1, description="Random PDE-residual points.")
    n_initial: int = Field(default=100, ge=2, description="Points on t = 0.")
    n_boundary: int = Field(default=100, ge=2, description="Times per boundary.")
    n_obs: int = Field(default=0, ge=0, description="Observations of the exact solution.")
    w_pde: float = Field(default=1.0, ge=0.0)
    w_ic: float = Field(default=1.0, ge=0.0)
    w_bc: float = Field(default=1.0, ge=0.0)
    w_data: float = Field(default=1.0, ge=0.0)
    log_every: int = Field(default=100, ge=1)
    threads: int = Field(default=1, ge=1, description="Torch CPU threads (1 = fair timing).")


class SolverConfig(BaseModel):
    """A solver and the grid its solution is reported on.

    For classical schemes ``nx`` and ``nt`` are the discretisation. For neural
    solvers they are only the evaluation grid, so every method is scored on the
    same points.
    """

    model_config = ConfigDict(extra="forbid")

    name: SOLVER_NAMES
    nx: int = Field(ge=5, description="Number of spatial grid POINTS, including both boundaries.")
    nt: int = Field(ge=3, description="Number of time levels, including t=0.")
    neural: NeuralConfig | None = None

    @model_validator(mode="after")
    def _neural_block_matches_solver(self) -> SolverConfig:
        if self.name not in NEURAL_SOLVERS:
            if self.neural is not None:
                raise ValueError(f"solver {self.name!r} takes no 'neural' block")
            return self
        if self.neural is None:
            self.neural = NeuralConfig()
        if self.name == "mlp" and self.neural.n_obs == 0:
            raise ValueError("solver 'mlp' learns from observations only: set neural.n_obs > 0")
        return self


class RunConfig(BaseModel):
    """Everything needed to reproduce one measurement."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    seed: int = 0
    out_dir: Path = Path("results")
    problem: ProblemConfig
    solver: SolverConfig

    @model_validator(mode="after")
    def _check_solver_applies_to_problem(self) -> RunConfig:
        allowed = COMPATIBLE[self.problem.name]
        if self.solver.name not in allowed:
            raise ValueError(
                f"solver {self.solver.name!r} cannot solve problem {self.problem.name!r}; "
                f"compatible solvers are {sorted(allowed)}"
            )
        return self

    def fingerprint(self) -> str:
        """Stable SHA-256 over the semantic content of the run.

        ``run_id`` and ``out_dir`` are excluded so that two runs that differ
        only in bookkeeping hash identically -- that is what lets us detect
        accidental duplicate work in a sweep.
        """
        payload: dict[str, Any] = self.model_dump(mode="json", exclude={"run_id", "out_dir"})
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_run_config(path: str | Path) -> RunConfig:
    """Load and validate a run config from YAML."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(raw)}")
    return RunConfig.model_validate(raw)
