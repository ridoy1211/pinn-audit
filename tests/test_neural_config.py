"""Neural solver configs. These need no PyTorch, so they run everywhere."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pinnaudit.config import (
    NeuralConfig,
    ProblemConfig,
    RunConfig,
    SolverConfig,
    load_run_config,
)
from pinnaudit.solvers import available_solvers

NEURAL_CONFIGS = [
    "configs/pinn_smoke.yaml",
    "configs/pinn_heat.yaml",
    "configs/pinn_wave.yaml",
    "configs/mlp_heat.yaml",
]


@pytest.mark.parametrize("path", NEURAL_CONFIGS)
def test_shipped_neural_configs_are_valid(path: str) -> None:
    cfg = load_run_config(path)
    assert cfg.solver.neural is not None


def test_pinn_gets_default_neural_settings() -> None:
    solver = SolverConfig(name="pinn", nx=21, nt=21)
    assert solver.neural == NeuralConfig()


def test_mlp_without_observations_is_rejected() -> None:
    with pytest.raises(ValidationError, match="n_obs"):
        SolverConfig(name="mlp", nx=21, nt=21)


def test_mlp_with_observations_is_accepted() -> None:
    solver = SolverConfig(name="mlp", nx=21, nt=21, neural=NeuralConfig(n_obs=50))
    assert solver.neural is not None
    assert solver.neural.n_obs == 50


def test_classical_solver_rejects_a_neural_block() -> None:
    with pytest.raises(ValidationError, match="takes no 'neural' block"):
        SolverConfig(name="fd_heat_cn", nx=21, nt=21, neural=NeuralConfig())


def test_unknown_neural_setting_is_rejected() -> None:
    with pytest.raises(ValidationError):
        NeuralConfig(widht=32)  # type: ignore[call-arg]


def test_neural_solvers_apply_to_both_problems() -> None:
    for name in ("heat1d", "wave1d"):
        RunConfig(
            run_id="x",
            problem=ProblemConfig(name=name),  # type: ignore[arg-type]
            solver=SolverConfig(name="pinn", nx=21, nt=21),
        )


def test_network_size_changes_the_fingerprint() -> None:
    def run(width: int) -> RunConfig:
        return RunConfig(
            run_id="x",
            problem=ProblemConfig(name="heat1d"),
            solver=SolverConfig(name="pinn", nx=21, nt=21, neural=NeuralConfig(width=width)),
        )

    assert run(16).fingerprint() != run(32).fingerprint()
    assert run(16).fingerprint() == run(16).fingerprint()


def test_registry_lists_neural_solvers_without_importing_torch() -> None:
    assert {"pinn", "mlp", "fd_heat_cn", "fd_wave_leapfrog"} <= set(available_solvers())
