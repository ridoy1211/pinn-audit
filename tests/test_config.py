"""The config layer must reject bad experiments before they burn compute."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pinnaudit.config import ProblemConfig, RunConfig, SineMode, SolverConfig, load_run_config


def test_smoke_config_loads() -> None:
    cfg = load_run_config("configs/smoke.yaml")
    assert cfg.problem.name == "heat1d"
    assert cfg.solver.name == "fd_heat_cn"


@pytest.mark.parametrize(
    "path", ["configs/smoke.yaml", "configs/heat_fd.yaml", "configs/wave_fd.yaml"]
)
def test_all_shipped_configs_are_valid(path: str) -> None:
    load_run_config(path)


def test_incompatible_solver_is_rejected(wave_cfg: ProblemConfig) -> None:
    with pytest.raises(ValidationError, match="cannot solve problem"):
        RunConfig(
            run_id="bad",
            problem=wave_cfg,
            solver=SolverConfig(name="fd_heat_cn", nx=21, nt=21),
        )


def test_typos_are_rejected_not_ignored() -> None:
    with pytest.raises(ValidationError):
        ProblemConfig(name="heat1d", alhpa=0.05)  # type: ignore[call-arg]


def test_zero_initial_condition_is_rejected() -> None:
    with pytest.raises(ValidationError, match="identically zero"):
        ProblemConfig(name="heat1d", displacement_modes=[], velocity_modes=[])


def test_mode_number_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        SineMode(n=0, amplitude=1.0)


def test_fingerprint_is_stable_and_ignores_run_id(heat_run: RunConfig) -> None:
    renamed = heat_run.model_copy(deep=True)
    renamed.run_id = "a_different_name"
    assert heat_run.fingerprint() == renamed.fingerprint()

    refined = heat_run.model_copy(deep=True)
    refined.solver.nx = 41
    assert heat_run.fingerprint() != refined.fingerprint()
