from __future__ import annotations

import pytest

from pinnaudit.config import ProblemConfig, RunConfig, SineMode, SolverConfig


@pytest.fixture
def heat_cfg() -> ProblemConfig:
    return ProblemConfig(
        name="heat1d",
        x_max=1.0,
        t_max=1.0,
        alpha=0.05,
        displacement_modes=[SineMode(n=1, amplitude=1.0), SineMode(n=3, amplitude=0.5)],
    )


@pytest.fixture
def wave_cfg() -> ProblemConfig:
    return ProblemConfig(
        name="wave1d",
        x_max=1.0,
        t_max=1.0,
        c=1.0,
        displacement_modes=[SineMode(n=1, amplitude=1.0), SineMode(n=3, amplitude=0.5)],
    )


@pytest.fixture
def heat_run(heat_cfg: ProblemConfig) -> RunConfig:
    return RunConfig(
        run_id="test_heat",
        problem=heat_cfg,
        solver=SolverConfig(name="fd_heat_cn", nx=21, nt=21),
    )
