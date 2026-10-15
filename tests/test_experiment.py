"""End-to-end: config in, scored measurement out."""

from __future__ import annotations

import pytest

from pinnaudit.config import RunConfig, load_run_config
from pinnaudit.experiment import run_once


def test_run_once_produces_all_metrics(heat_run: RunConfig) -> None:
    result = run_once(heat_run)
    for key in ("relative_l2", "max_abs_error", "rmse", "final_time_relative_l2"):
        assert key in result.metrics
        assert result.metrics[key] >= 0.0


def test_run_once_is_deterministic(heat_run: RunConfig) -> None:
    a = run_once(heat_run).metrics["relative_l2"]
    b = run_once(heat_run).metrics["relative_l2"]
    assert a == pytest.approx(b, rel=1e-15)


@pytest.mark.parametrize(
    "path", ["configs/smoke.yaml", "configs/heat_fd.yaml", "configs/wave_fd.yaml"]
)
def test_shipped_configs_run_and_are_accurate(path: str) -> None:
    result = run_once(load_run_config(path))
    assert result.metrics["relative_l2"] < 1e-2


def test_summary_is_human_readable(heat_run: RunConfig) -> None:
    text = run_once(heat_run).summary()
    assert "relL2" in text and "heat1d" in text
