"""Scoring the confirmation predictions: each rule is checked on data built to pass or fail it."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np

_SPEC = importlib.util.spec_from_file_location("confirm_report", "scripts/confirm_report.py")
assert _SPEC is not None
assert _SPEC.loader is not None
cr = importlib.util.module_from_spec(_SPEC)
sys.modules["confirm_report"] = cr
_SPEC.loader.exec_module(cr)

SEEDS_DEV = np.arange(5)
SEEDS_HELD = np.arange(5, 15)


def runs(values: list[float], seeds: np.ndarray, seconds: float = 100.0) -> object:
    return cr.Runs(seeds, np.array(values, dtype=float), np.full(len(values), seconds))


def make(**overrides: list[float]) -> tuple[dict, dict]:
    """A passing configuration of development and held-out data."""
    dev = {
        "heat": {
            "pinn": runs([1.4e-3, 2.0e-3, 2.9e-3, 4.4e-3, 5.0e-3], SEEDS_DEV),
            "pinn_lbfgs": runs([2.8e-4, 3.4e-4, 4.1e-4, 4.3e-4, 6.0e-4], SEEDS_DEV),
        },
        "wave": {
            "pinn": runs([0.484, 0.49, 0.495, 0.50, 0.58], SEEDS_DEV),
            "pinn_w100_lbfgs": runs([2.1e-2, 2.2e-2, 2.3e-2, 3.0e-2, 7.0e-2], SEEDS_DEV),
        },
    }
    held = {
        "heat": {
            "fd_41": runs([5.31e-4] * 10, SEEDS_HELD, 0.001),
            "fd_81": runs([1.33e-4] * 10, SEEDS_HELD, 0.001),
            "pinn": runs([2.0e-3 + 1e-4 * i for i in range(10)], SEEDS_HELD),
            "pinn_lbfgs": runs([3.0e-4 + 3e-5 * i for i in range(10)], SEEDS_HELD, 175.0),
        },
        "wave": {
            "fd_21": runs([1.65e-2] * 10, SEEDS_HELD, 0.0003),
            "pinn": runs([0.50 + 0.003 * i for i in range(10)], SEEDS_HELD),
            "pinn_w100_lbfgs": runs([2.5e-2 + 4e-3 * i for i in range(10)], SEEDS_HELD, 250.0),
        },
    }
    for key, values in overrides.items():
        problem, variant = key.split("__")
        held[problem][variant] = runs(values, SEEDS_HELD, held[problem][variant].times[0])
    return dev, held


def verdicts(dev: dict, held: dict) -> dict[str, bool]:
    return {statement.split(":")[0]: passed for statement, passed, _ in cr.evaluate(dev, held)}


def test_a_consistent_dataset_passes_every_prediction() -> None:
    dev, held = make()
    assert all(passed for _, passed, _ in cr.evaluate(dev, held))


def test_prediction_1_fails_when_a_median_leaves_the_development_range() -> None:
    dev, held = make(heat__pinn=[1.0e-5] * 10)
    results = {s: p for s, p, _ in cr.evaluate(dev, held)}
    assert not results["P1  heat pinn: held-out median inside the development range"]


def test_prediction_2_fails_if_the_tuned_wave_pinn_beats_the_coarsest_grid() -> None:
    dev, held = make(wave__pinn_w100_lbfgs=[1.0e-2] * 10)
    assert not verdicts(dev, held)["P2  wave"]


def test_prediction_3_fails_outside_the_classical_bracket() -> None:
    dev, held = make(heat__pinn_lbfgs=[9.0e-4] * 10)
    assert not verdicts(dev, held)["P3  heat"]


def test_prediction_4_needs_nine_of_ten_paired_wins() -> None:
    plain = [3.0e-3] * 10
    eight_wins = [1.0e-3] * 8 + [9.0e-3] * 2
    dev, held = make(heat__pinn=plain, heat__pinn_lbfgs=eight_wins)
    results = [(s, p) for s, p, _ in cr.evaluate(dev, held) if s.startswith("P4  heat")]
    assert results
    assert not results[0][1]


def test_prediction_4_reports_the_exact_p_value_for_ten_wins() -> None:
    dev, held = make()
    detail = next(d for s, _, d in cr.evaluate(dev, held) if s.startswith("P4  heat"))
    assert "10 of 10" in detail
    assert "0.0020" in detail


def test_mismatched_seeds_are_rejected() -> None:
    dev, held = make()
    held["heat"]["pinn"] = cr.Runs(np.arange(20, 30), np.full(10, 2e-3), np.full(10, 100.0))
    try:
        cr.evaluate(dev, held)
    except ValueError as exc:
        assert "same seeds" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


def test_slow_runs_are_flagged_and_normal_runs_are_not() -> None:
    _, held = make()
    assert cr.slow_runs(held) == []
    times = np.full(10, 250.0)
    times[3] = 6800.0
    held["wave"]["pinn_w100_lbfgs"] = cr.Runs(SEEDS_HELD, np.full(10, 3e-2), times)
    flagged = cr.slow_runs(held)
    assert len(flagged) == 1
    assert "seed 8" in flagged[0]
    assert "6800" in flagged[0]


def test_classical_runs_are_never_flagged_as_slow() -> None:
    _, held = make()
    times = np.full(10, 0.001)
    times[0] = 5.0
    held["heat"]["fd_41"] = cr.Runs(SEEDS_HELD, np.full(10, 5e-4), times)
    assert cr.slow_runs(held) == []
