"""Sweeps: validation up front, resumability, and surviving a failed run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from pinnaudit.sweep import (
    SweepConfig,
    Variant,
    apply_overrides,
    expand,
    load_sweep_config,
    run_sweep,
)

HEAT_FD = Path("configs/heat_fd.yaml")
WAVE_FD = Path("configs/wave_fd.yaml")


def _sweep(tmp_path: Path, seeds: list[int] | None = None) -> SweepConfig:
    return SweepConfig(
        name="t",
        seeds=seeds or [0, 1],
        out_dir=tmp_path,
        variants=[
            Variant(label="fd_21", config=HEAT_FD, overrides={"solver.nx": 21, "solver.nt": 21}),
            Variant(label="fd_41", config=HEAT_FD),
        ],
    )


# ------------------------------------------------------------ overrides -----


def test_overrides_set_nested_keys_without_touching_the_input() -> None:
    raw = {"solver": {"name": "pinn", "nx": 41, "neural": {"width": 32}}}
    out = apply_overrides(raw, {"solver.neural.lbfgs_iters": 500, "solver.nx": 81})
    assert out["solver"]["neural"] == {"width": 32, "lbfgs_iters": 500}
    assert out["solver"]["nx"] == 81
    assert raw["solver"]["nx"] == 41
    assert "lbfgs_iters" not in raw["solver"]["neural"]


def test_overrides_create_a_missing_block() -> None:
    out = apply_overrides({"solver": {"name": "pinn"}}, {"solver.neural.n_obs": 5})
    assert out["solver"]["neural"] == {"n_obs": 5}


def test_overriding_through_a_non_mapping_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a mapping"):
        apply_overrides({"solver": 3}, {"solver.nx": 1})


# ------------------------------------------------------------ expansion -----


def test_one_run_per_variant_and_seed(tmp_path: Path) -> None:
    runs = expand(_sweep(tmp_path, seeds=[0, 1, 2]))
    assert len(runs) == 6
    assert {cfg.run_id for _, cfg in runs} >= {"fd_21_seed0", "fd_41_seed2"}
    assert len({cfg.fingerprint() for _, cfg in runs}) == 6


def test_seed_limit_and_variant_filter(tmp_path: Path) -> None:
    runs = expand(_sweep(tmp_path, seeds=[0, 1, 2]), n_seeds=1, only=["fd_41"])
    assert [(label, cfg.seed) for label, cfg in runs] == [("fd_41", 0)]


def test_unknown_variant_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        expand(_sweep(tmp_path), only=["nope"])


def test_override_typo_is_caught_before_any_run(tmp_path: Path) -> None:
    sweep = SweepConfig(
        name="t",
        seeds=[0],
        out_dir=tmp_path,
        variants=[Variant(label="bad", config=HEAT_FD, overrides={"solver.nxx": 5})],
    )
    with pytest.raises(ValidationError):
        expand(sweep)


def test_duplicate_labels_are_rejected(tmp_path: Path) -> None:
    variant = Variant(label="same", config=HEAT_FD)
    with pytest.raises(ValidationError, match="unique"):
        SweepConfig(name="t", seeds=[0], out_dir=tmp_path, variants=[variant, variant])


def test_sweep_name_must_be_filename_safe(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        SweepConfig(
            name="has space",
            seeds=[0],
            out_dir=tmp_path,
            variants=[Variant(label="a", config=HEAT_FD)],
        )


@pytest.mark.parametrize(
    ("path", "n_runs"),
    [("configs/sweeps/forward_heat.yaml", 45), ("configs/sweeps/forward_wave.yaml", 60)],
)
def test_shipped_sweeps_are_valid(path: str, n_runs: int) -> None:
    runs = expand(load_sweep_config(path))
    assert len(runs) == n_runs
    by_label = {label: cfg for label, cfg in runs}
    assert by_label["pinn_lbfgs"].solver.neural is not None
    assert by_label["pinn_lbfgs"].solver.neural.lbfgs_iters == 500
    assert by_label["pinn_data200"].solver.neural is not None
    assert by_label["pinn_data200"].solver.neural.n_obs == 200
    assert by_label["mlp_data200"].solver.name == "mlp"
    assert by_label["fd_321"].solver.nx == 321


def test_wave_sweep_has_the_loss_weight_variants() -> None:
    by_label = {
        label: cfg for label, cfg in expand(load_sweep_config("configs/sweeps/forward_wave.yaml"))
    }
    for label, weight in (("pinn_w10", 10.0), ("pinn_w100", 100.0), ("pinn_w1000", 1000.0)):
        neural = by_label[label].solver.neural
        assert neural is not None
        assert (neural.w_ic, neural.w_bc, neural.w_pde) == (weight, weight, 1.0)


# -------------------------------------------------------------- running -----


def test_interrupted_sweep_resumes_without_repeating_work(tmp_path: Path) -> None:
    sweep = _sweep(tmp_path)
    first = run_sweep(sweep, echo=lambda _: None)
    assert (first.ran, first.skipped, first.failed) == (4, 0, 0)

    df = pl.read_parquet(first.results_path)
    assert df.height == 4
    assert {"sweep", "variant", "solution_file", "metric_relative_l2"} <= set(df.columns)
    saved = np.load(df["solution_file"][0])
    assert set(saved.files) == {"x", "t", "u"}

    second = run_sweep(sweep, echo=lambda _: None)
    assert (second.ran, second.skipped, second.failed) == (0, 4, 0)
    assert pl.read_parquet(first.results_path).height == 4


def test_a_failed_run_is_recorded_and_the_sweep_continues(tmp_path: Path) -> None:
    sweep = SweepConfig(
        name="t",
        seeds=[0],
        out_dir=tmp_path,
        variants=[
            # nx=41, nt=11 gives a Courant number of 4, which the solver refuses.
            Variant(label="bad", config=WAVE_FD, overrides={"solver.nt": 11}),
            Variant(label="good", config=HEAT_FD, overrides={"solver.nx": 21, "solver.nt": 21}),
        ],
    )
    outcome = run_sweep(sweep, echo=lambda _: None)
    assert (outcome.ran, outcome.failed) == (1, 1)
    failures = (tmp_path / "sweep_t_failures.txt").read_text(encoding="utf-8")
    assert "bad_seed0" in failures
    assert "CFL" in failures
