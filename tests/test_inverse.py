"""Inverse experiments: validation, resumability and the CLI (classical method only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from pinnaudit.cli import app
from pinnaudit.config import NeuralConfig, ProblemConfig
from pinnaudit.inverse import InverseConfig, job_key, load_inverse_config, run_inverse
from pinnaudit.summary import inverse_to_markdown, summarise_inverse


def _config(tmp_path: Path, **overrides: Any) -> InverseConfig:
    settings: dict[str, Any] = {
        "name": "t",
        "problem": ProblemConfig(name="heat1d"),
        "n_obs": [20],
        "noise": [0.0, 0.05],
        "seeds": [0, 1],
        "methods": ["lsq"],
        "out_dir": tmp_path,
    }
    settings.update(overrides)
    return InverseConfig(**settings)


# ----------------------------------------------------------- validation -----


@pytest.mark.parametrize(
    "path", ["configs/inverse/heat_known_ic.yaml", "configs/inverse/heat_unknown_ic.yaml"]
)
def test_shipped_inverse_configs_are_valid(path: str) -> None:
    cfg = load_inverse_config(path)
    assert cfg.methods == ["lsq", "pinn"]
    assert len(cfg.n_obs) * len(cfg.noise) * len(cfg.seeds) == 45


def test_only_the_heat_problem_is_supported(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="heat1d only"):
        _config(tmp_path, problem=ProblemConfig(name="wave1d"))


def test_initial_condition_flag_must_match_the_pinn_weight(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="ignore it"):
        _config(tmp_path, ic_known=True, neural=NeuralConfig(w_ic=0.0))
    with pytest.raises(ValidationError, match="must be 0"):
        _config(tmp_path, ic_known=False, neural=NeuralConfig(w_ic=1.0))
    _config(tmp_path, ic_known=False, neural=NeuralConfig(w_ic=0.0))


def test_bad_grids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        _config(tmp_path, noise=[-0.1])
    with pytest.raises(ValidationError, match="at least 1"):
        _config(tmp_path, n_obs=[0])
    with pytest.raises(ValidationError):
        _config(tmp_path, typo_field=1)


def test_job_keys_are_stable_and_sensitive(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    key = job_key(cfg, "lsq", 20, 0.05, 0)
    assert key == job_key(cfg, "lsq", 20, 0.05, 0)
    others = {
        job_key(cfg, "lsq", 20, 0.05, 1),
        job_key(cfg, "lsq", 20, 0.10, 0),
        job_key(cfg, "lsq", 50, 0.05, 0),
        job_key(cfg, "pinn", 20, 0.05, 0),
        job_key(_config(tmp_path, alpha0=0.3), "lsq", 20, 0.05, 0),
    }
    assert key not in others
    assert len(others) == 5


def test_network_settings_change_the_pinn_key_only(tmp_path: Path) -> None:
    a, b = _config(tmp_path), _config(tmp_path, neural=NeuralConfig(width=16))
    assert job_key(a, "pinn", 20, 0.0, 0) != job_key(b, "pinn", 20, 0.0, 0)
    assert job_key(a, "lsq", 20, 0.0, 0) == job_key(b, "lsq", 20, 0.0, 0)


# -------------------------------------------------------------- running -----


def test_inverse_sweep_runs_and_resumes(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = run_inverse(cfg, echo=lambda _: None)
    assert (first.ran, first.skipped, first.failed) == (4, 0, 0)

    df = pl.read_parquet(first.results_path)
    assert df.height == 4
    expected = {"method", "n_obs", "noise", "seed", "alpha_hat", "rel_err", "wall_time_s"}
    assert expected <= set(df.columns)
    assert df["rel_err"].max() < 0.05
    assert set(df["method"].to_list()) == {"lsq"}

    second = run_inverse(cfg, echo=lambda _: None)
    assert (second.ran, second.skipped, second.failed) == (0, 4, 0)
    assert pl.read_parquet(first.results_path).height == 4


def test_unconfigured_method_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        run_inverse(_config(tmp_path), methods=["pinn"], echo=lambda _: None)


def test_unknown_initial_condition_runs_end_to_end(tmp_path: Path) -> None:
    cfg = _config(
        tmp_path,
        ic_known=False,
        neural=NeuralConfig(w_ic=0.0),
        n_obs=[50],
        noise=[0.0],
        seeds=[0],
    )
    outcome = run_inverse(cfg, echo=lambda _: None)
    assert outcome.ran == 1
    row = pl.read_parquet(outcome.results_path).row(0, named=True)
    assert row["ic_known"] is False
    assert row["rel_err"] < 1e-3
    assert row["meta_n_params"] == 7


# ------------------------------------------------------------------ cli -----


def test_inverse_cli_runs_and_summarises(tmp_path: Path) -> None:
    config_path = tmp_path / "inverse.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(tmp_path).model_dump(mode="json")), encoding="utf-8"
    )
    runner = CliRunner()

    run = runner.invoke(app, ["inverse", "--config", str(config_path)])
    assert run.exit_code == 0, run.output
    assert "ran 4" in run.output
    assert "| lsq |" in run.output

    summary = runner.invoke(app, ["inverse-summary", "--name", "t", "--results", str(tmp_path)])
    assert summary.exit_code == 0, summary.output
    assert "| observations |" in summary.output


def test_inverse_cli_rejects_an_unknown_method(tmp_path: Path) -> None:
    config_path = tmp_path / "inverse.yaml"
    config_path.write_text(
        yaml.safe_dump(_config(tmp_path).model_dump(mode="json")), encoding="utf-8"
    )
    result = CliRunner().invoke(app, ["inverse", "--config", str(config_path), "--method", "pinn"])
    assert result.exit_code != 0


def test_inverse_summary_reports_a_missing_sweep(tmp_path: Path) -> None:
    args = ["inverse-summary", "--name", "nope", "--results", str(tmp_path)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0


# -------------------------------------------------------------- summary -----


def test_inverse_summary_statistics_and_markdown() -> None:
    df = pl.DataFrame(
        {
            "n_obs": [10, 10, 10, 10],
            "noise": [0.05, 0.05, 0.05, 0.05],
            "method": ["lsq", "lsq", "lsq", "pinn"],
            "seed": [0, 1, 2, 0],
            "rel_err": [0.01, 0.02, 0.03, 0.2],
            "wall_time_s": [1.0, 2.0, 3.0, 100.0],
        }
    )
    summary = summarise_inverse(df)
    assert summary["method"].to_list() == ["lsq", "pinn"]
    lsq = summary.filter(pl.col("method") == "lsq").row(0, named=True)
    assert lsq["n_seeds"] == 3
    assert lsq["err_median"] == pytest.approx(0.02)
    assert (lsq["err_min"], lsq["err_max"]) == (0.01, 0.03)
    assert lsq["time_median_s"] == pytest.approx(2.0)

    lines = inverse_to_markdown(summary).splitlines()
    assert lines[0].startswith("| observations")
    assert "| 10 | 5% | lsq | 3 | 2% | 1% to 3% | 2 |" in lines[2]
