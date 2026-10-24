"""Configuration and sweep runner for the inverse experiments.

An inverse sweep crosses the number of observations with the noise level and the
seed, and fits alpha with each requested method on identical data. Finished jobs
are recorded as they complete and skipped on re-run, as in ``pinnaudit.sweep``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pinnaudit.config import NeuralConfig, ProblemConfig
from pinnaudit.identify import IdentResult, Observations, fit_lsq, make_observations
from pinnaudit.io import append_records
from pinnaudit.problems.heat import Heat1D

Method = Literal["lsq", "pinn"]


def _both_methods() -> list[Method]:
    return ["lsq", "pinn"]


class InverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    problem: ProblemConfig
    ic_known: bool = Field(default=True, description="Is the initial condition given?")
    lsq_ic_modes: int = Field(default=6, ge=1, description="Sine modes for the classical fit.")
    n_obs: list[int] = Field(min_length=1)
    noise: list[float] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    methods: list[Method] = Field(default_factory=_both_methods, min_length=1)
    alpha0: float = Field(default=0.2, gt=0.0, description="Starting guess, same for all.")
    neural: NeuralConfig = Field(default_factory=NeuralConfig)
    out_dir: Path = Path("results")

    @model_validator(mode="after")
    def _check_consistency(self) -> InverseConfig:
        if self.problem.name != "heat1d":
            raise ValueError("inverse experiments are implemented for heat1d only")
        if any(n < 1 for n in self.n_obs):
            raise ValueError("every n_obs must be at least 1")
        if any(s < 0.0 for s in self.noise):
            raise ValueError("every noise level must be non-negative")
        if self.ic_known and self.neural.w_ic == 0.0:
            raise ValueError("ic_known is true but neural.w_ic is 0: the PINN would ignore it")
        if not self.ic_known and self.neural.w_ic != 0.0:
            raise ValueError("ic_known is false, so neural.w_ic must be 0")
        return self


@dataclass
class InverseOutcome:
    ran: int
    skipped: int
    failed: int
    results_path: Path


def load_inverse_config(path: str | Path) -> InverseConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(raw)}")
    return InverseConfig.model_validate(raw)


def job_key(cfg: InverseConfig, method: str, n_obs: int, noise: float, seed: int) -> str:
    """Stable hash of everything that can change one job's outcome."""
    payload: dict[str, Any] = {
        "problem": cfg.problem.model_dump(mode="json"),
        "ic_known": cfg.ic_known,
        "alpha0": cfg.alpha0,
        "method": method,
        "n_obs": n_obs,
        "noise": noise,
        "seed": seed,
    }
    if method == "pinn":
        payload["neural"] = cfg.neural.model_dump(mode="json")
    if method == "lsq" and not cfg.ic_known:
        payload["lsq_ic_modes"] = cfg.lsq_ic_modes
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _done_jobs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pl.read_parquet(path, columns=["job"])["job"].to_list())


def _fit(
    cfg: InverseConfig, problem: Heat1D, obs: Observations, method: str, seed: int
) -> IdentResult:
    if method == "lsq":
        modes = None if cfg.ic_known else cfg.lsq_ic_modes
        return fit_lsq(problem, obs, cfg.alpha0, ic_modes=modes)
    # PyTorch is an optional extra, so it is imported only when a PINN fit is requested.
    try:
        from pinnaudit.inverse_pinn import fit_pinn
    except ImportError as exc:
        raise ImportError('the PINN estimator needs PyTorch: pip install -e ".[nn]"') from exc
    return fit_pinn(problem, obs, cfg.alpha0, cfg.neural, seed, ic_known=cfg.ic_known)


def run_inverse(
    cfg: InverseConfig,
    n_seeds: int | None = None,
    methods: Sequence[str] | None = None,
    echo: Callable[[str], None] = print,
) -> InverseOutcome:
    """Run every (seed, observations, noise, method) job not already recorded."""
    if methods:
        unknown = set(methods) - set(cfg.methods)
        if unknown:
            raise ValueError(f"unknown method(s) {sorted(unknown)}; configured: {cfg.methods}")
    if n_seeds is not None and n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")

    seeds = cfg.seeds[:n_seeds] if n_seeds else cfg.seeds
    chosen = [m for m in cfg.methods if not methods or m in methods]
    jobs = [
        (m, n, noise, seed)
        for seed in seeds
        for n in cfg.n_obs
        for noise in cfg.noise
        for m in chosen
    ]

    problem = Heat1D(cfg.problem)
    name = f"inverse_{cfg.name}"
    results_path = cfg.out_dir / f"{name}.parquet"
    failures_path = cfg.out_dir / f"{name}_failures.txt"
    done = _done_jobs(results_path)

    ran = skipped = failed = 0
    for i, (method, n, noise, seed) in enumerate(jobs, start=1):
        key = job_key(cfg, method, n, noise, seed)
        if key in done:
            skipped += 1
            continue

        label = f"{method} N={n} noise={noise:g} seed={seed}"
        echo(f"[{i}/{len(jobs)}] {label} ...")
        obs = make_observations(problem, n, noise, seed)
        try:
            result = _fit(cfg, problem, obs, method, seed)
        except Exception as exc:
            failed += 1
            message = f"{label}: {type(exc).__name__}: {exc}"
            echo(f"    FAILED  {message}")
            cfg.out_dir.mkdir(parents=True, exist_ok=True)
            with failures_path.open("a", encoding="utf-8") as fh:
                fh.write(message + "\n")
            continue

        alpha_true = cfg.problem.alpha
        record: dict[str, Any] = {
            "sweep": cfg.name,
            "job": key,
            "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
            "method": method,
            "ic_known": cfg.ic_known,
            "n_obs": n,
            "noise": noise,
            "seed": seed,
            "alpha_true": alpha_true,
            "alpha0": cfg.alpha0,
            "alpha_hat": result.alpha_hat,
            "rel_err": abs(result.alpha_hat - alpha_true) / alpha_true,
            "wall_time_s": result.wall_time_s,
        }
        scalars = {k: v for k, v in result.meta.items() if not isinstance(v, dict | list)}
        record.update({f"meta_{k}": v for k, v in scalars.items()})
        append_records([record], cfg.out_dir, name=name)
        ran += 1
        echo(
            f"    alpha_hat={result.alpha_hat:.5f}  error={100 * record['rel_err']:.3g}%  "
            f"time={result.wall_time_s:.2f} s"
        )

    return InverseOutcome(ran=ran, skipped=skipped, failed=failed, results_path=results_path)


__all__ = [
    "InverseConfig",
    "InverseOutcome",
    "job_key",
    "load_inverse_config",
    "run_inverse",
]
