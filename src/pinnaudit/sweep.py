"""Run an experiment over many seeds and variants, resumably.

A sweep file lists *variants*: a base run config plus a few dotted overrides,
for example ``solver.neural.lbfgs_iters: 500``. Every run is validated before
the first one starts, so a typo costs seconds and not an hour. Each finished run
is appended to Parquet straight away, and a run whose fingerprint is already
there is skipped, so an interrupted sweep picks up where it stopped.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pinnaudit.config import RunConfig
from pinnaudit.experiment import run_once
from pinnaudit.io import append_records, build_record
from pinnaudit.solvers.base import Solution


class Variant(BaseModel):
    """One arm of a sweep: a base config and what to change in it."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    config: Path
    overrides: dict[str, Any] = Field(default_factory=dict)


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    seeds: list[int] = Field(min_length=1)
    out_dir: Path = Path("results")
    variants: list[Variant] = Field(min_length=1)

    @model_validator(mode="after")
    def _labels_are_unique(self) -> SweepConfig:
        labels = [v.label for v in self.variants]
        if len(labels) != len(set(labels)):
            raise ValueError(f"variant labels must be unique, got {labels}")
        return self


@dataclass
class SweepOutcome:
    ran: int
    skipped: int
    failed: int
    results_path: Path


def load_sweep_config(path: str | Path) -> SweepConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(raw)}")
    return SweepConfig.model_validate(raw)


def apply_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``raw`` with dotted keys such as ``solver.nx`` set."""
    out = copy.deepcopy(raw)
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        node = out
        for key in keys[:-1]:
            child = node.get(key)
            if child is None:
                child = {}
                node[key] = child
            if not isinstance(child, dict):
                raise ValueError(f"override {dotted!r}: {key!r} is not a mapping")
            node = child
        node[keys[-1]] = value
    return out


def expand(
    sweep: SweepConfig,
    n_seeds: int | None = None,
    only: Sequence[str] | None = None,
) -> list[tuple[str, RunConfig]]:
    """Turn a sweep into validated ``(variant label, run config)`` pairs."""
    labels = {v.label for v in sweep.variants}
    if only:
        unknown = set(only) - labels
        if unknown:
            raise ValueError(f"unknown variant(s) {sorted(unknown)}; known: {sorted(labels)}")
    if n_seeds is not None and n_seeds < 1:
        raise ValueError("n_seeds must be at least 1")

    seeds = sweep.seeds[:n_seeds] if n_seeds else sweep.seeds
    runs: list[tuple[str, RunConfig]] = []
    for variant in sweep.variants:
        if only and variant.label not in only:
            continue
        with variant.config.open("r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh)
        for seed in seeds:
            data = apply_overrides(base, variant.overrides)
            data["seed"] = seed
            data["run_id"] = f"{variant.label}_seed{seed}"
            data["out_dir"] = str(sweep.out_dir)
            runs.append((variant.label, RunConfig.model_validate(data)))
    return runs


def _done_fingerprints(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pl.read_parquet(path, columns=["fingerprint"])["fingerprint"].to_list())


def _save_solution(sweep: SweepConfig, cfg: RunConfig, solution: Solution) -> Path:
    folder = sweep.out_dir / "solutions" / sweep.name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{cfg.run_id}.npz"
    np.savez_compressed(path, x=solution.x, t=solution.t, u=solution.u)
    return path


def run_sweep(
    sweep: SweepConfig,
    n_seeds: int | None = None,
    only: Sequence[str] | None = None,
    echo: Callable[[str], None] = print,
) -> SweepOutcome:
    """Run every (variant, seed) pair that is not already in the results file."""
    runs = expand(sweep, n_seeds=n_seeds, only=only)
    name = f"sweep_{sweep.name}"
    results_path = sweep.out_dir / f"{name}.parquet"
    failures_path = sweep.out_dir / f"{name}_failures.txt"
    done = _done_fingerprints(results_path)

    ran = skipped = failed = 0
    for i, (label, cfg) in enumerate(runs, start=1):
        if cfg.fingerprint() in done:
            skipped += 1
            continue

        echo(f"[{i}/{len(runs)}] {cfg.run_id} ...")
        start = time.perf_counter()
        try:
            result = run_once(cfg)
        except Exception as exc:
            failed += 1
            message = f"{cfg.run_id}: {type(exc).__name__}: {exc}"
            echo(f"    FAILED  {message}")
            sweep.out_dir.mkdir(parents=True, exist_ok=True)
            with failures_path.open("a", encoding="utf-8") as fh:
                fh.write(message + "\n")
            continue

        record = build_record(cfg, result.solution, result.metrics)
        record.update(
            sweep=sweep.name,
            variant=label,
            solution_file=str(_save_solution(sweep, cfg, result.solution)),
        )
        append_records([record], sweep.out_dir, name=name)
        ran += 1
        echo(f"    {result.summary()}")
        echo(f"    finished in {time.perf_counter() - start:.1f} s")

    return SweepOutcome(ran=ran, skipped=skipped, failed=failed, results_path=results_path)
