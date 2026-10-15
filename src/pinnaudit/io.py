"""Run-record persistence.

One row per measurement, appended to Parquet. Figures are built later from
these files only -- never from numbers pasted out of a notebook -- so a plot
can always be traced back to the config that produced it.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pinnaudit.config import RunConfig
from pinnaudit.solvers.base import Solution


def build_record(
    cfg: RunConfig,
    solution: Solution,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Flatten a completed run into a single tabular row."""
    record: dict[str, Any] = {
        "run_id": cfg.run_id,
        "fingerprint": cfg.fingerprint(),
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "seed": cfg.seed,
        "problem": cfg.problem.name,
        "solver": solution.solver_name,
        "nx": len(solution.x),
        "nt": len(solution.t),
        "n_dof": solution.n_dof,
        "wall_time_s": solution.wall_time_s,
        "x_max": cfg.problem.x_max,
        "t_max": cfg.problem.t_max,
        "alpha": cfg.problem.alpha,
        "c": cfg.problem.c,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    record.update({f"metric_{k}": v for k, v in metrics.items()})
    record.update({f"meta_{k}": v for k, v in solution.meta.items() if not isinstance(v, dict)})
    return record


def append_records(records: list[dict[str, Any]], out_dir: Path, name: str) -> Path:
    """Append rows to ``out_dir/name.parquet``, creating it if absent."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"

    fresh = pl.DataFrame(records)
    if path.exists():
        existing = pl.read_parquet(path)
        fresh = pl.concat([existing, fresh], how="diagonal_relaxed")
    fresh.write_parquet(path)
    return path


def dump_config(cfg: RunConfig, out_dir: Path) -> Path:
    """Write the exact resolved config beside the results, for reproduction."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"config_{cfg.run_id}_{cfg.fingerprint()}.json"
    path.write_text(json.dumps(cfg.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path
