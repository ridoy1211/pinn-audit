"""Run-record persistence.

One row per measurement, appended to Parquet. Figures are built later from
these files only -- never from numbers pasted out of a notebook -- so a plot
can always be traced back to the config that produced it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pinnaudit.config import RunConfig
from pinnaudit.solvers.base import Solution


@lru_cache(maxsize=1)
def current_code_revision() -> str:
    """Identify the checked-out source plus its current Python implementation."""
    root = Path(__file__).resolve().parents[2]
    source = root / "src" / "pinnaudit"
    if not source.is_dir():
        # Also provide a source hash when installed from a wheel without its repo.
        source = Path(__file__).resolve().parent
        root = source.parent
    digest = hashlib.sha256()
    for path in sorted(source.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())

    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return f"{commit}+src-{digest.hexdigest()[:16]}"


def execution_fingerprint(config_fingerprint: str, code_revision: str | None = None) -> str:
    """Fingerprint one config under the exact source that executes it."""
    revision = code_revision or current_code_revision()
    payload = f"{config_fingerprint}:{revision}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_record(
    cfg: RunConfig,
    solution: Solution,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Flatten a completed run into a single tabular row."""
    completed_utc = dt.datetime.now(dt.UTC).isoformat()
    code_revision = current_code_revision()
    config_fingerprint = cfg.fingerprint()
    design_blob = json.dumps(
        cfg.model_dump(mode="json", exclude={"run_id", "out_dir", "seed"}), sort_keys=True
    )
    record: dict[str, Any] = {
        "run_id": cfg.run_id,
        "fingerprint": config_fingerprint,
        "execution_fingerprint": execution_fingerprint(config_fingerprint, code_revision),
        "config_fingerprint": config_fingerprint,
        "design_fingerprint": hashlib.sha256(design_blob.encode()).hexdigest()[:16],
        "code_revision": code_revision,
        "run_started_utc": solution.meta.get("run_started_utc"),
        "completed_utc": completed_utc,
        # Keep the historical field, which has always been written at completion.
        "timestamp_utc": completed_utc,
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
    scalars = {
        k: v
        for k, v in solution.meta.items()
        if k != "run_started_utc" and not isinstance(v, dict | list)
    }
    record.update({f"meta_{k}": v for k, v in scalars.items()})
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
