#!/usr/bin/env python3
"""Check that each prediction was committed before the runs it predicts.

For each experiment this compares the time of the commit that recorded the
prediction with the earliest recorded run start among the rows it concerns. Result
files are not versioned, so run this on the machine that produced them. Legacy
rows with completion timestamps only cannot verify chronology.

A prediction is VERIFIED only if a commit with the expected message exists and is
earlier than the first run. Otherwise it is reported as NOT VERIFIED, and the lab
notes must say so. Nothing here is edited by hand: the output is the evidence.

Usage:  python scripts/check_chronology.py
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    commit_subject: str  # exact message of the commit that recorded the prediction
    results: str  # parquet file holding the runs
    column: str  # column used to pick the runs
    values: tuple[str, ...]  # values of that column that belong to this experiment


@dataclass(frozen=True)
class RunStart:
    timestamp: datetime | None
    issue: str | None = None


CHECKS = [
    Check(
        "loss-weight test (wave)",
        "Write down the Day 3 hypothesis and prediction before running it",
        "results/sweep_forward_wave.parquet",
        "variant",
        ("pinn_w10", "pinn_w100", "pinn_w1000"),
    ),
    Check(
        "best shot (wave)",
        "Write down the best-shot prediction before running it",
        "results/sweep_forward_wave.parquet",
        "variant",
        ("pinn_w100_lbfgs",),
    ),
    Check(
        "best shot (heat)",
        "Write down the best-shot prediction before running it",
        "results/sweep_forward_heat.parquet",
        "variant",
        ("pinn_w100_lbfgs",),
    ),
    Check(
        "inverse problem, known initial condition",
        "Write down the inverse-problem prediction before running the PINN",
        "results/inverse_heat_alpha_known.parquet",
        "method",
        ("pinn",),
    ),
    Check(
        "inverse problem, unknown initial condition",
        "Write down the inverse-problem prediction before running the PINN",
        "results/inverse_heat_alpha_unknown.parquet",
        "method",
        ("pinn",),
    ),
    Check(
        "confirmation on held-out seeds (heat)",
        "Write down the confirmation-run predictions before running them",
        "results/sweep_forward_heat_confirm.parquet",
        "variant",
        ("pinn", "pinn_lbfgs"),
    ),
    Check(
        "confirmation on held-out seeds (wave)",
        "Write down the confirmation-run predictions before running them",
        "results/sweep_forward_wave_confirm.parquet",
        "variant",
        ("pinn", "pinn_w100_lbfgs"),
    ),
]


def commit_times(subject: str, repo: Path = Path(".")) -> list[datetime]:
    """Commit times of every commit whose message is exactly ``subject``."""
    out = subprocess.run(
        ["git", "log", "--format=%cI%x09%s"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    times = []
    for line in out.splitlines():
        stamp, _, message = line.partition("\t")
        if message == subject:
            times.append(datetime.fromisoformat(stamp))
    return times


def first_run(path: Path, column: str, values: tuple[str, ...]) -> RunStart:
    """Earliest explicit start time for selected rows, or why it is unavailable."""
    import polars as pl

    if not path.exists():
        return RunStart(None)
    df = pl.read_parquet(path)
    if column not in df.columns:
        return RunStart(None, f"selection column {column!r} is missing")
    selected = df.filter(pl.col(column).is_in(list(values)))
    if selected.is_empty():
        return RunStart(None)
    if "run_started_utc" not in selected.columns:
        return RunStart(None, "legacy rows have completion timestamps only")
    raw_stamps = selected["run_started_utc"].to_list()
    if any(not isinstance(stamp, str) or not stamp.strip() for stamp in raw_stamps):
        return RunStart(None, "one or more selected rows lack a recorded run start")
    try:
        stamps = [datetime.fromisoformat(stamp) for stamp in raw_stamps]
    except ValueError:
        return RunStart(None, "one or more selected run-start timestamps are malformed")
    if any(stamp.utcoffset() is None for stamp in stamps):
        return RunStart(None, "one or more selected run-start timestamps lack a timezone")
    return RunStart(min(stamps))


def verdict(commit: datetime | None, run: datetime | None) -> str:
    if run is None:
        return "no runs found"
    if commit is None:
        return "NOT VERIFIED (no commit recording the prediction)"
    if commit < run:
        return "VERIFIED (prediction committed before the first run)"
    return "NOT VERIFIED (the commit is later than the first run)"


def checked_verdict(commit: datetime | None, run_start: RunStart) -> str:
    """Do not infer a run start from a legacy completion timestamp."""
    if run_start.issue:
        return f"NOT VERIFIED (run start unavailable: {run_start.issue})"
    return verdict(commit, run_start.timestamp)


def main() -> None:
    for check in CHECKS:
        commits = commit_times(check.commit_subject)
        commit = min(commits) if commits else None
        run_start = first_run(Path(check.results), check.column, check.values)
        print(f"{check.name}")
        print(f"    prediction commit : {commit.isoformat() if commit else 'none'}")
        if run_start.timestamp:
            run_text = run_start.timestamp.isoformat()
        elif run_start.issue:
            run_text = f"unavailable ({run_start.issue})"
        else:
            run_text = "no matching rows"
        print(f"    first run start   : {run_text}")
        print(f"    verdict           : {checked_verdict(commit, run_start)}")


if __name__ == "__main__":
    main()
