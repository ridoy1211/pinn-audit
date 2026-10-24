#!/usr/bin/env python3
"""Check that each prediction was committed before the runs it predicts.

For each experiment this compares the time of the commit that recorded the
prediction with the earliest timestamp among the runs it concerns. Result files
are not versioned, so run this on the machine that produced them.

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


def first_run(path: Path, column: str, values: tuple[str, ...]) -> datetime | None:
    """Earliest ``timestamp_utc`` among the rows of ``path`` where ``column`` is in ``values``."""
    import polars as pl

    if not path.exists():
        return None
    df = pl.read_parquet(path, columns=["timestamp_utc", column])
    stamps = df.filter(pl.col(column).is_in(list(values)))["timestamp_utc"].to_list()
    return min(datetime.fromisoformat(s) for s in stamps) if stamps else None


def verdict(commit: datetime | None, run: datetime | None) -> str:
    if run is None:
        return "no runs found"
    if commit is None:
        return "NOT VERIFIED (no commit recording the prediction)"
    if commit < run:
        return "VERIFIED (prediction committed before the first run)"
    return "NOT VERIFIED (the commit is later than the first run)"


def main() -> None:
    for check in CHECKS:
        commits = commit_times(check.commit_subject)
        commit = min(commits) if commits else None
        run = first_run(Path(check.results), check.column, check.values)
        print(f"{check.name}")
        print(f"    prediction commit : {commit.isoformat() if commit else 'none'}")
        print(f"    first run (UTC)   : {run.isoformat() if run else 'none'}")
        print(f"    verdict           : {verdict(commit, run)}")


if __name__ == "__main__":
    main()
