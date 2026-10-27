#!/usr/bin/env python3
"""Score the pre-registered confirmation predictions and report the statistics.

The four predictions are the ones written in docs/LOG.md before the confirmation run
started. They are coded here so that scoring needs no hand arithmetic, and the
development reference values are read from the development results, not retyped.

It also lists any run whose wall-clock time is far above its siblings', which is how a
laptop that went to sleep mid-run shows up.

Usage:  python scripts/confirm_report.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pinnaudit.stats import bootstrap_ci, exact_signed_rank_p, paired_wins

PINN_VARIANTS = {"heat": ("pinn", "pinn_lbfgs"), "wave": ("pinn", "pinn_w100_lbfgs")}
TUNED = {"heat": "pinn_lbfgs", "wave": "pinn_w100_lbfgs"}
SLOW_FACTOR = 3.0  # a run this many times slower than its variant's median is flagged


@dataclass(frozen=True)
class Runs:
    seeds: np.ndarray
    errors: np.ndarray
    times: np.ndarray


Data = dict[str, dict[str, Runs]]  # problem -> variant -> runs, sorted by seed


def load(results: Path) -> tuple[Data, Data]:
    """Development and confirmation results, one entry per variant."""
    import polars as pl

    columns = ["variant", "run_id", "seed", "metric_relative_l2", "wall_time_s"]
    dev: Data = {}
    confirm: Data = {}
    for problem in ("heat", "wave"):
        for store, name in (
            (dev, f"sweep_forward_{problem}"),
            (confirm, f"sweep_forward_{problem}_confirm"),
        ):
            df = pl.read_parquet(results / f"{name}.parquet", columns=columns)
            store[problem] = {}
            for part in df.partition_by("variant", maintain_order=True):
                part = part.sort("seed")
                store[problem][str(part["variant"][0])] = Runs(
                    part["seed"].to_numpy(),
                    part["metric_relative_l2"].to_numpy(),
                    part["wall_time_s"].to_numpy(),
                )
    return dev, confirm


def evaluate(dev: Data, confirm: Data) -> list[tuple[str, bool, str]]:
    """The four pre-registered predictions: (statement, passed, the numbers behind it)."""
    out: list[tuple[str, bool, str]] = []

    for problem, variants in PINN_VARIANTS.items():
        for variant in variants:
            low, high = dev[problem][variant].errors.min(), dev[problem][variant].errors.max()
            median = float(np.median(confirm[problem][variant].errors))
            out.append(
                (
                    f"P1  {problem} {variant}: held-out median inside the development range",
                    bool(low <= median <= high),
                    f"median {median:.3g}; development range {low:.3g} to {high:.3g}",
                )
            )

    coarsest = float(np.median(confirm["wave"]["fd_21"].errors))
    tuned = float(np.median(confirm["wave"][TUNED["wave"]].errors))
    out.append(
        (
            "P2  wave: tuned PINN median above the 21-point classical error",
            tuned > coarsest,
            f"tuned median {tuned:.3g}; 21-point error {coarsest:.3g}",
        )
    )

    fd81 = float(np.median(confirm["heat"]["fd_81"].errors))
    fd41 = float(np.median(confirm["heat"]["fd_41"].errors))
    heat = float(np.median(confirm["heat"][TUNED["heat"]].errors))
    out.append(
        (
            "P3  heat: L-BFGS PINN median between the 81- and 41-point classical errors",
            fd81 <= heat <= fd41,
            f"median {heat:.3g}; 81-point {fd81:.3g}; 41-point {fd41:.3g}",
        )
    )

    for problem in ("heat", "wave"):
        plain, best = confirm[problem]["pinn"], confirm[problem][TUNED[problem]]
        if not np.array_equal(plain.seeds, best.seeds):
            raise ValueError(f"{problem}: plain and tuned runs do not cover the same seeds")
        wins, n = paired_wins(best.errors, plain.errors)
        p = exact_signed_rank_p(best.errors, plain.errors)
        out.append(
            (
                f"P4  {problem}: tuned beats plain in at least 9 of 10 seeds",
                bool(n == 10 and wins >= 9),
                f"{wins} of {n} seeds; exact paired p = {p:.4f}",
            )
        )
    return out


def slow_runs(confirm: Data) -> list[str]:
    """Neural runs far slower than their variant's median, e.g. after a laptop slept."""
    flagged = []
    for problem, variants in confirm.items():
        for variant, runs in variants.items():
            if variant.startswith("fd_"):
                continue
            limit = SLOW_FACTOR * float(np.median(runs.times))
            for seed, seconds in zip(runs.seeds, runs.times, strict=True):
                if seconds > limit:
                    flagged.append(f"{problem} {variant} seed {int(seed)}: {seconds:.0f} s")
    return flagged


def describe(runs: Runs) -> str:
    low, high = bootstrap_ci(runs.errors)
    return (
        f"median {np.median(runs.errors):.3g}  [95% bootstrap {low:.3g}, {high:.3g}]  "
        f"range {runs.errors.min():.3g} to {runs.errors.max():.3g}  "
        f"median time {np.median(runs.times):.4g} s  (n = {runs.errors.size})"
    )


def main(results: Path = Path("results")) -> None:
    dev, confirm = load(results)

    print("PRE-REGISTERED PREDICTIONS")
    for statement, passed, detail in evaluate(dev, confirm):
        print(f"  {'PASS' if passed else 'FAIL'}  {statement}")
        print(f"        {detail}")

    for title, data in (("HELD-OUT SEEDS 5 TO 14", confirm), ("DEVELOPMENT SEEDS 0 TO 4", dev)):
        print(f"\n{title}")
        for problem, variants in data.items():
            for variant, runs in variants.items():
                if variant in {v for vs in PINN_VARIANTS.values() for v in vs} or (
                    title.startswith("HELD") and variant.startswith("fd_")
                ):
                    print(f"  {problem:5} {variant:16} {describe(runs)}")

    print("\nTIMING CHECK")
    flagged = slow_runs(confirm)
    if flagged:
        print(f"  runs more than {SLOW_FACTOR:g}x slower than their variant's median:")
        for line in flagged:
            print(f"    {line}")
    else:
        print("  no run is unusually slow")


if __name__ == "__main__":
    main()
