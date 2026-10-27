#!/usr/bin/env python3
"""Accuracy against wall-clock cost, one panel per sweep.

Classical solvers are drawn as a curve (one point per grid resolution). Each
neural variant is drawn as its individual seeds (faint dots) plus their mean
(large marker), so the spread across seeds is visible and not hidden in a bar.
Closer to the bottom-left is better.

Usage:  python scripts/plot_pareto.py --name forward_heat forward_wave
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

Groups = dict[str, tuple[np.ndarray, np.ndarray]]  # variant -> (times, errors), one per seed


def load_groups(path: Path) -> tuple[str, Groups]:
    import polars as pl

    df = pl.read_parquet(path)
    groups: Groups = {}
    for part in df.partition_by("variant", maintain_order=True):
        groups[str(part["variant"][0])] = (
            part["wall_time_s"].to_numpy(),
            part["metric_relative_l2"].to_numpy(),
        )
    return str(df["problem"][0]), groups


def draw_panel(ax: Axes, title: str, groups: Groups) -> None:
    classical = {k: v for k, v in groups.items() if k.startswith("fd_")}
    neural = {k: v for k, v in groups.items() if not k.startswith("fd_")}

    if classical:
        points = sorted(
            (float(np.mean(t)), float(np.mean(e)), k) for k, (t, e) in classical.items()
        )
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            "ko-",
            label="finite differences (grid sweep)",
        )
        for time_s, error, key in points:
            ax.annotate(
                key.removeprefix("fd_"),
                (time_s, error),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )

    markers = ["o", "s", "^", "D", "v", "P"]
    for i, (label, (times, errors)) in enumerate(neural.items()):
        colour = f"C{i}"
        ax.scatter(times, errors, color=colour, alpha=0.35, s=18)
        ax.scatter(
            [np.mean(times)],
            [np.mean(errors)],
            color=colour,
            marker=markers[i % len(markers)],
            s=90,
            edgecolor="k",
            label=f"{label} (n={len(errors)})",
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="wall-clock time (s)", ylabel="relative $L^2$ error", title=title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)


def main(
    names: list[str], results: Path, out_dir: Path, filename: str = "accuracy_vs_time.png"
) -> None:
    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 5.2), squeeze=False)
    for ax, name in zip(axes[0], names, strict=True):
        problem, groups = load_groups(results / f"sweep_{name}.parquet")
        draw_panel(ax, f"{problem}: accuracy vs. cost", groups)
    fig.suptitle("Where does each method land? (bottom-left is better)", fontsize=12)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", nargs="+", required=True, help="sweep names")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--filename", default="accuracy_vs_time.png")
    args = ap.parse_args()
    main(args.name, args.results, args.out, args.filename)
