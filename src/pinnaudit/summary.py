"""Aggregate sweep results across seeds."""

from __future__ import annotations

import polars as pl


def summarise(df: pl.DataFrame) -> pl.DataFrame:
    """One row per variant: seed count, error mean/spread/range, mean time."""
    return df.group_by("variant", maintain_order=True).agg(
        pl.col("seed").n_unique().alias("n_seeds"),
        pl.col("metric_relative_l2").mean().alias("l2_mean"),
        pl.col("metric_relative_l2").std().alias("l2_std"),
        pl.col("metric_relative_l2").min().alias("l2_min"),
        pl.col("metric_relative_l2").max().alias("l2_max"),
        pl.col("wall_time_s").mean().alias("time_mean_s"),
    )


def to_markdown(summary: pl.DataFrame) -> str:
    """Render a summary as a Markdown table, ready to paste into the lab notes."""
    lines = [
        "| variant | seeds | rel. L2 mean | std | min | max | mean time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary.iter_rows(named=True):
        std = "n/a" if row["l2_std"] is None else f"{row['l2_std']:.2e}"
        lines.append(
            f"| {row['variant']} | {row['n_seeds']} | {row['l2_mean']:.2e} | {std} "
            f"| {row['l2_min']:.2e} | {row['l2_max']:.2e} | {row['time_mean_s']:.4g} |"
        )
    return "\n".join(lines)
