"""Aggregate sweep results across seeds."""

from __future__ import annotations

import polars as pl


def _require_unique_replicates(df: pl.DataFrame, keys: list[str]) -> None:
    """Prevent revised/repeated jobs from being silently pooled as independent seeds."""
    if df.select(keys).is_duplicated().any():
        raise ValueError(
            "duplicate replicate rows: filter one immutable experiment revision before summary"
        )
    group_keys = [key for key in keys if key != "seed"]
    # New records make code/design identity explicit. Missing identity in legacy
    # records is not reconstructed here; those need a separate evidence audit.
    for identity in ("code_revision", "design_fingerprint", "config_fingerprint"):
        if identity == "config_fingerprint" and "variant" in keys:
            continue  # Forward config fingerprints include the seed.
        if identity in df.columns:
            counts = df.group_by(group_keys).agg(pl.col(identity).n_unique().alias("n"))
            if counts.filter(pl.col("n") > 1).height:
                raise ValueError("mixed experiment identities: filter one immutable revision")


def summarise(df: pl.DataFrame) -> pl.DataFrame:
    """One row per variant: seed count, error mean/spread/range, mean time."""
    _require_unique_replicates(df, ["variant", "seed"])
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


def summarise_inverse(df: pl.DataFrame) -> pl.DataFrame:
    """One row per (observations, noise, method): error in alpha across seeds, and time."""
    _require_unique_replicates(df, ["n_obs", "noise", "method", "seed"])
    return (
        df.group_by("n_obs", "noise", "method")
        .agg(
            pl.col("seed").n_unique().alias("n_seeds"),
            pl.col("rel_err").median().alias("err_median"),
            pl.col("rel_err").min().alias("err_min"),
            pl.col("rel_err").max().alias("err_max"),
            pl.col("wall_time_s").median().alias("time_median_s"),
        )
        .sort("n_obs", "noise", "method")
    )


def inverse_to_markdown(summary: pl.DataFrame) -> str:
    """Markdown table of an inverse summary. Errors are relative errors in alpha, in percent."""
    lines = [
        "| observations | noise | method | seeds | median error | min to max | median time (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            f"| {row['n_obs']} | {100 * row['noise']:g}% | {row['method']} | {row['n_seeds']} "
            f"| {100 * row['err_median']:.3g}% "
            f"| {100 * row['err_min']:.3g}% to {100 * row['err_max']:.3g}% "
            f"| {row['time_median_s']:.4g} |"
        )
    return "\n".join(lines)
