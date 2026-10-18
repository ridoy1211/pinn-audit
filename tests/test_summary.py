"""Across-seed aggregation."""

from __future__ import annotations

import polars as pl
import pytest

from pinnaudit.summary import summarise, to_markdown


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "variant": ["a", "a", "a", "b"],
            "seed": [0, 1, 2, 0],
            "metric_relative_l2": [1.0, 2.0, 3.0, 5.0],
            "wall_time_s": [10.0, 20.0, 30.0, 4.0],
        }
    )


def test_summary_statistics() -> None:
    row = summarise(_frame()).filter(pl.col("variant") == "a").row(0, named=True)
    assert row["n_seeds"] == 3
    assert row["l2_mean"] == pytest.approx(2.0)
    assert row["l2_std"] == pytest.approx(1.0)  # sample standard deviation of 1, 2, 3
    assert row["l2_min"] == pytest.approx(1.0)
    assert row["l2_max"] == pytest.approx(3.0)
    assert row["time_mean_s"] == pytest.approx(20.0)


def test_a_single_seed_has_no_spread() -> None:
    row = summarise(_frame()).filter(pl.col("variant") == "b").row(0, named=True)
    assert row["n_seeds"] == 1
    assert row["l2_std"] is None


def test_variant_order_is_preserved() -> None:
    assert summarise(_frame())["variant"].to_list() == ["a", "b"]


def test_markdown_table_has_a_row_per_variant() -> None:
    text = to_markdown(summarise(_frame()))
    lines = text.splitlines()
    assert lines[0].startswith("| variant")
    assert len(lines) == 4
    assert "| a | 3 | 2.00e+00 | 1.00e+00 |" in lines[2]
    assert "n/a" in lines[3]
