from __future__ import annotations

import numpy as np
import pytest
from scripts.broader_analysis import METHODS, NN_METHODS, SCORE_METRICS, summarize_study

SEEDS = (101, 102, 103)


def make_cases(n: int = 10) -> list[dict[str, object]]:
    return [
        {
            "case_id": f"n10-{family}-{seed}",
            "n_obs": 10,
            "data_seed": 986000 + seed,
            "family": family,
        }
        for family in ("A", "B")
        for seed in range(n)
    ]


def make_records(cases: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for case in cases:
        for method in METHODS:
            method_scale = {"unreg6": 1.1, "fixed6": 1.0, "cv6": 0.9, "cv12": 0.8}.get(method, 0.0)
            fit_seeds = SEEDS if method in NN_METHODS else (None,)
            for ix, training_seed in enumerate(fit_seeds):
                scale = method_scale + (ix * 0.1 if method in NN_METHODS else 0)
                rows.append(
                    {
                        **case,
                        "method": method,
                        "training_seed": training_seed,
                        "status": "completed",
                        "relative_alpha_error": scale + 0.01,
                        "field_relative_l2": scale + 0.02,
                        "ic_relative_l2": scale + 0.03,
                        "fit_time_s": 1.0,
                        "selection_time_s": 0.2,
                        "evaluation_time_s": 0.1,
                    }
                )
    return rows


def find_contrast(result, method="pinn_200"):
    return next(
        item
        for item in result["paired_contrasts"]
        if item["family"] == "A" and item["neural_method"] == method
    )


def test_neural_case_score_uses_median_and_pair_count_is_not_start_count():
    cases = make_cases(10)[:10]
    rows = make_records(cases)
    result = summarize_study(rows, cases, SEEDS, bootstrap_reps=500)
    case_id = cases[0]["case_id"]
    score = next(
        item
        for item in result["dataset_scores"]
        if item["case_id"] == case_id and item["method"] == "pinn_200"
    )
    raw = [
        row["relative_alpha_error"]
        for row in rows
        if row["case_id"] == case_id and row["method"] == "pinn_200"
    ]
    assert score["scores"]["relative_alpha_error"] == pytest.approx(np.median(raw))
    contrast = find_contrast(result)
    assert contrast["complete_pairs"] == 10
    assert contrast["expected_pairs"] == 10
    assert (
        sum(contrast["metrics"]["relative_alpha_error"][key] for key in ("wins", "ties", "losses"))
        == 10
    )


def test_paired_bootstrap_is_for_median_of_paired_differences():
    cases = make_cases(10)[:10]
    rows = make_records(cases)
    # Paired percentage point differences are 1, 2, ..., 10; median is 5.5.
    for i, case in enumerate(cases):
        for row in rows:
            if row["case_id"] == case["case_id"] and row["method"] == "pinn_200":
                row["relative_alpha_error"] = 0.10 + (i + 1) / 100
            if row["case_id"] == case["case_id"] and row["method"] == "cv12":
                row["relative_alpha_error"] = 0.10
    result = summarize_study(rows, cases, SEEDS, bootstrap_reps=1000)
    stats = find_contrast(result)["metrics"]["relative_alpha_error"]
    assert stats["median_pp"] == pytest.approx(5.5)
    assert stats["mean_pp"] == pytest.approx(5.5)
    assert stats["bootstrap_95_percentile_ci_for_median_pp"] is not None
    assert (
        stats["bootstrap_95_percentile_ci_for_median_pp"]
        == summarize_study(rows, cases, SEEDS, bootstrap_reps=1000)["paired_contrasts"][0][
            "metrics"
        ]["relative_alpha_error"]["bootstrap_95_percentile_ci_for_median_pp"]
    )


def test_design_count_collapses_family_pairs():
    cases = make_cases(10)
    result = summarize_study(make_records(cases), cases, SEEDS, bootstrap_reps=10)
    counts = result["independent_design_counts"]
    assert counts["dataset_case_count"] == 20
    assert counts["independent_design_count"] == 10
    assert counts["by_n_obs"]["10"]["dataset_case_count"] == 20
    assert counts["by_n_obs"]["10"]["independent_design_count"] == 10
    assert all(contrast["complete_pairs"] == 10 for contrast in result["paired_contrasts"])


def test_missing_failed_and_invalid_jobs_stay_in_denominator_and_suppress_accuracy():
    cases = make_cases(10)[:10]
    rows = make_records(cases)
    # One missing long-budget initialization, one short-budget timeout, and one
    # completed row with a nonfinite score.
    rows = [
        row
        for row in rows
        if not (
            row["case_id"] == cases[0]["case_id"]
            and row["method"] == "pinn_1000"
            and row["training_seed"] == 103
        )
    ]
    for row in rows:
        if (
            row["case_id"] == cases[1]["case_id"]
            and row["method"] == "pinn_200"
            and row["training_seed"] == 101
        ):
            row["status"] = "timeout"
        if (
            row["case_id"] == cases[2]["case_id"]
            and row["method"] == "pinn_200"
            and row["training_seed"] == 102
        ):
            row["field_relative_l2"] = float("nan")
    result = summarize_study(rows, cases, SEEDS, bootstrap_reps=10)
    long_ledger = next(
        entry
        for entry in result["completeness_ledger"]
        if entry["case_id"] == cases[0]["case_id"] and entry["method"] == "pinn_1000"
    )
    assert long_ledger["expected_jobs"] == 3
    assert long_ledger["observed_jobs"] == 2
    assert long_ledger["status_counts"]["missing"] == 1
    assert long_ledger["complete"] is False
    short_summary = next(
        item
        for item in result["method_summaries"]
        if item["family"] == "A" and item["method"] == "pinn_200"
    )
    assert short_summary["expected_count"] == 10
    # Only the short-budget timeout and invalid score affect this procedure.
    # The missing long-budget start does not censor a valid short-budget case.
    assert short_summary["complete_count"] == 8
    assert short_summary["status_counts"]["timeout"] == 1
    assert short_summary["status_counts"]["invalid_scores"] == 1
    contrast = find_contrast(result)
    assert contrast["expected_pairs"] == 10
    assert contrast["complete_pairs"] == 8
    assert (
        contrast["metrics"]["relative_alpha_error"]["bootstrap_95_percentile_ci_for_median_pp"]
        is None
    )
    assert contrast["inference_complete"] is False


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("duplicate", "duplicate record"),
        ("metadata", "metadata disagreement"),
        ("unknown_case", "unexpected case_id"),
        ("unknown_seed", "unexpected training seed"),
    ],
)
def test_invalid_records_are_rejected(mutation, message):
    cases = make_cases(1)[:1]
    rows = make_records(cases)
    if mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "metadata":
        rows[0]["family"] = "other"
    elif mutation == "unknown_case":
        rows[0]["case_id"] = "not-expected"
    else:
        target = next(row for row in rows if row["method"] == "pinn_200")
        target["training_seed"] = 999
    with pytest.raises(ValueError, match=message):
        summarize_study(rows, cases, SEEDS, bootstrap_reps=10)


def test_metrics_are_percentage_point_error_differences():
    cases = make_cases(10)[:10]
    rows = make_records(cases)
    result = summarize_study(rows, cases, SEEDS, bootstrap_reps=10)
    metric = find_contrast(result)["metrics"][SCORE_METRICS[0]]
    # PINN median 0.11 vs CV12 0.81: a -70 percentage point difference.
    assert metric["median_pp"] == pytest.approx(-70.0)
    assert metric["wins"] == 10
