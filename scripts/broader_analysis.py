"""Descriptive analysis for the broader PINN comparison study.

This module only consumes recorded outputs. It does not fit models or import
PyTorch, and it keeps completeness separate from conditional accuracy summaries.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

METHODS = ("unreg6", "fixed6", "cv6", "cv12", "pinn_200", "pinn_1000")
NN_METHODS = ("pinn_200", "pinn_1000")
BASELINE_METHOD = "cv12"
SCORE_METRICS = ("relative_alpha_error", "field_relative_l2", "ic_relative_l2")
TIMING_METRICS = ("fit_time_s", "selection_time_s", "evaluation_time_s", "job_elapsed_s")
STATUSES = ("completed", "failed", "timeout", "interrupted")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.number))
        and not isinstance(value, bool)
        and np.isfinite(value)
    )


def _as_float(value: Any) -> float | None:
    return float(value) if _finite_number(value) else None


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _max(values: Sequence[float]) -> float | None:
    return float(np.max(values)) if values else None


def _validate_inputs(
    records: Iterable[Mapping[str, Any]],
    expected_cases: Sequence[Mapping[str, Any]],
    training_seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], tuple[int, ...]]:
    cases: dict[str, dict[str, Any]] = {}
    for raw_case in expected_cases:
        case = dict(raw_case)
        required = ("case_id", "n_obs", "data_seed", "family")
        if any(key not in case for key in required):
            raise ValueError(f"expected case requires fields {required}")
        case_id = str(case["case_id"])
        if case_id in cases:
            raise ValueError(f"duplicate expected case_id: {case_id}")
        if not isinstance(case["n_obs"], int) or case["n_obs"] <= 0:
            raise ValueError(f"invalid n_obs for case {case_id}")
        if not isinstance(case["data_seed"], int):
            raise ValueError(f"invalid data_seed for case {case_id}")
        case["case_id"] = case_id
        cases[case_id] = case
    if not cases:
        raise ValueError("expected_cases must not be empty")
    seeds = tuple(training_seeds)
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("training_seeds must contain exactly three distinct integers")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int | None]] = set()
    for raw_record in records:
        row = dict(raw_record)
        for key in ("case_id", "n_obs", "data_seed", "family", "method", "status"):
            if key not in row:
                raise ValueError(f"record missing required field: {key}")
        row["case_id"] = str(row["case_id"])
        case_id = row["case_id"]
        if case_id not in cases:
            raise ValueError(f"unexpected case_id: {case_id}")
        expected = cases[case_id]
        for key in ("n_obs", "data_seed", "family"):
            if row[key] != expected[key]:
                raise ValueError(f"metadata disagreement for {case_id}: {key}")
        method = row["method"]
        if method not in METHODS:
            raise ValueError(f"unexpected method: {method}")
        status = row["status"]
        if status not in STATUSES:
            raise ValueError(f"unexpected status for {case_id}/{method}: {status}")
        seed = row.get("training_seed")
        if method in NN_METHODS:
            if seed not in seeds:
                raise ValueError(f"unexpected training seed for {case_id}/{method}: {seed}")
        elif seed is not None:
            raise ValueError(f"classical record {case_id}/{method} must have training_seed=None")
        key = (case_id, method, seed)
        if key in seen:
            raise ValueError(f"duplicate record: {key}")
        seen.add(key)
        rows.append(row)
    return rows, cases, seeds


def _bootstrap_median_ci(
    differences: Sequence[float], *, reps: int, seed: int
) -> list[float] | None:
    if not differences:
        return None
    rng = np.random.default_rng(seed)
    values = np.asarray(differences, dtype=float)
    samples = rng.choice(values, size=(reps, len(values)), replace=True)
    bounds = np.percentile(np.median(samples, axis=1), [2.5, 97.5])
    return [float(bounds[0]), float(bounds[1])]


def summarize_study(
    records: Iterable[Mapping[str, Any]],
    expected_cases: Sequence[Mapping[str, Any]],
    training_seeds: Sequence[int],
    *,
    bootstrap_reps: int = 10_000,
    bootstrap_seed_offset: int = 42_000,
    min_complete_pairs: int = 10,
) -> dict[str, Any]:
    """Summarize declared pipelines while preserving missing and failed jobs.

    Neural case scores are medians across all three declared starts only when
    all three starts completed with finite scores. Contrasts compare that
    per-case median with the corresponding CV12 dataset score.
    """
    if bootstrap_reps < 1 or min_complete_pairs < 1:
        raise ValueError("bootstrap_reps and min_complete_pairs must be positive")
    rows, cases, seeds = _validate_inputs(records, expected_cases, training_seeds)
    row_index = {(row["case_id"], row["method"], row.get("training_seed")): row for row in rows}

    ledger: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        for method in METHODS:
            job_seeds: tuple[int | None, ...] = seeds if method in NN_METHODS else (None,)
            jobs: list[dict[str, Any]] = []
            counts: Counter[str] = Counter()
            for seed in job_seeds:
                row = row_index.get((case_id, method, seed))
                if row is None:
                    counts["missing"] += 1
                    jobs.append({"training_seed": seed, "status": "missing", "valid_scores": False})
                    continue
                score_values = {metric: _as_float(row.get(metric)) for metric in SCORE_METRICS}
                score_valid = row["status"] == "completed" and all(
                    value is not None for value in score_values.values()
                )
                reported_status = row["status"]
                counts[reported_status] += 1
                if reported_status == "completed" and not score_valid:
                    counts["invalid_scores"] += 1
                jobs.append(
                    {
                        "training_seed": seed,
                        "status": reported_status,
                        "valid_scores": score_valid,
                        "scores": score_values,
                    }
                )
            valid_count = sum(job["valid_scores"] for job in jobs)
            needed = len(job_seeds)
            ledger.append(
                {
                    "case_id": case_id,
                    "n_obs": case["n_obs"],
                    "data_seed": case["data_seed"],
                    "family": case["family"],
                    "method": method,
                    "expected_jobs": needed,
                    "observed_jobs": needed - counts["missing"],
                    "valid_completed_jobs": valid_count,
                    "complete": valid_count == needed,
                    "status_counts": dict(counts),
                    "jobs": jobs,
                }
            )

    dataset_scores: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        for method in METHODS:
            job_seeds = seeds if method in NN_METHODS else (None,)
            rows_for_case = [row_index.get((case_id, method, seed)) for seed in job_seeds]
            valid = [
                row
                for row in rows_for_case
                if row is not None
                and row["status"] == "completed"
                and all(_as_float(row.get(metric)) is not None for metric in SCORE_METRICS)
            ]
            complete = len(valid) == len(job_seeds)
            score_summary: dict[str, float | None] = {}
            for metric in SCORE_METRICS:
                vals = [_as_float(row[metric]) for row in valid]
                finite_vals = [value for value in vals if value is not None]
                score_summary[metric] = float(np.median(finite_vals)) if complete else None
            dataset_scores.append(
                {
                    "case_id": case_id,
                    "n_obs": case["n_obs"],
                    "data_seed": case["data_seed"],
                    "family": case["family"],
                    "method": method,
                    "complete": complete,
                    "score_rule": "median_of_three_starts"
                    if method in NN_METHODS
                    else "single_fit",
                    "initialization_scores": [
                        {
                            "training_seed": seed,
                            "status": row["status"] if row else "missing",
                            "scores": {
                                metric: _as_float(row.get(metric)) if row else None
                                for metric in SCORE_METRICS
                            },
                        }
                        for seed, row in zip(job_seeds, rows_for_case, strict=True)
                    ],
                    "scores": score_summary,
                }
            )

    by_case_method = {(item["case_id"], item["method"]): item for item in dataset_scores}
    strata = sorted({(case["n_obs"], case["family"]) for case in cases.values()})
    method_summaries: list[dict[str, Any]] = []
    for n_obs, family in strata:
        stratum_cases = [
            case for case in cases.values() if case["n_obs"] == n_obs and case["family"] == family
        ]
        for method in METHODS:
            items = [by_case_method[(case["case_id"], method)] for case in stratum_cases]
            eligible = [item for item in items if item["complete"]]
            status_counts: Counter[str] = Counter()
            for item in items:
                ledger_item = next(
                    entry
                    for entry in ledger
                    if entry["case_id"] == item["case_id"] and entry["method"] == method
                )
                status_counts.update(ledger_item["status_counts"])
            metrics = {}
            for metric in SCORE_METRICS:
                values = [float(item["scores"][metric]) for item in eligible]
                metrics[metric] = {
                    "median": _median(values),
                    "mean": _mean(values),
                    "max": _max(values),
                }
            timing = {}
            timing_rows = [
                row
                for row in rows
                if row["method"] == method
                and row["case_id"] in {case["case_id"] for case in stratum_cases}
            ]
            for metric in TIMING_METRICS:
                values = [
                    float(_as_float(row.get(metric)))
                    for row in timing_rows
                    if _as_float(row.get(metric)) is not None
                ]
                timing[metric] = {
                    "observed_values": len(values),
                    "total_s": float(sum(values)) if values else None,
                    "mean_per_observed_job_s": _mean(values),
                }
            method_summaries.append(
                {
                    "n_obs": n_obs,
                    "family": family,
                    "method": method,
                    "expected_count": len(items),
                    "complete_count": len(eligible),
                    "conditional_on_completion": True,
                    "metrics": metrics,
                    "timings_all_observed_jobs": timing,
                    "status_counts": dict(status_counts),
                }
            )

    paired_contrasts: list[dict[str, Any]] = []
    for n_obs, family in strata:
        stratum_cases = [
            case for case in cases.values() if case["n_obs"] == n_obs and case["family"] == family
        ]
        for method in NN_METHODS:
            pairs: list[dict[str, Any]] = []
            for case in stratum_cases:
                neural = by_case_method[(case["case_id"], method)]
                classical = by_case_method[(case["case_id"], BASELINE_METHOD)]
                if neural["complete"] and classical["complete"]:
                    differences = {
                        metric: (neural["scores"][metric] - classical["scores"][metric]) * 100.0
                        for metric in SCORE_METRICS
                    }
                    pairs.append(
                        {
                            "case_id": case["case_id"],
                            "data_seed": case["data_seed"],
                            "differences_pp": differences,
                        }
                    )
            metrics = {}
            for metric in SCORE_METRICS:
                values = [float(pair["differences_pp"][metric]) for pair in pairs]
                ties = sum(value == 0.0 for value in values)
                metrics[metric] = {
                    "median_pp": _median(values),
                    "mean_pp": _mean(values),
                    "max_pp": _max(values),
                    "wins": sum(value < 0.0 for value in values),
                    "ties": ties,
                    "losses": sum(value > 0.0 for value in values),
                    "bootstrap_95_percentile_ci_for_median_pp": (
                        _bootstrap_median_ci(
                            values,
                            reps=bootstrap_reps,
                            seed=bootstrap_seed_offset + n_obs,
                        )
                        if len(values) >= min_complete_pairs
                        else None
                    ),
                }
            paired_contrasts.append(
                {
                    "n_obs": n_obs,
                    "family": family,
                    "neural_method": method,
                    "baseline_method": BASELINE_METHOD,
                    "expected_pairs": len(stratum_cases),
                    "complete_pairs": len(pairs),
                    "pairing_unit": "dataset case (same case_id); family pairs are not pooled",
                    "inference_complete": len(pairs) >= min_complete_pairs,
                    "conditional_on_completion": True,
                    "metrics": metrics,
                    "pairs": pairs,
                }
            )

    designs_by_n: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for case in cases.values():
        designs_by_n[case["n_obs"]].add((case["n_obs"], case["data_seed"]))
    independent_design_counts = {
        "dataset_case_count": len(cases),
        "independent_design_count": len(
            {(case["n_obs"], case["data_seed"]) for case in cases.values()}
        ),
        "by_n_obs": {
            str(n_obs): {
                "dataset_case_count": sum(case["n_obs"] == n_obs for case in cases.values()),
                "independent_design_count": len(designs),
            }
            for n_obs, designs in sorted(designs_by_n.items())
        },
        "counting_rule": "unique (n_obs, data_seed); IC-family cases sharing a design count once",
    }
    return {
        "methods": list(METHODS),
        "neural_methods": list(NN_METHODS),
        "baseline_method": BASELINE_METHOD,
        "score_metrics": list(SCORE_METRICS),
        "training_seeds": list(seeds),
        "analysis_rules": {
            "neural_dataset_score": (
                "median of all three valid completed starts; otherwise incomplete"
            ),
            "accuracy_summaries": (
                "conditional on completion; no unconditional superiority inferred"
            ),
            "paired_difference_units": "percentage points of raw relative error fractions",
            "bootstrap": {
                "reps": bootstrap_reps,
                "seed": "bootstrap_seed_offset + n_obs",
                "seed_offset": bootstrap_seed_offset,
                "minimum_complete_pairs": min_complete_pairs,
                "statistic": (
                    "median of paired differences, resampling dataset cases "
                    "within each family stratum"
                ),
            },
        },
        "independent_design_counts": independent_design_counts,
        "completeness_ledger": ledger,
        "method_summaries": method_summaries,
        "dataset_scores": dataset_scores,
        "paired_contrasts": paired_contrasts,
    }


def format_markdown(result: Mapping[str, Any]) -> str:
    """Render a compact human-readable summary, retaining conditional labels."""
    lines = [
        "# Broader comparison analysis",
        "",
        "Accuracy summaries are conditional on completion; failures remain in the "
        "completeness ledger.",
        "",
        f"Dataset cases: {result['independent_design_counts']['dataset_case_count']}; "
        f"independent designs: {result['independent_design_counts']['independent_design_count']}.",
        "",
        "## Paired contrasts against CV12",
        "",
        "Differences are neural minus CV12 in percentage points; negative values "
        "favor the neural pipeline.",
        "",
        "| N | Family | Neural method | Metric | Complete pairs | Median pp | Mean pp | "
        "Wins / ties / losses | 95% percentile CI for median pp |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for contrast in result["paired_contrasts"]:
        for metric, stats in contrast["metrics"].items():
            ci = stats["bootstrap_95_percentile_ci_for_median_pp"]
            ci_text = "not estimated" if ci is None else f"[{ci[0]:.4g}, {ci[1]:.4g}]"
            lines.append(
                f"| {contrast['n_obs']} | {contrast['family']} | "
                f"{contrast['neural_method']} | {metric} | "
                f"{contrast['complete_pairs']} | {stats['median_pp']} | {stats['mean_pp']} | "
                f"{stats['wins']} / {stats['ties']} / {stats['losses']} | {ci_text} |"
            )
    return "\n".join(lines) + "\n"
