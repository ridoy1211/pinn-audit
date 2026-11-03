#!/usr/bin/env python3
"""Score saved fits on independent quadrature grids; no fitting or tuning occurs.

This scorer is specific to the archived unit-interval, two-sine-mode heat benchmark.
The known simulation truth is used for evaluation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def quadrature_weights(grid):
    weights = np.zeros_like(grid)
    widths = np.diff(grid)
    weights[:-1] += 0.5 * widths
    weights[1:] += 0.5 * widths
    return weights


def relative_error(estimate, reference, weights):
    return float(
        np.sqrt(np.sum(weights * (estimate - reference) ** 2) / np.sum(weights * reference**2))
    )


def true_field(x, t):
    return np.outer(np.exp(-0.05 * np.pi**2 * t), np.sin(np.pi * x)) + 0.5 * np.outer(
        np.exp(-0.05 * (3 * np.pi) ** 2 * t), np.sin(3 * np.pi * x)
    )


def predicted_field(x, t, alpha, amplitudes):
    k = np.arange(1, len(amplitudes) + 1, dtype=float) * np.pi
    return (np.exp(-alpha * t[:, None] * k**2) * amplitudes) @ np.sin(x[:, None] * k).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("results/sparse_inverse_diagnostic_v1")
    )
    parser.add_argument("--pilot", type=Path, default=Path("results/my_classical_pilot_v1"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/sparse_inverse_field_analysis_v1")
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Choose a fresh analysis directory to preserve earlier evidence")
    records_path = args.results / "records.json"
    config_path = args.results / "diagnostic_config.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["length"] != 1.0 or config["alpha_true_for_reporting_only"] != 0.05:
        raise ValueError(
            "This scorer is specific to the archived unit-interval alpha=0.05 benchmark"
        )
    used_paths = [records_path, config_path, Path(__file__).resolve()]
    for seed in sorted({r["data_seed"] for r in records}):
        path = args.pilot / f"observations_n10_seed{seed}.npz"
        with np.load(path) as saved:
            x, t = saved["x"], saved["t"]
            paired = np.sin(np.pi * x) * np.exp(-0.05 * np.pi**2 * t) + 0.5 * np.sin(
                3 * np.pi * x
            ) * np.exp(-0.05 * (3 * np.pi) ** 2 * t)
            np.testing.assert_allclose(paired, saved["clean"], atol=1e-13, rtol=1e-13)
        used_paths.append(path)
    args.out_dir.mkdir(parents=True)
    scores = []
    for rec in records:
        seed, ridge, lower, points = (
            rec[k] for k in ("data_seed", "ridge", "lower_bound", "profile_points")
        )
        filename = f"profile_seed{seed}_ridge{ridge:g}_low{lower:g}_p{points}.npz"
        profile_path = args.results / filename
        with np.load(profile_path) as saved:
            amplitudes = saved["amplitudes"].copy()
        used_paths.append(profile_path)
        scored = dict(rec)
        for density in (201, 401):
            x = np.linspace(0, 1, density)
            t = np.linspace(0, 1, density)
            wx, wt = quadrature_weights(x), quadrature_weights(t)
            np.testing.assert_allclose(wx.sum(), 1.0, atol=1e-14)
            reference = true_field(x, t)
            # Independent analytic construction cross-check before using the scorer.
            known = predicted_field(x, t, 0.05, np.array([1, 0, 0.5, 0, 0, 0]))
            np.testing.assert_allclose(known, reference, rtol=1e-13, atol=1e-13)
            prediction = predicted_field(x, t, rec["alpha_hat"], amplitudes)
            scored[f"field_relative_l2_{density}"] = relative_error(
                prediction, reference, np.outer(wt, wx)
            )
            scored[f"ic_relative_l2_{density}"] = relative_error(prediction[0], reference[0], wx)
        scores.append(scored)
    (args.out_dir / "field_scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "classification": "Post-hoc derived evaluation of saved fits; no new fitting or tuning",
        "reference": "alpha=0.05; IC=sin(pi*x)+0.5*sin(3*pi*x); x,t in [0,1]",
        "quadrature": "Tensor-product trapezoidal weights on 201 and 401 point grids",
        "input_and_scorer_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in used_paths
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# Independent field scoring of the saved diagnostic fits",
        "",
        "Post-hoc evaluation, not tuning. All errors below are percentages.",
        "Field/IC scores use 401-point trapezoidal quadrature; "
        "the 201-point scores are also saved.",
        "Only two selected development datasets are represented, regardless of the number of fits.",
        "",
        "| seed | lambda | lower alpha | search grid "
        "| alpha error | field L2 error | IC L2 error |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in scores:
        lines.append(
            f"| {r['data_seed']} | {r['ridge']:g} | {r['lower_bound']:g} | {r['profile_points']} "
            f"| {100 * r['relative_alpha_error']:.4f}% | {100 * r['field_relative_l2_401']:.4f}% "
            f"| {100 * r['ic_relative_l2_401']:.4f}% |"
        )
    change = max(abs(r["field_relative_l2_401"] - r["field_relative_l2_201"]) for r in scores)
    lines += [
        "",
        "Largest field-error change between quadrature grids: "
        f"{100 * change:.6g} percentage points.",
        "This is a quadrature consistency check, not a statistical confidence interval.",
    ]
    summary = "\n".join(lines) + "\n"
    (args.out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
