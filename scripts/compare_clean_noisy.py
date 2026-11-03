#!/usr/bin/env python3
"""Exploratory clean/noisy control on identical saved observation coordinates."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from pinnaudit.io import current_code_revision
from pinnaudit.spectral_inverse import fit_spectral, heat_design, project_amplitudes
from score_sparse_diagnostic import predicted_field, quadrature_weights, relative_error, true_field


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("results/my_classical_pilot_v1"))
    parser.add_argument(
        "--diagnostic", type=Path, default=Path("results/sparse_inverse_diagnostic_v1")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/clean_noisy_control_v1"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    module = inspect.getsourcefile(fit_spectral)
    if module is None or Path(module).resolve() != root / "src/pinnaudit/spectral_inverse.py":
        raise RuntimeError("Wrong imported checkout; reinstall this checkout's editable package")
    if args.out_dir.exists():
        raise FileExistsError("Use a fresh output directory; previous evidence is preserved")
    pilot_config = args.pilot / "config.json"
    settings = json.loads(pilot_config.read_text(encoding="utf-8"))
    if settings["length"] != 1 or settings["alpha_true_for_scoring_only"] != 0.05:
        raise ValueError("This control is specific to the archived two-mode benchmark")
    old_path = args.diagnostic / "records.json"
    old_records = json.loads(old_path.read_text(encoding="utf-8"))
    source_paths = [
        Path(__file__).resolve(),
        root / "scripts/score_sparse_diagnostic.py",
        root / "src/pinnaudit/spectral_inverse.py",
        root / "src/pinnaudit/io.py",
        root / "docs/CLEAN_NOISY_CONTROL_PLAN.md",
    ]
    observation_paths = [args.pilot / f"observations_n10_seed{s}.npz" for s in (0, 1)]
    inputs = [pilot_config, old_path, *observation_paths]
    data = {}
    for seed, path in enumerate(observation_paths):
        with np.load(path) as saved:
            data[seed] = {k: saved[k].copy() for k in ("x", "t", "clean", "y")}
        d = data[seed]
        if any(v.shape != (10,) or not np.isfinite(v).all() for v in d.values()):
            raise ValueError("Expected ten finite paired observations")
        known = heat_design(d["x"], d["t"], 0.05, 1.0, 6) @ np.array([1, 0, 0.5, 0, 0, 0])
        np.testing.assert_allclose(known, d["clean"], rtol=1e-13, atol=1e-13)
    config = {
        "status": "exploratory_selected_development_datasets",
        "seeds": [0, 1],
        "targets": ["clean", "noisy"],
        "modes": 6,
        "ridges": [0.0, 1e-6],
        "lower_bounds": [1e-3, 1e-4, 1e-5],
        "upper_bound": 1.0,
        "profile_points": 401,
        "field_quadrature_points": 401,
        "alpha_true_diagnostic_and_scoring_only": 0.05,
        "known_alpha_candidate_is_not_used_to_select_fit": True,
    }
    args.out_dir.mkdir(parents=True)
    write_json(args.out_dir / "config.json", config)
    manifest = {
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "in_progress",
        "code_revision": current_code_revision(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
    }
    write_json(args.out_dir / "manifest.json", manifest)
    grid = np.linspace(0, 1, 401)
    weights = quadrature_weights(grid)
    reference = true_field(grid, grid)
    records = []
    for seed, d in data.items():
        noise_rms = float(np.sqrt(np.mean((d["y"] - d["clean"]) ** 2)))
        for target in config["targets"]:
            y = d["clean"] if target == "clean" else d["y"]
            for ridge in config["ridges"]:
                for lower in config["lower_bounds"]:
                    started = datetime.now(UTC).isoformat()
                    fit = fit_spectral(
                        d["x"],
                        d["t"],
                        y,
                        modes=6,
                        ridge=ridge,
                        alpha_bounds=(lower, 1.0),
                        profile_points=401,
                    )
                    design = heat_design(d["x"], d["t"], fit.alpha_hat, 1.0, 6)
                    mse = float(np.mean((design @ fit.amplitudes - y) ** 2))
                    penalty = float(ridge * np.sum(np.arange(1, 7) ** 4 * fit.amplitudes**2))
                    np.testing.assert_allclose(mse + penalty, fit.objective, rtol=1e-8, atol=1e-14)
                    singular = np.linalg.svd(design, compute_uv=False)
                    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
                    prediction = predicted_field(grid, grid, fit.alpha_hat, fit.amplitudes)
                    # Diagnostic only, after fitting. Never fed back into the estimator.
                    _, true_alpha_objective = project_amplitudes(
                        heat_design(d["x"], d["t"], 0.05, 1.0, 6), y, ridge
                    )
                    gap = fit.objective - true_alpha_objective
                    record = {
                        "seed": seed,
                        "target": target,
                        "ridge": ridge,
                        "lower_bound": lower,
                        "alpha_hat": fit.alpha_hat,
                        "at_bound": fit.at_bound,
                        "relative_alpha_error": abs(fit.alpha_hat - 0.05) / 0.05,
                        "data_mse": mse,
                        "penalty": penalty,
                        "objective": fit.objective,
                        "field_relative_l2": relative_error(
                            prediction, reference, np.outer(weights, weights)
                        ),
                        "ic_relative_l2": relative_error(prediction[0], reference[0], weights),
                        "amplitude_design_rank": int(np.linalg.matrix_rank(design)),
                        "amplitude_design_singular_values": singular.tolist(),
                        "amplitude_design_condition_number": condition,
                        "true_alpha_projected_objective_diagnostic": true_alpha_objective,
                        "objective_gap_vs_true_alpha_candidate": gap,
                        "search_warning": bool(gap > 1e-12 + 1e-8 * true_alpha_objective),
                        "realized_noise_rms": noise_rms,
                        "run_started_utc": started,
                        "completed_utc": datetime.now(UTC).isoformat(),
                        "fit_wall_time_s": fit.wall_time_s,
                    }
                    if target == "noisy":
                        old = next(
                            r
                            for r in old_records
                            if r["data_seed"] == seed
                            and r["ridge"] == ridge
                            and r["lower_bound"] == lower
                            and r["profile_points"] == 401
                        )
                        np.testing.assert_allclose(fit.alpha_hat, old["alpha_hat"], rtol=1e-5)
                        np.testing.assert_allclose(fit.objective, old["objective"], rtol=1e-7)
                    records.append(record)
                    write_json(args.out_dir / "records.json", records)
                    np.savez_compressed(
                        args.out_dir / f"seed{seed}_{target}_ridge{ridge:g}_low{lower:g}.npz",
                        profile_alpha=fit.profile_alpha,
                        profile_objective=fit.profile_objective,
                        alpha_hat=fit.alpha_hat,
                        amplitudes=fit.amplitudes,
                        x=d["x"],
                        t=d["t"],
                        target=y,
                    )
    lines = [
        "# Clean/noisy control (exploratory)",
        "",
        "Same coordinates; two selected development datasets; 24 classical fits; no PINNs.",
        "All recovery errors below are percentages. True alpha is not used for selection.",
        "",
        "| seed | target | lambda | lower alpha | alpha estimate | bound hit "
        "| alpha error (%) | field error (%) | IC error (%) | search warning |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['seed']} | {r['target']} | {r['ridge']:g} | {r['lower_bound']:g} "
            f"| {r['alpha_hat']:.8g} | {r['at_bound']} "
            f"| {100 * r['relative_alpha_error']:.5g} | {100 * r['field_relative_l2']:.5g} "
            f"| {100 * r['ic_relative_l2']:.5g} | {r['search_warning']} |"
        )
    lines += [
        "",
        "Noisy estimates reproduce the archived diagnostic within declared tolerances.",
        "Search warning means a fitted objective is materially worse than a separately",
        "evaluated feasible candidate at true alpha; it is not a confidence statement.",
        "Rank/conditioning in records.json concerns amplitudes conditional on fitted alpha.",
        "See docs/CLEAN_NOISY_CONTROL_PLAN.md for interpretation and limitations.",
    ]
    summary = "\n".join(lines) + "\n"
    (args.out_dir / "summary.md").write_text(summary, encoding="utf-8")
    manifest.update(status="complete", completed_utc=datetime.now(UTC).isoformat())
    write_json(args.out_dir / "manifest.json", manifest)
    print(summary)


if __name__ == "__main__":
    main()
