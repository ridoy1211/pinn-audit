#!/usr/bin/env python3
"""Development-only data-based tuning; see docs/DATA_ONLY_TUNING_PLAN.md."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from pinnaudit.config import ProblemConfig
from pinnaudit.identify import make_observations
from pinnaudit.io import current_code_revision
from pinnaudit.problems.heat import Heat1D
from pinnaudit.spectral_inverse import fit_spectral
from score_sparse_diagnostic import predicted_field, quadrature_weights, relative_error, true_field


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def balanced_folds(n, seed):
    if n < 10:
        raise ValueError("This five-fold study requires at least ten observations")
    labels = np.empty(n, dtype=int)
    labels[np.random.default_rng(seed).permutation(n)] = np.arange(n) % 5
    return labels


def cross_validate(x, t, y, labels, ridges):
    """Select from noisy observations only; simulation truth is not an argument."""
    if not (x.shape == t.shape == y.shape == labels.shape):
        raise ValueError("Coordinates, target and fold labels must have matching shapes")
    if set(np.unique(labels)) != set(range(5)):
        raise ValueError("All five validation folds must be present")
    losses, fits, details = [], [], []
    for ridge in ridges:
        squared_error_sum = 0.0
        candidate_fits = []
        for fold in range(5):
            validation = labels == fold
            training = ~validation
            fit = fit_spectral(
                x[training],
                t[training],
                y[training],
                modes=6,
                ridge=ridge,
                alpha_bounds=(1e-5, 1.0),
                profile_points=401,
            )
            residual = fit.predict(x[validation], t[validation]) - y[validation]
            loss = float(np.sum(residual**2))
            if not np.isfinite(loss):
                raise RuntimeError("Nonfinite validation loss; retain and investigate this run")
            squared_error_sum += loss
            candidate_fits.append(fit)
            details.append(
                {
                    "ridge": ridge,
                    "fold": fold,
                    "n_validation": int(validation.sum()),
                    "validation_sse": loss,
                    "alpha_hat": fit.alpha_hat,
                    "at_bound": fit.at_bound,
                    "fit_wall_time_s": fit.wall_time_s,
                }
            )
        losses.append(squared_error_sum / y.size)
        fits.append(candidate_fits)
    selected = min(range(len(ridges)), key=lambda j: (losses[j], ridges[j]))
    return selected, losses, fits, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/data_only_tuning_development_v1")
    )
    parser.add_argument("--smoke", action="store_true", help="Separate software-check seeds only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    imported = inspect.getsourcefile(fit_spectral)
    if imported is None or Path(imported).resolve() != root / "src/pinnaudit/spectral_inverse.py":
        raise RuntimeError("Wrong imported checkout; reinstall this checkout's editable package")
    if args.out_dir.exists():
        raise FileExistsError("Choose a fresh directory; previous evidence must be preserved")
    config = {
        "status": "software_smoke_only" if args.smoke else "exploratory_development",
        "datasets": (
            [[10, 987000], [50, 987001]]
            if args.smoke
            else [[10, s] for s in range(1000, 1010)] + [[50, s] for s in range(2000, 2010)]
        ),
        "ridges": [0.0, 1e-8, 1e-6, 1e-4, 1e-2],
        "modes": 6,
        "alpha_bounds": [1e-5, 1.0],
        "profile_points": 401,
        "noise_fraction_of_sample_clean_std": 0.05,
        "fold_seed_offset": 30000,
        "folds": 5,
        "selection": "minimum pooled validation MSE; exact tie: smaller lambda",
        "alpha_true_scoring_and_generation_only": 0.05,
    }
    source_paths = [
        Path(__file__).resolve(),
        root / "scripts/score_sparse_diagnostic.py",
        root / "docs/DATA_ONLY_TUNING_PLAN.md",
        root / "src/pinnaudit/config.py",
        root / "src/pinnaudit/identify.py",
        root / "src/pinnaudit/spectral_inverse.py",
        root / "src/pinnaudit/io.py",
        root / "src/pinnaudit/problems/heat.py",
    ]
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
        "config_sha256": hashlib.sha256((args.out_dir / "config.json").read_bytes()).hexdigest(),
    }
    write_json(args.out_dir / "manifest.json", manifest)
    grid = np.linspace(0, 1, 401)
    weights = quadrature_weights(grid)
    truth = true_field(grid, grid)
    records, selections = [], []
    try:
        for index, (n, seed) in enumerate(config["datasets"]):
            started = datetime.now(UTC).isoformat()
            clock = time.perf_counter()
            obs = make_observations(Heat1D(ProblemConfig(name="heat1d")), n, 0.05, seed)
            labels = balanced_folds(n, 30000 + seed)
            stem = f"n{n}_seed{seed}"
            # Save inputs before fitting so even an interrupted dataset remains auditable.
            np.savez_compressed(
                args.out_dir / f"observations_{stem}.npz",
                x=obs.x,
                t=obs.t,
                y=obs.u,
                clean=obs.clean,
                sigma=obs.sigma,
                fold_labels=labels,
            )
            cv_clock = time.perf_counter()
            selected, losses, cv_fits, fold_records = cross_validate(
                obs.x, obs.t, obs.u, labels, config["ridges"]
            )
            tuning_seconds = time.perf_counter() - cv_clock
            # Fit every candidate before truth-based evaluation. Choice is already fixed.
            full_fits = [
                fit_spectral(
                    obs.x,
                    obs.t,
                    obs.u,
                    modes=6,
                    ridge=ridge,
                    alpha_bounds=(1e-5, 1.0),
                    profile_points=401,
                )
                for ridge in config["ridges"]
            ]
            for j, (ridge, fit) in enumerate(zip(config["ridges"], full_fits, strict=True)):
                prediction = predicted_field(grid, grid, fit.alpha_hat, fit.amplitudes)
                records.append(
                    {
                        "n_obs": n,
                        "data_seed": seed,
                        "ridge": ridge,
                        "cv_selected": j == selected,
                        "validation_mse": losses[j],
                        "alpha_hat": fit.alpha_hat,
                        "at_bound": fit.at_bound,
                        "relative_alpha_error": abs(fit.alpha_hat - 0.05) / 0.05,
                        "field_relative_l2": relative_error(
                            prediction, truth, np.outer(weights, weights)
                        ),
                        "ic_relative_l2": relative_error(prediction[0], truth[0], weights),
                        "full_fit_wall_time_s": fit.wall_time_s,
                    }
                )
            np.savez_compressed(
                args.out_dir / f"fits_{stem}.npz",
                full_alpha=np.array([f.alpha_hat for f in full_fits]),
                full_amplitudes=np.array([f.amplitudes for f in full_fits]),
                full_profile_alpha=np.array([f.profile_alpha for f in full_fits]),
                full_profile_objective=np.array([f.profile_objective for f in full_fits]),
                cv_alpha=np.array([[f.alpha_hat for f in group] for group in cv_fits]),
                cv_amplitudes=np.array([[f.amplitudes for f in group] for group in cv_fits]),
                cv_profile_alpha=np.array([[f.profile_alpha for f in group] for group in cv_fits]),
                cv_profile_objective=np.array(
                    [[f.profile_objective for f in group] for group in cv_fits]
                ),
            )
            write_json(args.out_dir / f"fold_scores_{stem}.json", fold_records)
            selections.append(
                {
                    "n_obs": n,
                    "data_seed": seed,
                    "selected_ridge": config["ridges"][selected],
                    "validation_mse": losses,
                    "tuning_elapsed_s": tuning_seconds,
                    "dataset_total_elapsed_s": time.perf_counter() - clock,
                    "run_started_utc": started,
                    "completed_utc": datetime.now(UTC).isoformat(),
                }
            )
            write_json(args.out_dir / "records.json", records)
            write_json(args.out_dir / "selections.json", selections)
            print(
                f"Completed dataset {index + 1}/{len(config['datasets'])}: N={n}, seed={seed}",
                flush=True,
            )
        lines = [
            "# Data-only tuning development study",
            "",
            f"Status: {config['status']}",
            "All recovery errors below are percentages. No PINNs trained.",
            "Datasets are development data, not held-out confirmation.",
            "",
            "| N | method | datasets | median alpha error | mean alpha error | max alpha error "
            "| median field error | median IC error | bound hits |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        comparison_notes = []
        for n in (10, 50):
            for method in ("unregularized", "fixed_1e-6", "CV_selected"):
                rows = [
                    r
                    for r in records
                    if r["n_obs"] == n
                    and (
                        (method == "CV_selected" and r["cv_selected"])
                        or (method == "unregularized" and r["ridge"] == 0)
                        or (method == "fixed_1e-6" and r["ridge"] == 1e-6)
                    )
                ]
                errors = [r["relative_alpha_error"] for r in rows]
                lines.append(
                    f"| {n} | {method} | {len(rows)} | {100 * np.median(errors):.4g} "
                    f"| {100 * np.mean(errors):.4g} | {100 * max(errors):.4g} "
                    f"| {100 * np.median([r['field_relative_l2'] for r in rows]):.4g} "
                    f"| {100 * np.median([r['ic_relative_l2'] for r in rows]):.4g} "
                    f"| {sum(r['at_bound'] for r in rows)} |"
                )
            chosen = [s["selected_ridge"] for s in selections if s["n_obs"] == n]
            counts = {f"{v:g}": chosen.count(v) for v in config["ridges"]}
            comparison_notes.append(f"N={n} selected-lambda counts: {counts}")
            baseline = {r["data_seed"]: r for r in records if r["n_obs"] == n and r["ridge"] == 0}
            selected_rows = [r for r in records if r["n_obs"] == n and r["cv_selected"]]
            for metric in ("relative_alpha_error", "field_relative_l2"):
                differences = [r[metric] - baseline[r["data_seed"]][metric] for r in selected_rows]
                wins = sum(d < 0 for d in differences)
                ties = sum(d == 0 for d in differences)
                losses = sum(d > 0 for d in differences)
                comparison_notes.append(
                    f"N={n}, {metric}, CV vs unregularized paired wins/ties/losses: "
                    f"{wins}/{ties}/{losses} "
                    "(strict numerical comparisons, not a significance test)."
                )
        lines += ["", *comparison_notes, ""]
        lines += [
            "All candidates, fold fits, inputs, and selected fits are retained.",
            "A lower validation error does not guarantee better coefficient recovery.",
            "See docs/DATA_ONLY_TUNING_PLAN.md; this is not a fair timing benchmark.",
        ]
        summary = "\n".join(lines) + "\n"
        (args.out_dir / "summary.md").write_text(summary, encoding="utf-8")
        manifest.update(
            status="complete",
            completed_utc=datetime.now(UTC).isoformat(),
            output_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.out_dir.iterdir())
                if p.name != "manifest.json"
            },
        )
        write_json(args.out_dir / "manifest.json", manifest)
        print(summary)
    except Exception as exc:
        manifest.update(status="failed", failed_utc=datetime.now(UTC).isoformat(), error=repr(exc))
        write_json(args.out_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
