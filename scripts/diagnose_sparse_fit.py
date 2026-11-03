#!/usr/bin/env python3
"""Post-hoc bound/search diagnostic on two existing development datasets.

No observations are generated and no PINNs are trained. See
docs/SPARSE_INVERSE_DIAGNOSTIC_PLAN.md before interpreting these selected examples.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import scipy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinnaudit.io import current_code_revision
from pinnaudit.spectral_inverse import fit_spectral, heat_design


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/my_classical_pilot_v1"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/sparse_inverse_diagnostic_v1")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected_module = root / "src/pinnaudit/spectral_inverse.py"
    actual_module = inspect.getsourcefile(fit_spectral)
    if actual_module is None or Path(actual_module).resolve() != expected_module.resolve():
        raise RuntimeError(
            "Python is importing a different checkout. Reinstall this project's editable package."
        )
    pilot_config_path = args.results / "config.json"
    pilot_config = json.loads(pilot_config_path.read_text(encoding="utf-8"))
    inputs = {seed: args.results / f"observations_n10_seed{seed}.npz" for seed in (0, 1)}
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.out_dir.exists():
        raise FileExistsError("Choose a fresh output directory to preserve previous evidence")
    args.out_dir.mkdir(parents=True)
    config = {
        "status": "post_hoc_exploratory_selected_examples",
        "n_obs": 10,
        "data_seeds": [0, 1],
        "modes": 6,
        "ridges": [0.0, 1e-6],
        "alpha_lower_bounds": [1e-3, 1e-4, 1e-5],
        "alpha_upper_bound": 1.0,
        "profile_points": 401,
        "grid_check_points": 1601,
        "alpha_true_for_reporting_only": pilot_config["alpha_true_for_scoring_only"],
        "length": pilot_config["length"],
        "input_results_directory": str(args.results.resolve()),
    }
    (args.out_dir / "diagnostic_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    source_paths = [
        Path(__file__).resolve(),
        root / "src/pinnaudit/spectral_inverse.py",
        root / "docs/SPARSE_INVERSE_DIAGNOSTIC_PLAN.md",
    ]
    manifest = {
        "started_utc": datetime.now(UTC).isoformat(),
        "code_revision": current_code_revision(),
        "estimator_source_file": str(Path(actual_module).resolve()),
        "script_status": "Exploratory working script; exact script/plan hashes are recorded below",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "source_sha256": {str(p.relative_to(root)): digest(p) for p in source_paths},
        "input_sha256": {
            str(p.resolve()): digest(p) for p in [pilot_config_path, *inputs.values()]
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    data = {}
    for seed, path in inputs.items():
        with np.load(path) as saved:
            data[seed] = (saved["x"].copy(), saved["t"].copy(), saved["y"].copy())

    records = []
    for seed in config["data_seeds"]:
        x, t, y = data[seed]
        for ridge in config["ridges"]:
            for lower in config["alpha_lower_bounds"]:
                grids = [401, 1601] if seed == 1 and lower == 1e-5 else [401]
                for points in grids:
                    start = datetime.now(UTC).isoformat()
                    fit = fit_spectral(
                        x,
                        t,
                        y,
                        length=config["length"],
                        modes=6,
                        ridge=ridge,
                        alpha_bounds=(lower, 1.0),
                        profile_points=points,
                    )
                    prediction = heat_design(x, t, fit.alpha_hat, fit.length, 6) @ fit.amplitudes
                    mse = float(np.mean((prediction - y) ** 2))
                    roughness = float(np.sum(np.arange(1, 7, dtype=float) ** 4 * fit.amplitudes**2))
                    penalty = ridge * roughness
                    if not np.isclose(mse + penalty, fit.objective, rtol=1e-8, atol=1e-14):
                        raise RuntimeError("Objective decomposition disagrees with estimator")
                    alpha_true = config["alpha_true_for_reporting_only"]
                    record = {
                        "data_seed": seed,
                        "ridge": ridge,
                        "lower_bound": lower,
                        "upper_bound": 1.0,
                        "profile_points": points,
                        "alpha_hat": fit.alpha_hat,
                        "at_bound": fit.at_bound,
                        "relative_alpha_error": abs(fit.alpha_hat - alpha_true) / alpha_true,
                        "data_mse": mse,
                        "initial_field_roughness": roughness,
                        "penalty": penalty,
                        "objective": fit.objective,
                        "run_started_utc": start,
                        "completed_utc": datetime.now(UTC).isoformat(),
                        "wall_time_s": fit.wall_time_s,
                    }
                    records.append(record)
                    (args.out_dir / "records.json").write_text(
                        json.dumps(records, indent=2), encoding="utf-8"
                    )
                    np.savez_compressed(
                        args.out_dir
                        / f"profile_seed{seed}_ridge{ridge:g}_low{lower:g}_p{points}.npz",
                        alpha=fit.profile_alpha,
                        objective=fit.profile_objective,
                        amplitudes=fit.amplitudes,
                    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for row, seed in enumerate(config["data_seeds"]):
        for col, ridge in enumerate(config["ridges"]):
            ax = axes[row, col]
            profile_path = args.out_dir / f"profile_seed{seed}_ridge{ridge:g}_low1e-05_p401.npz"
            with np.load(profile_path) as saved:
                ax.loglog(
                    saved["alpha"], saved["objective"], color="#1f5576", label="Profile objective"
                )
            for lower, marker in zip(config["alpha_lower_bounds"], ("o", "s", "^"), strict=True):
                rec = next(
                    r
                    for r in records
                    if r["data_seed"] == seed
                    and r["ridge"] == ridge
                    and r["lower_bound"] == lower
                    and r["profile_points"] == 401
                )
                ax.scatter(
                    rec["alpha_hat"],
                    rec["objective"],
                    marker=marker,
                    s=55,
                    label=f"Fit: lower={lower:g}",
                    zorder=4,
                )
            ax.axvline(
                config["alpha_true_for_reporting_only"],
                color="#986300",
                linestyle="--",
                label="True alpha (diagnostic only)",
            )
            ax.set_title(f"Data seed {seed}; lambda={ridge:g}")
            ax.set_xlabel("Diffusivity alpha")
            ax.set_ylabel("Mean data error + roughness penalty")
            ax.grid(alpha=0.2, which="both")
            ax.legend(fontsize=7)
    fig.suptitle("Exploratory diagnostic: separate objectives for each lambda", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out_dir / "alpha_profiles.png", dpi=160)
    plt.close(fig)

    lines = [
        "# Sparse-fit diagnostic (exploratory)",
        "",
        "Same saved development observations; selected examples; no PINNs trained.",
        "Relative coefficient errors below are percentages. True alpha is used only for reporting.",
        "",
        "| seed | lambda | lower alpha | grid points | alpha estimate | bound hit "
        "| error (%) | data MSE | penalty | total objective |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['data_seed']} | {r['ridge']:g} | {r['lower_bound']:g} | {r['profile_points']} "
            f"| {r['alpha_hat']:.7g} | {r['at_bound']} | {100 * r['relative_alpha_error']:.4g} "
            f"| {r['data_mse']:.6g} | {r['penalty']:.6g} | {r['objective']:.6g} |"
        )
    lines += ["", "## Grid-resolution check", ""]
    for ridge in config["ridges"]:
        chosen = [
            r
            for r in records
            if r["data_seed"] == 1 and r["ridge"] == ridge and r["lower_bound"] == 1e-5
        ]
        coarse, fine = sorted(chosen, key=lambda r: r["profile_points"])
        relative_change = abs(fine["alpha_hat"] - coarse["alpha_hat"]) / coarse["alpha_hat"]
        lines.append(
            f"- lambda={ridge:g}: 401 to 1601 points changes alpha by "
            f"{relative_change:.6g} relative; "
            f"objective changes from {coarse['objective']:.9g} to {fine['objective']:.9g}."
        )
    lines += [
        "",
        "Interpret using docs/SPARSE_INVERSE_DIAGNOSTIC_PLAN.md.",
        "Bound sensitivity does not alone prove",
        "structural non-identifiability; grid agreement does not prove global optimality.",
        "Penalized profiles are not confidence intervals. Preserve all results.",
    ]
    summary = "\n".join(lines) + "\n"
    (args.out_dir / "summary.md").write_text(summary, encoding="utf-8")
    manifest["completed_utc"] = datetime.now(UTC).isoformat()
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
