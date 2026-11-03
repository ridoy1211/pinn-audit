#!/usr/bin/env python3
"""Development-only classical pilot. This does not establish PINN superiority.

Run from the repository root: python scripts/pilot_regularization.py
Only seeds 0..4 are used. All declared ridge values are reported; no selection by
the known truth occurs. Future tuning/confirmation needs a separately frozen plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from pinnaudit.config import ProblemConfig
from pinnaudit.identify import fit_lsq, make_observations
from pinnaudit.problems.heat import Heat1D
from pinnaudit.spectral_inverse import fit_spectral


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/spectral_pilot"))
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "records.json").exists():
        raise FileExistsError("Use a fresh output directory to preserve earlier pilot evidence")
    config = {
        "status": "exploratory_classical_only",
        "observations": [10, 50],
        "noise": 0.05,
        "data_seeds": list(range(5)),
        "modes": 6,
        "ridges": [0.0, 1e-6, 1e-4, 1e-2],
        "alpha_bounds": [0.001, 1.0],
        "profile_points": 101,
        "alpha_true_for_scoring_only": 0.05,
        "length": 1.0,
        "fd_grid": [161, 161],
        "fd_alpha0": 0.2,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
        )
        revision = revision_result.stdout.strip() if revision_result.returncode == 0 else "unknown"
        tracked_diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"], check=False, capture_output=True, text=True
        ).stdout
    except OSError:
        revision, tracked_diff = "unknown", ""
    (out / "tracked_changes.patch").write_text(tracked_diff, encoding="utf-8")
    manifest = {
        "started_utc": datetime.now(UTC).isoformat(),
        "base_git_revision": revision,
        "working_tree_may_include_new_files": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for pattern in (
                "src/pinnaudit/*.py",
                "src/pinnaudit/problems/*.py",
                "src/pinnaudit/solvers/*.py",
                "scripts/pilot_regularization.py",
            )
            for p in Path(".").glob(pattern)
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    problem = Heat1D(ProblemConfig(name="heat1d"))
    records = []
    for n in config["observations"]:
        for seed in config["data_seeds"]:
            obs = make_observations(problem, n, config["noise"], seed)
            np.savez_compressed(
                out / f"observations_n{n}_seed{seed}.npz",
                x=obs.x,
                t=obs.t,
                y=obs.u,
                clean=obs.clean,
                sigma=obs.sigma,
            )
            started = datetime.now(UTC).isoformat()
            fd = fit_lsq(problem, obs, alpha0=0.2, ic_modes=6)
            records.append(
                {
                    "method": "fd_unregularized",
                    "n_obs": n,
                    "data_seed": seed,
                    "ridge": None,
                    "alpha_hat": fd.alpha_hat,
                    "rel_alpha_error": abs(fd.alpha_hat - 0.05) / 0.05,
                    "wall_time_s": fd.wall_time_s,
                    "success": fd.meta["success"],
                    "run_started_utc": started,
                    "completed_utc": datetime.now(UTC).isoformat(),
                }
            )
            for ridge in config["ridges"]:
                started = datetime.now(UTC).isoformat()
                fit = fit_spectral(obs.x, obs.t, obs.u, modes=6, ridge=ridge)
                records.append(
                    {
                        "method": "spectral",
                        "n_obs": n,
                        "data_seed": seed,
                        "ridge": ridge,
                        "alpha_hat": fit.alpha_hat,
                        "rel_alpha_error": abs(fit.alpha_hat - 0.05) / 0.05,
                        "wall_time_s": fit.wall_time_s,
                        "at_bound": fit.at_bound,
                        "run_started_utc": started,
                        "completed_utc": datetime.now(UTC).isoformat(),
                    }
                )
                np.savez_compressed(
                    out / f"profile_n{n}_seed{seed}_ridge{ridge:g}.npz",
                    alpha=fit.profile_alpha,
                    objective=fit.profile_objective,
                    amplitudes=fit.amplitudes,
                )
            (out / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = [
        "# Exploratory classical-only pilot",
        "",
        "Five development data seeds; no PINNs were trained by this script. All lambda values",
        "are shown. The smallest error is not a justified selection rule for confirmation.",
        "",
        "| N | method | ridge | median relative alpha error | median seconds |",
        "|---|---|---|---|---|",
    ]
    for n in config["observations"]:
        for method, ridge in [
            ("fd_unregularized", None),
            *[("spectral", v) for v in config["ridges"]],
        ]:
            subset = [
                r
                for r in records
                if r["n_obs"] == n and r["method"] == method and r["ridge"] == ridge
            ]
            error = np.median([r["rel_alpha_error"] for r in subset])
            seconds = np.median([r["wall_time_s"] for r in subset])
            lines.append(f"| {n} | {method} | {ridge} | {error:.6g} | {seconds:.6g} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
