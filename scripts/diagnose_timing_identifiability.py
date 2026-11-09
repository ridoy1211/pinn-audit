"""Post-evaluation explanatory diagnostics; no PINN retraining or model selection."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pinnaudit.spectral_inverse import heat_design

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    source = ROOT / "results/timing_evaluation_v1"
    out = ROOT / "results/timing_identifiability_v1"
    if out.exists():
        raise FileExistsError(f"Preserve existing diagnostics: {out}")
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Expected a completed evaluation.")
    print("Verifying evaluation artifacts...", flush=True)
    for name, expected in manifest["output_sha256"].items():
        if sha(source / name) != expected:
            raise RuntimeError(f"Artifact changed: {name}")
    cfg = json.loads((source / "config.json").read_text())
    records = json.loads((source / "records.json").read_text())
    classical = [r for r in records if r["method"] == "cv12"]
    neural = {r["case_id"]: r for r in records if r["method"] == "pinn_1000"}
    if len(classical) != 360 or len(neural) != 360:
        raise RuntimeError("Expected 360 paired datasets.")
    out.mkdir()
    (out / "profiles").mkdir()
    sensitivity, ranks, boundary = [], [], []
    with threadpool_limits(limits=1):
        for number, r in enumerate(classical, 1):
            case = r["case_id"]
            identity = {k: r[k] for k in ("case_id", "n_obs", "family", "timing", "data_seed")}
            with np.load(source / f"data/{case}.npz", allow_pickle=False) as d:
                x, t, y = (d[k].copy() for k in ("x", "t", "y"))
            folder = source / r["artifact_path"]
            selection = json.loads((folder / "selection.json").read_text())
            payload = {}
            with np.load(folder / "candidate_fits.npz", allow_pickle=False) as fits:
                for i, ridge in enumerate(cfg["ridges"]):
                    prefix = f"full{i}_"
                    alpha = float(fits[prefix + "alpha"])
                    a = fits[prefix + "amplitudes"]
                    B = heat_design(x, t, alpha, 1.0, 12)
                    residual_mse = float(np.mean((B @ a - y) ** 2))
                    penalty = float(ridge * np.sum(np.arange(1, 13, dtype=float) ** 4 * a**2))
                    np.testing.assert_allclose(
                        residual_mse + penalty, fits[prefix + "objective"], rtol=1e-9, atol=1e-13
                    )
                    sensitivity.append(
                        {
                            **identity,
                            "ridge": ridge,
                            "alpha_hat": alpha,
                            "selected_by_cv": i == selection["selected_index"],
                            "validation_mse": selection["validation_mse"][i],
                            "data_mse": residual_mse,
                            "penalty": penalty,
                            "amplitude_l2": float(np.linalg.norm(a)),
                        }
                    )
                    payload[f"ridge_{i}_alpha"] = fits[prefix + "profile_alpha"].copy()
                    payload[f"ridge_{i}_objective"] = fits[prefix + "profile_objective"].copy()
            payload["ridges"] = np.array(cfg["ridges"])
            grid = np.unique(
                np.r_[
                    np.geomspace(*cfg["alpha_bounds"], 121),
                    cfg["alpha_true_generation_and_scoring_only"],
                    r["alpha_hat"],
                    neural[case]["alpha_hat"],
                ]
            )
            raw_profiles = []
            for alpha in grid:
                B = heat_design(x, t, float(alpha), 1.0, 12)
                singular = np.linalg.svd(B, compute_uv=False)
                tol = max(B.shape) * np.finfo(float).eps * singular[0]
                rank = int(np.sum(singular > tol))
                a, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
                mse = float(np.mean((B @ a - y) ** 2))
                raw_profiles.append(mse)
                ranks.append(
                    {
                        **identity,
                        "alpha": float(alpha),
                        "modes": 12,
                        "numerical_rank": rank,
                        "rank_tolerance": float(tol),
                        "column_nullity_at_tolerance": 12 - rank,
                        "full_row_rank_at_tolerance": rank == len(x),
                        "largest_singular_value": float(singular[0]),
                        "smallest_returned_singular_value": float(singular[-1]),
                        "rectangular_singular_ratio": float(singular[0] / singular[-1])
                        if singular[-1] > 0
                        else "inf",
                        "unregularized_data_mse": mse,
                        "unregularized_amplitude_l2": float(np.linalg.norm(a)),
                    }
                )
            payload["unregularized_alpha"] = grid
            payload["unregularized_mse"] = np.array(raw_profiles)
            np.savez_compressed(out / "profiles" / f"{case}.npz", **payload)
            for record in (r, neural[case]):
                lo, hi = cfg["alpha_bounds"]
                alpha = record["alpha_hat"]
                boundary.append(
                    {
                        **identity,
                        "method": record["method"],
                        "alpha_hat": alpha,
                        "alpha_over_lower_bound": alpha / lo,
                        "upper_bound_over_alpha": hi / alpha,
                        "fraction_of_log_interval": float(np.log(alpha / lo) / np.log(hi / lo)),
                        "reported_at_bound": record["at_bound"],
                    }
                )
            if number % 30 == 0:
                print(f"Diagnosed {number}/360 datasets", flush=True)
    csv_write(out / "regularization_sensitivity.csv", sensitivity)
    csv_write(out / "rank_profiles.csv", ranks)
    csv_write(out / "boundary_distances.csv", boundary)
    notes = """# Identifiability diagnostics: stage 1

These are post-evaluation explanatory analyses, not new confirmatory tests.
Saved penalized profiles minimize data MSE plus lambda*sum(j^4*a_j^2).
They are not likelihood-only profiles or confidence intervals.
Unregularized profiles use NumPy least squares with its default rank cutoff.
Rank loss in floating-point arithmetic is not proof of exact structural rank loss.
For N=10 and M=12, full row rank implies at least two coefficient null directions.
A finite ratio of returned singular values does not imply unique coefficients.
When full row rank holds across nearby alpha, free coefficients can interpolate
the finite data for each such alpha. Ridge restricts the solutions but does not
create data-only identification. Extremely large coefficients may be required.
Boundary distances are continuous diagnostics; no new binary failure threshold
has been selected from the evaluation results.

Still required: initial-state-flexibility sensitivity and independent PINN rollout.
"""
    (out / "README.md").write_text(notes, encoding="utf-8")
    provenance = {
        "input_manifest_sha256": sha(source / "manifest.json"),
        "script_sha256": sha(Path(__file__)),
        "numpy": np.__version__,
        "output_sha256": {
            p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("IDENTIFIABILITY STAGE 1 COMPLETE")
    print(f"360 profile bundles; {len(sensitivity)} regularization fits inspected.")
    print(f"{len(ranks)} rank/profile diagnostic points; 720 boundary-distance records.")
    print(f"Output: {out}")
    print("No original results changed; no PINNs retrained.")


if __name__ == "__main__":
    main()
