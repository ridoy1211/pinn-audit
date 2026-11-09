"""Post-evaluation basis sensitivity: CV-selected 6 and 24 sine modes."""

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from broader_common import CVFailure, cv_spectral, write_json
from pinnaudit.spectral_inverse import fit_spectral
from run_broader_study import evaluate, pack_fit, save_cv, save_npz

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "results/timing_evaluation_v1"
    out = ROOT / "results/timing_basis_flexibility_v1"
    if out.exists():
        raise FileExistsError(f"Preserve the existing diagnostic run: {out}")
    manifest = json.loads((source / "manifest.json").read_text())
    print("Verifying original evaluation...", flush=True)
    for name, h in manifest["output_sha256"].items():
        if sha(source / name) != h:
            raise RuntimeError(f"Changed evaluation artifact: {name}")
    cfg = json.loads((source / "config.json").read_text())
    original = json.loads((source / "records.json").read_text())
    cases = [r for r in original if r["method"] == "cv12"]
    if len(cases) != 360:
        raise RuntimeError("Expected all 360 classical evaluation cases.")
    out.mkdir()
    write_json(
        out / "protocol.json",
        {
            "classification": "post_evaluation_explanatory_sensitivity",
            "modes": [6, 24],
            "reference_modes": 12,
            "cases": "all_360",
            "ridges": cfg["ridges"],
            "selection": "unchanged_five_fold_CV",
            "interpretation": "Changing basis size changes the initial-state assumption. "
            "Six modes exclude family B mode seven. "
            "This is not a truth-selected replacement estimator.",
            "runtime_note": "Timing not directly comparable to original job cost: "
            "only the CV-selected full-data fit is computed here.",
            "input_manifest_sha256": sha(source / "manifest.json"),
            "script_sha256": sha(Path(__file__)),
        },
    )
    records = []
    with threadpool_limits(limits=1):
        for number, reference in enumerate(cases, 1):
            case = reference["case_id"]
            with np.load(source / f"data/{case}.npz", allow_pickle=False) as d:
                obs = {k: d[k].copy() for k in ("x", "t", "y", "fold_labels")}
            for modes in (6, 24):
                folder = out / "jobs" / case / f"cv{modes}"
                folder.mkdir(parents=True)
                row = {
                    k: reference[k] for k in ("case_id", "n_obs", "family", "timing", "data_seed")
                }
                row.update(modes=modes, reference_alpha=reference["alpha_hat"])
                started = time.perf_counter()
                try:
                    cv = cv_spectral(
                        **obs,
                        modes=modes,
                        ridges=cfg["ridges"],
                        alpha_bounds=tuple(cfg["alpha_bounds"]),
                        profile_points=cfg["profile_points"],
                    )
                    save_cv(folder / "candidate_fits.npz", cv.fold_fits)
                    write_json(folder / "fold_scores.json", cv.fold_records)
                    ridge = cfg["ridges"][cv.selected_index]
                    write_json(
                        folder / "selection.json",
                        {
                            "selected_ridge": ridge,
                            "selected_index": cv.selected_index,
                            "validation_mse": cv.losses,
                        },
                    )
                    fit = fit_spectral(
                        obs["x"],
                        obs["t"],
                        obs["y"],
                        modes=modes,
                        ridge=ridge,
                        alpha_bounds=tuple(cfg["alpha_bounds"]),
                        profile_points=cfg["profile_points"],
                    )
                    arrays = {}
                    pack_fit(arrays, "selected", fit)
                    save_npz(folder / "model.npz", **arrays)
                    scores = evaluate(
                        fit.predict, fit.alpha_hat, obs, cfg, reference, folder, False
                    )
                    row.update(
                        status="completed",
                        alpha_hat=fit.alpha_hat,
                        selected_ridge=ridge,
                        at_bound=bool(fit.at_bound),
                        **scores,
                    )
                except CVFailure as exc:
                    save_cv(folder / "candidate_fits.npz", exc.fold_fits)
                    write_json(folder / "fold_scores.json", exc.fold_records)
                    row.update(status="failed", error=repr(exc))
                except Exception as exc:
                    row.update(status="failed", error=repr(exc))
                row["job_seconds"] = time.perf_counter() - started
                row["artifact_sha256"] = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
                write_json(folder / "job.json", row)
                records.append(row)
                write_json(out / "records.json", records)
            if number % 10 == 0:
                print(f"Completed attempts for {number}/360 datasets", flush=True)
    fields = [
        "case_id",
        "n_obs",
        "family",
        "timing",
        "data_seed",
        "modes",
        "status",
        "reference_alpha",
        "alpha_hat",
        "selected_ridge",
        "at_bound",
        "relative_alpha_error",
        "ic_relative_l2",
        "field_relative_l2",
        "job_seconds",
        "error",
    ]
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    write_json(
        out / "provenance.json",
        {
            "output_sha256": {
                p.relative_to(out).as_posix(): sha(p) for p in out.rglob("*") if p.is_file()
            },
            "completed": sum(r["status"] == "completed" for r in records),
            "unsuccessful": sum(r["status"] != "completed" for r in records),
        },
    )
    print("BASIS FLEXIBILITY CHECK COMPLETE")
    print(f"Completed: {sum(r['status'] == 'completed' for r in records)}/720")
    print(f"Unsuccessful: {sum(r['status'] != 'completed' for r in records)}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
