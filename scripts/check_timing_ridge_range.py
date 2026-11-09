"""Development-only CV12 sensitivity check; preserves the original timing run."""

import argparse
import copy
import inspect
import json
import platform
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from broader_common import write_json
from broader_provenance import check_hashes, sha256, source_hashes
from pinnaudit.spectral_inverse import fit_spectral
from run_broader_study import run_jobs
from run_timing_study import ROOT, build_jobs, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    original = ROOT / "results/timing_development_v1"
    out = ROOT / "results/timing_ridge_check_v1"
    if out.exists():
        raise FileExistsError(f"Output already exists; preserve it: {out}")
    config_path = ROOT / "configs/inverse/timing_development.json"
    base = load_config(config_path)
    data = json.loads((original / "data_manifest.json").read_text())
    if data["config"] != base or data["jobs"] != build_jobs(base):
        raise RuntimeError("Original data configuration or schedule changed.")
    check_hashes(ROOT, data["source_sha256"])
    check_hashes(original, data["data_sha256"])
    if (
        Path(inspect.getsourcefile(fit_spectral)).resolve()
        != (ROOT / "src/pinnaudit/spectral_inverse.py").resolve()
    ):
        raise RuntimeError("Imported package belongs to another checkout.")

    old_records = json.loads((original / "records.json").read_text())
    old = {r["case_id"]: r for r in old_records if r["method"] == "cv12"}
    if len(old) != 36 or any(r["status"] != "completed" for r in old.values()):
        raise RuntimeError("Expected 36 completed original CV12 fits.")
    for record in old.values():
        check_hashes(original / record["artifact_path"], record["artifact_sha256"])

    cfg = copy.deepcopy(base)
    cfg["study_id"] = "joint_heat_timing_ridge_check_v1"
    cfg["ridges"] = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
    jobs = [j for j in build_jobs(cfg) if j["method"] == "cv12"]
    sources = source_hashes(ROOT, config_path)
    sources.update(data["source_sha256"])
    sources["scripts/check_timing_ridge_range.py"] = sha256(Path(__file__))
    out.mkdir()
    (out / "data").mkdir()
    for name in data["data_sha256"]:
        shutil.copy2(original / name, out / name)
    check_hashes(out, data["data_sha256"])
    write_json(out / "config.json", cfg)
    write_json(out / "schedule.json", jobs)
    manifest = {
        "mode": "development_regularization_sensitivity_only",
        "expected_jobs": 36,
        "source_sha256": sources,
        "data_sha256": data["data_sha256"],
        "original_data_manifest_sha256": sha256(original / "data_manifest.json"),
        "config_sha256": sha256(out / "config.json"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    with threadpool_limits(limits=1):
        manifest["threadpools"] = threadpool_info()
        write_json(out / "manifest.json", manifest)
        # Only classical jobs are scheduled; no PINN is imported or trained here.
        records = run_jobs(ROOT, out, cfg, [], jobs, manifest, None)

    comparisons = []
    for record in records:
        if record["status"] != "completed":
            continue
        case = record["case_id"]
        old_folder = original / old[case]["artifact_path"]
        new_folder = out / record["artifact_path"]
        before = json.loads((old_folder / "selection.json").read_text())
        after = json.loads((new_folder / "selection.json").read_text())
        shared = [after["validation_mse"][cfg["ridges"].index(v)] for v in base["ridges"]]
        consistent = bool(np.allclose(shared, before["validation_mse"], rtol=1e-9, atol=1e-12))
        comparisons.append(
            {
                "case_id": case,
                "old_ridge": before["selected_ridge"],
                "new_ridge": after["selected_ridge"],
                "old_best_validation_mse": min(before["validation_mse"]),
                "new_best_validation_mse": min(after["validation_mse"]),
                "shared_candidate_scores_reproduced": consistent,
                "new_alpha_hat": record["alpha_hat"],
                "new_at_bound": record["at_bound"],
            }
        )
    write_json(out / "comparisons.json", comparisons)
    failures = sum(r["status"] != "completed" for r in records)
    mismatches = sum(not c["shared_candidate_scores_reproduced"] for c in comparisons)
    manifest.update(status="finished", failed_jobs=failures, shared_candidate_mismatches=mismatches)
    write_json(out / "manifest.json", manifest)
    print("\nREGULARIZATION CHECK SUMMARY")
    print(f"Completed: {len(comparisons)}/36; unsuccessful: {failures}")
    print(f"Shared-candidate validation mismatches: {mismatches}")
    print("Selected penalties:", dict(Counter(c["new_ridge"] for c in comparisons)))
    for c in comparisons:
        if c["old_ridge"] != c["new_ridge"]:
            print(
                f"{c['case_id']}: {c['old_ridge']:g} -> {c['new_ridge']:g}; "
                f"validation MSE {c['old_best_validation_mse']:.6g} -> "
                f"{c['new_best_validation_mse']:.6g}"
            )
    print(f"Results: {out}")
    print("No evaluation configuration has been changed.")
    if failures or mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
