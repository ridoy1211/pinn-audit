"""Save development observations and a manifest. Does not train models."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from broader_common import make_dataset
from run_timing_study import build_jobs, load_config
from timing_common import generate_timing_designs

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config_path = ROOT / "configs/inverse/timing_development.json"
    cfg = load_config(config_path)
    jobs = build_jobs(cfg)

    out = ROOT / "results/timing_development_v1"
    if out.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {out}")

    source_names = [
        "configs/inverse/timing_development.json",
        "configs/inverse/broader_comparison.json",
        "docs/TIMING_PROTOCOL.md",
        "scripts/timing_common.py",
        "scripts/broader_common.py",
        "scripts/run_timing_study.py",
        "scripts/prepare_timing_data.py",
    ]
    source_hashes = {name: sha256(ROOT / name) for name in source_names}
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        text=True,
    )

    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=False)
    data_hashes = {}

    for n in (10, 50):
        for seed in cfg["designs"][str(n)]:
            designs = generate_timing_designs(n, seed, cfg["fold_seed_offset"])
            for family, modes in cfg["families"].items():
                for timing, design in designs.items():
                    case_id = f"n{n}_seed{seed}_{family}_{timing}"
                    obs = make_dataset(
                        design,
                        cfg["alpha_true_generation_and_scoring_only"],
                        modes,
                        cfg["noise_sigma_absolute"],
                    )

                    # Preserve the pairing information for later verification.
                    obs["base_time_draw"] = design["base_time_draw"]
                    obs["mixed_early_mask"] = design["mixed_early_mask"]

                    path = data_dir / f"{case_id}.npz"
                    np.savez_compressed(path, **obs)

                    # Verify that saved arrays round-trip without changes.
                    with np.load(path, allow_pickle=False) as saved:
                        if set(saved.files) != set(obs):
                            raise RuntimeError(f"Missing saved fields: {case_id}")
                        for key, value in obs.items():
                            np.testing.assert_array_equal(saved[key], value)

                    data_hashes[path.relative_to(out).as_posix()] = sha256(path)

    expected = {f"data/{job['case_id']}.npz" for job in jobs}
    if set(data_hashes) != expected or len(data_hashes) != 36:
        raise RuntimeError("Saved datasets do not match the job plan.")

    manifest = {
        "study_id": cfg["study_id"],
        "phase": "development",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "git_status_at_generation": git_status,
        "source_sha256": source_hashes,
        "config": cfg,
        "jobs": jobs,
        "data_sha256": data_hashes,
        "numpy_version": np.__version__,
        "note": "Development data only; no fitting performed.",
    }
    (out / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Saved and reloaded successfully: {len(data_hashes)} datasets")
    print(f"Manifest contains: {len(jobs)} planned fitting jobs")
    print(f"Output: {out}")
    print("No training was performed.")


if __name__ == "__main__":
    main()
