"""Controlled timing study: job-planning stage.

This version validates the development configuration and prints jobs.
It does not generate observations or run fitting.
"""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = ("early", "mixed", "late")


def load_config(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))

    if cfg["phase"] != "development":
        raise ValueError("This runner currently supports development only.")
    if set(cfg["families"]) != {"A", "B"}:
        raise ValueError("Expected initial fields A and B.")
    if set(cfg["designs"]) != {"10", "50"}:
        raise ValueError("Expected observation counts 10 and 50.")
    if cfg["classical_methods"] != ["cv12"]:
        raise ValueError("Expected only CV12.")
    if cfg["training_seeds"] != [101]:
        raise ValueError("Expected primary training seed 101.")
    if set(cfg["neural_budgets"]) != {"pinn_1000"}:
        raise ValueError("Expected only PINN-1000.")

    expected_schedules = {
        "early": {"window": [0.01, 0.10]},
        "mixed": {
            "early_window": [0.01, 0.10],
            "late_window": [0.20, 0.80],
            "early_fraction": 0.5,
        },
        "late": {"window": [0.20, 0.80]},
    }
    if cfg["timing_schedules"] != expected_schedules:
        raise ValueError("Timing configuration does not match the generator.")

    all_seeds = []
    for seeds in cfg["designs"].values():
        if len(seeds) != 3:
            raise ValueError("Expected three development blocks per N.")
        if any(type(seed) is not int or seed < 0 for seed in seeds):
            raise ValueError("Data seeds must be nonnegative integers.")
        all_seeds.extend(seeds)

    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Development data seeds must be distinct.")

    # Ensure inherited scientific settings have not changed accidentally.
    original_path = ROOT / "configs/inverse/broader_comparison.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    changed_keys = {
        "study_id", "designs", "training_seeds",
        "neural_budgets", "min_complete_pairs",
    }
    for key, value in original.items():
        if key not in changed_keys and cfg.get(key) != value:
            raise ValueError(f"Unexpected change to original setting: {key}")

    if cfg["neural_budgets"]["pinn_1000"] != original["neural_budgets"]["pinn_1000"]:
        raise ValueError("PINN-1000 optimization budget changed.")

    original_seeds = {
        seed for seeds in original["designs"].values() for seed in seeds
    }
    if original_seeds.intersection(all_seeds):
        raise ValueError("Development seeds overlap the original study.")

    return cfg


def build_jobs(cfg):
    jobs = []
    for n in (10, 50):
        for seed in cfg["designs"][str(n)]:
            for family in ("A", "B"):
                for timing in SCHEDULES:
                    case_id = f"n{n}_seed{seed}_{family}_{timing}"
                    for method, training_seed in (
                        ("cv12", None),
                        ("pinn_1000", 101),
                    ):
                        suffix = (
                            method if training_seed is None
                            else f"{method}_train{training_seed}"
                        )
                        jobs.append({
                            "case_id": case_id,
                            "job_id": f"{case_id}/{suffix}",
                            "n_obs": n,
                            "data_seed": seed,
                            "family": family,
                            "timing": timing,
                            "method": method,
                            "training_seed": training_seed,
                        })

    identifiers = [job["job_id"] for job in jobs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate job identifiers.")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/inverse/timing_development.json",
    )
    parser.add_argument("--plan", action="store_true", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    jobs = build_jobs(cfg)
    cases = {job["case_id"] for job in jobs}

    if len(cases) != 36 or len(jobs) != 72:
        raise ValueError("Expected exactly 36 datasets and 72 fitting jobs.")

    print(f"Study: {cfg['study_id']}")
    print(f"Datasets: {len(cases)}")
    print(f"Fitting jobs: {len(jobs)}")
    for method in ("cv12", "pinn_1000"):
        count = sum(job["method"] == method for job in jobs)
        print(f"  {method}: {count}")

    print("\nFirst six jobs:")
    for job in jobs[:6]:
        print(f"  {job['job_id']}")

    print("\nPlan checks passed. No data generated or training performed.")


if __name__ == "__main__":
    main()