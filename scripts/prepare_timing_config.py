"""Prepare the timing development configuration without running fits."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "configs/inverse/broader_comparison.json"
    destination = root / "configs/inverse/timing_development.json"

    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing configuration: {destination}"
        )

    cfg = json.loads(source.read_text(encoding="utf-8"))

    original_seeds = {
        seed for seeds in cfg["designs"].values() for seed in seeds
    }
    development_designs = {
        "10": [806000, 806001, 806002],
        "50": [807000, 807001, 807002],
    }
    development_seeds = {
        seed
        for seeds in development_designs.values()
        for seed in seeds
    }
    assert development_seeds.isdisjoint(original_seeds)
    assert development_seeds.isdisjoint({876543, 876544})

    cfg["study_id"] = "joint_heat_timing_development_v1"
    cfg["phase"] = "development"
    cfg["designs"] = development_designs
    cfg["classical_methods"] = ["cv12"]
    cfg["training_seeds"] = [101]
    cfg["neural_budgets"] = {
        "pinn_1000": cfg["neural_budgets"]["pinn_1000"]
    }
    cfg["timing_schedules"] = {
        "early": {"window": [0.01, 0.10]},
        "mixed": {
            "early_window": [0.01, 0.10],
            "late_window": [0.20, 0.80],
            "early_fraction": 0.5,
        },
        "late": {"window": [0.20, 0.80]},
    }
    cfg["min_complete_pairs"] = 3

    text = json.dumps(cfg, indent=2, allow_nan=False) + "\n"
    destination.write_text(text, encoding="utf-8", newline="\n")

    blocks = sum(len(seeds) for seeds in development_designs.values())
    datasets = blocks * len(cfg["families"]) * len(cfg["timing_schedules"])

    print(f"Created: {destination.relative_to(root)}")
    print(f"Independent base blocks: {blocks}")
    print(f"Datasets: {datasets}")
    print(f"Planned fits: {datasets} CV12 + {datasets} PINN-1000")
    print("Development only. No data generated and no training performed.")


if __name__ == "__main__":
    main()