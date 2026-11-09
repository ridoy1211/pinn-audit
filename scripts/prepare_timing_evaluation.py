"""Prepare an evaluation configuration without generating observations."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "configs/inverse"
    original = json.loads((folder / "broader_comparison.json").read_text())
    cfg = json.loads((folder / "timing_development.json").read_text())
    destination = folder / "timing_evaluation.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")

    used = {876543, 876544}
    for previous in (original, cfg):
        used.update(seed for seeds in previous["designs"].values() for seed in seeds)
    designs = {
        "10": list(range(906000, 906030)),
        "50": list(range(907000, 907030)),
    }
    seeds = [seed for group in designs.values() for seed in group]
    if len(set(seeds)) != 60 or used.intersection(seeds):
        raise ValueError("Evaluation seed lists overlap each other or prior listed seeds.")

    cfg.update(
        study_id="joint_heat_timing_evaluation_v1",
        phase="evaluation",
        designs=designs,
        ridges=[1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
        min_complete_pairs=30,
    )
    cfg["analysis_plan"] = {
        "primary_stratum": {"n_obs": 10, "family": "B"},
        "primary_outcome": "diffusivity_relative_absolute_error_percent",
        "primary_contrast": "median_of_blockwise_D_mixed_minus_D_late",
        "D_definition": "PINN_error_percent_minus_CV12_error_percent",
        "bootstrap_unit": "whole_design_noise_block_within_observation_count",
        "bootstrap_reps": 10000,
        "bootstrap_seed": 908000,
        "interval": "percentile_95_percent",
        "secondary_intervals": "pointwise_not_simultaneous",
        "missing_estimates": "report_failures_and_conditional_complete_block_contrasts",
        "finite_capped_estimates": "include_and_flag",
    }
    destination.write_text(
        json.dumps(cfg, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Created: {destination}")
    print("N=10 seeds: 906000 through 906029")
    print("N=50 seeds: 907000 through 907029")
    print("Independent base blocks: 60; datasets: 360; planned fits: 720")
    print("Primary PINN training seed: 101")
    print("Seed overlap checks passed against original broader study, development,")
    print("and software-check data seeds. No evaluation observations generated.")
    print("Runner validation and a source/config freeze are still required.")


if __name__ == "__main__":
    main()
