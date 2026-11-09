"""Prepare separate robustness evaluation; no observations or fitting."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT / "configs/inverse"
    destination = folder / "offbasis_evaluation.json"
    decision = ROOT / "docs/OFFBASIS_DEVELOPMENT_DECISIONS.md"
    if destination.exists() or decision.exists():
        raise FileExistsError("Configuration or decision record exists; preserve it.")
    cfg = json.loads((folder / "offbasis_development.json").read_text())
    source = ROOT / "results/offbasis_development_v1"
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["config"] != cfg or manifest["status"] != "complete":
        raise RuntimeError("Expected completed pilot matching the current configuration.")
    for name, h in manifest["output_sha256"].items():
        if sha(source / name) != h:
            raise RuntimeError(f"Changed development artifact: {name}")
    rows = json.loads((source / "records.json").read_text())
    if len(rows) != 16 or any(r["status"] != "completed" for r in rows):
        raise RuntimeError("Expected 16 completed pilot fits.")
    neural = [r for r in rows if r["method"] == "pinn_1000"]
    if any(r["final_training_trace"]["stage"] != "completed_budget" for r in neural):
        raise RuntimeError("Investigate pilot training termination before evaluation.")
    used = {876543, 876544}
    for name in (
        "broader_comparison.json",
        "timing_development.json",
        "timing_evaluation.json",
        "offbasis_development.json",
    ):
        previous = json.loads((folder / name).read_text())
        used.update(s for group in previous["designs"].values() for s in group)
    designs = {"10": list(range(1106000, 1106030)), "50": list(range(1107000, 1107030))}
    seeds = [s for group in designs.values() for s in group]
    if len(set(seeds)) != 60 or used.intersection(seeds):
        raise RuntimeError("Seed overlap detected.")
    cfg.update(
        study_id="offbasis_heat_evaluation_v1",
        phase="evaluation",
        designs=designs,
        min_complete_pairs=30,
    )
    cfg["analysis_plan"] = {
        "classification": "secondary_offbasis_robustness",
        "main_outcome": "diffusivity_relative_absolute_error_percent",
        "other_outcomes": ["initial_field_relative_L2_percent", "full_field_relative_L2_percent"],
        "D": "PINN_error_minus_CV12_error_within_block_and_schedule",
        "I": "D_mixed_minus_D_late_within_block",
        "estimator": "median_of_paired_block_differences",
        "bootstrap_unit": "whole_design_noise_block_within_N",
        "bootstrap_reps": 10000,
        "bootstrap_seed": 1108000,
        "interval": "95_percent_percentile_pointwise_not_simultaneous",
        "failed_estimates": "report_failures_and_conditional_complete_block_contrasts",
        "finite_capped_estimates": "include_and_flag",
        "relation_to_original": "does_not_replace_original_primary_test",
    }
    differences = [
        100 * abs(r[k] - v) for r in rows for k, v in r.get("refinement_scores", {}).items()
    ]
    selected = [r["selected_ridge"] for r in rows if r["method"] == "cv12"]
    endpoints = sum(v in (min(cfg["ridges"]), max(cfg["ridges"])) for v in selected)
    if endpoints:
        raise RuntimeError("Investigate endpoint selections before freezing this decision.")
    text = f"""# Off-basis development decisions

All 16 development fits completed. All eight PINNs completed their optimization
budget. No classical fit selected an endpoint of the seven-candidate ridge grid.

Across the checked first blocks, the maximum absolute change in reported field
or initial-field relative errors between 401- and 801-point scoring grids was
{max(differences):.8f} percentage points. This is a numerical refinement check,
not a universal error bound. Retain the existing scoring and refinement settings.

Recovery was difficult in several pilot cases. Preserve these outcomes; do not
change the initial field, noise, schedules, model settings, or optimization budget
to improve truth-based performance.

Decision: retain the verified initial field and all pilot fitting settings.
Proceed with 30 fresh blocks per observation count and two schedules:
120 evaluation datasets, 240 fits, 60 independent design/noise blocks.
This is a planning sample size, not a power guarantee.

Data seeds: N=10, 1106000 through 1106029; N=50, 1107000 through 1107029.
Bootstrap seed: 1108000. PINN initialization remains 101.
Comparisons are secondary robustness analyses with pointwise intervals.
The evaluation source/configuration freeze must precede data generation.

Development manifest SHA256: {sha(source / "manifest.json")}
"""
    destination.write_text(
        json.dumps(cfg, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    decision.write_text(text, encoding="utf-8", newline="\n")
    print("Created configs/inverse/offbasis_evaluation.json")
    print("Created docs/OFFBASIS_DEVELOPMENT_DECISIONS.md")
    print("120 datasets; 240 fits; evaluation seeds checked for separation.")
    print("Pilot fitting settings retained. No evaluation observations generated.")


if __name__ == "__main__":
    main()
