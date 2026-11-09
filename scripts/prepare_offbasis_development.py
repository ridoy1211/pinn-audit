"""Prepare off-basis pilot configuration after independent truth verification."""

import hashlib
import json
from pathlib import Path

import offbasis_truth as truth

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = ROOT / "configs/inverse/offbasis_development.json"
    if destination.exists():
        raise FileExistsError(f"Preserve existing configuration: {destination}")
    verified = ROOT / "results/offbasis_truth_verification_v1"
    provenance = json.loads((verified / "provenance.json").read_text())
    for name, expected in provenance["output_sha256"].items():
        if sha(verified / name) != expected:
            raise RuntimeError(f"Truth-verification artifact changed: {name}")
    if sha(Path(truth.__file__)) != provenance["truth_source_sha256"]:
        raise RuntimeError("Truth generator changed after verification.")
    verification = json.loads((verified / "verification.json").read_text())
    if verification["passed"] is not True:
        raise RuntimeError("Truth verification has not passed.")
    source = ROOT / "configs/inverse/timing_evaluation.json"
    cfg = json.loads(source.read_text())
    previous_names = (
        "broader_comparison.json",
        "timing_development.json",
        "timing_evaluation.json",
    )
    used = {876543, 876544}
    for name in previous_names:
        previous = json.loads((ROOT / "configs/inverse" / name).read_text())
        used.update(s for group in previous["designs"].values() for s in group)
    designs = {"10": [1006000, 1006001], "50": [1007000, 1007001]}
    seeds = [s for group in designs.values() for s in group]
    if len(set(seeds)) != 4 or used.intersection(seeds):
        raise RuntimeError("Pilot seeds overlap listed prior data seeds.")
    cfg.update(
        study_id="offbasis_heat_development_v1",
        phase="development",
        designs=designs,
        min_complete_pairs=2,
    )
    # Do not encode the true field as the original finite-sine family structure.
    cfg.pop("families")
    cfg.pop("analysis_plan")
    cfg["truth"] = {
        "label": "C_localized",
        "definition": "x*(1-x)*exp(-(x-center)^2/(2*width^2))/normalization",
        "center": truth.CENTER,
        "width": truth.WIDTH,
        "peak_location": truth.PEAK_LOCATION,
        "normalization": truth.NORMALIZATION,
        "spectral_modes": 256,
        "quadrature_points": 1024,
        "initial_scoring": "direct_definition",
        "verified_generator_sha256": provenance["truth_source_sha256"],
        "verification_provenance_sha256": sha(verified / "provenance.json"),
    }
    cfg["timing_schedules"] = {k: cfg["timing_schedules"][k] for k in ("mixed", "late")}
    cfg["scoring"] = {
        "initial_field": "direct_definition",
        "full_field": "verified_spectral_solution_with_direct_t0",
        "report_signal_rms_over_noise_sigma": True,
    }
    destination.write_text(
        json.dumps(cfg, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Created: {destination}")
    print("Independent development blocks: 4")
    print("Datasets: 8; planned fits: 8 CV12 + 8 PINN-1000")
    print("Data seeds: N=10 [1006000,1006001]; N=50 [1007000,1007001]")
    print("PINN training seed: 101; schedules: mixed, late")
    print("Verified truth source and prior data-seed separation.")
    print("No observations generated or fitting performed.")


if __name__ == "__main__":
    main()
