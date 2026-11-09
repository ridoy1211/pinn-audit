"""Consolidate post-evaluation identifiability and physical-consistency evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TIMINGS = ("early", "mixed", "late")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_read(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure_save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=220)
    fig.savefig(out / f"{name}.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Replay the timing diagnostic report.")
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS,
        help="Results directory containing the saved timing evaluation and diagnostic directories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="New report directory. Defaults to timing_fix2_report_v1 under --results.",
    )
    args = parser.parse_args()
    results = args.results.resolve()
    out = (args.out_dir or results / "timing_fix2_report_v1").resolve()
    if out.exists():
        parser.error(f"--out-dir already exists: {out}. Choose a new directory.")
    inputs = {}
    names = (
        "timing_identifiability_v1",
        "timing_basis_flexibility_v1",
        "timing_rollout_v1",
        "timing_rollout_refinement_v1",
    )
    input_folders = (*names, "timing_evaluation_v1")
    if any(out.is_relative_to((results / name).resolve()) for name in input_folders):
        parser.error("--out-dir must be outside the input evidence directories.")
    for name in names:
        folder = results / name
        print(f"Verifying {name}...", flush=True)
        provenance = read(folder / "provenance.json")
        for filename, expected in provenance["output_sha256"].items():
            if sha(folder / filename) != expected:
                raise RuntimeError(f"Changed diagnostic artifact: {name}/{filename}")
        inputs[name] = sha(folder / "provenance.json")
    evaluation = results / "timing_evaluation_v1"
    manifest = read(evaluation / "manifest.json")
    for filename in ("config.json", "records.json"):
        if sha(evaluation / filename) != manifest["output_sha256"][filename]:
            raise RuntimeError(f"Changed evaluation file: {filename}")
    inputs["evaluation_manifest"] = sha(evaluation / "manifest.json")
    cfg = read(evaluation / "config.json")
    reference = [r for r in read(evaluation / "records.json") if r["method"] == "cv12"]
    basis = read(results / "timing_basis_flexibility_v1/records.json")
    if len(basis) != 720 or any(r["status"] != "completed" for r in basis):
        raise RuntimeError("This report expects all 720 basis fits completed.")
    basis += [{**r, "modes": 12} for r in reference]
    rollout = read(results / "timing_rollout_v1/records.json")
    refinements = read(results / "timing_rollout_refinement_v1/records.json")
    replacements = {r["case_id"]: r for r in refinements}
    failed = {r["case_id"] for r in rollout if not r["refinement_screen_pass"]}
    if set(replacements) != failed:
        raise RuntimeError("Refinements do not match the original failed screens.")
    rollout = [
        {**replacements.get(r["case_id"], r), "used_refined_result": r["case_id"] in replacements}
        for r in rollout
    ]
    if len(rollout) != 360 or not all(r["refinement_screen_pass"] for r in rollout):
        raise RuntimeError("Unresolved rollout numerical screens.")
    ranks = csv_read(results / "timing_identifiability_v1/rank_profiles.csv")
    sensitivities = csv_read(results / "timing_identifiability_v1/regularization_sensitivity.csv")
    out.mkdir()
    basis_summary, rollout_summary, rank_summary, lambda_summary = [], [], [], []
    for n in (10, 50):
        for family in ("A", "B"):
            for timing in TIMINGS:
                identity = {"n_obs": n, "family": family, "timing": timing}
                for modes in (6, 12, 24):
                    group = [
                        r
                        for r in basis
                        if (r["n_obs"], r["family"], r["timing"], r["modes"])
                        == (n, family, timing, modes)
                    ]
                    assert len(group) == 30
                    row = {**identity, "modes": modes, "blocks": 30}
                    for name, key in (
                        ("alpha", "relative_alpha_error"),
                        ("initial", "ic_relative_l2"),
                        ("field", "field_relative_l2"),
                    ):
                        row[f"median_{name}_error_percent"] = float(
                            np.median([100 * r[key] for r in group])
                        )
                    row["bound_hits"] = sum(bool(r["at_bound"]) for r in group)
                    basis_summary.append(row)
                group = [
                    r
                    for r in rollout
                    if (r["n_obs"], r["family"], r["timing"]) == (n, family, timing)
                ]
                values = np.array([100 * r["rollout_network_relative_l2"] for r in group])
                rollout_summary.append(
                    {
                        **identity,
                        "blocks": len(group),
                        "median_discrepancy_percent": float(np.median(values)),
                        "q25_percent": float(np.quantile(values, 0.25)),
                        "q75_percent": float(np.quantile(values, 0.75)),
                        "max_discrepancy_percent": float(np.max(values)),
                        "max_forward_refinement_percent": max(
                            100 * r["forward_refinement_relative_l2"] for r in group
                        ),
                    }
                )
                group = [
                    r
                    for r in sensitivities
                    if (int(r["n_obs"]), r["family"], r["timing"]) == (n, family, timing)
                ]
                ratios = []
                for seed in cfg["designs"][str(n)]:
                    values = [float(r["alpha_hat"]) for r in group if int(r["data_seed"]) == seed]
                    assert len(values) == 7
                    ratios.append(max(values) / min(values))
                lambda_summary.append(
                    {
                        **identity,
                        "median_max_over_min_alpha_across_grid": float(np.median(ratios)),
                        "note": "All candidate penalties, including poorly validating candidates; "
                        "not estimator uncertainty.",
                    }
                )
        for timing in TIMINGS:
            # A and B share designs: count each geometric design only once.
            group = [
                r
                for r in ranks
                if int(r["n_obs"]) == n
                and r["family"] == "A"
                and r["timing"] == timing
                and np.isclose(float(r["alpha"]), 0.05, rtol=0, atol=1e-14)
            ]
            assert len(group) == 30
            rank_summary.append(
                {
                    "n_obs": n,
                    "timing": timing,
                    "unique_designs": 30,
                    "alpha": 0.05,
                    "minimum_numerical_rank": min(int(r["numerical_rank"]) for r in group),
                    "maximum_numerical_rank": max(int(r["numerical_rank"]) for r in group),
                    "full_row_rank_designs": sum(
                        r["full_row_rank_at_tolerance"] == "True" for r in group
                    ),
                }
            )
    csv_write(out / "basis_summary.csv", basis_summary)
    csv_write(out / "rollout_summary.csv", rollout_summary)
    csv_write(out / "rank_summary.csv", rank_summary)
    csv_write(out / "regularization_range_summary.csv", lambda_summary)
    # Retain one consolidated per-model table with explicit refinement provenance.
    keys = [
        "case_id",
        "n_obs",
        "family",
        "timing",
        "data_seed",
        "alpha_hat",
        "rollout_network_relative_l2",
        "forward_refinement_relative_l2",
        "initial_projection_relative_l2",
        "scoring_grid_change_absolute",
        "refinement_screen_pass",
        "used_refined_result",
    ]
    csv_write(out / "rollout_all_models.csv", [{k: r[k] for k in keys} for r in rollout])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (n, family) in zip(
        axes.flat, ((10, "A"), (10, "B"), (50, "A"), (50, "B")), strict=False
    ):
        for modes in (6, 12, 24):
            vals = [
                next(
                    r["median_alpha_error_percent"]
                    for r in basis_summary
                    if (r["n_obs"], r["family"], r["timing"], r["modes"]) == (n, family, t, modes)
                )
                for t in TIMINGS
            ]
            ax.plot(TIMINGS, vals, "o-", label=f"{modes} modes")
        ax.set_yscale("log")
        ax.set_title(f"N={n}, field {family}")
        ax.set_ylabel("Median diffusivity error (%)")
        ax.legend()
    figure_save(fig, out, "basis_sensitivity")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (n, family) in zip(
        axes.flat, ((10, "A"), (10, "B"), (50, "A"), (50, "B")), strict=False
    ):
        for i, t in enumerate(TIMINGS):
            values = sorted(
                100 * r["rollout_network_relative_l2"]
                for r in rollout
                if (r["n_obs"], r["family"], r["timing"]) == (n, family, t)
            )
            ax.scatter(np.full(len(values), i), values, s=12, alpha=0.55)
            ax.plot([i - 0.15, i + 0.15], [np.median(values)] * 2, color="black", linewidth=2)
        ax.set_xticks(range(3), TIMINGS)
        ax.set_yscale("log")
        ax.set_title(f"N={n}, field {family}")
        ax.set_ylabel("Network\N{EN DASH}rollout discrepancy (%)")
    figure_save(fig, out, "physical_consistency")
    # First seed in each N stratum, field B: chosen by schedule order, not performance.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, n in enumerate((10, 50)):
        seed = cfg["designs"][str(n)][0]
        for col, t in enumerate(TIMINGS):
            ax = axes[row, col]
            case = f"n{n}_seed{seed}_B_{t}"
            with np.load(results / "timing_identifiability_v1/profiles" / f"{case}.npz") as p:
                for i, ridge in enumerate(p["ridges"]):
                    ax.loglog(
                        p[f"ridge_{i}_alpha"],
                        np.maximum(p[f"ridge_{i}_objective"], 1e-30),
                        label=f"{ridge:g}",
                    )
                ax.loglog(
                    p["unregularized_alpha"],
                    np.maximum(p["unregularized_mse"], 1e-30),
                    "k--",
                    label="unregularized",
                )
            ax.axvline(0.05, color="gray", linestyle=":", label="true alpha")
            ax.set_title(f"N={n}, B, {t}; seed {seed}")
            ax.set_xlabel("Diffusivity")
            ax.set_ylabel("Profile objective / data MSE")
    axes[0, 0].legend(fontsize=7, ncol=2)
    figure_save(fig, out, "profile_examples")

    lines = [
        "# Fix 2: explanatory diagnostic report",
        "",
        "These analyses were specified after evaluation and explain the frozen procedures; "
        "they do not replace them.",
        "All 720 added basis fits completed. All 360 rollout numerical screens pass after "
        "replacing one result with its separately archived refinement.",
        "",
        "## Identifiability",
        "",
        "At alpha=0.05, all 30 distinct N=10 designs per schedule have numerical row rank 10 "
        "with 12 coefficients. "
        "A and B share these designs and are not independent replications of this rank check. "
        "For a full-row-rank matrix B(alpha), a=B^T(BB^T)^(-1)y gives an exact interpolant "
        "in exact arithmetic. "
        "Continuity preserves full row rank locally, so nearby diffusivities can also "
        "interpolate using different coefficients. "
        "This is established linear algebra, not a new theorem. Numerical rank uses the "
        "recorded floating-point cutoff; "
        "ill-conditioning and large coefficient norms can limit numerical interpolation.",
        "At N=50, full column rank 12 at the true diffusivity is not by itself proof of joint "
        "identification of alpha and the coefficients.",
        "Positive regularization can select a solution without establishing data-only "
        "identifiability. Penalized profile curvature includes the prior penalty.",
        "",
        "## Initial-state flexibility",
        "",
        "| N | Field | Timing | 6-mode alpha error (%) | 12-mode | 24-mode |",
        "|---|---|---|---:|---:|---:|",
    ]
    for n in (10, 50):
        for family in ("A", "B"):
            for t in TIMINGS:
                values = [
                    next(
                        r["median_alpha_error_percent"]
                        for r in basis_summary
                        if (r["n_obs"], r["family"], r["timing"], r["modes"]) == (n, family, t, m)
                    )
                    for m in (6, 12, 24)
                ]
                lines.append(
                    f"| {n} | {family} | {t} | " + " | ".join(f"{v:.3f}" for v in values) + " |"
                )
    lines += [
        "",
        "Six modes omit field B's seventh mode. More modes increase flexibility but do not "
        "guarantee better recovery. "
        "Each basis uses the same data-only CV rule and candidate penalties; the prior is not "
        "representation-neutral. "
        "The tables are descriptive post-evaluation sensitivities, "
        "not truth-based model selection.",
        "",
        "## Independent physical consistency",
        "",
        "| N | Field | Timing | Median discrepancy (%) | Maximum (%) |",
        "|---|---|---|---:|---:|",
    ]
    for r in rollout_summary:
        lines.append(
            f"| {r['n_obs']} | {r['family']} | {r['timing']} | "
            f"{r['median_discrepancy_percent']:.4f} | {r['max_discrepancy_percent']:.4f} |"
        )
    lines += [
        "",
        "Discrepancy is the space-time L2 norm of PINN minus independent rollout, divided by "
        "the rollout norm. "
        "The rollout uses the learned initial field and learned diffusivity, "
        "not their true values. "
        "Refinement screens are numerical checks, not thresholds defining physical acceptability. "
        "Comparisons are global in space-time and can hide local errors. "
        "Small consistency error does not establish accurate parameter or initial-state recovery.",
        "",
        "## Figure and reporting notes",
        "",
        "- basis_sensitivity: marginal median diffusivity errors; "
        "no inferential intervals are claimed.",
        "- physical_consistency: every model is shown; black marks are medians.",
        "- profile_examples: field B, first evaluation seed per N, all schedules. "
        "The selection uses seed order, not outcome. "
        "Profiles re-fit coefficients at each diffusivity. Each positive-penalty curve is a "
        "different objective; their absolute levels must not be used to select lambda. "
        "Zero plotting values are floored at 1e-30.",
        "- The regularization-range table includes poorly validating candidates. "
        "It describes sensitivity across the grid, not statistical uncertainty.",
        "- These diagnostics do not uniquely separate optimization error from "
        "identifiability and prior effects.",
        "",
        "Fix 2 calculations and reporting are complete. Manuscript integration and external "
        "archiving remain separate tasks.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    provenance = {
        "input_provenance_sha256": inputs,
        "script_sha256": sha(Path(__file__)),
        "example_selection": "field_B_first_seed_per_N_all_schedules",
        "output_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("FIX 2 REPORT COMPLETE")
    print(f"Output: {out}")
    print("5 CSV tables, 3 figures in PNG/SVG, report, and provenance record.")


if __name__ == "__main__":
    main()
