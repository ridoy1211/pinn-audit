#!/usr/bin/env python3
"""Post-evaluation integrity/numerical review and figures; no fitting or selection."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

from broader_provenance import check_hashes, git, relative_path, sha256, verify_run_outputs

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = ("unreg6", "fixed6", "cv6", "cv12", "pinn_200", "pinn_1000")
METRICS = ("relative_alpha_error", "field_relative_l2", "ic_relative_l2")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def exact_spectral_errors(alpha_hat, amplitudes, alpha_true, modes):
    """Independent exact L2 integration using sine orthogonality on the unit domain."""
    size = max(len(amplitudes), max(m["n"] for m in modes))
    fitted, truth = np.zeros(size), np.zeros(size)
    fitted[: len(amplitudes)] = amplitudes
    for mode in modes:
        truth[mode["n"] - 1] = mode["amplitude"]
    k2 = (np.arange(1, size + 1) * np.pi) ** 2

    def integral(beta):
        return -np.expm1(-beta) / beta

    denominator = np.sum(truth**2 * integral(2 * alpha_true * k2))
    numerator = np.sum(
        fitted**2 * integral(2 * alpha_hat * k2)
        + truth**2 * integral(2 * alpha_true * k2)
        - 2 * fitted * truth * integral((alpha_hat + alpha_true) * k2)
    )
    if denominator <= 0 or numerator < -1e-12:
        raise ValueError("Invalid analytic norm")
    return {
        "field_relative_l2": float(np.sqrt(max(0, numerator) / denominator)),
        "ic_relative_l2": float(np.linalg.norm(fitted - truth) / np.linalg.norm(truth)),
    }


def verify_cv(folder, obs, cfg, selected_record):
    selection = read_json(folder / "selection.json")
    details = read_json(folder / "fold_scores.json")
    labels = obs["fold_labels"]
    losses = []
    with np.load(folder / "candidate_fits.npz") as fits:
        for candidate, ridge in enumerate(cfg["ridges"]):
            total = 0.0
            for fold in range(5):
                held = labels == fold
                prefix = f"cv{candidate}_fold{fold}"
                amplitudes = fits[prefix + "_amplitudes"]
                alpha = float(fits[prefix + "_alpha"])
                k = np.arange(1, len(amplitudes) + 1) * np.pi
                pred = (
                    np.sin(obs["x"][held, None] * k) * np.exp(-alpha * obs["t"][held, None] * k**2)
                ) @ amplitudes
                sse = float(np.sum((pred - obs["y"][held]) ** 2))
                recorded = next(
                    r for r in details if r["candidate_index"] == candidate and r["fold"] == fold
                )
                assert recorded["ridge"] == ridge
                np.testing.assert_allclose(sse, recorded["validation_sse"], rtol=1e-8, atol=1e-15)
                total += sse
            losses.append(total / len(labels))
    chosen = min(range(len(losses)), key=lambda j: (losses[j], cfg["ridges"][j]))
    np.testing.assert_allclose(losses, selection["validation_mse"], rtol=1e-8, atol=1e-15)
    assert chosen == selection["selected_index"]
    assert cfg["ridges"][chosen] == selected_record["selected_ridge"]
    return 25


def verify_models(root, folder, cfg, records):
    import torch

    import pilot_bounded_pinn as pilot

    torch.set_num_threads(1)
    exact, neural, folds, dense_checks = [], [], 0, 0
    for index, record in enumerate(records):
        assert record["status"] == "completed"
        job = relative_path(folder, record["artifact_path"])
        obs_path = folder / "data" / (record["case_id"] + ".npz")
        with np.load(obs_path) as saved:
            obs = {k: saved[k].copy() for k in ("x", "t", "y", "fold_labels")}
        if record["training_seed"] is None:
            with np.load(job / "model.npz") as saved:
                alpha = float(saved["selected_alpha"])
                amplitudes = saved["selected_amplitudes"].copy()
            np.testing.assert_allclose(alpha, record["alpha_hat"], rtol=1e-12)
            scores = exact_spectral_errors(
                alpha,
                amplitudes,
                cfg["alpha_true_generation_and_scoring_only"],
                cfg["families"][record["family"]],
            )
            np.testing.assert_allclose(
                scores["ic_relative_l2"], record["ic_relative_l2"], rtol=1e-10, atol=1e-13
            )
            exact.append(
                {
                    "case_id": record["case_id"],
                    "method": record["method"],
                    **scores,
                    "field_quadrature_difference_pp": 100
                    * (record["field_relative_l2"] - scores["field_relative_l2"]),
                    "ic_quadrature_difference_pp": 100
                    * (record["ic_relative_l2"] - scores["ic_relative_l2"]),
                }
            )
            if record["method"] in ("cv6", "cv12"):
                folds += verify_cv(job, obs, cfg, record)
        else:
            state = torch.load(job / "checkpoint.pt", map_location="cpu", weights_only=True)
            with torch.random.fork_rng(devices=[]):
                model = pilot.BoundedHeatField(state["settings"])
                model.load_state_dict(state["model_state"])
            np.testing.assert_allclose(
                float(model.alpha().detach()), record["alpha_hat"], rtol=1e-12
            )
            prediction = pilot.predict_paired(model, obs["x"], obs["t"])
            with np.load(job / "predictions.npz") as saved:
                np.testing.assert_allclose(
                    prediction, saved["data_predictions"], rtol=1e-10, atol=1e-12
                )
                if "u" in saved.files:
                    xx, tt = np.meshgrid(saved["x"], saved["t"])
                    dense = pilot.predict_paired(model, xx.ravel(), tt.ravel()).reshape(xx.shape)
                    np.testing.assert_allclose(dense, saved["u"], rtol=1e-10, atol=1e-12)
                    dense_checks += 1
            np.testing.assert_allclose(
                np.mean((prediction - obs["y"]) ** 2), record["data_mse"], rtol=1e-9
            )
            neural.append(
                {
                    "case_id": record["case_id"],
                    "method": record["method"],
                    "training_seed": record["training_seed"],
                    "iterations": state["lbfgs_iterations"],
                    "data_prediction_reload_verified": True,
                }
            )
        if (index + 1) % 40 == 0:
            print(f"Verified saved states {index + 1}/{len(records)}", flush=True)
    return {
        "classical_exact_norms": exact,
        "neural_checkpoints": neural,
        "held_out_cv_predictions_recomputed": folds,
        "neural_dense_grids_reconstructed": dense_checks,
    }


def figures(out, analysis):
    scores = analysis["dataset_scores"]
    strata = ((10, "A"), (10, "B"), (50, "A"), (50, "B"))
    labels = ("Unreg6", "Fixed6", "CV6", "CV12", "PINN200", "PINN1000")
    colors = ("#999999", "#789d57", "#3f8e62", "#1c6687", "#dc9a36", "#ba4b3a")
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharey="row")
    names = ("Diffusivity error (%)", "Field error (%)", "Initial-field error (%)")
    for col, (n, family) in enumerate(strata):
        for row, (metric, title) in enumerate(zip(METRICS, names, strict=True)):
            ax = axes[row, col]
            groups = [
                np.array(
                    [
                        100 * r["scores"][metric]
                        for r in scores
                        if r["n_obs"] == n
                        and r["family"] == family
                        and r["method"] == method
                        and r["complete"]
                    ]
                )
                for method in METHODS
            ]
            ax.boxplot(groups, showfliers=False, widths=0.55, medianprops={"color": "black"})
            for j, values in enumerate(groups):
                offsets = np.linspace(-0.12, 0.12, len(values))
                ax.scatter(
                    np.full(len(values), j + 1) + offsets,
                    values,
                    s=16,
                    color=colors[j],
                    alpha=0.8,
                    zorder=3,
                )
            if row == 2 and family == "B":
                ax.hlines(100 / np.sqrt(21), 0.5, 3.5, color="black", linestyle="--", linewidth=1)
            ax.set_yscale("log")
            ax.set_xticks(range(1, 7), labels, rotation=40, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.2)
            if row == 0:
                ax.set_title(f"N={n}; family {family}")
            if col == 0:
                ax.set_ylabel(title)
    fig.suptitle(
        "Fixed-pipeline evaluation: ten designs per stratum; neural points summarize three starts",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.005,
        "Dashed IC line: six-mode representation floor for family B (21.82%). "
        "Boxes show quartiles; dots show every design.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(out / "method_errors.png", dpi=180)
    fig.savefig(out / "method_errors.pdf")
    plt.close(fig)
    contrasts = analysis["paired_contrasts"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 6), sharey=True)
    titles = ("Diffusivity (co-primary)", "Field (secondary)", "Initial field (co-primary)")
    ylabels = []
    for i, c in enumerate(contrasts):
        ylabels.append(
            f"N={c['n_obs']} / {c['family']} / {c['neural_method'].removeprefix('pinn_')}"
        )
        color = "#245d82" if c["neural_method"] == "pinn_200" else "#ba541c"
        for ax, metric in zip(axes, METRICS, strict=True):
            values = c["metrics"][metric]
            ci = values["bootstrap_95_percentile_ci_for_median_pp"]
            if ci is not None:
                ax.plot(ci, [i, i], color=color, linewidth=2)
            ax.scatter(values["median_pp"], i, color=color, s=40, zorder=3)
    for ax, title in zip(axes, titles, strict=True):
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="PINN - CV12 error (percentage points)")
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_yticks(range(len(contrasts)), ylabels)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Median paired differences and pointwise 95% percentile bootstrap intervals", fontsize=12
    )
    fig.text(
        0.5,
        0.01,
        "Negative favors PINN; positive favors CV12. Ten paired designs per row; "
        "intervals are not multiplicity-adjusted.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out / "paired_contrasts.png", dpi=180)
    fig.savefig(out / "paired_contrasts.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/broader_study_v1"))
    parser.add_argument("--analysis", type=Path, default=Path("results/broader_analysis_v1"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/broader_results_review_v1"))
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Preserve the existing review; choose a fresh output directory")
    root = Path(__file__).resolve().parents[1]
    manifest = verify_run_outputs(args.results)
    check_hashes(root, manifest["source_sha256"])
    analysis = read_json(args.analysis / "analysis.json")
    for name, expected in read_json(args.analysis / "manifest.json")["output_sha256"].items():
        assert sha256(args.analysis / name) == expected, name
    assert analysis["input_manifest_sha256"] == sha256(args.results / "manifest.json")
    cfg, records = (read_json(args.results / name) for name in ("config.json", "records.json"))
    freeze = manifest["freeze"]["freeze_record"]
    run_commit = manifest["freeze"]["run_commit"]
    freeze_path = "docs/BROADER_COMPARISON_FREEZE.json"
    import hashlib

    assert (
        hashlib.sha256(git(root, "show", f"{run_commit}:{freeze_path}")).hexdigest()
        == manifest["freeze"]["freeze_sha256"]
    )
    date = git(root, "show", "-s", "--format=%cI", run_commit).decode().strip()
    assert datetime.fromisoformat(date) <= datetime.fromisoformat(manifest["started_utc"])
    assert datetime.fromisoformat(freeze["created_utc"]) <= datetime.fromisoformat(
        manifest["first_data_generated_utc"]
    )
    assert Counter(r["status"] for r in records) == {"completed": 400}
    audit = verify_models(root, args.results, cfg, records)
    refinement = {
        metric: max(
            100 * abs(r["refinement_scores"][metric] - r[metric])
            for r in records
            if "refinement_scores" in r
        )
        for metric in METRICS[1:]
    }
    audit.update(
        classification="post_evaluation_derived_audit_no_training_or_selection",
        review_script_sha256=sha256(Path(__file__).resolve()),
        input_run_manifest_sha256=sha256(args.results / "manifest.json"),
        input_analysis_manifest_sha256=sha256(args.analysis / "manifest.json"),
        source_hashes_verified=len(manifest["source_sha256"]),
        run_output_hashes_verified=len(manifest["output_sha256"]),
        freeze_created_utc=freeze["created_utc"],
        freeze_commit=run_commit,
        freeze_commit_time=date,
        first_data_generated_utc=manifest["first_data_generated_utc"],
        run_elapsed_hours=(
            datetime.fromisoformat(manifest["completed_utc"])
            - datetime.fromisoformat(manifest["started_utc"])
        ).total_seconds()
        / 3600,
        planned_refinement_max_change_pp=refinement,
    )
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "verification.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8", newline="\n"
    )
    rows = []
    for summary in analysis["method_summaries"]:
        raw = [
            r
            for r in records
            if r["n_obs"] == summary["n_obs"]
            and r["family"] == summary["family"]
            and r["method"] == summary["method"]
        ]
        case_times = [
            sum(r["fit_time_s"] + r["selection_time_s"] for r in raw if r["case_id"] == case_id)
            for case_id in sorted({r["case_id"] for r in raw})
        ]
        rows.append(
            {
                "n_obs": summary["n_obs"],
                "family": summary["family"],
                "method": summary["method"],
                "median_per_fit_seconds": float(np.median([r["fit_time_s"] for r in raw])),
                "median_selection_seconds": float(np.median([r["selection_time_s"] for r in raw])),
                "median_all_starts_fit_plus_selection_seconds_per_design": float(
                    np.median(case_times)
                ),
            }
        )
    with (args.out_dir / "timing.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures(args.out_dir, analysis)
    print(
        f"Reviewed {len(records)} completed jobs; verified "
        f"{audit['held_out_cv_predictions_recomputed']} held-out fold predictions."
    )
    print(
        f"Reloaded {len(audit['neural_checkpoints'])} neural checkpoints; "
        f"reconstructed {audit['neural_dense_grids_reconstructed']} saved dense grids."
    )
    print(f"Figures and numerical audit saved to {args.out_dir}.")


if __name__ == "__main__":
    main()
