#!/usr/bin/env python3
"""Verify saved neural budget experiments without training or selecting models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

from pilot_bounded_pinn import BoundedHeatField, column, predict_paired
from pinnaudit.solvers.neural import heat_residual
from score_sparse_diagnostic import quadrature_weights, relative_error, true_field

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("results/bounded_pinn_pilot_v1"))
    parser.add_argument("--new", type=Path, default=Path("results/bounded_pinn_budget_check_v1"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/pinn_budget_review_v1"))
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Preserve existing reviews; choose a fresh output directory")
    root = Path(__file__).resolve().parents[1]
    manifests = []
    for folder in (args.old, args.new):
        manifest = read_json(folder / "manifest.json")
        assert manifest["status"] == "complete"
        for section in ("source_sha256", "input_sha256"):
            for name, expected in manifest[section].items():
                assert digest(root / name) == expected, name
        for name, expected in manifest["output_sha256"].items():
            assert digest(folder / name) == expected, name
        manifests.append(manifest)
    old_config, new_config = (read_json(p / "config.json") for p in (args.old, args.new))
    for key in old_config:
        if key not in ("stage", "lbfgs_iters", "lbfgs_max_eval"):
            assert old_config[key] == new_config[key], key
    assert digest(args.old / "manifest.json") == new_config["original_result_manifest_sha256"]
    assert digest(root / new_config["protocol_note"]) == new_config["protocol_sha256"]
    before, after = (read_json(p / "records.json") for p in (args.old, args.new))
    assert len(before) == len(after) == 6
    torch.set_num_threads(1)
    with (
        np.load(args.old / "pde_points.npz") as old_points,
        np.load(args.new / "pde_points.npz") as points,
    ):
        for name in ("training", "evaluation"):
            np.testing.assert_array_equal(old_points[name], points[name])
        collocation = points["training"].copy()
        evaluation = points["evaluation"].copy()
    pairs = []
    for row in after:
        assert row["status"] == "completed_budget"
        key = (row["n_obs"], row["data_seed"], row["training_seed"])
        original = next(
            r for r in before if (r["n_obs"], r["data_seed"], r["training_seed"]) == key
        )
        name = f"n{key[0]}_data{key[1]}_train{key[2]}"
        old_trace, new_trace = (read_json(p / name / "trace.json") for p in (args.old, args.new))
        old_adam, new_adam = (
            next(r for r in tr if r["stage"] == "adam_end") for tr in (old_trace, new_trace)
        )
        for metric in ("alpha", "loss", "data_mse", "pde_mse"):
            np.testing.assert_allclose(old_adam[metric], new_adam[metric], rtol=1e-11, atol=1e-13)
        saved = torch.load(args.new / name / "checkpoint.pt", map_location="cpu", weights_only=True)
        model = BoundedHeatField(saved["settings"])
        model.load_state_dict(saved["model_state"])
        np.testing.assert_allclose(float(model.alpha().detach()), row["alpha_hat"], rtol=1e-12)
        with np.load(args.new / name / "predictions.npz") as predictions:
            gx, gt = predictions["x"].copy(), predictions["t"].copy()
            xx, tt = np.meshgrid(gx, gt)
            field = predict_paired(model, xx.ravel(), tt.ravel()).reshape(xx.shape)
            np.testing.assert_allclose(field, predictions["u"], rtol=1e-10, atol=1e-12)
            reference = true_field(gx, gt)
            wx, wt = quadrature_weights(gx), quadrature_weights(gt)
            for actual, expected in (
                (relative_error(field, reference, np.outer(wt, wx)), row["field_relative_l2"]),
                (relative_error(field[0], reference[0], wx), row["ic_relative_l2"]),
            ):
                np.testing.assert_allclose(actual, expected, rtol=1e-10)
            residual = heat_residual(
                model, column(evaluation[:, 0]), column(evaluation[:, 1]), model.alpha()
            )
            np.testing.assert_allclose(
                residual.detach().numpy()[:, 0],
                predictions["independent_pde_residual"],
                rtol=1e-9,
                atol=1e-12,
            )
        path = root / new_config["data_directory"] / f"observations_n{key[0]}_seed{key[1]}.npz"
        with np.load(path) as obs:
            data = (model(column(obs["x"]), column(obs["t"])) - column(obs["y"])).square().mean()
        pde = (
            heat_residual(
                model, column(collocation[:, 0]), column(collocation[:, 1]), model.alpha()
            )
            .square()
            .mean()
        )
        total = new_config["data_weight"] * data + new_config["pde_weight"] * pde
        np.testing.assert_allclose(
            float(total.detach()), row["final_training_trace"]["loss"], rtol=1e-9
        )
        grad = torch.autograd.grad(total, tuple(model.parameters()))
        pairs.append(
            {
                "n_obs": key[0],
                "data_seed": key[1],
                "training_seed": key[2],
                "short": original,
                "long": row,
                "long_lbfgs_iterations": saved["lbfgs_iterations"],
                "long_gradient_max_abs": max(float(g.detach().abs().max()) for g in grad),
                "adam_phase_reproduced": True,
            }
        )
    args.out_dir.mkdir(parents=True)
    audit = {
        "classification": "post_hoc_derived_evaluation_no_training_or_model_selection",
        "review_script_sha256": digest(Path(__file__).resolve()),
        "input_manifest_sha256": {
            str(p): digest(p / "manifest.json") for p in (args.old, args.new)
        },
        "output_hashes_verified": [len(m["output_sha256"]) for m in manifests],
        "long_checkpoint_grids_reconstructed": 6,
        "pairs": pairs,
    }
    (args.out_dir / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# Paired neural budget review",
        "",
        "Two fixed development datasets; all recovery errors are percentages.",
        "PINN medians summarize three initializations of each dataset, not three datasets.",
        "",
        "| N | L-BFGS limit | median alpha error | median field error "
        "| median IC error | median training seconds |",
        "|---|---|---|---|---|---|",
    ]
    for n in (10, 50):
        for stage, limit in (("short", 200), ("long", 1000)):
            rows = [r[stage] for r in pairs if r["n_obs"] == n]
            values = [
                100 * float(np.median([r[k] for r in rows]))
                for k in ("relative_alpha_error", "field_relative_l2", "ic_relative_l2")
            ]
            seconds = float(np.median([r["training_elapsed_s"] for r in rows]))
            lines.append(
                f"| {n} | {limit} | {values[0]:.4g} | {values[1]:.4g} "
                f"| {values[2]:.4g} | {seconds:.2f} |"
            )
    lines += [
        "",
        "All six Adam phases reproduce. Longer-budget training objectives "
        "and recovery metrics are retained together.",
        "No budget is selected by truth error, and no new fit was run by this review.",
    ]
    summary = "\n".join(lines) + "\n"
    (args.out_dir / "summary.md").write_text(summary, encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, metric, title in zip(
        axes,
        ("relative_alpha_error", "field_relative_l2", "ic_relative_l2"),
        ("Diffusivity", "Temperature field", "Initial field"),
        strict=True,
    ):
        for n, color in ((10, "#245d82"), (50, "#ba541c")):
            group = [r for r in pairs if r["n_obs"] == n]
            values = np.array(
                [[100 * r[stage][metric] for stage in ("short", "long")] for r in group]
            )
            for v in values:
                ax.plot([200, 1000], v, color=color, alpha=0.25, marker="o", linewidth=0.8)
            ax.plot(
                [200, 1000],
                np.median(values, axis=0),
                color=color,
                marker="o",
                linewidth=2,
                label=f"N={n}, median",
            )
        ax.set(
            title=title,
            xlabel="L-BFGS iteration limit",
            ylabel="Relative error (%)",
            yscale="log",
            xticks=[200, 1000],
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Budget sensitivity on two fixed development cases; faint lines are initializations",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "budget_sensitivity.png", dpi=180)
    fig.savefig(args.out_dir / "budget_sensitivity.pdf")
    plt.close(fig)
    print(summary)


if __name__ == "__main__":
    main()
