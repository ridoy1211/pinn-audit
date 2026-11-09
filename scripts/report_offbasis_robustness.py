"""Render the off-basis results and measure omitted modes at the saved sensors."""

import argparse
import csv
import io
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analyze_timing_evaluation import digest, read

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results/offbasis_analysis_v1"
CONFIG = ROOT / "configs/inverse/offbasis_evaluation.json"
SCHEDULES = ("mixed", "late")
GROUPS = [(n, timing) for n in (10, 50) for timing in SCHEDULES]
COLORS = {"mixed": "#1f6488", "late": "#9a430b"}
BLOCKS = 30
PROJECTION_MODES = 12


def checked_path(folder, name, expected):
    path = folder / name
    if digest(path) != expected:
        raise ValueError(f"Archived file has changed: {path}")
    return path


def archived_rows(name, hashes):
    path = checked_path(ANALYSIS, name, hashes[name])
    with path.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["metric"] == "diffusivity"]


def load_projection(evidence, config):
    folder = evidence / "offbasis_truth_verification_v1"
    truth = config["truth"]
    path = checked_path(folder, "provenance.json", truth["verification_provenance_sha256"])
    provenance = read(path)
    if provenance["truth_source_sha256"] != truth["verified_generator_sha256"]:
        raise ValueError("Truth verification uses a different generator.")
    hashes = provenance["output_sha256"]
    verification = read(checked_path(folder, "verification.json", hashes["verification.json"]))
    if not verification["passed"]:
        raise ValueError("The archived truth verification did not pass.")
    with np.load(
        checked_path(folder, "initial_field.npz", hashes["initial_field.npz"]),
        allow_pickle=False,
    ) as arrays:
        x, initial, projected, coefficients = (
            arrays[k] for k in ("x", "initial", "projection12", "coefficients256")
        )
    if len(coefficients) != truth["spectral_modes"]:
        raise ValueError("The verified coefficient count differs from the frozen configuration.")
    return x, initial, projected, coefficients, verification["twelve_mode_projection_relative_l2"]


def sensor_diagnostics(folder, config, hashes, coefficients):
    alpha = config["alpha_true_generation_and_scoring_only"]
    sigma = config["noise_sigma_absolute"]
    wave_numbers = np.pi * np.arange(1, len(coefficients) + 1)
    rows = []
    for n in (10, 50):
        seeds = config["designs"][str(n)]
        if len(seeds) != BLOCKS or len(set(seeds)) != BLOCKS:
            raise ValueError(f"Expected {BLOCKS} distinct design seeds for N={n}.")
        for seed in seeds:
            designs = []
            for timing in SCHEDULES:
                case = f"n{n}_seed{seed}_C_{timing}"
                name = f"data/{case}.npz"
                path = checked_path(folder, name, hashes[name])
                with np.load(path, allow_pickle=False) as arrays:
                    x, t, clean, noise = (arrays[key] for key in ("x", "t", "clean", "unit_noise"))
                    if float(arrays["sigma"]) != sigma or x.shape != (n,) or np.any(t <= 0):
                        raise ValueError(f"Unexpected observation design for {case}.")
                    np.testing.assert_allclose(
                        arrays["y"], clean + sigma * noise, atol=1e-13, rtol=0
                    )
                designs.append((x, noise))
                modes = (
                    np.sin(x[:, None] * wave_numbers)
                    * np.exp(-alpha * t[:, None] * wave_numbers**2)
                    * coefficients
                )
                np.testing.assert_allclose(modes.sum(axis=1), clean, atol=2e-13, rtol=0)
                # Sum the omitted modes directly to retain small differences at late times.
                tail = modes[:, PROJECTION_MODES:].sum(axis=1)
                rms = float(np.sqrt(np.mean(tail**2)))
                if not np.isfinite(rms) or rms <= 0:
                    raise ValueError(f"Invalid omitted-mode RMS for {case}.")
                rows.append(
                    {
                        "case_id": case,
                        "n_obs": n,
                        "seed": seed,
                        "schedule": timing,
                        "tail_rms": rms,
                        "tail_rms_over_noise_sd": rms / sigma,
                        "max_absolute_tail": float(np.max(np.abs(tail))),
                    }
                )
            np.testing.assert_array_equal(designs[0][0], designs[1][0])
            np.testing.assert_array_equal(designs[0][1], designs[1][1])
    return rows


def render_figure(projection, ratios, differences):
    x, initial, projected, _, projection_error = projection
    with plt.rc_context({"font.size": 9, "svg.hashsalt": "pinn-audit-offbasis-v1"}):
        fig, axes = plt.subplot_mosaic(
            [["field", "sensors"], ["gap", "gap"]],
            figsize=(7.2, 5.5),
            height_ratios=(1.28, 1),
            layout="constrained",
        )
        fig.suptitle("Off-basis robustness evaluation, initial field C", fontsize=11)
        field = axes["field"]
        field.axhline(0, color="0.82", linewidth=0.8)
        field.plot(x, initial, color=COLORS["mixed"], label="Field C")
        field.plot(x, projected, "--", color=COLORS["late"], label="12-mode projection")
        field.set(
            title="A  Initial temperature",
            xlabel="Position x",
            ylabel="Normalized temperature",
            xlim=(0, 1),
        )
        field.legend(loc="upper right", frameon=False, fontsize=8)
        field.text(
            0.98,
            0.69,
            f"{100 * projection_error:.2f}% relative $L^2$ error",
            transform=field.transAxes,
            ha="right",
            fontsize=8,
        )

        sensors = axes["sensors"]
        sensors.set_yscale("log")
        sensors.axhline(1, color="0.46", linestyle="--", linewidth=0.9)
        sensors.text(
            0.98,
            1,
            "Noise SD",
            transform=sensors.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
        )
        for index, group in enumerate(GROUPS):
            color = COLORS[group[1]]
            values = ratios[group]
            sensors.scatter(
                index + np.linspace(-0.16, 0.16, len(values)),
                values,
                color=color,
                alpha=0.55,
                s=13,
                linewidths=0,
            )
            sensors.plot(
                index, np.median(values), "D", color=color, markerfacecolor="white", markersize=4.5
            )
        sensors.set(title="B  Discrepancy at the sensors", ylabel="Omitted-mode RMS / noise SD")
        sensors.set_xticks(
            range(len(GROUPS)), [f"{n} obs.\n{timing.capitalize()}" for n, timing in GROUPS]
        )
        sensors.set_ylim(
            min(1e-15, min(v.min() for v in ratios.values()) / 2),
            max(3, 1.5 * max(v.max() for v in ratios.values())),
        )
        sensors.set_yticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1])
        sensors.grid(axis="y", which="major", color="0.9")

        gap = axes["gap"]
        gap.axvline(0, color="0.46", linestyle="--", linewidth=0.9)
        for i, group in enumerate(GROUPS):
            row = differences[group]
            estimate, low, high = (
                float(row[key]) for key in ("median_difference_pp", "ci95_low_pp", "ci95_high_pp")
            )
            gap.errorbar(
                estimate,
                i,
                xerr=[[estimate - low], [high - estimate]],
                fmt="D",
                color=COLORS[group[1]],
                markerfacecolor="white",
                capsize=3,
                markersize=5,
            )
        gap.set(
            title="C  Median paired error gaps with pointwise 95% intervals",
            xlabel="PINN-1000 minus CV12 diffusivity error (percentage points)",
            ylim=(len(GROUPS) - 0.55, -0.45),
        )
        gap.set_yticks(range(len(GROUPS)), [f"{n} obs., {timing}" for n, timing in GROUPS])
        for ax in axes.values():
            ax.spines[["top", "right"]].set_visible(False)

        png = io.BytesIO()
        svg = io.StringIO()
        fig.savefig(png, format="png", dpi=360)
        fig.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(fig)
    svg_text = "\n".join(line.rstrip() for line in svg.getvalue().split("\n"))
    return png.getvalue(), svg_text.encode("utf-8")


def report_text(config, projection_error, rows, ratios, absolute, differences, interactions):
    sigma = config["noise_sigma_absolute"]
    late_max = max(row["max_absolute_tail"] for row in rows if row["schedule"] == "late")
    lines = [
        "# Off-basis robustness results",
        "",
        f"The twelve-mode projection of field C has {100 * projection_error:.2f}% relative "
        f"L2 error at time zero. Across the late observations, the maximum absolute "
        f"temperature discrepancy from the omitted modes was {late_max:.3g}, "
        f"with a measurement noise SD of {sigma:g}.",
        "",
        "![Off-basis projection, sensor discrepancy, and paired error gaps](offbasis.png)",
        "",
        "**Figure caption.** Panel A shows the localized initial temperature and its "
        "twelve-mode orthogonal projection. The percentage is the relative L2 error "
        "of the projection at time zero. Panel B shows the RMS contribution of modes "
        "13 through 256 at the saved observation points, divided by the measurement "
        "noise SD. The calculation uses the true diffusivity. Each group has 30 "
        "points and a median diamond. The vertical scale is logarithmic and the dashed "
        "line marks a ratio of one. Panel C shows the medians of paired PINN-1000 minus "
        "CV12 diffusivity errors and pointwise 95% percentile bootstrap intervals. "
        "Negative gaps indicate lower PINN-1000 errors. Each interval uses 10,000 "
        "whole-block resamples.",
        "",
        "Method errors in the table are relative absolute diffusivity errors in percent. "
        "The two schedules and methods share 30 design and noise blocks at each "
        "observation count.",
        "",
        "| N | Schedule | CV12 median [Q1, Q3] (%) | PINN-1000 median [Q1, Q3] (%) | "
        "Median paired gap [95% CI] (pp) | Median omitted-mode RMS / noise SD |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for n, timing in GROUPS:
        cells = []
        for method in ("cv12", "pinn_1000"):
            row = absolute[n, timing, method]
            cells.append(
                f"{float(row['median_error_percent']):.2f} "
                f"[{float(row['q25_percent']):.2f}, {float(row['q75_percent']):.2f}]"
            )
        row = differences[n, timing]
        gap = (
            f"{float(row['median_difference_pp']):.4f} "
            f"[{float(row['ci95_low_pp']):.4f}, {float(row['ci95_high_pp']):.4f}]"
        )
        lines.append(
            f"| {n} | {timing.capitalize()} | {cells[0]} | {cells[1]} | {gap} | "
            f"{np.median(ratios[n, timing]):.3g} |"
        )
    lines += [
        "",
        "The secondary timing interaction subtracts the late gap from the mixed gap "
        "within each block before taking the median.",
        "",
    ]
    for row in interactions:
        lines.append(
            f"- N={row['n_obs']}, {float(row['median_interaction_pp']):.4f} pp with pointwise "
            f"95% interval [{float(row['ci95_low_pp']):.4f}, {float(row['ci95_high_pp']):.4f}]."
        )
    lines += [
        "",
        "Both interaction intervals span zero. The sensor calculation describes the "
        "orthogonal projection at the true diffusivity. CV12 estimates diffusivity and "
        "initial coefficients from noisy data. The evaluation covers one localized field "
        "and two schedules. Its initial L2 energy differs from that of fields A and B "
        "under the same absolute noise SD.",
        "",
        "[Archived off-basis analysis](../../results/offbasis_analysis_v1/REPORT.md)",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "results",
        help="Results directory containing the restored off-basis evidence.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures/publication_offbasis_v1",
        help="New directory for the figure, table, and sensor diagnostic.",
    )
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    out = args.out_dir.resolve()
    if out.exists() or out.is_relative_to(evidence) or out.is_relative_to(ANALYSIS):
        parser.error("--out-dir must be a new directory outside the evidence and analysis.")

    folder = evidence / "offbasis_evaluation_v1"
    analysis_path = ANALYSIS / "provenance.json"
    analysis = read(analysis_path)
    manifest_path = checked_path(folder, "manifest.json", analysis["input_manifest_sha256"])
    manifest = read(manifest_path)
    if (
        manifest["status"] != "complete"
        or manifest["failed_jobs"]
        or manifest["terminal_jobs"] != 240
    ):
        raise ValueError("Expected the completed archive of 240 off-basis fits.")
    hashes = manifest["output_sha256"]
    config = read(checked_path(folder, "config.json", hashes["config.json"]))
    if config != read(CONFIG) or config["analysis_plan"] != analysis["analysis_plan"]:
        raise ValueError("The evidence does not match the frozen off-basis configuration.")

    projection = load_projection(evidence, config)
    rows = sensor_diagnostics(folder, config, hashes, projection[3])
    ratios = {
        group: np.array(
            [r["tail_rms_over_noise_sd"] for r in rows if (r["n_obs"], r["schedule"]) == group]
        )
        for group in GROUPS
    }
    tables = analysis["output_sha256"]
    absolute = {
        (int(r["n_obs"]), r["timing"], r["method"]): r
        for r in archived_rows("absolute_errors.csv", tables)
    }
    differences = {
        (int(r["n_obs"]), r["timing"]): r for r in archived_rows("paired_differences.csv", tables)
    }
    interactions = archived_rows("timing_interactions.csv", tables)
    if any(
        int(row["blocks"]) != BLOCKS
        for row in (*absolute.values(), *differences.values(), *interactions)
    ):
        raise ValueError("The archived summaries must contain 30 blocks per stratum.")
    png, svg = render_figure(projection, ratios, differences)
    report = report_text(config, projection[4], rows, ratios, absolute, differences, interactions)

    out.mkdir(parents=True)
    (out / "offbasis.png").write_bytes(png)
    (out / "offbasis.svg").write_bytes(svg)
    (out / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    with (out / "sensor_tail.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "classification": "publication display and descriptive sensor diagnostic",
        "inputs": {
            "evaluation_manifest_sha256": digest(manifest_path),
            "analysis_provenance_sha256": digest(analysis_path),
            "truth_verification_provenance_sha256": config["truth"][
                "verification_provenance_sha256"
            ],
        },
        "script_sha256": digest(Path(__file__)),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "output_sha256": {
            name: digest(out / name)
            for name in ("offbasis.png", "offbasis.svg", "REPORT.md", "sensor_tail.csv")
        },
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Verified {len(rows)} observation datasets and the archived off-basis analysis.")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
