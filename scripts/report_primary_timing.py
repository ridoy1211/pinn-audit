"""Build the publication figure and table for the prespecified timing result."""

import argparse
import csv
import io
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from analyze_timing_evaluation import TIMINGS, digest, interval, read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/inverse/timing_evaluation.json"
PRIMARY = ("10", "B", "diffusivity")
BLOCKS = 30


def checked_rows(source, hashes, name):
    path = source / name
    if digest(path) != hashes[name]:
        raise ValueError(f"Archived analysis file has changed: {name}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_results(source):
    provenance_path = source / "analysis_provenance.json"
    provenance = read(provenance_path)
    plan = provenance["analysis_plan"]
    config = read(CONFIG)
    if plan != config["analysis_plan"] or plan["primary_stratum"] != {"n_obs": 10, "family": "B"}:
        raise ValueError("The analysis plan does not match the frozen primary stratum.")
    seeds = config["designs"]["10"]
    if len(seeds) != BLOCKS or len(set(seeds)) != BLOCKS:
        raise ValueError("Expected 30 distinct design seeds in the primary stratum.")

    hashes = provenance["output_sha256"]
    block_rows = checked_rows(source, hashes, "paired_blocks.csv")
    selected = [r for r in block_rows if (r["n_obs"], r["family"], r["metric"]) == PRIMARY]
    errors, gaps = {}, {}
    for timing in TIMINGS:
        rows = [r for r in selected if r["timing"] == timing]
        by_seed = {int(r["seed"]): r for r in rows}
        if len(rows) != BLOCKS or set(by_seed) != set(seeds):
            raise ValueError(f"Expected the same 30 design seeds for {timing}.")
        values = np.array(
            [
                [
                    float(by_seed[seed][key])
                    for key in ("cv12_error_percent", "pinn_error_percent", "difference_pp")
                ]
                for seed in seeds
            ]
        )
        errors[timing] = values[:, :2]
        gaps[timing] = values[:, 2]
        if not np.isfinite(values).all() or np.any(errors[timing] < 0):
            raise ValueError(f"Invalid error values for {timing}.")
        np.testing.assert_allclose(
            errors[timing][:, 1] - errors[timing][:, 0], gaps[timing], rtol=0, atol=1e-10
        )

    # The frozen analysis draws the N=10 resample indices before the N=50 indices.
    indices = np.random.default_rng(plan["bootstrap_seed"]).integers(
        0, BLOCKS, size=(plan["bootstrap_reps"], BLOCKS)
    )
    interactions = gaps["mixed"] - gaps["late"]
    primary = interval(interactions, indices)
    rows = checked_rows(source, hashes, "timing_interactions.csv")
    recorded = [r for r in rows if r["primary"] == "True"]
    if len(recorded) != 1:
        raise ValueError("Expected one prespecified primary interaction.")
    row = recorded[0]
    if (row["n_obs"], row["family"], row["metric"]) != PRIMARY or int(row["blocks"]) != BLOCKS:
        raise ValueError("The archived primary row has an unexpected stratum or block count.")
    np.testing.assert_allclose(
        primary,
        [float(row[key]) for key in ("median_interaction_pp", "ci95_low_pp", "ci95_high_pp")],
        rtol=0,
        atol=1e-10,
    )

    quantiles = {t: np.quantile(errors[t], (0.5, 0.25, 0.75), axis=0).T for t in TIMINGS}
    differences = {t: interval(gaps[t], indices) for t in TIMINGS}
    wins = {t: int(np.count_nonzero(gaps[t] < 0)) for t in TIMINGS}
    inputs = {
        "config_sha256": digest(CONFIG),
        "analysis_provenance_sha256": digest(provenance_path),
        "analysis_output_sha256": {
            name: hashes[name] for name in ("paired_blocks.csv", "timing_interactions.csv")
        },
    }
    return errors, quantiles, differences, wins, interactions, primary, inputs


def render_figure(errors, quantiles, interactions, primary):
    with plt.rc_context({"font.size": 9, "svg.hashsalt": "pinn-audit-primary-timing-v1"}):
        fig, (absolute_ax, interaction_ax) = plt.subplots(
            1, 2, figsize=(7.2, 3.65), layout="constrained", width_ratios=(1, 1.12)
        )
        fig.suptitle("10 observations, initial field B, 30 paired blocks", fontsize=11)
        jitter = np.linspace(-0.075, 0.075, BLOCKS)
        for i, timing in enumerate(TIMINGS):
            for method, (label, color, marker, offset) in enumerate(
                (("CV12", "#1f6488", "o", -0.18), ("PINN-1000", "#9a430b", "^", 0.18))
            ):
                center = i + offset
                median, q1, q3 = quantiles[timing][method]
                absolute_ax.scatter(
                    center + jitter,
                    errors[timing][:, method],
                    color=color,
                    marker=marker,
                    s=14,
                    alpha=0.55,
                    linewidths=0,
                    label=label if i == 0 else None,
                    zorder=3,
                )
                absolute_ax.errorbar(
                    center,
                    median,
                    yerr=[[median - q1], [q3 - median]],
                    fmt="D",
                    color=color,
                    markerfacecolor="white",
                    markersize=4.5,
                    linewidth=1.3,
                    capsize=2.5,
                    zorder=4,
                )
        absolute_ax.set(
            title="A  Absolute diffusivity errors",
            ylabel="Relative absolute error (%)",
            xlim=(-0.45, 2.45),
            ylim=(-3, 1.05 * max(v.max() for v in errors.values())),
        )
        absolute_ax.set_xticks(range(len(TIMINGS)), [t.capitalize() for t in TIMINGS])
        absolute_ax.grid(axis="y", color="0.89")
        absolute_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncols=2, frameon=False)

        estimate, low, high = primary
        interaction_ax.axhspan(-6, -1, color="0.96")
        interaction_ax.axvline(0, color="0.48", linestyle="--", linewidth=0.9)
        interaction_ax.scatter(
            np.sort(interactions), np.arange(1, BLOCKS + 1), s=16, color="#1f6488", zorder=3
        )
        interaction_ax.errorbar(
            estimate,
            -3.5,
            xerr=[[estimate - low], [high - estimate]],
            fmt="D",
            color="#8d320c",
            markerfacecolor="white",
            markersize=5,
            linewidth=1.5,
            capsize=3.5,
            zorder=4,
        )
        xmin = min(0, low, interactions.min())
        xmax = max(0, high, interactions.max())
        pad = 0.055 * (xmax - xmin)
        interaction_ax.set(
            title="B  Prespecified timing interaction",
            xlabel="Mixed-minus-late change in error gap (pp)",
            ylabel="Blocks sorted by interaction",
            xlim=(min(xmin - pad, -25), xmax + pad),
            ylim=(-6, BLOCKS + 2),
        )
        interaction_ax.set_yticks(
            [-3.5, 1, 10, 20, 30], ["Median\n(95% CI)", "1", "10", "20", "30"]
        )
        interaction_ax.xaxis.set_major_locator(MultipleLocator(25))
        interaction_ax.text(
            0.98,
            -3.5,
            f"{estimate:.2f} [{low:.2f}, {high:.2f}]".replace("-", "\N{MINUS SIGN}"),
            transform=interaction_ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
        )
        for ax in (absolute_ax, interaction_ax):
            ax.spines[["top", "right"]].set_visible(False)

        png = io.BytesIO()
        svg = io.StringIO()
        fig.savefig(png, format="png", dpi=360)
        fig.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(fig)
    svg_text = "\n".join(line.rstrip() for line in svg.getvalue().split("\n"))
    return png.getvalue(), svg_text.encode("utf-8")


def report_text(quantiles, differences, wins, primary):
    estimate, low, high = primary
    lines = [
        "# Prespecified timing result",
        "",
        f"For 10 observations and initial field B, the median interaction was {estimate:.4f} "
        f"percentage points (95% percentile bootstrap interval [{low:.4f}, {high:.4f}]). "
        "The interval spans zero.",
        "",
        "![Absolute diffusivity errors and the paired timing interaction](primary_timing.png)",
        "",
        "**Figure caption.** Panel A shows relative absolute diffusivity errors "
        "(100 times the absolute estimation error divided by the true diffusivity). "
        "Each small mark represents one block. Hollow diamonds and bars show medians "
        "and interquartile ranges. Panel B shows the interaction for each of the 30 blocks "
        "and the median with its 95% interval. For each block and schedule, the error gap "
        "is PINN-1000 error minus CV12 error in percentage points. "
        "The interaction subtracts the late gap from "
        "the mixed gap within that block. A positive interaction means the error gap is "
        "larger under mixed observations. The interval uses 10,000 whole-block resamples.",
        "",
        "## Schedule-level results",
        "",
        "| Schedule | CV12 median [Q1, Q3] (%) | PINN-1000 median [Q1, Q3] (%) | "
        "Median paired gap [95% CI] (pp) | Blocks with lower PINN error |",
        "|---|---:|---:|---:|---:|",
    ]
    for timing in TIMINGS:
        cv, pinn = (f"{m:.2f} [{q1:.2f}, {q3:.2f}]" for m, q1, q3 in quantiles[timing])
        gap, lower, upper = differences[timing]
        lines.append(
            f"| {timing.capitalize()} | {cv} | {pinn} | "
            f"{gap:.3f} [{lower:.3f}, {upper:.3f}] | {wins[timing]}/{BLOCKS} |"
        )
    lines += [
        "",
        "The two methods and the schedules share the same 30 independent design and "
        "noise blocks. The within-schedule intervals are pointwise secondary summaries.",
        "",
        "[Archived controlled timing analysis]"
        "(../../results/timing_evaluation_analysis_v1/summary.md)",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results/timing_evaluation_analysis_v1",
        help="Directory containing the archived timing analysis and provenance.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures/publication_timing_v1",
        help="New directory for the publication figure, report, and provenance.",
    )
    args = parser.parse_args()
    source = args.results.resolve()
    out = args.out_dir.resolve()
    if out.exists() or out.is_relative_to(source):
        parser.error("--out-dir must be a new directory outside --results.")

    errors, quantiles, differences, wins, interactions, primary, inputs = load_results(source)
    png, svg = render_figure(errors, quantiles, interactions, primary)
    report = report_text(quantiles, differences, wins, primary)
    out.mkdir(parents=True)
    (out / "primary_timing.png").write_bytes(png)
    (out / "primary_timing.svg").write_bytes(svg)
    (out / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    provenance = {
        "classification": "publication display of the archived prespecified analysis",
        "inputs": inputs,
        "script_sha256": digest(Path(__file__)),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "output_sha256": {
            name: digest(out / name)
            for name in ("primary_timing.png", "primary_timing.svg", "REPORT.md")
        },
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Verified all 30 paired blocks and the archived primary interval. Output: {out}")


if __name__ == "__main__":
    main()
