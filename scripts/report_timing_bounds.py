"""Plot continuous bound proximity using the archived controlled timing fits."""

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analyze_timing_evaluation import METHODS, TIMINGS, digest, read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/inverse/timing_evaluation.json"
ANCHOR = ROOT / "results/timing_fix2_report_v1/provenance.json"
OUTPUT = ROOT / "figures/publication_bounds_v1"
COUNTS = (10, 50)
FAMILIES = ("A", "B")
STYLES = {
    "cv12": ("CV12", "#1f6488", "o", 0.20),
    "pinn_1000": ("PINN-1000", "#9a430b", "^", -0.20),
}


def checked(path, expected_digest):
    if digest(path) != expected_digest:
        raise ValueError(f"Archived file hash mismatch: {path}")
    return path


def load_estimates(evidence):
    anchors = read(ANCHOR)["input_provenance_sha256"]
    evaluation = evidence / "timing_evaluation_v1"
    diagnostics = evidence / "timing_identifiability_v1"
    manifest_path = checked(evaluation / "manifest.json", anchors["evaluation_manifest"])
    diagnostic_path = checked(diagnostics / "provenance.json", anchors["timing_identifiability_v1"])
    manifest, diagnostic = read(manifest_path), read(diagnostic_path)
    if diagnostic["input_manifest_sha256"] != digest(manifest_path):
        raise ValueError("Boundary diagnostics refer to a different evaluation manifest.")
    hashes = manifest["output_sha256"]
    config_path = checked(evaluation / "config.json", hashes["config.json"])
    records_path = checked(evaluation / "records.json", hashes["records.json"])
    boundary_path = checked(
        diagnostics / "boundary_distances.csv",
        diagnostic["output_sha256"]["boundary_distances.csv"],
    )
    config = read(config_path)
    if config != read(CONFIG) or config != manifest["freeze"]["config"]:
        raise ValueError("The archived configuration differs from the frozen timing configuration.")
    if config["alpha_bounds"] != config["neural"]["alpha_bounds"]:
        raise ValueError("The two methods must have the same configured diffusivity bounds.")
    expected = {
        (f"n{n}_seed{seed}_{family}_{timing}", method): (n, seed, family, timing)
        for n in COUNTS
        for seed in config["designs"][str(n)]
        for family in FAMILIES
        for timing in TIMINGS
        for method in METHODS
    }
    if (
        len(expected) != 720
        or manifest["expected_jobs"] != len(expected)
        or manifest["terminal_jobs"] != len(expected)
        or manifest["status"] != "complete"
        or manifest["failed_jobs"] != 0
    ):
        raise ValueError("Expected all 720 original timing fits to be complete.")
    records = read(records_path)
    indexed = {(r["case_id"], r["method"]): r for r in records}
    if len(records) != len(expected) or indexed.keys() != expected.keys():
        raise ValueError("The saved records do not match the frozen timing job set.")

    lo, hi = config["alpha_bounds"]
    tolerance = 8 * np.finfo(float).eps
    groups = defaultdict(list)
    seen = set()
    with boundary_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = row["case_id"], row["method"]
            if key in seen or key not in expected:
                raise ValueError(f"Unexpected or duplicate boundary row: {key}")
            seen.add(key)
            record = indexed[key]
            n, _, family, timing = expected[key]
            alpha = float(row["alpha_hat"])
            if (
                (int(row["n_obs"]), int(row["data_seed"]), row["family"], row["timing"])
                != expected[key]
                or (record["n_obs"], record["data_seed"], record["family"], record["timing"])
                != expected[key]
                or record["status"] != "completed"
                or alpha != record["alpha_hat"]
                or row["reported_at_bound"] != str(record["at_bound"])
                or not lo * (1 - tolerance) <= alpha <= hi * (1 + tolerance)
            ):
                raise ValueError(f"Boundary row does not match its completed fit: {key}")
            saved = [
                float(row[column])
                for column in (
                    "alpha_over_lower_bound",
                    "upper_bound_over_alpha",
                    "fraction_of_log_interval",
                )
            ]
            if not np.allclose(
                saved,
                [alpha / lo, hi / alpha, np.log(alpha / lo) / np.log(hi / lo)],
                rtol=1e-12,
                atol=1e-14,
            ):
                raise ValueError(f"Boundary distances disagree with the configured bounds: {key}")
            groups[n, family, timing, key[1]].append(record)
    if seen != expected.keys() or any(len(group) != 30 for group in groups.values()):
        raise ValueError("Expected 30 archived estimates per count, field, schedule and method.")
    input_sha256 = {
        f"repository/{path.relative_to(ROOT).as_posix()}": digest(path)
        for path in (ANCHOR, CONFIG, ROOT / "scripts/analyze_timing_evaluation.py")
    }
    for path in (manifest_path, config_path, records_path, diagnostic_path, boundary_path):
        input_sha256[f"evidence/{path.relative_to(evidence).as_posix()}"] = digest(path)
    return groups, config, input_sha256


def make_figure(groups, bounds, truth):
    lo, hi = bounds
    with plt.rc_context({"font.size": 9, "svg.hashsalt": "pinn-audit-bound-proximity-v1"}):
        fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.15, 4.5))
        fig.subplots_adjust(
            left=0.125, right=0.988, top=0.93, bottom=0.205, hspace=0.27, wspace=0.08
        )
        rng = np.random.default_rng(8103)
        for row, family in enumerate(FAMILIES):
            for col, n in enumerate(COUNTS):
                ax = axes[row, col]
                ax.set_title(f"Field {family}, $N={n}$", fontsize=10)
                ax.set_xscale("log")
                ax.set_xlim(lo / 1.65, hi * 1.65)
                ax.set_ylim(-0.55, 2.55)
                ax.set_yticks((2, 1, 0), labels=[timing.title() for timing in TIMINGS])
                ax.tick_params(axis="y", length=0)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color="0.91", linewidth=0.65)
                ax.axvline(truth, color="0.22", linestyle="--", linewidth=0.95)
                for bound in bounds:
                    ax.axvline(bound, color="0.46", linestyle=":", linewidth=1.1)
                for y, timing in zip((2, 1, 0), TIMINGS, strict=True):
                    for method in METHODS:
                        _, color, marker, offset = STYLES[method]
                        fits = sorted(
                            groups[n, family, timing, method], key=lambda r: r["data_seed"]
                        )
                        alpha = [r["alpha_hat"] for r in fits]
                        jitter = rng.uniform(-0.115, 0.115, size=len(alpha))
                        ax.scatter(
                            alpha,
                            y + offset + jitter,
                            color=color,
                            marker=marker,
                            s=11,
                            alpha=0.54,
                            linewidths=0,
                            zorder=3,
                        )
                        ax.scatter(
                            np.median(alpha),
                            y + offset,
                            color=color,
                            marker="D",
                            edgecolors="white",
                            linewidths=0.7,
                            s=29,
                            zorder=4,
                        )
        fig.supxlabel(
            r"Fitted diffusivity $\hat{\alpha}$ (log scale)", x=0.557, y=0.094, fontsize=10
        )
        handles = [
            Line2D(
                [], [], color=color, linestyle="none", marker=marker, label=label, markersize=4.5
            )
            for label, color, marker, _ in STYLES.values()
        ]
        handles.extend(
            [
                Line2D(
                    [],
                    [],
                    color="0.3",
                    linestyle="none",
                    marker="D",
                    label="Median",
                    markersize=4.5,
                ),
                Line2D([], [], color="0.22", linestyle="--", label="True diffusivity"),
                Line2D([], [], color="0.46", linestyle=":", label="Fitting bounds"),
            ]
        )
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5)
        png, svg = io.BytesIO(), io.StringIO()
        fig.savefig(png, format="png", dpi=360)
        fig.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(fig)
    clean_svg = "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n"
    return png.getvalue(), clean_svg.encode("utf-8")


def make_report(groups, config):
    lo, hi = config["alpha_bounds"]
    truth = config["alpha_true_generation_and_scoring_only"]
    lines = [
        "# Diffusivity estimates and fitting bounds",
        "",
        "![Fitted diffusivity by sample count, field, schedule, and method](bounds.png)",
        "",
        "**Figure caption.** Each point is a completed fit from the controlled timing evaluation. "
        "Each method and schedule has 30 fits in each panel. Families and schedules share the "
        "same design/noise blocks at fixed N. Vertical offsets separate methods and reduce "
        "overplotting. Diamonds mark medians. Dashed lines mark the "
        f"true diffusivity ({truth:g}). Dotted lines mark the common bounds ({lo:g}, {hi:g}). "
        "All four panels use the same horizontal scale.",
        "",
        "The table gives the median and interquartile range of fitted diffusivity divided by the "
        "lower bound. A ratio of one lies at the lower bound. Counts use the original numerical "
        "flags for proximity to either bound.",
        "",
        "| N | Field | Schedule | CV12 ratio, median [Q1, Q3] | PINN-1000 ratio, median [Q1, Q3] "
        "| CV12 flagged / 30 | PINN-1000 flagged / 30 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for n in COUNTS:
        for family in FAMILIES:
            for timing in TIMINGS:
                summaries, flags = [], []
                for method in METHODS:
                    fits = groups[n, family, timing, method]
                    q1, median, q3 = np.quantile(
                        [r["alpha_hat"] / lo for r in fits], [0.25, 0.5, 0.75]
                    )
                    summaries.append(f"{median:.5g} [{q1:.5g}, {q3:.5g}]")
                    flags.append(sum(r["at_bound"] for r in fits))
                cells = [str(n), family, timing.title(), *summaries, *map(str, flags)]
                lines.append("| " + " | ".join(cells) + " |")

    cv_fits = [r for (*_, method), fits in groups.items() if method == "cv12" for r in fits]
    pinn_fits = [r for (*_, method), fits in groups.items() if method == "pinn_1000" for r in fits]
    cv_flags = [r for r in cv_fits if r["at_bound"]]
    lower_flags = sum(np.isclose(r["alpha_hat"] / lo, 1, atol=0, rtol=1e-12) for r in cv_flags)
    pinn_flags = sum(r["at_bound"] for r in pinn_fits)
    closest = min((r for r in pinn_fits if not r["at_bound"]), key=lambda r: r["alpha_hat"])
    pct_above = 100 * (closest["alpha_hat"] / lo - 1)
    primary_early = np.median([r["alpha_hat"] for r in groups[10, "B", "early", "pinn_1000"]])
    lines.extend(
        [
            "",
            f"CV12 flagged {len(cv_flags)} of {len(cv_fits)} fits. {lower_flags} were at the lower "
            "bound to floating point precision. "
            f"PINN-1000 flagged {pinn_flags} of {len(pinn_fits)}. "
            f"Its lowest unflagged estimate was {closest['alpha_hat']:.7g} ({pct_above:.3f}% above "
            f"the lower bound, `{closest['case_id']}`). For field B with ten early observations, "
            f"the PINN median was {primary_early:.5g}, about {primary_early / lo:.2f} times the "
            "lower bound.",
            "",
            "CV12 can return an endpoint of its search interval. PINN-1000 maps an unconstrained "
            "parameter into the allowed log-diffusivity interval using a sigmoid. Continuous "
            "distances show how near each estimator came to a bound even when its flag was false. "
            "These are descriptive distributions from the frozen timing runs. See the paired "
            "analysis for comparisons within blocks.",
            "",
            "[Paired timing analysis](../../results/timing_evaluation_analysis_v1/summary.md)",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "results", help="Restored results root."
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT, help="New figure output directory.")
    args = parser.parse_args()
    evidence, out = args.evidence.resolve(), args.out_dir.resolve()
    if out.exists() or out.is_relative_to(evidence):
        parser.error("--out-dir must be a new directory outside the archived evidence.")
    groups, config, input_sha256 = load_estimates(evidence)
    png, svg = make_figure(
        groups, config["alpha_bounds"], config["alpha_true_generation_and_scoring_only"]
    )
    report = make_report(groups, config)
    out.mkdir(parents=True)
    (out / "bounds.png").write_bytes(png)
    (out / "bounds.svg").write_bytes(svg)
    (out / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    provenance = {
        "study_id": config["study_id"],
        "analysis_role": "post-evaluation descriptive display of the frozen timing estimates",
        "input_sha256": input_sha256,
        "script_sha256": digest(Path(__file__)),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "output_sha256": {
            name: digest(out / name) for name in ("bounds.png", "bounds.svg", "REPORT.md")
        },
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("Verified all 720 diffusivity estimates against the archived fit records.")
    print(f"Output written to {out}")


if __name__ == "__main__":
    main()
