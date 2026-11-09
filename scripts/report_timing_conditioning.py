"""Render the saved rank, residual, and ridge diagnostics for publication."""

import argparse
import csv
import io
import json
from inspect import getfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analyze_timing_evaluation import TIMINGS, digest, read
from pinnaudit.spectral_inverse import heat_design

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/inverse/timing_evaluation.json"
ANCHOR = ROOT / "results/timing_fix2_report_v1/provenance.json"
MODES = 12
N_EXAMPLE = 10
BLOCKS = 30
DISPLAY_FLOOR = 1e-30
STYLES = {
    "early": ("#1f6488", "-", "o"),
    "mixed": ("#287a5b", "--", "s"),
    "late": ("#9a430b", ":", "^"),
}


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Archived file has changed: {path}")
    return path


def table_rows(folder, hashes, name):
    path = checked(folder / name, hashes[name])
    with path.open(newline="", encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["family"] == "B"]


def rank_summary(ranks, config):
    true_alpha = config["alpha_true_generation_and_scoring_only"]
    at_truth = [r for r in ranks if float(r["alpha"]) == true_alpha]
    summaries = []
    for n in (10, 50):
        seeds = config["designs"][str(n)]
        for timing in TIMINGS:
            group = [r for r in at_truth if (r["n_obs"], r["timing"]) == (str(n), timing)]
            if len(group) != BLOCKS or {int(r["data_seed"]) for r in group} != set(seeds):
                raise ValueError(f"Unexpected design set for N={n}, {timing}.")
            rank = min(n, MODES)
            if any(
                (int(r["modes"]), int(r["numerical_rank"]), int(r["column_nullity_at_tolerance"]))
                != (MODES, rank, MODES - rank)
                for r in group
            ):
                raise ValueError(f"Unexpected rank or nullity for N={n}, {timing}.")
            smallest_over_cutoff = [
                float(r["smallest_returned_singular_value"]) / float(r["rank_tolerance"])
                for r in group
            ]
            norms = [float(r["unregularized_amplitude_l2"]) for r in group]
            summaries.append(
                (n, timing, rank, MODES - rank, np.median(smallest_over_cutoff), np.median(norms))
            )
    return at_truth, summaries


def load_examples(ranks, at_truth, ridge_rows, source, hashes, config):
    seed = config["designs"][str(N_EXAMPLE)][0]
    true_alpha = config["alpha_true_generation_and_scoring_only"]
    truth_by_case = {r["case_id"]: r for r in at_truth}
    examples, observation_hashes = {}, {}
    for timing in TIMINGS:
        case = f"n{N_EXAMPLE}_seed{seed}_B_{timing}"
        name = f"data/{case}.npz"
        observation_hashes[name] = hashes[name]
        with np.load(checked(source / name, hashes[name]), allow_pickle=False) as data:
            design = heat_design(data["x"], data["t"], true_alpha, config["length"], MODES)
        singular = np.linalg.svd(design, compute_uv=False)
        row = truth_by_case[case]
        cutoff = float(row["rank_tolerance"])
        np.testing.assert_allclose(
            [singular[0], singular[-1]],
            [float(row["largest_singular_value"]), float(row["smallest_returned_singular_value"])],
            rtol=1e-6,
            atol=0,
        )
        if np.count_nonzero(singular > cutoff) != int(row["numerical_rank"]):
            raise ValueError(
                f"Recomputed singular values disagree with the archived rank for {case}."
            )
        profile = sorted(
            (r for r in ranks if r["case_id"] == case), key=lambda r: float(r["alpha"])
        )
        candidates = sorted(
            (r for r in ridge_rows if r["case_id"] == case), key=lambda r: float(r["ridge"])
        )
        ridges = [float(r["ridge"]) for r in candidates]
        if ridges != config["ridges"]:
            raise ValueError(f"Unexpected ridge grid for {case}.")
        validation = np.array([float(r["validation_mse"]) for r in candidates])
        selected = [i for i, r in enumerate(candidates) if r["selected_by_cv"] == "True"]
        if selected != [int(np.argmin(validation))]:
            raise ValueError(
                f"Archived ridge selection differs from the validation minimum for {case}."
            )
        examples[timing] = {
            "spectrum": singular / singular[0],
            "cutoff": cutoff / singular[0],
            "candidate_alpha": np.array([float(r["alpha"]) for r in profile]),
            "mse": np.array([float(r["unregularized_data_mse"]) for r in profile]),
            "rank_drop": np.array([int(r["numerical_rank"]) < N_EXAMPLE for r in profile]),
            "ridges": ridges,
            "validation": validation,
            "fitted_alpha": [float(r["alpha_hat"]) for r in candidates],
            "selected": selected[0],
        }
    return seed, examples, observation_hashes


def render_figure(examples, config, seed):
    true_alpha = config["alpha_true_generation_and_scoring_only"]
    with plt.rc_context({"font.size": 9, "svg.hashsalt": "pinn-audit-conditioning-v1"}):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), layout="constrained")
        spectrum, profile, cv, fitted = axes.flat
        fig.suptitle(
            f"Twelve-mode spectral inversion, 10 observations, field B, seed {seed}", fontsize=11
        )
        for timing, series in examples.items():
            color, style, marker = STYLES[timing]
            spectrum.semilogy(
                range(1, N_EXAMPLE + 1),
                series["spectrum"],
                color=color,
                linestyle=style,
                marker=marker,
                markersize=3,
                label=timing.capitalize(),
            )
            mse = np.maximum(series["mse"], DISPLAY_FLOOR)
            alpha = series["candidate_alpha"]
            profile.loglog(alpha, mse, color=color, linestyle=style)
            dropped = series["rank_drop"]
            profile.plot(alpha[dropped], mse[dropped], "x", color=color, markersize=3, alpha=0.65)
            for ax, key in ((cv, "validation"), (fitted, "fitted_alpha")):
                ax.loglog(
                    series["ridges"],
                    series[key],
                    color=color,
                    linestyle=style,
                    marker=marker,
                    markersize=3,
                )
                i = series["selected"]
                ax.plot(
                    series["ridges"][i],
                    series[key][i],
                    "D",
                    color=color,
                    markerfacecolor="white",
                    markersize=5.5,
                    zorder=4,
                )
        cutoff = examples["early"]["cutoff"]
        spectrum.axhline(cutoff, color="0.48", linestyle="--", linewidth=0.9)
        spectrum.text(
            0.98,
            cutoff,
            "Rank cutoff",
            transform=spectrum.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
        )
        spectrum.set(
            title="A  Singular spectrum at true diffusivity",
            xlabel="Returned singular value index",
            ylabel="Singular value / largest",
            ylim=(cutoff / 6, 2.5),
        )
        spectrum.set_xticks([1, 3, 5, 7, 10])
        spectrum.legend(loc="lower left", bbox_to_anchor=(0, 0.12), frameon=False, fontsize=8)
        profile.axvline(true_alpha, color="0.5", linestyle="--", linewidth=0.9)
        profile.set(
            title="B  Unregularized residual profile",
            xlabel="Candidate diffusivity",
            ylabel="Profiled data MSE",
        )
        cv.set(title="C  Cross-validation error", xlabel="Ridge weight", ylabel="Validation MSE")
        fitted.axhline(true_alpha, color="0.5", linestyle="--", linewidth=0.9)
        fitted.axhline(config["alpha_bounds"][0], color="0.55", linestyle=":", linewidth=0.9)
        fitted.set(
            title="D  Fit to all observations",
            xlabel="Ridge weight",
            ylabel="Fitted diffusivity",
        )
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)

        png = io.BytesIO()
        svg = io.StringIO()
        fig.savefig(png, format="png", dpi=360)
        fig.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(fig)
    svg_text = "\n".join(line.rstrip() for line in svg.getvalue().split("\n"))
    return png.getvalue(), svg_text.encode("utf-8")


def report_text(seed, config, summaries, examples):
    true_alpha = config["alpha_true_generation_and_scoring_only"]
    lines = [
        "# Conditioning and ridge selection",
        "",
        f"The rank summaries below use the true diffusivity {true_alpha:g} and one "
        "copy of each design. Fields A and B share the same observation geometries. "
        "Numerical rank counts singular values above `max(N, 12) * eps * sigma_max`. "
        "The coefficient norms use field B and the minimum-norm least-squares solution.",
        "",
        "| N | Schedule | Designs | Numerical rank | Coefficient nullity | "
        "Median smallest returned singular value / cutoff | Median coefficient L2 norm |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for n, timing, rank, nullity, ratio, norm in summaries:
        lines.append(
            f"| {n} | {timing.capitalize()} | {BLOCKS} | {rank} | {nullity} | "
            f"{ratio:.3g} | {norm:.3g} |"
        )
    selections = []
    for timing, series in examples.items():
        i = series["selected"]
        selections.append(f"{timing} ({series['ridges'][i]:.3g}, {series['fitted_alpha'][i]:.5g})")
    lines += [
        "",
        "A full-row-rank 10 by 12 matrix has two coefficient null directions. "
        "Exact full row rank persists near any interior diffusivity with a nonzero "
        "10 by 10 minor, because that minor varies continuously with diffusivity. "
        "Free coefficients can therefore interpolate the same observations at nearby "
        "diffusivities under that condition. For 50 observations, full column rank "
        "concerns the coefficients at fixed diffusivity.",
        "",
        "![Singular spectra, residual profiles, and ridge selection](conditioning.png)",
        "",
        f"**Figure caption.** Field B and the first frozen N=10 seed ({seed}) give the "
        "same example block for all three schedules. Panel A shows the ten returned "
        f"singular values at diffusivity {true_alpha:g}, scaled by the largest. The dashed "
        "line shows the numerical rank cutoff on that scale. Panel B profiles the data "
        "MSE over unrestricted coefficients at each candidate diffusivity. Crosses mark "
        f"numerical rank below ten. Values below {DISPLAY_FLOOR:g} are drawn at the "
        "display floor. Panels C and D show the archived validation MSE and full-data "
        "diffusivity estimate for each candidate ridge weight. Diamonds mark the ridge "
        "weight selected by cross-validation. Dashed lines in B and D mark the true "
        "diffusivity. The dotted horizontal line in D marks the lower bound.",
        "",
        "For this block the selected (ridge weight, fitted diffusivity) pairs are "
        + ", ".join(selections)
        + ".",
        "",
        "[Archived timing diagnostics](../../results/timing_fix2_report_v1/REPORT.md)",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "results",
        help="Results directory containing the restored timing evidence and diagnostics.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "figures/publication_conditioning_v1",
        help="New directory for the figure, table, and provenance.",
    )
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    out = args.out_dir.resolve()
    if out.exists() or out.is_relative_to(evidence) or out.is_relative_to(ROOT / "results"):
        parser.error("--out-dir must be a new directory outside the evidence.")

    source = evidence / "timing_evaluation_v1"
    diagnostics = evidence / "timing_identifiability_v1"
    anchors = read(ANCHOR)["input_provenance_sha256"]
    manifest_path = checked(source / "manifest.json", anchors["evaluation_manifest"])
    diagnostic_path = checked(diagnostics / "provenance.json", anchors["timing_identifiability_v1"])
    manifest = read(manifest_path)
    archived = read(diagnostic_path)
    if archived["input_manifest_sha256"] != anchors["evaluation_manifest"]:
        raise ValueError("The diagnostics and evaluation have different source manifests.")
    hashes = manifest["output_sha256"]
    config = read(checked(source / "config.json", hashes["config.json"]))
    if config != read(CONFIG) or manifest["status"] != "complete":
        raise ValueError("Expected the completed frozen timing evaluation.")
    design_hash = manifest["source_sha256"]["src/pinnaudit/spectral_inverse.py"]
    checked(Path(getfile(heat_design)), design_hash)

    tables = archived["output_sha256"]
    ranks = table_rows(diagnostics, tables, "rank_profiles.csv")
    ridge_rows = table_rows(diagnostics, tables, "regularization_sensitivity.csv")
    at_truth, summaries = rank_summary(ranks, config)
    seed, examples, data_hashes = load_examples(ranks, at_truth, ridge_rows, source, hashes, config)
    png, svg = render_figure(examples, config, seed)
    report = report_text(seed, config, summaries, examples)

    out.mkdir(parents=True)
    (out / "conditioning.png").write_bytes(png)
    (out / "conditioning.svg").write_bytes(svg)
    (out / "REPORT.md").write_text(report, encoding="utf-8", newline="\n")
    provenance = {
        "classification": "publication display of post-evaluation conditioning diagnostics",
        "example": {
            "n_obs": N_EXAMPLE,
            "family": "B",
            "seed": seed,
            "selection": "first listed seed",
        },
        "inputs": {
            "evaluation_manifest_sha256": anchors["evaluation_manifest"],
            "diagnostic_provenance_sha256": anchors["timing_identifiability_v1"],
            "design_source_sha256": design_hash,
            "observation_sha256": data_hashes,
        },
        "script_sha256": digest(Path(__file__)),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "output_sha256": {
            name: digest(out / name)
            for name in ("conditioning.png", "conditioning.svg", "REPORT.md")
        },
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Verified {len(summaries) * BLOCKS} design and schedule combinations.")
    print(f"Output written to {out}")


if __name__ == "__main__":
    main()
