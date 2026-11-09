"""Draw the frozen heat inversion design using verified configuration and saved points."""

import argparse
import io
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from analyze_timing_evaluation import TIMINGS, digest, read

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "figures/publication_design_v1"
N_EXAMPLE = 10
WINDOW_COLORS = ("#32749b", "#38796a")
PAIR_KEYS = ("x", "base_time_draw", "unit_noise", "mixed_early_mask", "fold_labels")
ALPHA_KEY = "alpha_true_generation_and_scoring_only"


def record_input(path, base, prefix, inputs, expected=None):
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Archived file hash mismatch: {path}")
    inputs[f"{prefix}/{path.relative_to(base).as_posix()}"] = actual
    return path


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_study(evidence, name, analysis_path, inputs):
    anchor = record_input(RESULTS / analysis_path, ROOT, "repository", inputs)
    folder = evidence / f"{name}_evaluation_v1"
    manifest = read(
        record_input(
            folder / "manifest.json",
            evidence,
            "evidence",
            inputs,
            read(anchor)["input_manifest_sha256"],
        )
    )
    config = read(
        record_input(
            folder / "config.json",
            evidence,
            "evidence",
            inputs,
            manifest["output_sha256"]["config.json"],
        )
    )
    committed = record_input(
        ROOT / f"configs/inverse/{name}_evaluation.json", ROOT, "repository", inputs
    )
    if config != read(committed) or config != manifest["freeze"]["config"]:
        raise ValueError(f"The saved {name} configuration does not match the freeze.")
    points = record_input(
        folder / "pde_points.npz", evidence, "evidence", inputs, manifest["pde_points_sha256"]
    )
    if manifest["status"] != "complete" or manifest["failed_jobs"] != 0:
        raise ValueError(f"The {name} evaluation is incomplete.")
    return config, manifest, folder, points


def load_block(evidence, folder, manifest, config, families, windows, inputs):
    n = N_EXAMPLE
    seed = config["designs"][str(n)][0]
    block = {}
    ref = None
    for family in families:
        for timing in config["timing_schedules"]:
            relative = f"data/n{n}_seed{seed}_{family}_{timing}.npz"
            path = record_input(
                folder / relative, evidence, "evidence", inputs, manifest["data_sha256"][relative]
            )
            obs = load_npz(path)
            if ref is None:
                ref = obs
            if any(not np.array_equal(obs[key], ref[key]) for key in PAIR_KEYS):
                raise ValueError(f"Observation pairing differs within block {seed}: {path.name}")
            early, late = [left + (right - left) * obs["base_time_draw"] for left, right in windows]
            mask = obs["mixed_early_mask"]
            expected_t = {"early": early, "mixed": np.where(mask, early, late), "late": late}[
                timing
            ]
            sigma = config["noise_sigma_absolute"]
            if (
                obs["x"].shape != (n,)
                or mask.sum() != n // 2
                or not np.allclose(obs["t"], expected_t, rtol=0, atol=2e-15)
                or obs["sigma"].item() != sigma
                or not np.array_equal(obs["y"], obs["clean"] + sigma * obs["unit_noise"])
            ):
                raise ValueError(f"Archived observations differ from the design: {path.name}")
            block[family, timing] = obs
    return block, seed


def load_design(evidence):
    inputs = {}
    timing, timing_manifest, timing_dir, points_path = load_study(
        evidence, "timing", "timing_evaluation_analysis_v1/analysis_provenance.json", inputs
    )
    offbasis, offbasis_manifest, offbasis_dir, offbasis_points = load_study(
        evidence, "offbasis", "offbasis_analysis_v1/provenance.json", inputs
    )
    shared = ("length", "final_time", ALPHA_KEY, "noise_sigma_absolute", "collocation_points")
    schedules = timing["timing_schedules"]
    if (
        any(timing[k] != offbasis[k] for k in shared)
        or digest(points_path) != digest(offbasis_points)
        or set(schedules) != set(TIMINGS)
        or offbasis["timing_schedules"] != {k: schedules[k] for k in ("mixed", "late")}
    ):
        raise ValueError("Expected the common frozen settings and the mixed/late off-basis subset.")
    windows = schedules["early"]["window"], schedules["late"]["window"]
    if schedules["mixed"] != dict(
        early_window=windows[0], late_window=windows[1], early_fraction=0.5
    ):
        raise ValueError(
            "The mixed design does not split observations equally between the windows."
        )
    for config in (timing, offbasis):
        if set(config["designs"]) != {"10", "50"} or any(
            len(set(seeds)) != 30 or len(seeds) != 30 for seeds in config["designs"].values()
        ):
            raise ValueError("Expected 30 blocks at each of N=10 and N=50 in each evaluation.")
    if any(set(timing["designs"][n]) & set(offbasis["designs"][n]) for n in timing["designs"]):
        raise ValueError("The two evaluations must use separate design/noise blocks.")

    truth_dir = RESULTS / "offbasis_truth_verification_v1"
    c = offbasis["truth"]
    truth_provenance = record_input(
        truth_dir / "provenance.json",
        ROOT,
        "repository",
        inputs,
        c["verification_provenance_sha256"],
    )
    truth_hashes = read(truth_provenance)["output_sha256"]
    initial = load_npz(
        record_input(
            truth_dir / "initial_field.npz",
            ROOT,
            "repository",
            inputs,
            truth_hashes["initial_field.npz"],
        )
    )
    x = initial["x"]
    direct = x * (1 - x) * np.exp(-((x - c["center"]) ** 2) / (2 * c["width"] ** 2))
    if not np.allclose(initial["initial"], direct / c["normalization"], rtol=1e-13, atol=1e-15):
        raise ValueError("The saved initial field disagrees with the verified definition.")
    profiles = {
        name: sum(
            row["amplitude"] * np.sin(row["n"] * np.pi * x / timing["length"]) for row in terms
        )
        for name, terms in timing["families"].items()
    }
    profiles["C"] = initial["initial"]
    block, seed = load_block(
        evidence, timing_dir, timing_manifest, timing, ("A", "B"), windows, inputs
    )
    _, offbasis_seed = load_block(
        evidence, offbasis_dir, offbasis_manifest, offbasis, ("C",), windows, inputs
    )
    training = load_npz(points_path)["training"]
    if training.shape != (timing["collocation_points"], 2):
        raise ValueError("Unexpected saved collocation shape.")
    record_input(ROOT / "scripts/analyze_timing_evaluation.py", ROOT, "repository", inputs)
    return (
        timing,
        offbasis,
        x,
        profiles,
        windows,
        block["B", "mixed"],
        training,
        seed,
        offbasis_seed,
        inputs,
    )


def draw(config, x, profiles, windows, obs, training):
    with plt.rc_context({"font.size": 8, "svg.hashsalt": "pinn-audit-study-design-v1"}):
        fig, (initial, schedule, points) = plt.subplots(1, 3, figsize=(7.15, 3.1))
        fig.subplots_adjust(left=0.084, right=0.982, bottom=0.31, top=0.873, wspace=0.48)
        initial.set_title("A  Initial fields", loc="left", fontsize=10)
        for label, color, style in (
            ("A", "#1f6488", "--"),
            ("B", "#9a430b", "-"),
            ("C", "#714a82", "-."),
        ):
            initial.plot(x, profiles[label], color=color, ls=style, lw=1.45, label=label)
        initial.set(
            xlim=(0, config["length"]), xlabel=r"Position $x$", ylabel=r"Temperature $u(x,0)$"
        )
        initial.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])
        initial.axhline(0, linewidth=0.6, color="0.8", zorder=0)
        initial.legend(loc="upper center", bbox_to_anchor=(0.5, -0.29), ncol=3, frameon=False)

        schedule.set_title("B  Timing schedules", loc="left", fontsize=10)
        rows = (((0, r"$N$"),), ((0, r"$N/2$"), (1, r"$N/2$")), ((1, r"$N$"),))
        for y, segments in zip((2, 1, 0), rows, strict=True):
            schedule.axhline(y, color="0.91", linewidth=0.7, zorder=0)
            for i, label in segments:
                left, right = windows[i]
                schedule.plot(
                    [left, right], [y, y], lw=5, color=WINDOW_COLORS[i], solid_capstyle="butt"
                )
                schedule.text((left + right) / 2, y + 0.16, label, ha="center", va="bottom")
        schedule.set_ylim(-0.49, 2.60)
        schedule.set_yticks([2, 1, 0], labels=[label.title() for label in TIMINGS])
        schedule.tick_params(axis="y", length=0)
        for name in ("top", "left", "right"):
            schedule.spines[name].set_visible(False)
        schedule.text(
            0.5,
            -0.32,
            f"$N=10$ or $50$\nAbsolute noise SD $={config['noise_sigma_absolute']:g}$",
            transform=schedule.transAxes,
            ha="center",
            va="top",
            linespacing=1.45,
        )

        points.set_title("C  Field B, mixed", loc="left", fontsize=10)
        for (left, right), color in zip(windows, WINDOW_COLORS, strict=True):
            points.axvspan(left, right, color=color, alpha=0.10, linewidth=0)
        scatter_obs = points.scatter(
            obs["t"],
            obs["x"],
            marker="D",
            s=19,
            color="#9a430b",
            edgecolors="white",
            lw=0.4,
            label=f"{len(obs['x'])} observations",
            zorder=3,
        )
        scatter_pde = points.scatter(
            training[:, 1],
            training[:, 0],
            s=3,
            color="#595f65",
            linewidths=0,
            alpha=0.52,
            label=f"{len(training)} PDE collocation points",
            zorder=2,
        )
        points.set(ylim=(0, config["length"]), ylabel=r"Position $x$")
        points.set_yticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])
        points.legend(
            handles=[scatter_obs, scatter_pde],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.29),
            frameon=False,
            markerscale=1.15,
        )
        for ax in (schedule, points):
            ax.set(xlim=(-0.025, config["final_time"] + 0.025), xlabel=r"Time $t$")
            ax.set_xticks(np.linspace(0, config["final_time"], 6))
        png, svg = io.BytesIO(), io.StringIO()
        fig.savefig(png, format="png", dpi=360)
        fig.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(fig)
    clean_svg = "\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n"
    return png.getvalue(), clean_svg.encode("utf-8")


def caption(config, offbasis, windows, seed):
    (e0, e1), (l0, l1) = windows
    c = offbasis["truth"]
    return (
        "# Controlled heat inversion design\n\n"
        "![Initial fields, observation schedules, and a saved sensor design](design.png)\n\n"
        "**Figure caption.** The one-dimensional heat equation uses zero temperature at both ends "
        f"of the unit interval and true diffusivity {config[ALPHA_KEY]:g}. The initial field and "
        "diffusivity are unknown to the fitted procedures. **A:** A and B are the prescribed sine "
        f"sums. C is a localized field proportional to $x(1-x)$ times a Gaussian centered at "
        f"{c['center']:g} with width {c['width']:g}, scaled to unit peak. **B:** Each dataset has "
        f"$N=10$ or $N=50$ observations in the indicated windows. Early is [{e0:.2f}, {e1:.2f}], "
        f"late is [{l0:.2f}, {l1:.2f}], and mixed assigns half to each window. Field C uses only "
        "mixed and late. **C:** The plotted observations are from the first frozen $N=10$ "
        f"block (field B, mixed schedule, seed {seed}). Both fits use these temperature "
        f"measurements. The {config['collocation_points']} gray points carry no temperature "
        "measurements and are used to evaluate the PINN's PDE loss during training.\n\n"
        "Each study has 30 independent design/noise blocks per observation count. Fields A and B "
        "share the same blocks. Within a block the schedules reuse spatial coordinates, normalized "
        "time draws, standard normal noise draws, and cross-validation folds. The off-basis "
        "evaluation uses fresh blocks. Independent standard normal draws are multiplied by the "
        f"fixed absolute noise standard deviation {config['noise_sigma_absolute']:g} and added to "
        "the simulated temperatures. Because the clean signals differ across schedules and fields, "
        "this convention does not fix signal-to-noise ratio. The schedules vary delay and time "
        "spread together.\n\n"
        "[Timing protocol](../../docs/TIMING_PROTOCOL.md) · "
        "[Off-basis protocol](../../docs/OFFBASIS_PROTOCOL.md)\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=RESULTS, help="Restored results root.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT, help="New figure output directory.")
    args = parser.parse_args()
    evidence, out = args.evidence.resolve(), args.out_dir.resolve()
    if out.exists() or out.is_relative_to(evidence):
        parser.error("--out-dir must be a new directory outside the archived evidence.")
    config, offbasis, x, profiles, windows, obs, training, seed, offbasis_seed, inputs = (
        load_design(evidence)
    )
    png, svg = draw(config, x, profiles, windows, obs, training)
    out.mkdir(parents=True)
    (out / "design.png").write_bytes(png)
    (out / "design.svg").write_bytes(svg)
    (out / "REPORT.md").write_text(
        caption(config, offbasis, windows, seed), encoding="utf-8", newline="\n"
    )
    provenance = {
        "analysis_role": "study design schematic from frozen configurations and saved points",
        "example_seed": seed,
        "offbasis_pairing_check_seed": offbasis_seed,
        "input_sha256": inputs,
        "script_sha256": digest(Path(__file__)),
        "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
        "output_sha256": {
            name: digest(out / name) for name in ("design.png", "design.svg", "REPORT.md")
        },
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        "Verified both frozen configurations, the saved initial profile, and both example blocks."
    )
    print(f"Output written to {out}")


if __name__ == "__main__":
    main()
