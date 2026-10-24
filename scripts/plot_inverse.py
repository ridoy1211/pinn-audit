#!/usr/bin/env python3
"""Error in the recovered diffusivity against noise: classical fit versus PINN.

One row per setting (initial condition known or unknown), one column per number
of observations. Faint dots are individual seeds and the lines join the medians,
so spread is visible. The legend gives each method's median wall-clock time.
Lower is better.

Usage:  python scripts/plot_inverse.py --name heat_alpha_known heat_alpha_unknown
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# method -> (noise level -> errors in percent, one per seed), median time in seconds
PanelData = dict[str, tuple[dict[float, np.ndarray], float]]

LABELS = {
    "heat_alpha_known": "initial condition known",
    "heat_alpha_unknown": "initial condition unknown",
}
COLOURS = {"lsq": "C0", "pinn": "C3"}
OFFSETS = {"lsq": -0.06, "pinn": 0.06}


def load(name: str, results: Path) -> dict[int, PanelData]:
    import polars as pl

    df = pl.read_parquet(results / f"inverse_{name}.parquet")
    panels: dict[int, PanelData] = {}
    for n in sorted(df["n_obs"].unique().to_list()):
        sub = df.filter(pl.col("n_obs") == n)
        panel: PanelData = {}
        for method in sub["method"].unique(maintain_order=True).to_list():
            rows = sub.filter(pl.col("method") == method)
            by_noise = {
                float(noise): rows.filter(pl.col("noise") == noise)["rel_err"].to_numpy() * 100.0
                for noise in sorted(rows["noise"].unique().to_list())
            }
            median_time = rows["wall_time_s"].median()
            panel[str(method)] = (by_noise, float(median_time))  # type: ignore[arg-type]
        panels[int(n)] = panel
    return panels


def draw(settings: dict[str, dict[int, PanelData]], out_path: Path) -> None:
    rows = list(settings)
    cols = sorted({n for panels in settings.values() for n in panels})
    fig, axes = plt.subplots(
        len(rows), len(cols), figsize=(4.3 * len(cols), 3.7 * len(rows)), squeeze=False, sharey=True
    )
    for r, setting in enumerate(rows):
        for c, n in enumerate(cols):
            ax = axes[r][c]
            panel = settings[setting].get(n, {})
            for method, (by_noise, median_time) in panel.items():
                noises = sorted(by_noise)
                xs = np.arange(len(noises)) + OFFSETS.get(method, 0.0)
                for x, noise in zip(xs, noises, strict=True):
                    ax.scatter(
                        np.full(by_noise[noise].size, x),
                        by_noise[noise],
                        color=COLOURS.get(method, "grey"),
                        alpha=0.3,
                        s=16,
                    )
                ax.plot(
                    xs,
                    [float(np.median(by_noise[noise])) for noise in noises],
                    "o-",
                    color=COLOURS.get(method, "grey"),
                    label=f"{method} (median {median_time:.3g} s)",
                )
                ax.set_xticks(np.arange(len(noises)))
                ax.set_xticklabels([f"{100 * noise:g}%" for noise in noises])
            ax.set_yscale("log")
            ax.set_title(f"{LABELS.get(setting, setting)}\n{n} observations", fontsize=9.5)
            ax.set_xlabel("noise level")
            if c == 0:
                ax.set_ylabel("error in alpha (%)")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=7.5)
    fig.suptitle("Recovering the diffusivity from noisy data (lower is better)", fontsize=11.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main(names: list[str], results: Path, out_dir: Path) -> None:
    draw({name: load(name, results) for name in names}, out_dir / "inverse_error_vs_noise.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", nargs="+", required=True, help="inverse sweep names")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    a = ap.parse_args()
    main(a.name, a.results, a.out)
