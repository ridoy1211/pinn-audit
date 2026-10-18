#!/usr/bin/env python3
"""Which Fourier modes carry the neural solvers' error on the wave problem?

Top row: the amplitude of mode 1 and mode 3 over time for one example run,
against the exact answer. Bottom: each mode's contribution to the relative L2
error, averaged over seeds, for every neural variant. If a PINN really fails
because of spectral bias, mode 3 should dominate the error and its amplitude
curve should be flattened toward zero.

Usage:  python scripts/plot_wave_modes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinnaudit.modes import mode_amplitudes, mode_error_contributions


def draw(
    t: np.ndarray,
    exact_amps: np.ndarray,
    shown_amps: np.ndarray,
    shown_label: str,
    contributions: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """``contributions`` maps variant -> array of shape (n_seeds, n_modes)."""
    n_modes = next(iter(contributions.values())).shape[1]

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(2, 2)
    for col, mode in enumerate((1, 3)):
        ax = fig.add_subplot(grid[0, col])
        ax.plot(t, exact_amps[:, mode - 1], "k--", lw=2, label="exact")
        ax.plot(t, shown_amps[:, mode - 1], lw=1.6, label=shown_label)
        ax.set(xlabel="t", ylabel=f"amplitude of mode {mode}", title=f"mode {mode} over time")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    ax = fig.add_subplot(grid[1, :])
    labels = list(contributions)
    width = 0.8 / len(labels)
    modes = np.arange(1, n_modes + 1)
    for i, label in enumerate(labels):
        values = contributions[label]
        spread = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(n_modes)
        ax.bar(
            modes + (i - (len(labels) - 1) / 2) * width,
            values.mean(axis=0),
            width,
            yerr=spread,
            capsize=2,
            label=f"{label} (n={values.shape[0]})",
        )
    ax.set(
        xlabel="Fourier mode n",
        ylabel="share of relative $L^2$ error",
        title="Where the error lives (the exact solution contains modes 1 and 3 only)",
    )
    ax.set_xticks(modes)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def main(
    name: str,
    problem_config: Path,
    variants: list[str],
    show: str,
    n_modes: int,
    results: Path,
    out_dir: Path,
) -> None:
    import polars as pl

    from pinnaudit.config import load_run_config
    from pinnaudit.problems import build_problem

    problem = build_problem(load_run_config(problem_config).problem)
    df = pl.read_parquet(results / f"sweep_{name}.parquet")

    contributions: dict[str, np.ndarray] = {}
    shown: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    for variant in variants:
        rows = df.filter(pl.col("variant") == variant).sort("seed")
        if rows.height == 0:
            print(f"warning: no runs for variant {variant!r}, skipping")
            continue
        per_seed = []
        for path in rows["solution_file"].to_list():
            sol = np.load(path)
            u_ref = problem.exact(sol["x"], sol["t"])
            per_seed.append(mode_error_contributions(sol["u"], u_ref, sol["x"], n_modes))
            if variant == show and shown is None:
                shown = (sol["x"], sol["t"], sol["u"])
        contributions[variant] = np.array(per_seed)

    if not contributions:
        raise SystemExit("no matching runs found; run the sweep first")
    if shown is None:
        raise SystemExit(f"variant {show!r} has no runs to plot")

    x, t, u = shown
    exact_amps = mode_amplitudes(problem.exact(x, t), x, n_modes)
    draw(
        t,
        exact_amps,
        mode_amplitudes(u, x, n_modes),
        f"{show} (seed 0)",
        contributions,
        out_dir / "wave_modes.png",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="forward_wave")
    ap.add_argument("--problem-config", type=Path, default=Path("configs/pinn_wave.yaml"))
    ap.add_argument(
        "--variants", nargs="+", default=["pinn", "pinn_lbfgs", "pinn_data200", "mlp_data200"]
    )
    ap.add_argument("--show", default="pinn")
    ap.add_argument("--modes", type=int, default=8)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    a = ap.parse_args()
    main(a.name, a.problem_config, a.variants, a.show, a.modes, a.results, a.out)
