#!/usr/bin/env python3
"""Classical baselines against their analytic solutions.

Deliberately a script, not a notebook: figures must be regenerable with one
command from a clean checkout.

Usage:  python scripts/plot_baselines.py [--out figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinnaudit.config import ProblemConfig
from pinnaudit.metrics import observed_order, relative_l2
from pinnaudit.problems import Heat1D, Wave1D
from pinnaudit.solvers import CrankNicolsonHeat, LeapfrogWave


def nt_for_courant(nx: int, lam: float, c: float, t_max: float, x_max: float) -> int:
    h = x_max / (nx - 1)
    return round(t_max / (lam * h / c)) + 1


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    heat = Heat1D(ProblemConfig(name="heat1d"))
    wave = Wave1D(ProblemConfig(name="wave1d"))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for row, (problem, solver, label) in enumerate(
        [
            (heat, CrankNicolsonHeat(), "heat1d / Crank-Nicolson"),
            (wave, LeapfrogWave(), "wave1d / leapfrog"),
        ]
    ):
        nx = 81
        nt = nx if row == 0 else nt_for_courant(nx, 0.5, 1.0, 1.0, 1.0)
        sol = solver.solve(problem, nx=nx, nt=nt)
        ref = problem.exact(sol.x, sol.t)

        ax = axes[row, 0]
        im = ax.pcolormesh(sol.x, sol.t, sol.u, shading="auto", cmap="RdBu_r")
        ax.set(xlabel="x", ylabel="t", title=f"{label}\nnumerical solution")
        fig.colorbar(im, ax=ax)

        ax = axes[row, 1]
        for frac in (0.0, 0.25, 0.5, 1.0):
            n = int(frac * (nt - 1))
            ax.plot(sol.x, ref[n], "k-", lw=2.5, alpha=0.3)
            ax.plot(sol.x, sol.u[n], "--", lw=1.4, label=f"t = {sol.t[n]:.2f}")
        ax.set(xlabel="x", ylabel="u", title="slices: analytic (grey) vs numerical (dashed)")
        ax.legend(fontsize=8)

        ax = axes[row, 2]
        hs, errs = [], []
        for n in [21, 41, 81, 161]:
            m = n if row == 0 else nt_for_courant(n, 0.5, 1.0, 1.0, 1.0)
            s = solver.solve(problem, nx=n, nt=m)
            hs.append(1.0 / (n - 1))
            errs.append(relative_l2(s.u, problem.exact(s.x, s.t)))
        orders = observed_order(errs)
        ax.loglog(hs, errs, "o-", label="measured")
        ax.loglog(hs, [errs[0] * (h / hs[0]) ** 2 for h in hs], "k--", alpha=0.6, label="$O(h^2)$")
        ax.set(
            xlabel="h",
            ylabel="relative $L^2$ error",
            title=f"convergence: observed order {orders[-1]:.2f}",
        )
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("Classical baselines vs. analytic solutions", fontsize=13)
    fig.tight_layout()
    path = out_dir / "baselines.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    print(
        f"peak |u| heat={np.max(np.abs(heat.exact(np.linspace(0, 1, 101), np.array([0.0])))):.4f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("figures"))
    main(ap.parse_args().out)
