"""Command line entry point: ``pinnaudit <command>``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pinnaudit import __version__
from pinnaudit.config import load_run_config
from pinnaudit.experiment import run_once
from pinnaudit.io import append_records, build_record, dump_config
from pinnaudit.metrics import observed_order
from pinnaudit.problems import available_problems
from pinnaudit.solvers import available_solvers

app = typer.Typer(
    add_completion=False,
    help="Audit of physics-informed neural networks versus classical PDE solvers.",
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"pinnaudit {__version__}")


@app.command(name="list")
def list_components() -> None:
    """List registered problems and solvers."""
    typer.echo("problems: " + ", ".join(available_problems()))
    typer.echo("solvers:  " + ", ".join(available_solvers()))


@app.command()
def run(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
    save: Annotated[bool, typer.Option("--save/--no-save", help="Write a Parquet record.")] = True,
) -> None:
    """Run one experiment described by a YAML config."""
    cfg = load_run_config(config)
    result = run_once(cfg)
    typer.echo(result.summary())

    if save:
        record = build_record(cfg, result.solution, result.metrics)
        path = append_records([record], cfg.out_dir, name="runs")
        dump_config(cfg, cfg.out_dir)
        typer.echo(f"wrote {path}")


@app.command()
def convergence(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
    levels: Annotated[int, typer.Option(min=2, max=8, help="Number of refinements.")] = 4,
) -> None:
    """Refine the grid in the config and report the observed convergence order.

    Both nx and nt are doubled at each level, so the mesh ratio and Courant
    number stay fixed and the measured order is the scheme's true order.
    """
    base = load_run_config(config)
    errors: list[float] = []
    rows = []

    for level in range(levels):
        factor = 2**level
        cfg = base.model_copy(deep=True)
        cfg.run_id = f"{base.run_id}_L{level}"
        cfg.solver.nx = (base.solver.nx - 1) * factor + 1
        cfg.solver.nt = (base.solver.nt - 1) * factor + 1

        result = run_once(cfg)
        errors.append(result.metrics["relative_l2"])
        rows.append(build_record(cfg, result.solution, result.metrics))
        typer.echo(result.summary())

    orders = observed_order(errors)
    typer.echo("")
    typer.echo("observed orders: " + ", ".join(f"{o:.3f}" for o in orders))
    typer.echo(f"asymptotic order: {orders[-1]:.3f}")

    path = append_records(rows, base.out_dir, name="convergence")
    typer.echo(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
