"""Command line entry point: ``pinnaudit <command>``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from pinnaudit import __version__
from pinnaudit.config import NEURAL_SOLVERS, load_run_config
from pinnaudit.experiment import run_once
from pinnaudit.io import append_records, build_record, dump_config
from pinnaudit.metrics import observed_order
from pinnaudit.modes import load_mode_contributions
from pinnaudit.problems import available_problems, build_problem
from pinnaudit.solvers import available_solvers
from pinnaudit.summary import summarise, to_markdown
from pinnaudit.sweep import load_sweep_config, run_sweep

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
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Log training progress.")
    ] = False,
) -> None:
    """Run one experiment described by a YAML config."""
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
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
    if base.solver.name in NEURAL_SOLVERS:
        raise typer.BadParameter("convergence studies apply to classical solvers only")
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


@app.command()
def sweep(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
    n_seeds: Annotated[int | None, typer.Option(min=1, help="Use only the first N seeds.")] = None,
    only: Annotated[list[str] | None, typer.Option(help="Run only these variant labels.")] = None,
) -> None:
    """Run every variant over every seed. Safe to interrupt and re-run."""
    sweep_cfg = load_sweep_config(config)
    try:
        outcome = run_sweep(sweep_cfg, n_seeds=n_seeds, only=only, echo=typer.echo)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo("")
    typer.echo(
        f"ran {outcome.ran}, skipped {outcome.skipped} already done, failed {outcome.failed}"
    )
    if outcome.results_path.exists():
        typer.echo(f"results: {outcome.results_path}")
        typer.echo("")
        typer.echo(to_markdown(summarise(pl.read_parquet(outcome.results_path))))


@app.command()
def summary(
    name: Annotated[str, typer.Option("--name", "-n", help="Sweep name.")],
    results: Annotated[Path, typer.Option(help="Results directory.")] = Path("results"),
) -> None:
    """Print the across-seed summary table for a finished sweep."""
    path = results / f"sweep_{name}.parquet"
    if not path.exists():
        raise typer.BadParameter(f"no results at {path}; run the sweep first")
    typer.echo(to_markdown(summarise(pl.read_parquet(path))))


@app.command()
def modes(
    name: Annotated[str, typer.Option("--name", "-n", help="Sweep name.")],
    problem_config: Annotated[
        Path,
        typer.Option(help="Run config that defines the problem.", exists=True, dir_okay=False),
    ] = Path("configs/pinn_wave.yaml"),
    variant: Annotated[
        list[str] | None, typer.Option(help="Variant to show (repeatable). Default: all neural.")
    ] = None,
    n_modes: Annotated[int, typer.Option("--modes", min=1, help="Modes to report.")] = 6,
    results: Annotated[Path, typer.Option(help="Results directory.")] = Path("results"),
) -> None:
    """Show which Fourier modes carry each variant's error (mean over seeds)."""
    path = results / f"sweep_{name}.parquet"
    if not path.exists():
        raise typer.BadParameter(f"no results at {path}; run the sweep first")
    df = pl.read_parquet(path)
    problem = build_problem(load_run_config(problem_config).problem)

    labels = variant or [
        v for v in df["variant"].unique(maintain_order=True).to_list() if not v.startswith("fd_")
    ]
    header = "  ".join(f"mode{n}".rjust(6) for n in range(1, n_modes + 1))
    typer.echo(f"variant        runs  {header}")
    for label in labels:
        files = df.filter(pl.col("variant") == label).sort("seed")["solution_file"].to_list()
        if not files:
            typer.echo(f"{label:<14} no runs")
            continue
        mean = load_mode_contributions(files, problem, n_modes).mean(axis=0)
        typer.echo(f"{label:<14} {len(files):>4}  " + "  ".join(f"{v:6.3f}" for v in mean))


if __name__ == "__main__":  # pragma: no cover
    app()
