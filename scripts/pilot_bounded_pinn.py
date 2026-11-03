#!/usr/bin/env python3
"""Data-only PINN feasibility pilot with saved fields and bounded diffusivity."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pinnaudit.io import current_code_revision
from pinnaudit.solvers.neural import DTYPE, MLP, heat_residual
from score_sparse_diagnostic import quadrature_weights, relative_error, true_field


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def column(values):
    return torch.tensor(np.asarray(values).reshape(-1, 1), dtype=DTYPE, device="cpu")


class BoundedHeatField(nn.Module):
    """Unknown initial field; exact unit-interval zero BC; bounded learnable alpha."""

    def __init__(self, settings):
        super().__init__()
        lower, upper = settings["alpha_bounds"]
        alpha0 = settings["alpha0"]
        if not (0 < lower < alpha0 < upper):
            raise ValueError("alpha0 must be strictly inside positive ordered bounds")
        self.log_lower, self.log_upper = math.log(lower), math.log(upper)
        fraction = (math.log(alpha0) - self.log_lower) / (self.log_upper - self.log_lower)
        self.raw_alpha = nn.Parameter(
            torch.tensor(math.log(fraction / (1 - fraction)), dtype=DTYPE)
        )
        self.network = MLP(settings["width"], settings["depth"], (0.0, 1.0), (0.0, 1.0))
        self.to(device="cpu", dtype=DTYPE)

    def alpha(self):
        return torch.exp(
            self.log_lower + (self.log_upper - self.log_lower) * torch.sigmoid(self.raw_alpha)
        )

    def forward(self, x, t):
        return x * (1 - x) * self.network(x, t)


def train_one(x, t, noisy_y, collocation, settings, training_seed, job_dir, echo=print):
    """Fit using noisy data and PDE only. No simulation truth is accepted."""
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(training_seed)
            return _train_one(x, t, noisy_y, collocation, settings, training_seed, job_dir, echo)
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


def _train_one(x, t, noisy_y, collocation, settings, training_seed, job_dir, echo):
    arrays = [np.asarray(v, dtype=float) for v in (x, t, noisy_y)]
    if any(v.ndim != 1 or not np.isfinite(v).all() for v in arrays):
        raise ValueError("Expected finite 1D observations")
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape) or not len(arrays[0]):
        raise ValueError("Observation shapes must match and be nonempty")
    if any(np.any((v < 0) | (v > 1)) for v in arrays[:2]):
        raise ValueError("Observation coordinates must lie in the unit square")
    if np.asarray(collocation).ndim != 2 or np.asarray(collocation).shape[1] != 2:
        raise ValueError("Collocation coordinates must have shape (n,2)")
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=False)
    model = BoundedHeatField(settings)
    ox, ot, oy = (column(v) for v in arrays)
    cx, ct = column(collocation[:, 0]), column(collocation[:, 1])
    trace, completed_adam, closure_calls = [], 0, 0
    lbfgs = None
    start = time.perf_counter()

    def budget():
        if time.perf_counter() - start > settings["max_train_seconds"]:
            raise TimeoutError("Declared training time budget exceeded")

    def loss_fn():
        data = torch.mean((model(ox, ot) - oy) ** 2)
        pde = torch.mean(heat_residual(model, cx, ct, model.alpha()) ** 2)
        total = settings["data_weight"] * data + settings["pde_weight"] * pde
        if not torch.isfinite(total):
            raise RuntimeError("Nonfinite training loss")
        return total, data, pde

    def checkpoint(status, error=None):
        payload = {
            "model_state": model.state_dict(),
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "settings": settings,
            "training_seed": training_seed,
            "status": status,
            "error": error,
            "adam_steps_completed": completed_adam,
            "lbfgs_closure_calls": closure_calls,
            "lbfgs_iterations": (
                int(lbfgs.state[next(iter(model.parameters()))].get("n_iter", 0))
                if lbfgs is not None
                else 0
            ),
            "training_elapsed_s": time.perf_counter() - start,
        }
        name = "failed_checkpoint.pt" if error is not None else "checkpoint.pt"
        temporary = job_dir / (name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(job_dir / name)
        write_json(job_dir / "trace.json", trace)

    def snapshot(stage):
        total, data, pde = loss_fn()
        row = {
            "stage": stage,
            "adam_steps": completed_adam,
            "lbfgs_closure_calls": closure_calls,
            "alpha": float(model.alpha().detach()),
            "loss": float(total.detach()),
            "data_mse": float(data.detach()),
            "pde_mse": float(pde.detach()),
            "elapsed_s": time.perf_counter() - start,
        }
        trace.append(row)
        checkpoint(stage)
        echo(f"  {stage}: Adam={completed_adam}, alpha={row['alpha']:.6g}, loss={row['loss']:.5g}")

    def finite_gradients():
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite training gradient")

    try:
        snapshot("initial")
        adam = torch.optim.Adam(
            [
                {"params": list(model.network.parameters()), "lr": settings["lr"]},
                {"params": [model.raw_alpha], "lr": settings["lr"] * settings["alpha_lr_factor"]},
            ]
        )
        for step in range(settings["adam_steps"]):
            budget()
            adam.zero_grad(set_to_none=True)
            total, _, _ = loss_fn()
            total.backward()
            finite_gradients()
            adam.step()
            completed_adam = step + 1
            if completed_adam % settings["trace_every"] == 0:
                snapshot("adam")
        snapshot("adam_end")
        if settings["lbfgs_iters"]:
            lbfgs = torch.optim.LBFGS(
                list(model.parameters()),
                lr=1.0,
                max_iter=settings["lbfgs_iters"],
                max_eval=settings["lbfgs_max_eval"],
                history_size=settings["lbfgs_history_size"],
                tolerance_grad=settings["tolerance_grad"],
                tolerance_change=settings["tolerance_change"],
                line_search_fn="strong_wolfe",
            )

            def closure():
                nonlocal closure_calls
                budget()
                lbfgs.zero_grad(set_to_none=True)
                total, _, _ = loss_fn()
                total.backward()
                finite_gradients()
                closure_calls += 1
                return total

            echo("  L-BFGS stage started")
            lbfgs.step(closure)
        budget()
        snapshot("completed_budget")
        return model, trace, time.perf_counter() - start
    except BaseException as exc:
        checkpoint("timeout" if isinstance(exc, TimeoutError) else "failed", repr(exc))
        raise


def predict_paired(model, x, t):
    output = []
    with torch.no_grad():
        for start in range(0, len(x), 4096):
            output.append(
                model(column(x[start : start + 4096]), column(t[start : start + 4096])).numpy()[
                    :, 0
                ]
            )
    return np.concatenate(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/inverse/bounded_pinn_pilot.json")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/bounded_pinn_pilot_v1"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    imported = inspect.getsourcefile(MLP)
    if imported is None or Path(imported).resolve() != root / "src/pinnaudit/solvers/neural.py":
        raise RuntimeError("Wrong imported checkout; reinstall this checkout's editable package")
    if args.out_dir.exists():
        raise FileExistsError("Choose a fresh output directory; preserve earlier runs")
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.smoke:
        settings.update(
            stage="software_smoke_only",
            data_directory="audit/data_only_tuning_smoke",
            datasets=[[10, 987000]],
            training_seeds=[91],
            width=8,
            depth=2,
            adam_steps=3,
            lbfgs_iters=2,
            lbfgs_max_eval=5,
            collocation_points=16,
            residual_eval_points=64,
            grid_points=21,
            trace_every=1,
            max_train_seconds=60.0,
        )
    data_dir = root / settings["data_directory"]
    data_paths = [
        data_dir / f"observations_n{n}_seed{seed}.npz" for n, seed in settings["datasets"]
    ]
    context_path = data_dir / "records.json"
    source_paths = [
        Path(__file__).resolve(),
        args.config.resolve(),
        root / "scripts/score_sparse_diagnostic.py",
        root / "src/pinnaudit/solvers/neural.py",
        root / "src/pinnaudit/io.py",
        root / "docs/BOUNDED_PINN_PILOT_PLAN.md",
    ]
    for path in [*data_paths, context_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out_dir.mkdir(parents=True)
    write_json(args.out_dir / "config.json", settings)
    manifest = {
        "started_utc": datetime.now(UTC).isoformat(),
        "status": "in_progress",
        "code_revision": current_code_revision(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float64",
        "torch_threads": 1,
        "deterministic_algorithms": True,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_paths
        },
        "input_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*data_paths, context_path]
        },
    }
    write_json(args.out_dir / "manifest.json", manifest)
    collocation = np.random.default_rng(settings["collocation_seed"]).uniform(
        0, 1, (settings["collocation_points"], 2)
    )
    evaluation = np.random.default_rng(settings["residual_eval_seed"]).uniform(
        0, 1, (settings["residual_eval_points"], 2)
    )
    np.savez_compressed(
        args.out_dir / "pde_points.npz", training=collocation, evaluation=evaluation
    )
    grid = np.linspace(0, 1, settings["grid_points"])
    xx, tt = np.meshgrid(grid, grid)
    weights = quadrature_weights(grid)
    records = []
    try:
        for (n, data_seed), path in zip(settings["datasets"], data_paths, strict=True):
            with np.load(path) as obs:
                x, t, y = (obs[k].copy() for k in ("x", "t", "y"))
            for training_seed in settings["training_seeds"]:
                name = f"n{n}_data{data_seed}_train{training_seed}"
                print(f"Starting {name}", flush=True)
                record = {
                    "n_obs": n,
                    "data_seed": data_seed,
                    "training_seed": training_seed,
                    "run_started_utc": datetime.now(UTC).isoformat(),
                }
                try:
                    model, trace, elapsed = train_one(
                        x,
                        t,
                        y,
                        collocation,
                        settings,
                        training_seed,
                        args.out_dir / name,
                        echo=lambda value: print(value, flush=True),
                    )
                    prediction = predict_paired(model, xx.ravel(), tt.ravel()).reshape(xx.shape)
                    data_prediction = predict_paired(model, x, t)
                    saved = torch.load(
                        args.out_dir / name / "checkpoint.pt", map_location="cpu", weights_only=True
                    )
                    with torch.random.fork_rng(devices=[]):
                        restored = BoundedHeatField(saved["settings"])
                        restored.load_state_dict(saved["model_state"])
                    np.testing.assert_allclose(
                        predict_paired(restored, x, t), data_prediction, rtol=1e-12, atol=1e-12
                    )
                    residual = (
                        heat_residual(
                            model, column(evaluation[:, 0]), column(evaluation[:, 1]), model.alpha()
                        )
                        .detach()
                        .numpy()[:, 0]
                    )
                    boundary = predict_paired(
                        model,
                        np.concatenate([np.zeros_like(grid), np.ones_like(grid)]),
                        np.tile(grid, 2),
                    )
                    alpha = float(model.alpha().detach())
                    reference = true_field(grid, grid)  # Evaluation only, after training completes.
                    np.savez_compressed(
                        args.out_dir / name / "predictions.npz",
                        x=grid,
                        t=grid,
                        u=prediction,
                        data_predictions=data_prediction,
                        independent_pde_residual=residual,
                    )
                    record.update(
                        status="completed_budget",
                        n_parameters=sum(p.numel() for p in model.parameters()),
                        alpha_hat=alpha,
                        relative_alpha_error=abs(alpha - 0.05) / 0.05,
                        field_relative_l2=relative_error(
                            prediction, reference, np.outer(weights, weights)
                        ),
                        ic_relative_l2=relative_error(prediction[0], reference[0], weights),
                        data_mse=float(np.mean((data_prediction - y) ** 2)),
                        independent_pde_mse=float(np.mean(residual**2)),
                        boundary_max_abs=float(np.max(np.abs(boundary))),
                        training_elapsed_s=elapsed,
                        final_training_trace=trace[-1],
                        at_bound=bool(
                            any(
                                np.isclose(math.log(alpha), math.log(b))
                                for b in settings["alpha_bounds"]
                            )
                        ),
                        checkpoint_reload_verified=True,
                    )
                except Exception as exc:
                    record.update(
                        status="timeout" if isinstance(exc, TimeoutError) else "failed",
                        error=repr(exc),
                    )
                    print(f"  {record['status']}: {exc}", flush=True)
                record["completed_utc"] = datetime.now(UTC).isoformat()
                records.append(record)
                write_json(args.out_dir / "records.json", records)
    except KeyboardInterrupt:
        manifest.update(status="interrupted", completed_utc=datetime.now(UTC).isoformat())
        write_json(args.out_dir / "manifest.json", manifest)
        raise
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context_rows = [
        r
        for r in context
        if [r["n_obs"], r["data_seed"]] in settings["datasets"]
        and (r["ridge"] in (0, 1e-6) or r["cv_selected"])
    ]
    write_json(args.out_dir / "classical_context.json", context_rows)
    lines = [
        "# Bounded PINN feasibility pilot",
        "",
        f"Stage: {settings['stage']}",
        "Fixed development cases; repeated initializations are not independent datasets.",
        "All recovery errors are percentages; completed_budget does not certify convergence.",
        "",
        "| N | data seed | training seed | status | alpha | alpha error "
        "| field error | IC error | training seconds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        if r["status"] == "completed_budget":
            lines.append(
                f"| {r['n_obs']} | {r['data_seed']} | {r['training_seed']} | {r['status']} "
                f"| {r['alpha_hat']:.6g} | {100 * r['relative_alpha_error']:.4g} "
                f"| {100 * r['field_relative_l2']:.4g} | {100 * r['ic_relative_l2']:.4g} "
                f"| {r['training_elapsed_s']:.2f} |"
            )
        else:
            lines.append(
                f"| {r['n_obs']} | {r['data_seed']} | {r['training_seed']} "
                f"| {r['status']} | - | - | - | - | - |"
            )
    lines += [
        "",
        "Classical context for these exact datasets is saved separately.",
        "This is an untuned feasibility pilot, not evidence of method superiority.",
    ]
    summary = "\n".join(lines) + "\n"
    (args.out_dir / "summary.md").write_text(summary, encoding="utf-8")
    failed = any(r["status"] != "completed_budget" for r in records)
    manifest.update(
        status="complete_with_failures" if failed else "complete",
        completed_utc=datetime.now(UTC).isoformat(),
        output_sha256={
            str(p.relative_to(args.out_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.out_dir.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        },
    )
    write_json(args.out_dir / "manifest.json", manifest)
    print(summary)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
