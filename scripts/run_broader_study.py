#!/usr/bin/env python3
"""Run the fixed joint-heat comparison, gated by a committed source/config freeze."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from broader_common import (
    CVFailure,
    cv_spectral,
    generate_design,
    make_dataset,
    score_field,
    write_json,
)
from broader_provenance import (
    canonical_hash,
    check_hashes,
    evaluation_guard,
    relative_path,
    sha256,
    source_hashes,
)
from pinnaudit.spectral_inverse import fit_spectral

CLASSICAL = ("unreg6", "fixed6", "cv6", "cv12")


def load_config(path):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["length"] != 1 or cfg["final_time"] != 1 or cfg["folds"] != 5:
        raise ValueError("The trainer supports the declared unit-square, five-fold protocol")
    if set(cfg["families"]) != {"A", "B"} or set(cfg["designs"]) != {"10", "50"}:
        raise ValueError("Expected both families and both observation-count strata")
    if cfg["classical_modes"] != [6, 12] or len(cfg["training_seeds"]) != 3:
        raise ValueError("Expected 6/12 classical modes and three neural initializations")
    if len(set(cfg["training_seeds"])) != 3 or any(s < 0 for s in cfg["training_seeds"]):
        raise ValueError("Training seeds must be distinct and nonnegative")
    if any(len(seeds) != 10 or len(set(seeds)) != 10 for seeds in cfg["designs"].values()):
        raise ValueError("The full study requires ten distinct designs per observation count")
    if not all(np.isfinite(r) and r > 0 for r in cfg["ridges"]):
        raise ValueError("The richer-basis CV requires strictly positive ridge candidates")
    if cfg["fixed_ridge"] not in cfg["ridges"] or cfg["noise_sigma_absolute"] <= 0:
        raise ValueError("Invalid fixed ridge or absolute noise standard deviation")
    lo, hi = cfg["alpha_bounds"]
    if not 0 < lo < cfg["alpha_true_generation_and_scoring_only"] < hi:
        raise ValueError("Simulation coefficient must be inside the declared positive bounds")
    if cfg["neural"]["alpha_bounds"] != cfg["alpha_bounds"]:
        raise ValueError("Neural and classical bounds must match")
    for modes in cfg["families"].values():
        ns = [m["n"] for m in modes]
        if not modes or len(set(ns)) != len(ns) or any(not isinstance(n, int) or n < 1 for n in ns):
            raise ValueError("IC modes must have unique positive integer indices")
        if not all(np.isfinite(m["amplitude"]) for m in modes) or not any(
            m["amplitude"] for m in modes
        ):
            raise ValueError("IC amplitudes must be finite and define a nonzero reference")
    return cfg


def smoke_config(original):
    cfg = copy.deepcopy(original)
    cfg.update(
        study_id="joint_heat_software_smoke",
        designs={"10": [987100], "50": [987101]},
        training_seeds=[901, 902, 903],
        profile_points=21,
        grid_points=21,
        refinement_grid_points=41,
        collocation_points=16,
        residual_eval_points=64,
    )
    cfg["neural"].update(width=8, depth=2, adam_steps=3, trace_every=1, max_train_seconds=60)
    cfg["neural_budgets"] = {
        "pinn_200": {"lbfgs_iters": 2, "lbfgs_max_eval": 5},
        "pinn_1000": {"lbfgs_iters": 4, "lbfgs_max_eval": 7},
    }
    return cfg


def cases_and_schedule(cfg):
    cases, schedule = [], []
    for n_string, seeds in cfg["designs"].items():
        n = int(n_string)
        for seed in seeds:
            for family in cfg["families"]:
                case = {
                    "case_id": f"n{n}_seed{seed}_{family}",
                    "n_obs": n,
                    "data_seed": seed,
                    "family": family,
                }
                cases.append(case)
                for method in CLASSICAL:
                    schedule.append({**case, "method": method, "training_seed": None})
                for method in cfg["neural_budgets"]:
                    for training_seed in cfg["training_seeds"]:
                        schedule.append({**case, "method": method, "training_seed": training_seed})
    return cases, schedule


def job_name(job):
    return job["method"] + (
        f"_train{job['training_seed']}" if job["training_seed"] is not None else ""
    )


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def pack_fit(arrays, prefix, fit):
    for key, value in {
        "alpha": fit.alpha_hat,
        "amplitudes": fit.amplitudes,
        "objective": fit.objective,
        "profile_alpha": fit.profile_alpha,
        "profile_objective": fit.profile_objective,
        "at_bound": fit.at_bound,
        "fit_time_s": fit.wall_time_s,
        "ridge": fit.ridge,
        "modes": fit.modes,
        "length": fit.length,
    }.items():
        arrays[f"{prefix}_{key}"] = value


def save_cv(path, fits, full_fits=()):
    arrays = {}
    for candidate, group in enumerate(fits):
        for fold, fit in enumerate(group):
            pack_fit(arrays, f"cv{candidate}_fold{fold}", fit)
    for candidate, fit in enumerate(full_fits):
        pack_fit(arrays, f"full{candidate}", fit)
    save_npz(path, **arrays)


def predict_grid(predict, size):
    grid = np.linspace(0, 1, size)
    xx, tt = np.meshgrid(grid, grid)
    x, t = xx.ravel(), tt.ravel()
    pieces = [predict(x[i : i + 4096], t[i : i + 4096]) for i in range(0, len(x), 4096)]
    return grid, np.concatenate(pieces).reshape(xx.shape)


def evaluate(predict, alpha_hat, obs, cfg, job, job_dir, is_first):
    start = time.perf_counter()
    grid, field = predict_grid(predict, cfg["grid_points"])
    alpha_true = cfg["alpha_true_generation_and_scoring_only"]
    scores = score_field(field, grid, grid, alpha_true, cfg["families"][job["family"]])
    scores["relative_alpha_error"] = abs(alpha_hat - alpha_true) / alpha_true
    data_predictions = predict(obs["x"], obs["t"])
    scores["data_mse"] = float(np.mean((data_predictions - obs["y"]) ** 2))
    payload = {"data_predictions": data_predictions}
    if is_first:
        fine_grid, fine = predict_grid(predict, cfg["refinement_grid_points"])
        scores["refinement_scores"] = score_field(
            fine, fine_grid, fine_grid, alpha_true, cfg["families"][job["family"]]
        )
        if cfg["save_dense_fields_for_first_design_only"]:
            payload.update(x=grid, t=grid, u=field)
    save_npz(job_dir / "predictions.npz", **payload)
    scores["evaluation_time_s"] = time.perf_counter() - start
    return scores


def classical_job(obs, cfg, job, folder):
    folder.mkdir(parents=True, exist_ok=False)
    method = job["method"]
    options = {"alpha_bounds": tuple(cfg["alpha_bounds"]), "profile_points": cfg["profile_points"]}
    selection_time = 0.0
    ridge = 0.0 if method == "unreg6" else cfg["fixed_ridge"]
    modes = 12 if method == "cv12" else 6
    diagnostics_time = 0.0
    if method in ("cv6", "cv12"):
        start = time.perf_counter()
        try:
            cv = cv_spectral(
                obs["x"],
                obs["t"],
                obs["y"],
                obs["fold_labels"],
                modes=modes,
                ridges=cfg["ridges"],
                **options,
            )
        except CVFailure as exc:
            save_cv(folder / "candidate_fits.npz", exc.fold_fits)
            write_json(folder / "fold_scores.json", exc.fold_records)
            raise
        selection_time = time.perf_counter() - start
        ridge = cfg["ridges"][cv.selected_index]
        save_cv(folder / "candidate_fits.npz", cv.fold_fits)
        write_json(folder / "fold_scores.json", cv.fold_records)
        write_json(
            folder / "selection.json",
            {
                "selected_index": cv.selected_index,
                "selected_ridge": ridge,
                "validation_mse": cv.losses,
            },
        )
        # Selection is complete before any full-data fit or truth-based scoring.
        full_fits = []
        for candidate_ridge in cfg["ridges"]:
            full_fits.append(
                fit_spectral(
                    obs["x"], obs["t"], obs["y"], modes=modes, ridge=candidate_ridge, **options
                )
            )
            save_cv(folder / "candidate_fits.npz", cv.fold_fits, full_fits)
        fit = full_fits[cv.selected_index]
        diagnostics_time = sum(
            f.wall_time_s for j, f in enumerate(full_fits) if j != cv.selected_index
        )
        save_cv(folder / "candidate_fits.npz", cv.fold_fits, full_fits)
        write_json(folder / "fold_scores.json", cv.fold_records)
        write_json(
            folder / "selection.json",
            {
                "selected_index": cv.selected_index,
                "selected_ridge": ridge,
                "validation_mse": cv.losses,
            },
        )
    else:
        fit = fit_spectral(obs["x"], obs["t"], obs["y"], modes=modes, ridge=ridge, **options)
    arrays = {}
    pack_fit(arrays, "selected", fit)
    save_npz(folder / "model.npz", **arrays)
    return (
        fit.predict,
        fit.alpha_hat,
        {
            "selected_ridge": ridge,
            "basis_modes": modes,
            "at_bound": fit.at_bound,
            "fit_time_s": fit.wall_time_s,
            "selection_time_s": selection_time,
            "nonselected_full_fit_diagnostics_time_s": diagnostics_time,
        },
    )


def neural_job(obs, cfg, job, folder, collocation, evaluation, pilot):
    import torch

    settings = copy.deepcopy(cfg["neural"])
    settings.update(cfg["neural_budgets"][job["method"]])
    model, trace, elapsed = pilot.train_one(
        obs["x"],
        obs["t"],
        obs["y"],
        collocation,
        settings,
        job["training_seed"],
        folder,
        echo=lambda value: print(value, flush=True),
    )
    saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        restored = pilot.BoundedHeatField(saved["settings"])
        restored.load_state_dict(saved["model_state"])
    np.testing.assert_allclose(
        pilot.predict_paired(model, obs["x"], obs["t"]),
        pilot.predict_paired(restored, obs["x"], obs["t"]),
        rtol=1e-12,
        atol=1e-12,
    )
    residual = pilot.heat_residual(
        model, pilot.column(evaluation[:, 0]), pilot.column(evaluation[:, 1]), model.alpha()
    )
    residual = residual.detach().numpy()[:, 0]
    save_npz(folder / "independent_residual.npz", residual=residual)
    alpha = float(model.alpha().detach())
    boundary = pilot.predict_paired(
        model, np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])
    )
    return (
        lambda x, t: pilot.predict_paired(model, x, t),
        alpha,
        {
            "fit_time_s": elapsed,
            "selection_time_s": 0.0,
            "checkpoint_reload_verified": True,
            "independent_pde_mse": float(np.mean(residual**2)),
            "boundary_max_abs": float(np.max(np.abs(boundary))),
            "final_training_trace": trace[-1],
            "lbfgs_iterations": saved["lbfgs_iterations"],
            "at_bound": bool(
                any(np.isclose(math.log(alpha), math.log(b)) for b in cfg["alpha_bounds"])
            ),
        },
    )


def prepare_data(out, cfg, manifest):
    index = manifest.setdefault("data_sha256", {})
    for n_string, seeds in cfg["designs"].items():
        n = int(n_string)
        for seed in seeds:
            design = generate_design(n, seed, cfg["fold_seed_offset"])
            for family, modes in cfg["families"].items():
                name = f"data/n{n}_seed{seed}_{family}.npz"
                path = out / name
                if path.exists():
                    if index.get(name) != sha256(path):
                        raise RuntimeError(f"Existing input is unindexed or changed: {name}")
                    continue
                if name in index:
                    raise RuntimeError(f"Previously indexed input disappeared: {name}")
                dataset = make_dataset(
                    design,
                    cfg["alpha_true_generation_and_scoring_only"],
                    modes,
                    cfg["noise_sigma_absolute"],
                )
                save_npz(path, **dataset)
                index[name] = sha256(path)
                write_json(out / "manifest.json", manifest)


def run_jobs(root, out, cfg, cases, schedule, manifest, pilot):
    points_path = out / "pde_points.npz"
    if points_path.exists():
        if manifest.get("pde_points_sha256") != sha256(points_path):
            raise RuntimeError("PDE point file changed or is unindexed")
        with np.load(points_path) as saved:
            collocation, evaluation = saved["training"].copy(), saved["evaluation"].copy()
    else:
        collocation = np.random.default_rng(cfg["collocation_seed"]).uniform(
            size=(cfg["collocation_points"], 2)
        )
        evaluation = np.random.default_rng(cfg["residual_eval_seed"]).uniform(
            size=(cfg["residual_eval_points"], 2)
        )
        save_npz(points_path, training=collocation, evaluation=evaluation)
        manifest["pde_points_sha256"] = sha256(points_path)
        write_json(out / "manifest.json", manifest)
    records = []
    for number, job in enumerate(schedule):
        check_hashes(root, manifest["source_sha256"])
        folder = out / "jobs" / job["case_id"] / job_name(job)
        status_path = folder / "job.json"
        if status_path.exists():
            record = json.loads(status_path.read_text(encoding="utf-8"))
            if any(record[k] != job[k] for k in job):
                raise RuntimeError("Recorded job identity does not match the frozen schedule")
            actual_names = {
                p.relative_to(folder).as_posix()
                for p in folder.rglob("*")
                if p.is_file() and p.name != "job.json"
            }
            if actual_names != set(record["artifact_sha256"]):
                raise RuntimeError("The sealed job artifact set changed; resume refused")
            for name, expected in record["artifact_sha256"].items():
                if sha256(relative_path(folder, name)) != expected:
                    raise RuntimeError("A sealed job artifact changed; resume refused")
            records.append(record)
            continue
        start_path = out / "progress" / job["case_id"] / (job_name(job) + ".json")
        if start_path.exists():
            record = json.loads(start_path.read_text(encoding="utf-8"))
            if any(record[k] != job[k] for k in job):
                raise RuntimeError("Interrupted attempt identity changed")
            record.update(
                status="interrupted", error="Unsealed previous attempt retained; not rerun"
            )
        else:
            if folder.exists():
                raise RuntimeError("Unindexed job directory; preserve it for investigation")
            record = {
                **job,
                "run_started_utc": datetime.now(UTC).isoformat(),
                "artifact_path": folder.relative_to(out).as_posix(),
            }
            write_json(start_path, record)
            name = f"data/{job['case_id']}.npz"
            if manifest["data_sha256"][name] != sha256(out / name):
                raise RuntimeError("Observation file changed before fitting")
            with np.load(out / name) as saved:
                # Truth arrays remain in the evidence file and never enter either trainer.
                obs = {k: saved[k].copy() for k in ("x", "t", "y", "fold_labels")}
            print(
                f"Job {number + 1}/{len(schedule)}: {job['case_id']} / {job_name(job)}", flush=True
            )
            started = time.perf_counter()
            try:
                if job["training_seed"] is None:
                    predict, alpha, details = classical_job(obs, cfg, job, folder)
                else:
                    predict, alpha, details = neural_job(
                        obs, cfg, job, folder, collocation, evaluation, pilot
                    )
                first = job["data_seed"] == cfg["designs"][str(job["n_obs"])][0]
                scores = evaluate(predict, alpha, obs, cfg, job, folder, first)
                record.update(status="completed", alpha_hat=alpha, **details, **scores)
            except KeyboardInterrupt:
                folder.mkdir(parents=True, exist_ok=True)
                record.update(
                    status="interrupted",
                    error="User interruption preserved",
                    completed_utc=datetime.now(UTC).isoformat(),
                )
                record["artifact_sha256"] = {
                    p.relative_to(folder).as_posix(): sha256(p)
                    for p in folder.rglob("*")
                    if p.is_file() and p.name != "job.json"
                }
                write_json(status_path, record)
                records.append(record)
                write_json(out / "records.json", records)
                raise
            except Exception as exc:
                folder.mkdir(parents=True, exist_ok=True)
                record.update(
                    status="timeout" if isinstance(exc, TimeoutError) else "failed", error=repr(exc)
                )
                print(f"  {record['status']}: {exc}", flush=True)
            record["job_elapsed_s"] = time.perf_counter() - started
        record["completed_utc"] = datetime.now(UTC).isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        record["artifact_sha256"] = {
            p.relative_to(folder).as_posix(): sha256(p)
            for p in folder.rglob("*")
            if p.is_file() and p.name != "job.json"
        }
        write_json(status_path, record)
        records.append(record)
        write_json(out / "records.json", records)
        manifest["terminal_jobs"] = len(records)
        write_json(out / "manifest.json", manifest)
        check_hashes(root, manifest["source_sha256"])
    write_json(out / "records.json", records)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/inverse/broader_comparison.json")
    )
    parser.add_argument("--freeze", type=Path, default=Path("docs/BROADER_COMPARISON_FREEZE.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/broader_study_v1"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    # Evaluation cannot reach data generation unless the exact committed freeze validates.
    freeze = None if args.smoke else evaluation_guard(root, args.config, args.freeze)
    if args.smoke:
        cfg = smoke_config(cfg)
    sources = source_hashes(root, args.config)
    mode = "software_smoke_only" if args.smoke else "prospective_fixed_pipeline_evaluation"
    identity = canonical_hash(
        {
            "mode": mode,
            "config": cfg,
            "sources": sources,
            "freeze_sha256": None if freeze is None else freeze["freeze_sha256"],
        }
    )
    if args.out_dir.exists() and not args.resume:
        raise FileExistsError(
            "Output exists. Preserve it; --resume handles only unfinished scheduled jobs"
        )
    if args.resume and not args.out_dir.is_dir():
        raise FileNotFoundError("No existing run to resume")
    cases, schedule = cases_and_schedule(cfg)
    if args.resume:
        manifest = json.loads((args.out_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["run_identity"] != identity:
            raise RuntimeError("Resume refused: source, config, mode or freeze identity changed")
        if manifest["status"] in ("complete", "complete_with_failures"):
            raise RuntimeError("The scheduled run is already complete; analyze its saved evidence")
        if json.loads((args.out_dir / "config.json").read_text(encoding="utf-8")) != cfg:
            raise RuntimeError("Saved configuration changed")
        manifest.setdefault("resumed_utc", []).append(datetime.now(UTC).isoformat())
    else:
        args.out_dir.mkdir(parents=True)
        manifest = {
            "started_utc": datetime.now(UTC).isoformat(),
            "status": "prepared",
            "mode": mode,
            "run_identity": identity,
            "source_sha256": sources,
            "freeze": freeze,
            "expected_jobs": len(schedule),
            "expected_cases": len(cases),
            "independent_designs": len({(c["n_obs"], c["data_seed"]) for c in cases}),
        }
        write_json(args.out_dir / "config.json", cfg)
        write_json(args.out_dir / "schedule.json", schedule)
        write_json(args.out_dir / "cases.json", cases)
        write_json(args.out_dir / "manifest.json", manifest)
    lock = args.out_dir / ".run.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        import torch
        from threadpoolctl import threadpool_info, threadpool_limits

        import pilot_bounded_pinn as pilot

        if (
            Path(inspect.getsourcefile(fit_spectral)).resolve()
            != root / "src/pinnaudit/spectral_inverse.py"
        ):
            raise RuntimeError("Wrong imported checkout; reinstall the current editable package")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):
            manifest.update(
                status="running",
                python=platform.python_version(),
                numpy=np.__version__,
                torch=str(torch.__version__),
                platform=platform.platform(),
                threadpools=threadpool_info(),
                torch_threads=1,
                dtype="float64",
                device="cpu",
            )
            write_json(args.out_dir / "manifest.json", manifest)
            check_hashes(root, sources)
            manifest.setdefault("first_data_generated_utc", datetime.now(UTC).isoformat())
            write_json(args.out_dir / "manifest.json", manifest)
            prepare_data(args.out_dir, cfg, manifest)
            records = run_jobs(root, args.out_dir, cfg, cases, schedule, manifest, pilot)
        failed = any(r["status"] != "completed" for r in records)
        manifest.update(
            status="complete_with_failures" if failed else "complete",
            completed_utc=datetime.now(UTC).isoformat(),
            terminal_jobs=len(records),
            output_sha256={
                p.relative_to(args.out_dir).as_posix(): sha256(p)
                for p in sorted(args.out_dir.rglob("*"))
                if p.is_file() and p.name not in ("manifest.json", ".run.lock")
            },
        )
        write_json(args.out_dir / "manifest.json", manifest)
        print(f"Finished {len(records)} jobs. Mode: {mode}. Status: {manifest['status']}.")
        print("Run analyze_broader_study.py on this directory for the declared analysis.")
    except BaseException as exc:
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "aborted",
            stopped_utc=datetime.now(UTC).isoformat(),
            error=repr(exc),
        )
        write_json(args.out_dir / "manifest.json", manifest)
        raise
    finally:
        lock.unlink()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
