"""Gate untouched data and retain interrupted attempts without replacement fits."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_broader_study")


def software_config(runner):
    return runner.smoke_config(runner.load_config(Path("configs/inverse/broader_comparison.json")))


def test_smoke_matrix_has_four_cases_and_24_neural_fits_without_reserved_seeds(runner):
    cfg = software_config(runner)
    cases, schedule = runner.cases_and_schedule(cfg)
    assert len(cases) == 4
    assert len(schedule) == 40
    assert sum(job["training_seed"] is not None for job in schedule) == 24
    assert {c["data_seed"] for c in cases} == {987100, 987101}
    assert "alpha_true_generation_and_scoring_only" not in cfg["neural"]
    assert "families" not in cfg["neural"]


def test_missing_freeze_blocks_main_before_data_or_output_creation(runner, tmp_path, monkeypatch):
    output = tmp_path / "uncreated"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_broader_study.py",
            "--freeze",
            str(tmp_path / "missing.json"),
            "--out-dir",
            str(output),
        ],
    )

    def forbidden(*args):
        raise AssertionError("Data generation must not run")

    monkeypatch.setattr(runner, "prepare_data", forbidden)
    with pytest.raises(FileNotFoundError):
        runner.main()
    assert not output.exists()


def fixture_run(runner, tmp_path):
    cfg = software_config(runner)
    cases, schedule = runner.cases_and_schedule(cfg)
    job = schedule[0]
    out = tmp_path / "run"
    out.mkdir()
    runner.save_npz(
        out / "pde_points.npz", training=np.array([[0.1, 0.2]]), evaluation=np.array([[0.3, 0.4]])
    )
    manifest = {"source_sha256": {}, "pde_points_sha256": runner.sha256(out / "pde_points.npz")}
    folder = out / "jobs" / job["case_id"] / runner.job_name(job)
    folder.mkdir(parents=True)
    return cfg, cases, job, out, manifest, folder


def test_resume_rejects_altered_artifact_without_retraining(runner, tmp_path):
    cfg, cases, job, out, manifest, folder = fixture_run(runner, tmp_path)
    artifact = folder / "model.npz"
    artifact.write_bytes(b"original artifact")
    runner.write_json(
        folder / "job.json",
        {**job, "status": "completed", "artifact_sha256": {"model.npz": runner.sha256(artifact)}},
    )
    artifact.write_bytes(b"altered artifact")
    with pytest.raises(RuntimeError, match="artifact changed"):
        runner.run_jobs(tmp_path, out, cfg, cases, [job], manifest, None)


def test_resume_preserves_original_start_of_unsealed_attempt(runner, tmp_path):
    cfg, cases, job, out, manifest, folder = fixture_run(runner, tmp_path)
    start = "2020-01-01T00:00:00+00:00"  # Synthetic software-test metadata only.
    runner.write_json(
        out / "progress" / job["case_id"] / (runner.job_name(job) + ".json"),
        {**job, "run_started_utc": start},
    )
    records = runner.run_jobs(tmp_path, out, cfg, cases, [job], manifest, None)
    assert records[0]["status"] == "interrupted"
    assert records[0]["run_started_utc"] == start
    assert not (folder / "model.npz").exists()
