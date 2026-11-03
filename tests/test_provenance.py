"""Regression checks that result rows retain a true pre-computation time."""

from datetime import UTC, datetime

from pinnaudit.config import RunConfig
from pinnaudit.experiment import run_once
from pinnaudit.io import build_record, execution_fingerprint


def test_record_start_survives_the_solve_and_precedes_completion(heat_run: RunConfig) -> None:
    before = datetime.now(UTC)
    result = run_once(heat_run)
    record = build_record(heat_run, result.solution, result.metrics)
    after = datetime.now(UTC)
    started = datetime.fromisoformat(record["run_started_utc"])
    completed = datetime.fromisoformat(record["completed_utc"])
    assert before <= started <= completed <= after
    assert record["timestamp_utc"] == record["completed_utc"]
    assert "+src-" in record["code_revision"]


def test_source_revision_changes_execution_identity_for_an_unchanged_config() -> None:
    assert execution_fingerprint("same_config", "source_a") != execution_fingerprint(
        "same_config", "source_b"
    )
