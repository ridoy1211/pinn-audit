"""The chronology checker: its verdict logic and its git parsing."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("check_chronology", "scripts/check_chronology.py")
assert _SPEC is not None
assert _SPEC.loader is not None
chrono = importlib.util.module_from_spec(_SPEC)
sys.modules["check_chronology"] = chrono  # dataclasses need the module to be registered
_SPEC.loader.exec_module(chrono)

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def test_commit_before_the_first_run_is_verified() -> None:
    assert chrono.verdict(T0, T0 + timedelta(minutes=1)).startswith("VERIFIED")


def test_commit_after_the_first_run_is_not_verified() -> None:
    assert chrono.verdict(T0 + timedelta(minutes=1), T0).startswith("NOT VERIFIED")


def test_missing_commit_is_not_verified() -> None:
    assert "no commit" in chrono.verdict(None, T0)


def test_missing_results_are_reported() -> None:
    assert chrono.verdict(T0, None) == "no runs found"


def test_timezone_offsets_are_compared_correctly() -> None:
    """A commit at 12:00 in UTC+6 is 06:00 UTC, so it is earlier than a 08:00 UTC run."""
    commit = datetime.fromisoformat("2026-01-01T12:00:00+06:00")
    run = datetime.fromisoformat("2026-01-01T08:00:00+00:00")
    assert chrono.verdict(commit, run).startswith("VERIFIED")


def _git(repo: Path, *args: str, when: str | None = None) -> None:
    env = {**os.environ, "GIT_COMMITTER_DATE": when} if when else dict(os.environ)
    base = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run([*base, *args], cwd=repo, check=True, capture_output=True, env=env)


def test_commit_times_finds_commits_by_exact_message(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("1", encoding="utf-8")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "Write down the plan", when="2026-01-01T10:00:00+00:00")
    (tmp_path / "a.txt").write_text("2", encoding="utf-8")
    later = "2026-01-02T10:00:00+00:00"
    _git(tmp_path, "commit", "-q", "-am", "Write down the plan and more", when=later)

    times = chrono.commit_times("Write down the plan", repo=tmp_path)
    assert times == [datetime(2026, 1, 1, 10, 0, tzinfo=UTC)]
    assert chrono.commit_times("No such message", repo=tmp_path) == []


def test_every_check_names_a_real_results_file_pattern() -> None:
    for check in chrono.CHECKS:
        assert check.results.startswith("results/")
        assert check.results.endswith(".parquet")
        assert check.values
