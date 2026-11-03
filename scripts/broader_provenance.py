"""Prospective source/config freeze and portable evidence checks for the new study."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

SOURCE_FILES = (
    "scripts/broader_common.py",
    "scripts/broader_analysis.py",
    "scripts/broader_provenance.py",
    "scripts/run_broader_study.py",
    "scripts/analyze_broader_study.py",
    "scripts/freeze_broader_study.py",
    "scripts/pilot_bounded_pinn.py",
    "scripts/score_sparse_diagnostic.py",
    "src/pinnaudit/spectral_inverse.py",
    "src/pinnaudit/solvers/neural.py",
    "src/pinnaudit/config.py",
    "src/pinnaudit/io.py",
    "src/pinnaudit/problems/base.py",
    "docs/BROADER_COMPARISON_PROTOCOL.md",
)


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_path(base, name):
    parsed = PureWindowsPath(name)
    if parsed.anchor or not parsed.parts or ".." in parsed.parts:
        raise ValueError(f"Expected relative evidence path: {name!r}")
    return base.joinpath(*parsed.parts)


def git(root, *arguments):
    return subprocess.run(
        ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def clean_revision(root):
    changes = git(root, "status", "--porcelain", "--untracked-files=all").decode().strip()
    if changes:
        raise RuntimeError(
            "Commit the verified code/config/protocol and smoke evidence "
            "before freezing or evaluation"
        )
    return git(root, "rev-parse", "HEAD").decode().strip()


def source_hashes(root, config_path):
    config_name = config_path.resolve().relative_to(root.resolve()).as_posix()
    names = sorted(set((*SOURCE_FILES, config_name)))
    return {name: sha256(relative_path(root, name)) for name in names}


def check_hashes(root, expected):
    for name, wanted in expected.items():
        if sha256(relative_path(root, name)) != wanted:
            raise RuntimeError(f"Frozen source/config changed: {name}")


def create_freeze(root, config_path, output):
    if output.exists():
        raise FileExistsError("Preserve previous freeze records; do not overwrite them")
    revision = clean_revision(root)
    sources = source_hashes(root, config_path)
    for name, expected in sources.items():
        committed = git(root, "show", f"HEAD:{name}")
        if hashlib.sha256(committed).hexdigest() != expected:
            raise RuntimeError(f"Working bytes differ from committed bytes: {name}")
    return {
        "schema": 1,
        "classification": "prospective_versioned_protocol_not_formal_preregistration",
        "created_utc": datetime.now(UTC).isoformat(),
        "implementation_commit": revision,
        "config_path": config_path.resolve().relative_to(root.resolve()).as_posix(),
        "config_sha256": sha256(config_path),
        "source_sha256": sources,
        "instruction": "Commit this freeze record before generating evaluation data",
    }


def evaluation_guard(root, config_path, freeze_path):
    # This runs before any evaluation observations, output directory, or model is created.
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    revision = clean_revision(root)
    name = freeze_path.resolve().relative_to(root.resolve()).as_posix()
    committed = git(root, "show", f"HEAD:{name}")
    if hashlib.sha256(committed).hexdigest() != sha256(freeze_path):
        raise RuntimeError("The freeze record must be committed unchanged")
    git(root, "merge-base", "--is-ancestor", freeze["implementation_commit"], revision)
    if freeze["config_path"] != config_path.resolve().relative_to(root.resolve()).as_posix():
        raise RuntimeError("Configuration path does not match the freeze")
    expected = source_hashes(root, config_path)
    if freeze["source_sha256"] != expected or freeze["config_sha256"] != sha256(config_path):
        raise RuntimeError("Current implementation/config/protocol does not match the freeze")
    check_hashes(root, freeze["source_sha256"])
    return {"freeze_sha256": sha256(freeze_path), "freeze_record": freeze, "run_commit": revision}


def verify_run_outputs(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] not in ("complete", "complete_with_failures"):
        raise RuntimeError("Finish the declared job schedule before the planned analysis")
    for name, expected in manifest["output_sha256"].items():
        if sha256(relative_path(folder, name)) != expected:
            raise RuntimeError(f"Output hash mismatch: {name}")
    return manifest
