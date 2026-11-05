"""A frozen configuration must be committed and immutable before evaluation."""

import json
import subprocess

import pytest
from scripts import broader_provenance as provenance


def command(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


@pytest.fixture
def frozen_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    command(root, "init")
    command(root, "config", "user.name", "Software verification")
    command(root, "config", "user.email", "smoke@example.invalid")
    (root / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    for name in provenance.SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"toy software-check content\n")
    config = root / "configs/inverse/broader_comparison.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_bytes(b'{"software_test":true}\n')
    command(root, "add", ".")
    command(root, "commit", "-m", "Software-only fixture")
    return root, config, root / "docs/freeze.json"


def test_uncommitted_freeze_is_rejected_then_committed_freeze_validates(frozen_repo):
    root, config, path = frozen_repo
    freeze = provenance.create_freeze(root, config, path)
    path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Commit"):
        provenance.evaluation_guard(root, config, path)
    command(root, "add", "docs/freeze.json")
    command(root, "commit", "-m", "Commit software-only freeze")
    result = provenance.evaluation_guard(root, config, path)
    assert result["freeze_sha256"] == provenance.sha256(path)


def test_source_modification_invalidates_the_frozen_identity(frozen_repo):
    root, config, path = frozen_repo
    path.write_text(json.dumps(provenance.create_freeze(root, config, path)), encoding="utf-8")
    command(root, "add", ".")
    command(root, "commit", "-m", "Freeze fixture")
    (root / provenance.SOURCE_FILES[0]).write_bytes(b"changed after freeze\n")
    command(root, "add", ".")
    command(root, "commit", "-m", "Changed software fixture")
    with pytest.raises(RuntimeError, match="does not match"):
        provenance.evaluation_guard(root, config, path)


def test_missing_freeze_cannot_create_any_evaluation_output(frozen_repo):
    root, config, path = frozen_repo
    with pytest.raises(FileNotFoundError):
        provenance.evaluation_guard(root, config, path)
    assert not (root / "results").exists()
