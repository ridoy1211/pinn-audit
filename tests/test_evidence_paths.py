"""Cross-platform reading of the original Windows-recorded evidence paths."""

import importlib
from pathlib import Path, PurePosixPath

import pytest

pytest.importorskip("torch")


@pytest.fixture
def review(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("review_pinn_budget_check")


@pytest.mark.parametrize(
    "name", [r"scripts\pilot_bounded_pinn.py", "scripts/pilot_bounded_pinn.py"]
)
def test_windows_archive_key_resolves_to_posix_components(review, name):
    assert review.evidence_path(PurePosixPath("/repo"), name) == PurePosixPath(
        "/repo/scripts/pilot_bounded_pinn.py"
    )


def test_same_file_hash_for_both_manifest_separator_styles(review, tmp_path):
    folder = tmp_path / "results" / "case"
    folder.mkdir(parents=True)
    target = folder / "records.json"
    target.write_bytes(b'{"preserved":true}\r\n')
    expected = review.digest(target)
    for name in (r"results\case\records.json", "results/case/records.json"):
        assert review.digest(review.evidence_path(tmp_path, name)) == expected


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "../outside",
        r"a\..\outside",
        "/outside",
        r"C:\outside",
        r"C:outside",
        r"\\server\share\outside",
    ],
)
def test_absolute_and_parent_traversal_names_are_rejected(review, tmp_path, name):
    with pytest.raises(ValueError, match="relative evidence path"):
        review.evidence_path(tmp_path, name)
