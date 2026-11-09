"""Ensure Ruff exclusions remain limited to the preserved study sources."""

import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ruff_exclusions_match_archived_sources():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    provenance = ROOT / "docs/source_provenance_addendum_v1/source_provenance.json"
    recorded = json.loads(provenance.read_bytes())
    expected = {row["source_path"]: row["lf_normalized_sha256"] for row in recorded["sources"]}

    for path in config["tool"]["ruff"]["extend-exclude"]:
        assert path in expected, f"Missing source provenance for {path}"
        content = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == expected[path], path
