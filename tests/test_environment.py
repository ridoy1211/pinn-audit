"""The environment record."""

from __future__ import annotations

import importlib.util
import sys

_PATH = "scripts/record_environment.py"
_SPEC = importlib.util.spec_from_file_location("record_environment", _PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
env = importlib.util.module_from_spec(_SPEC)
sys.modules["record_environment"] = env
_SPEC.loader.exec_module(env)


def test_collect_reports_the_key_facts() -> None:
    info = env.collect()
    expected = ("repository commit", "operating system", "logical CPUs", "python", "numpy", "scipy")
    for key in expected:
        assert key in info
    assert info["numpy"] != "not installed"


def test_markdown_has_a_row_per_item() -> None:
    info = {"python": "3.11.0", "numpy": "2.0.0"}
    text = env.to_markdown(info)
    assert text.startswith("# Environment")
    assert "| python | 3.11.0 |" in text
    assert "| numpy | 2.0.0 |" in text
