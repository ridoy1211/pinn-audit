#!/usr/bin/env python3
"""Write docs/ENVIRONMENT.md: the machine and library versions behind the results.

Run it on the machine that produced the results, after the runs. Timings depend on
the hardware, so a reader needs this to judge them.

Usage:  python scripts/record_environment.py
"""

from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

PACKAGES = ["numpy", "scipy", "polars", "pydantic", "typer", "torch", "matplotlib", "pyyaml"]


def collect() -> dict[str, str]:
    """Machine and software facts, in the order they are written."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    info = {
        "recorded (UTC)": datetime.now(UTC).strftime("%Y-%m-%d %H:%M"),
        "repository commit": commit,
        "operating system": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical CPUs": str(os.cpu_count()),
        "python": platform.python_version(),
    }
    for name in PACKAGES:
        try:
            info[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            info[name] = "not installed"
    return info


def to_markdown(info: dict[str, str]) -> str:
    lines = [
        "# Environment",
        "",
        "The machine and software behind the results in this repository. All experiments run",
        "on the CPU in float64 with one torch thread, so wall-clock times depend mostly on",
        "the single-core speed of this machine.",
        "",
        "| item | value |",
        "|---|---|",
    ]
    lines += [f"| {key} | {value} |" for key, value in info.items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    path = Path("docs/ENVIRONMENT.md")
    path.write_text(to_markdown(collect()), encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
