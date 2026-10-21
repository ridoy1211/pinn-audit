"""The `pinnaudit modes` command, exercised with the classical solver so no PyTorch is needed."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pinnaudit.cli import app
from pinnaudit.sweep import SweepConfig, Variant, run_sweep


def test_modes_command_prints_a_row_per_variant(tmp_path: Path) -> None:
    sweep = SweepConfig(
        name="m",
        seeds=[0, 1],
        out_dir=tmp_path,
        variants=[Variant(label="fd_41", config=Path("configs/wave_fd.yaml"))],
    )
    run_sweep(sweep, echo=lambda _: None)

    args = ["modes", "--name", "m", "--results", str(tmp_path)]
    args += ["--variant", "fd_41", "--modes", "4"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[0].startswith("variant")
    assert "mode4" in lines[0]
    assert lines[1].startswith("fd_41")
    assert " 2 " in lines[1]  # two runs averaged


def test_modes_command_rejects_a_missing_sweep(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["modes", "--name", "nope", "--results", str(tmp_path)])
    assert result.exit_code != 0
