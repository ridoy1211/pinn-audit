"""The held-out confirmation sweeps must stay held out and must repeat the development settings."""

from __future__ import annotations

import pytest

from pinnaudit.sweep import expand, load_sweep_config

PAIRS = [
    ("configs/sweeps/forward_heat_confirm.yaml", "configs/sweeps/forward_heat.yaml"),
    ("configs/sweeps/forward_wave_confirm.yaml", "configs/sweeps/forward_wave.yaml"),
]


@pytest.mark.parametrize(("confirm", "dev"), PAIRS)
def test_confirmation_uses_only_held_out_seeds(confirm: str, dev: str) -> None:
    runs = expand(load_sweep_config(confirm))
    assert len(runs) == 50
    assert {cfg.seed for _, cfg in runs} == set(range(5, 15))
    dev_seeds = {cfg.seed for _, cfg in expand(load_sweep_config(dev))}
    assert dev_seeds == set(range(5))
    assert not dev_seeds & {cfg.seed for _, cfg in runs}


@pytest.mark.parametrize(("confirm", "dev"), PAIRS)
def test_confirmed_variants_repeat_the_development_settings(confirm: str, dev: str) -> None:
    confirm_runs = {label: cfg for label, cfg in expand(load_sweep_config(confirm), n_seeds=1)}
    dev_runs = {label: cfg for label, cfg in expand(load_sweep_config(dev), n_seeds=1)}
    assert set(confirm_runs) <= set(dev_runs)
    for label, cfg in confirm_runs.items():
        reference = dev_runs[label]
        assert cfg.problem == reference.problem, label
        assert cfg.solver == reference.solver, label


def test_confirmation_results_do_not_mix_with_development_results() -> None:
    paths = [path for pair in PAIRS for path in pair]
    names = {load_sweep_config(path).name for path in paths}
    assert len(names) == 4
