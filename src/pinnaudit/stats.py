"""Small statistics helpers for comparing methods across seeds.

Everything here works on a handful of seeds, so the helpers are deliberately plain:
a percentile bootstrap for a median, a win count for paired seeds, and the exact
Wilcoxon signed-rank test. With n pairs the smallest possible two-sided exact
p-value is 2 / 2**n, which is 0.0625 for five pairs and about 0.002 for ten. That is
why the confirmation run uses ten seeds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

from pinnaudit.problems.base import Array


def bootstrap_ci(
    values: Sequence[float] | Array,
    statistic: Callable[..., Any] = np.median,
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic`` (the median by default)."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        raise ValueError("need at least two values")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    rng = np.random.default_rng(seed)
    resampled = v[rng.integers(0, v.size, size=(n_resamples, v.size))]
    stats = statistic(resampled, axis=1)
    lo, hi = np.quantile(stats, [(1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0])
    return float(lo), float(hi)


def paired_wins(better: Sequence[float] | Array, worse: Sequence[float] | Array) -> tuple[int, int]:
    """How many of the paired seeds have ``better`` strictly below ``worse``, and the total."""
    a, b = np.asarray(better, dtype=float), np.asarray(worse, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    return int(np.sum(a < b)), int(a.size)


def exact_signed_rank_p(a: Sequence[float] | Array, b: Sequence[float] | Array) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired samples (exact for small n)."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    if x.size == 0 or np.all(x == y):
        return 1.0
    return float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
