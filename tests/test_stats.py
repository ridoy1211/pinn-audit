"""Statistics helpers: the properties the confirmation run relies on."""

from __future__ import annotations

import numpy as np
import pytest

from pinnaudit.stats import bootstrap_ci, exact_signed_rank_p, paired_wins


def test_five_pairs_can_never_reach_p_below_0_05() -> None:
    """The reason the confirmation run uses ten seeds: five pairs give at best 2/32."""
    a, b = [1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]
    assert exact_signed_rank_p(a, b) == pytest.approx(2 / 32)


def test_ten_pairs_in_one_direction_reach_p_of_about_0_002() -> None:
    a = [float(i) for i in range(1, 11)]
    b = [10.0 * x for x in a]
    assert exact_signed_rank_p(a, b) == pytest.approx(2 / 1024)


def test_identical_samples_give_p_of_one() -> None:
    assert exact_signed_rank_p([1.0, 2.0], [1.0, 2.0]) == 1.0


def test_paired_samples_must_have_equal_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        exact_signed_rank_p([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="same length"):
        paired_wins([1.0, 2.0], [1.0])


def test_paired_wins_counts_strictly_better_seeds() -> None:
    assert paired_wins([1.0, 2.0, 3.0, 9.0], [2.0, 3.0, 4.0, 5.0]) == (3, 4)
    assert paired_wins([1.0, 2.0], [1.0, 2.0]) == (0, 2)


def test_bootstrap_interval_brackets_the_median() -> None:
    values = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 0.99])
    low, high = bootstrap_ci(values)
    assert low <= np.median(values) <= high
    assert low >= 0.9 and high <= 1.1


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    values = np.linspace(0.0, 1.0, 12)
    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)


def test_outliers_widen_the_interval() -> None:
    tight = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 0.99])
    loose = np.append(tight, [5.0, 0.1])
    narrow, wide = bootstrap_ci(tight), bootstrap_ci(loose)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_bootstrap_works_for_other_statistics() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    low, high = bootstrap_ci(values, statistic=np.mean)
    assert low <= np.mean(values) <= high


def test_bootstrap_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least two"):
        bootstrap_ci([1.0])
    with pytest.raises(ValueError, match="level"):
        bootstrap_ci([1.0, 2.0], level=1.5)
