"""Paired observation designs for the controlled timing experiment."""

import numpy as np

from broader_common import generate_design


SCHEDULES = ("early", "mixed", "late")


def generate_timing_designs(n, seed, fold_seed_offset=30000):
    """Return three schedules sharing locations, noise, and CV folds."""
    if n not in (10, 50):
        raise ValueError("This protocol supports n=10 or n=50.")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")

    # Reuse the original, tested design generator.
    base = generate_design(n, seed, fold_seed_offset)
    q = base["t"].copy()

    # Separate random stream for assigning half the indices to early times.
    assignment_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 90210])
    )
    early_mask = np.zeros(n, dtype=bool)
    early_mask[assignment_rng.permutation(n)[: n // 2]] = True

    early_times = 0.01 + 0.09 * q
    late_times = 0.20 + 0.60 * q
    mixed_times = np.where(early_mask, early_times, late_times)

    designs = {}
    for name, times in zip(
        SCHEDULES, (early_times, mixed_times, late_times)
    ):
        design = {key: value.copy() for key, value in base.items()}
        design["t"] = times.copy()
        design["base_time_draw"] = q.copy()
        design["mixed_early_mask"] = early_mask.copy()
        designs[name] = design

    return designs


def check_generator():
    """Check timing, pairing, reproducibility, and array independence."""
    for n in (10, 50):
        designs = generate_timing_designs(n, seed=876543)
        repeated = generate_timing_designs(n, seed=876543)

        early = designs["early"]
        mixed = designs["mixed"]
        late = designs["late"]
        mask = mixed["mixed_early_mask"]

        assert int(mask.sum()) == n // 2
        assert np.all((early["t"] >= 0.01) & (early["t"] <= 0.10))
        assert np.all((late["t"] >= 0.20) & (late["t"] <= 0.80))
        np.testing.assert_array_equal(mixed["t"][mask], early["t"][mask])
        np.testing.assert_array_equal(mixed["t"][~mask], late["t"][~mask])

        for name in SCHEDULES:
            design = designs[name]
            assert design["t"].shape == (n,)
            assert np.isfinite(design["t"]).all()

            for key in design:
                np.testing.assert_array_equal(
                    design[key], repeated[name][key]
                )

            for key in (
                "x",
                "unit_noise",
                "fold_labels",
                "base_time_draw",
                "mixed_early_mask",
            ):
                np.testing.assert_array_equal(design[key], early[key])

            counts = np.bincount(design["fold_labels"], minlength=5)
            np.testing.assert_array_equal(counts, np.full(5, n // 5))

        # Changing one schedule must not accidentally change another.
        original_x = mixed["x"].copy()
        early["x"][0] = -1.0
        np.testing.assert_array_equal(mixed["x"], original_x)

        different = generate_timing_designs(n, seed=876544)
        assert not np.array_equal(
            mixed["x"], different["mixed"]["x"]
        )

        print(f"PASS: N={n}; timing, pairing, folds, reproducibility")

    print("Generator checks passed. No training was performed.")


if __name__ == "__main__":
    check_generator()