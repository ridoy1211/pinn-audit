"""Check timing datasets against an independent analytic expression."""

import numpy as np

from broader_common import make_dataset
from timing_common import SCHEDULES, generate_timing_designs


FAMILIES = {
    "A": [
        {"n": 1, "amplitude": 1.0},
        {"n": 3, "amplitude": 0.5},
    ],
    "B": [
        {"n": 1, "amplitude": 1.0},
        {"n": 3, "amplitude": 0.5},
        {"n": 7, "amplitude": 0.25},
    ],
}

ALPHA = 0.05
SIGMA = 0.01


def reference_temperature(x, t, family):
    """Explicit solution, independent of the production evaluator."""
    result = (
        np.sin(np.pi * x) * np.exp(-ALPHA * np.pi**2 * t)
        + 0.5 * np.sin(3 * np.pi * x)
        * np.exp(-ALPHA * (3 * np.pi)**2 * t)
    )
    if family == "B":
        result = result + (
            0.25 * np.sin(7 * np.pi * x)
            * np.exp(-ALPHA * (7 * np.pi)**2 * t)
        )
    return result


def main():
    checked = 0

    for n in (10, 50):
        designs = generate_timing_designs(n, seed=876543)
        reference_noise = SIGMA * designs["early"]["unit_noise"]
        reference_x = designs["early"]["x"]

        for family, modes in FAMILIES.items():
            for schedule in SCHEDULES:
                design = designs[schedule]
                obs = make_dataset(design, ALPHA, modes, SIGMA)

                expected_clean = reference_temperature(
                    obs["x"], obs["t"], family
                )

                np.testing.assert_allclose(
                    obs["clean"], expected_clean,
                    rtol=1e-13, atol=1e-14,
                )
                np.testing.assert_allclose(
                    obs["y"], expected_clean + reference_noise,
                    rtol=1e-13, atol=1e-14,
                )
                np.testing.assert_allclose(
                    obs["y"] - obs["clean"], reference_noise,
                    rtol=0, atol=1e-14,
                )
                np.testing.assert_array_equal(obs["x"], reference_x)
                np.testing.assert_array_equal(obs["t"], design["t"])
                np.testing.assert_array_equal(
                    obs["fold_labels"], design["fold_labels"]
                )

                assert obs["y"].shape == (n,)
                assert np.isfinite(obs["y"]).all()
                checked += 1

            print(f"PASS: N={n}, family={family}, all three schedules")

    print(f"All {checked} datasets passed. No training was performed.")


if __name__ == "__main__":
    main()