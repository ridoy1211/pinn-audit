"""Error metrics. Every solver in the audit is scored with these, unchanged.

The headline number is the relative L2 error, because it is scale-free and is
the metric the PINN literature almost always reports -- using anything else
would make our numbers incomparable to published ones.
"""

from __future__ import annotations

import numpy as np

from pinnaudit.problems.base import Array


def _check(u: Array, u_ref: Array) -> tuple[Array, Array]:
    u = np.asarray(u, dtype=float)
    u_ref = np.asarray(u_ref, dtype=float)
    if u.shape != u_ref.shape:
        raise ValueError(f"shape mismatch: {u.shape} vs {u_ref.shape}")
    if not np.all(np.isfinite(u)):
        raise ValueError("solution contains non-finite values (diverged?)")
    return u, u_ref


def relative_l2(u: Array, u_ref: Array) -> float:
    """||u - u_ref||_2 / ||u_ref||_2 over all space-time samples."""
    u, u_ref = _check(u, u_ref)
    denom = float(np.linalg.norm(u_ref.ravel()))
    if denom == 0.0:
        raise ValueError("reference solution is identically zero; relative L2 undefined")
    return float(np.linalg.norm((u - u_ref).ravel()) / denom)


def max_abs_error(u: Array, u_ref: Array) -> float:
    """Worst pointwise error. Catches localised failures that L2 averages away."""
    u, u_ref = _check(u, u_ref)
    return float(np.max(np.abs(u - u_ref)))


def rmse(u: Array, u_ref: Array) -> float:
    u, u_ref = _check(u, u_ref)
    return float(np.sqrt(np.mean((u - u_ref) ** 2)))


def final_time_relative_l2(u: Array, u_ref: Array) -> float:
    """Relative L2 at the last time level only, where error has accumulated most."""
    u, u_ref = _check(u, u_ref)
    return relative_l2(u[-1], u_ref[-1])


def observed_order(errors: list[float], refinement_factor: float = 2.0) -> list[float]:
    """Empirical convergence order between successive grid refinements.

    For errors e_i on grids refined by ``refinement_factor``, the local order is
    ``log(e_i / e_{i+1}) / log(factor)``. A second-order scheme should give
    values near 2.0 in the asymptotic regime.
    """
    if len(errors) < 2:
        raise ValueError("need at least two errors to estimate an order")
    if any(e <= 0.0 for e in errors):
        raise ValueError("errors must be positive to take a log ratio")
    return [
        float(np.log(errors[i] / errors[i + 1]) / np.log(refinement_factor))
        for i in range(len(errors) - 1)
    ]
