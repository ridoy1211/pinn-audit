"""Which Fourier modes carry a solver's error?

Both test problems have zero Dirichlet boundaries, so every time slice can be
written as a sine series. Projecting a solver's output onto the first few modes
shows *where* its error lives: a PINN that learns the smooth mode and misses
the fast one fails in a very different way from one that is merely noisy.

On a uniform grid with zero boundary values the discrete sine transform is
exactly orthogonal, so the per-mode errors recombine into the ordinary relative
L2 error (see ``tests/test_modes.py``). That identity is what makes the
decomposition a faithful breakdown of the headline metric and not a separate
number.
"""

from __future__ import annotations

import numpy as np

from pinnaudit.problems.base import Array


def mode_amplitudes(u: Array, x: Array, n_modes: int) -> Array:
    """Sine-series coefficients a_n(t) of every time slice.

    Returns an array of shape ``(nt, n_modes)`` where column ``n - 1`` holds the
    coefficient of ``sin(n * pi * (x - x0) / L)``.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    if n_modes > x.size - 2:
        raise ValueError(f"{n_modes} modes cannot be resolved on {x.size} grid points")

    length = x[-1] - x[0]
    n = np.arange(1, n_modes + 1)
    basis = np.sin(np.pi * np.outer(n, x - x[0]) / length)  # (n_modes, nx)

    dx = np.diff(x)
    weights = np.zeros_like(x)  # trapezoid rule
    weights[:-1] += 0.5 * dx
    weights[1:] += 0.5 * dx
    return (2.0 / length) * (u * weights) @ basis.T


def mode_error_contributions(u: Array, u_ref: Array, x: Array, n_modes: int) -> Array:
    """Each mode's share of the relative L2 error, shape ``(n_modes,)``.

    Entry ``n - 1`` is ``||a_n(u) - a_n(u_ref)|| / ||a(u_ref)||``, with norms taken
    over time and the denominator over all modes. With ``n_modes = nx - 2`` and
    zero boundary values, ``sqrt(sum(contributions**2))`` equals the relative L2
    error exactly. A solver whose boundary values are not exactly zero (a PINN)
    will come out slightly lower, because boundary error lives outside the series.
    """
    a = mode_amplitudes(u, x, n_modes)
    a_ref = mode_amplitudes(u_ref, x, n_modes)
    scale = float(np.sqrt(np.sum(a_ref**2)))
    if scale == 0.0:
        raise ValueError("reference has no energy in the requested modes")
    return np.sqrt(np.sum((a - a_ref) ** 2, axis=0)) / scale
