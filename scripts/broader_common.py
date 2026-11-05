"""Shared, truth-isolated helpers for broader joint heat inverse studies."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pinnaudit.spectral_inverse import SpectralFit, fit_spectral

FloatArray = NDArray[np.float64]


def write_json(path: str | Path, value: Any) -> None:
    """Atomically write strict JSON, with stable indentation and LF newlines."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_modes(modes: list[dict[str, float | int]]) -> tuple[FloatArray, FloatArray]:
    ns = np.asarray([mode["n"] for mode in modes], dtype=float)
    amplitudes = np.asarray([mode["amplitude"] for mode in modes], dtype=float)
    if not modes or not np.isfinite(ns).all() or not np.isfinite(amplitudes).all():
        raise ValueError("modes must contain finite n and amplitude values")
    if np.any(ns < 1) or np.any(ns != np.floor(ns)):
        raise ValueError("mode indices n must be positive integers")
    return ns, amplitudes


def exact_paired(
    x: FloatArray, t: FloatArray, alpha: float, modes: list[dict[str, float | int]]
) -> FloatArray:
    """Evaluate a zero-Dirichlet heat solution at paired coordinates."""
    x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    ns, amplitudes = _validated_modes(modes)
    if x.ndim != 1 or t.shape != x.shape or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("x and t must be paired vectors and alpha must be positive")
    return np.sum(
        amplitudes[None, :]
        * np.sin(np.pi * x[:, None] * ns[None, :])
        * np.exp(-alpha * t[:, None] * (np.pi * ns[None, :]) ** 2),
        axis=1,
    )


def exact_grid(
    x: FloatArray, t: FloatArray, alpha: float, modes: list[dict[str, float | int]]
) -> FloatArray:
    """Evaluate the same solution on a tensor grid, returning shape (time, x)."""
    x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    ns, amplitudes = _validated_modes(modes)
    if x.ndim != 1 or t.ndim != 1 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("x and t must be vectors and alpha must be positive")
    return np.sum(
        amplitudes[None, None, :]
        * np.sin(np.pi * x[None, :, None] * ns[None, None, :])
        * np.exp(-alpha * t[:, None, None] * (np.pi * ns[None, None, :]) ** 2),
        axis=2,
    )


def generate_design(n: int, seed: int, fold_seed_offset: int = 30000) -> dict[str, NDArray]:
    """Generate paired uniform coordinates, fixed standard-normal noise, and five folds."""
    if not isinstance(n, int) or n < 5:
        raise ValueError("n must be an integer of at least five")
    rng = np.random.default_rng(seed)
    fold_rng = np.random.default_rng(fold_seed_offset + seed)
    labels = np.tile(np.arange(5, dtype=int), int(np.ceil(n / 5)))[:n]
    fold_labels = labels[fold_rng.permutation(n)]
    return {
        "x": rng.uniform(0.0, 1.0, n),
        "t": rng.uniform(0.0, 1.0, n),
        "unit_noise": rng.standard_normal(n),
        "fold_labels": fold_labels,
    }


def make_dataset(
    design: dict[str, NDArray],
    alpha: float,
    modes: list[dict[str, float | int]],
    sigma: float,
) -> dict[str, NDArray | float]:
    """Apply a truth and noise scale to a reusable design without drawing randomness."""
    x, t, unit_noise = (np.asarray(design[key], dtype=float) for key in ("x", "t", "unit_noise"))
    if x.ndim != 1 or x.shape != t.shape or x.shape != unit_noise.shape:
        raise ValueError("design x, t, and unit_noise must be paired vectors")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    clean = exact_paired(x, t, alpha, modes)
    return {
        "x": x.copy(),
        "t": t.copy(),
        "y": clean + sigma * unit_noise,
        "clean": clean,
        "unit_noise": unit_noise.copy(),
        "fold_labels": np.asarray(design["fold_labels"]).copy(),
        "sigma": float(sigma),
    }


def _trapz_weights(grid: FloatArray) -> FloatArray:
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all():
        raise ValueError("quadrature grids must be finite vectors with at least two points")
    widths = np.diff(grid)
    if np.any(widths <= 0):
        raise ValueError("quadrature grids must be strictly increasing")
    weights = np.zeros_like(grid)
    weights[:-1] += widths / 2
    weights[1:] += widths / 2
    return weights


def _relative_l2(estimate: FloatArray, reference: FloatArray, weights: FloatArray) -> float:
    denominator = float(np.sum(weights * reference**2))
    numerator = float(np.sum(weights * (estimate - reference) ** 2))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("relative L2 is undefined for a zero or nonfinite reference norm")
    return float(np.sqrt(numerator / denominator))


def score_field(
    prediction: FloatArray,
    x: FloatArray,
    t: FloatArray,
    alpha: float,
    modes: list[dict[str, float | int]],
) -> dict[str, float]:
    """Score field and initial-condition relative L2 errors with trapezoid weights."""
    prediction = np.asarray(prediction, dtype=float)
    x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
    reference = exact_grid(x, t, alpha, modes)
    if prediction.shape != reference.shape or t[0] != 0:
        raise ValueError("prediction must have shape (len(t), len(x)) and t[0] must be zero")
    if not np.isfinite(prediction).all():
        raise ValueError("prediction must contain only finite values")
    wx, wt = _trapz_weights(x), _trapz_weights(t)
    return {
        "field_relative_l2": _relative_l2(prediction, reference, wt[:, None] * wx[None, :]),
        "ic_relative_l2": _relative_l2(prediction[0], reference[0], wx),
    }


@dataclass(frozen=True)
class CVResult:
    selected_index: int
    losses: list[float]
    fold_fits: list[list[SpectralFit]]
    fold_records: list[dict[str, float | int | bool]]


class CVFailure(RuntimeError):
    """A candidate/fold fit failed; retain all completed evidence for the caller."""

    def __init__(
        self,
        message: str,
        *,
        fold_fits: list[list[SpectralFit]],
        fold_records: list[dict[str, Any]],
    ) -> None:
        super().__init__(message)
        self.fold_fits = fold_fits
        self.fold_records = fold_records


def cv_spectral(
    x: FloatArray,
    t: FloatArray,
    y: FloatArray,
    fold_labels: NDArray,
    *,
    modes: int,
    ridges: list[float],
    alpha_bounds: tuple[float, float],
    profile_points: int,
) -> CVResult:
    """Select ridge by pooled held-out SSE, fitting each candidate only on training folds."""
    x, t, y = (np.asarray(value, dtype=float) for value in (x, t, y))
    labels = np.asarray(fold_labels)
    if x.ndim != 1 or x.shape != t.shape or x.shape != y.shape or x.shape != labels.shape:
        raise ValueError("x, t, y, and fold_labels must be paired vectors")
    if not np.isfinite(x).all() or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError("CV inputs must be finite")
    folds = np.unique(labels)
    if len(folds) < 2 or len(ridges) == 0:
        raise ValueError("at least two folds and one ridge are required")
    fold_fits: list[list[SpectralFit]] = [[] for _ in ridges]
    fold_records: list[dict[str, Any]] = []
    loss_sse = np.zeros(len(ridges), dtype=float)
    loss_n = np.zeros(len(ridges), dtype=int)
    for candidate_index, ridge in enumerate(ridges):
        for fold in folds:
            validation = labels == fold
            record: dict[str, Any] = {
                "candidate_index": candidate_index,
                "ridge": float(ridge),
                "fold": int(fold),
                "n_validation": int(validation.sum()),
            }
            started = time.perf_counter()
            try:
                fit = fit_spectral(
                    x[~validation],
                    t[~validation],
                    y[~validation],
                    modes=modes,
                    ridge=float(ridge),
                    alpha_bounds=alpha_bounds,
                    profile_points=profile_points,
                )
                residual = fit.predict(x[validation], t[validation]) - y[validation]
                validation_sse = float(np.sum(residual**2))
                if not np.isfinite(validation_sse):
                    raise ValueError("validation SSE is not finite")
                fold_fits[candidate_index].append(fit)
                loss_sse[candidate_index] += validation_sse
                loss_n[candidate_index] += int(validation.sum())
                record.update(
                    validation_sse=validation_sse,
                    alpha_hat=float(fit.alpha_hat),
                    at_bound=bool(fit.at_bound),
                    fit_wall_time_s=float(fit.wall_time_s),
                )
            except Exception as error:
                record.update(
                    failure_type=type(error).__name__,
                    failure_message=str(error),
                    fit_wall_time_s=float(time.perf_counter() - started),
                )
                fold_records.append(record)
                raise CVFailure(
                    f"spectral CV failed for candidate {candidate_index}, fold {fold}: {error}",
                    fold_fits=fold_fits,
                    fold_records=fold_records,
                ) from error
            fold_records.append(record)
    losses = (loss_sse / loss_n).tolist()
    selected = min(range(len(ridges)), key=lambda i: (losses[i], float(ridges[i])))
    return CVResult(selected, losses, fold_fits, fold_records)
