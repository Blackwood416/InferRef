"""Numerical comparison metrics (SPEC §26, §27).

All metrics operate on the tensors' **logical values in canonical contiguous
order**, which is what an ``.irtensor`` payload holds (IR §20). Layout
differences are reported separately by :mod:`inferref.compare.layout` so that
"same numbers, different stride" never masquerades as a value mismatch
(SPEC §29).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Metric names produced by :func:`compute_metrics` (SPEC §27).
METRIC_NAMES = (
    "exact_match",
    "max_abs_error",
    "max_rel_error",
    "mean_abs_error",
    "rmse",
    "cosine_similarity",
    "nan_count",
    "inf_count",
    "mismatch_count",
)


def _finite_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


@dataclass
class Metrics:
    """Numerical comparison result for one tensor pair (SPEC §27)."""

    exact_match: bool = False
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    mean_abs_error: float = 0.0
    rmse: float = 0.0
    cosine_similarity: float = 1.0

    #: NaN/Inf counts, reported per side so an engine producing NaN is obvious.
    nan_count: int = 0
    inf_count: int = 0
    reference_nan_count: int = 0
    reference_inf_count: int = 0

    #: Elements outside tolerance.
    mismatch_count: int = 0
    element_count: int = 0

    #: Flat index and unravelled index of the first element outside tolerance.
    first_mismatch_flat_index: int | None = None
    first_mismatch_index: tuple[int, ...] | None = None
    first_mismatch_reference: float | None = None
    first_mismatch_actual: float | None = None

    #: Set when the two sides disagree on NaN/Inf placement.
    nan_mismatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "exact_match": self.exact_match,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "mean_abs_error": self.mean_abs_error,
            "rmse": self.rmse,
            "cosine_similarity": self.cosine_similarity,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "reference_nan_count": self.reference_nan_count,
            "reference_inf_count": self.reference_inf_count,
            "mismatch_count": self.mismatch_count,
            "element_count": self.element_count,
            "nan_mismatch": self.nan_mismatch,
        }
        if self.first_mismatch_index is not None:
            out["first_mismatch"] = {
                "flat_index": self.first_mismatch_flat_index,
                "index": list(self.first_mismatch_index),
                "reference": self.first_mismatch_reference,
                "actual": self.first_mismatch_actual,
            }
        return out


def compute_metrics(
    reference: np.ndarray,
    actual: np.ndarray,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
    equal_nan: bool = True,
) -> Metrics:
    """Compare two same-shaped arrays.

    ``atol``/``rtol`` follow the ``numpy.isclose`` convention:
    ``|a - r| <= atol + rtol * |r|``.
    """
    ref = np.asarray(reference)
    act = np.asarray(actual)
    if ref.shape != act.shape:
        raise ValueError(f"shape mismatch: {ref.shape} vs {act.shape}")

    metrics = Metrics(element_count=int(ref.size))
    if ref.size == 0:
        metrics.exact_match = True
        return metrics

    is_complex = np.iscomplexobj(ref) or np.iscomplexobj(act)
    ref_f = ref.astype(np.complex128) if is_complex else ref.astype(np.float64)
    act_f = act.astype(np.complex128) if is_complex else act.astype(np.float64)

    metrics.nan_count = int(np.isnan(act_f).sum())
    metrics.inf_count = int(np.isinf(act_f).sum())
    metrics.reference_nan_count = int(np.isnan(ref_f).sum())
    metrics.reference_inf_count = int(np.isinf(ref_f).sum())

    metrics.exact_match = bool(np.array_equal(ref_f, act_f, equal_nan=equal_nan))

    # Non-finite entries would poison every aggregate, so they are scored
    # separately: matching NaN/Inf placement is fine, differing placement is a
    # mismatch in its own right. inf - inf legitimately produces nan here, and
    # those elements are excluded from the aggregates below.
    finite = _finite_pairs(ref_f, act_f)
    nonfinite_agree = _nonfinite_agreement(ref_f, act_f, equal_nan=equal_nan)
    metrics.nan_mismatch = bool((~finite & ~nonfinite_agree).any())

    with np.errstate(invalid="ignore"):
        diff = np.abs(act_f - ref_f)
    if finite.any():
        finite_diff = diff[finite]
        metrics.max_abs_error = float(finite_diff.max())
        metrics.mean_abs_error = float(finite_diff.mean())
        metrics.rmse = float(np.sqrt(np.mean(np.square(finite_diff))))
        denom = np.abs(ref_f[finite])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(denom > 0, finite_diff / denom, 0.0)
        metrics.max_rel_error = float(np.nanmax(rel)) if rel.size else 0.0
        metrics.cosine_similarity = _cosine(ref_f[finite], act_f[finite])
    else:
        metrics.cosine_similarity = 1.0 if metrics.exact_match else 0.0

    # Tolerance test, treating disagreeing non-finite entries as mismatches.
    with np.errstate(invalid="ignore"):
        within = np.zeros(ref_f.shape, dtype=bool)
        within[finite] = diff[finite] <= (atol + rtol * np.abs(ref_f[finite]))
        within[~finite] = nonfinite_agree[~finite]

    mismatches = ~within
    metrics.mismatch_count = int(mismatches.sum())
    if metrics.mismatch_count:
        flat = int(np.argmax(mismatches.reshape(-1)))
        metrics.first_mismatch_flat_index = flat
        metrics.first_mismatch_index = tuple(
            int(i) for i in np.unravel_index(flat, ref_f.shape)
        )
        metrics.first_mismatch_reference = _scalar(ref_f.reshape(-1)[flat])
        metrics.first_mismatch_actual = _scalar(act_f.reshape(-1)[flat])
    return metrics


def _nonfinite_agreement(
    ref: np.ndarray, act: np.ndarray, *, equal_nan: bool
) -> np.ndarray:
    """Elementwise: do the two sides agree about NaN/+Inf/-Inf?"""
    both_nan = np.isnan(ref) & np.isnan(act) if equal_nan else np.zeros(ref.shape, bool)
    if np.iscomplexobj(ref) or np.iscomplexobj(act):
        same_inf = np.isinf(ref) & np.isinf(act) & (ref == act)
    else:
        same_inf = np.isinf(ref) & np.isinf(act) & (np.sign(ref) == np.sign(act))
    return both_nan | same_inf


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    value = float(np.real(np.vdot(a, b)) / (na * nb))
    # Guard against tiny floating-point excursions outside [-1, 1].
    return max(-1.0, min(1.0, value))


def _scalar(value: Any) -> float:
    if isinstance(value, complex) or np.iscomplexobj(value):
        return complex(value).real
    result = float(value)
    return result if math.isfinite(result) else result
