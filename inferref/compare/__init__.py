"""Numerical comparison between reference and engine outputs (SPEC §26-§31)."""

from inferref.compare.compare import (
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_MISSING,
    STATUS_PASS,
    ComparisonReport,
    TensorComparison,
    compare_tensors,
    compare_testcase,
    compare_traces,
    upstream_context,
)
from inferref.compare.layout import LayoutDiff, diff_layout
from inferref.compare.metrics import METRIC_NAMES, Metrics, compute_metrics
from inferref.compare.report import render_report, render_short
from inferref.compare.tolerance import DEFAULT_TOLERANCES, TolerancePolicy

__all__ = [
    "DEFAULT_TOLERANCES",
    "METRIC_NAMES",
    "STATUS_ERROR",
    "STATUS_FAIL",
    "STATUS_MISSING",
    "STATUS_PASS",
    "ComparisonReport",
    "LayoutDiff",
    "Metrics",
    "TensorComparison",
    "TolerancePolicy",
    "compare_tensors",
    "compare_testcase",
    "compare_traces",
    "compute_metrics",
    "diff_layout",
    "render_report",
    "render_short",
    "upstream_context",
]
