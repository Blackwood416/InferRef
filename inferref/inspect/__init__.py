"""Trace inspection and coverage analysis (SPEC §25, §34)."""

from inferref.inspect.analyze import Analysis, analyze, render_analysis
from inferref.inspect.text import render_tensor_detail, render_trace, trace_to_dict

__all__ = [
    "Analysis",
    "analyze",
    "render_analysis",
    "render_tensor_detail",
    "render_trace",
    "trace_to_dict",
]
