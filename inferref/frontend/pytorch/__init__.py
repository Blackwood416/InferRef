"""PyTorch frontend for InferRef (SPEC §7).

This is the only package that imports torch. Everything downstream of the Trace
IR is framework-independent (SPEC §60).
"""

from inferref.frontend.pytorch.hierarchical import (
    HierarchicalTraceError,
    LeafQuantization,
    compile_step_trace,
)
from inferref.frontend.pytorch.session import TraceSession, trace

__all__ = [
    "HierarchicalTraceError",
    "LeafQuantization",
    "TraceSession",
    "compile_step_trace",
    "trace",
]
