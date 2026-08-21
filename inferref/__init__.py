"""InferRef — reference tracing, testcase extraction and numerical comparison.

The top-level package deliberately imports only the stdlib-based IR layer.
:func:`trace` is resolved lazily so that ``import inferref`` works in an
environment without PyTorch (Trace IR v0.1 acceptance criterion #10).
"""

from __future__ import annotations

from typing import Any

from inferref.ir.package import TracePackage
from inferref.ir.version import (
    FORMAT,
    FORMAT_VERSION,
    INFERREF_VERSION,
    TENSOR_FORMAT_VERSION,
)

__version__ = INFERREF_VERSION

__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "INFERREF_VERSION",
    "TENSOR_FORMAT_VERSION",
    "TracePackage",
    "TraceSession",
    "__version__",
    "load_trace",
    "trace",
]


def load_trace(path: Any) -> TracePackage:
    """Load a trace package from ``path``. Requires no framework."""
    return TracePackage.load(path)


def trace(*args: Any, **kwargs: Any):
    """Trace a PyTorch execution (SPEC §54).

    ::

        with inferref.trace(output="trace/", scope="model.layers.0"):
            output = model(**inputs)

    Requires the ``torch`` extra.
    """
    from inferref.frontend.pytorch.session import trace as _trace

    return _trace(*args, **kwargs)


def __getattr__(name: str) -> Any:
    # Lazy re-export so `inferref.TraceSession` works without eagerly importing torch.
    if name == "TraceSession":
        from inferref.frontend.pytorch.session import TraceSession

        return TraceSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
