"""Two valid contracts and one intentionally broken entry point.

This package never imports InferRef core: descriptors are plain JSON dicts, so
the fixture distribution can be installed and discovered exactly like a
third-party pack without pulling core into the test environment.
"""

from __future__ import annotations

SWIGLU = {
    "format": "inferref-contract",
    "format_version": "0.1",
    "id": "swiglu/fused/v1",
    "description": "SwiGLU fusion over the last dimension",
    "inputs": {"x": {"kind": "tensor"}, "gate": {"kind": "tensor"}},
    "outputs": {"y": {"kind": "tensor"}},
    "relations": [
        "y.shape == x.shape",
        "y.dtype == x.dtype",
        "gate.shape == x.shape",
    ],
    "features": ["multiple_outputs"],
    "effects": ["pure"],
}

GDN = {
    "format": "inferref-contract",
    "format_version": "0.1",
    "id": "gated-deltanet/step/v1",
    "description": "one Gated DeltaNet recurrence step",
    "inputs": {"state": {"kind": "tensor"}, "x": {"kind": "tensor"}},
    "outputs": {"state_out": {"kind": "tensor"}},
    "relations": ["state_out.shape == state.shape", "state_out.dtype == state.dtype"],
}


def build() -> list[dict]:
    return [SWIGLU, GDN]


def build_broken() -> None:
    raise RuntimeError("intentionally broken fixture entry point")
