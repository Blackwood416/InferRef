"""Read a quantized trace value without borrowing the other side's parameters."""

from __future__ import annotations

from dataclasses import replace

from inferref.quantization.reference import dequantize as dequantize_tensor
from inferref.quantization.spec import QuantizationError
from inferref.tensor import codec


def trace_tensor(package, value_id: int, *, dequantize: bool = True):
    value = package.graph.value(value_id)
    if value.capture.mode != "full" or not value.capture.payload:
        raise QuantizationError(f"value {value_id}: full payload required")
    view = codec.read(package.tensor_payload_path(value.capture.payload))
    if (view.dtype, view.shape, view.stride, view.storage_offset) != (
        value.dtype, value.shape, value.stride, value.storage_offset_elements
    ):
        raise QuantizationError(f"value {value_id}: payload metadata differs from graph")
    if value.quantization is None:
        return view
    spec = value.quantization
    parameters = {i: trace_tensor(package, i, dequantize=False) for i in spec.dependencies}
    # Validate parameters even in raw-code mode; malformed artifacts are never
    # legitimate numerical comparisons.
    decoded = dequantize_tensor(view, spec, values=parameters)
    return replace(view, dtype="float32", data=decoded) if dequantize else view
