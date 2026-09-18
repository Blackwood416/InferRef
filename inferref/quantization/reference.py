"""Numpy-only affine/NF4 reference arithmetic and scheme dispatch.

No framework quantization state is inferred from byte carriers. Callers must
provide an explicit descriptor and immutable scale/zero-point tensor values.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np

from inferref.quantization.spec import QuantizationError, QuantizationSpec
from inferref.tensor.codec import TensorView, encode
from inferref.tensor.dtype_numpy import numpy_dtype_for


def _tensor(value: TensorView, dtype: str, shape: tuple[int, ...], label: str) -> np.ndarray:
    if not isinstance(value, TensorView):
        raise QuantizationError(f"{label} must be an unquantized TensorView")
    if value.dtype != dtype or value.shape != shape:
        raise QuantizationError(f"{label} must have dtype {dtype} and shape {shape}")
    data = value.data
    if not isinstance(data, np.ndarray) or data.dtype != numpy_dtype_for(dtype) or data.shape != shape:
        raise QuantizationError(f"{label} carrier dtype/shape does not match its metadata")
    return data


def _parameter(
    values: Mapping[int, TensorView], value_id: int, dtype: str,
    shape: tuple[int, ...], label: str,
) -> np.ndarray:
    try:
        value = values[value_id]
    except KeyError:
        raise QuantizationError(f"missing {label} value ID {value_id}") from None
    return _tensor(value, dtype, shape, label)


def _expand(data: np.ndarray, spec: QuantizationSpec) -> np.ndarray:
    if spec.granularity == "per_axis":
        assert spec.axis is not None
        shape = [1] * len(spec.logical_shape)
        shape[spec.axis] = spec.logical_shape[spec.axis]
        data = data.reshape(shape)
    elif spec.granularity == "per_group":
        assert spec.axis is not None and spec.group_size is not None
        # Index rather than repeat: a huge group size must not cause a huge
        # temporary allocation for a tensor whose last group is very short.
        indices = np.arange(spec.logical_shape[spec.axis], dtype=np.int64) // spec.group_size
        data = np.take(data, indices, axis=spec.axis)
    return np.broadcast_to(data, spec.logical_shape)


def _parameters(
    spec: QuantizationSpec, values: Mapping[int, TensorView],
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(spec, QuantizationSpec):
        raise QuantizationError("spec must be a validated QuantizationSpec")
    if spec.scheme == "nf4/v1":
        raise QuantizationError("NF4 is a codebook scheme, not affine quantization")
    scale = _parameter(values, spec.scale_value_id, "float32", spec.parameter_shape, "scale")
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise QuantizationError("scale values must be finite and strictly positive")
    if spec.zero_point_value_id is None:
        zero = np.zeros(spec.parameter_shape, dtype=np.float32)
    else:
        zero = _parameter(values, spec.zero_point_value_id, spec.zero_point_dtype, spec.parameter_shape,
                          "zero point").astype(np.float32)
        low, high = spec.code_range
        if np.any(zero < low) or np.any(zero > high):
            raise QuantizationError("zero point outside storage code range")
    return _expand(scale, spec), _expand(zero, spec)


def _float32(values: np.ndarray, shape: tuple[int, ...] | None, label: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32"):
        raise QuantizationError(f"{label} must be a float32 numpy array (no implicit cast)")
    if shape is not None and values.shape != shape:
        raise QuantizationError(f"{label} shape must be {shape}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise QuantizationError(f"{label} must contain only finite values")
    return values


def dequantize_affine(
    codes: TensorView, spec: QuantizationSpec, *, values: Mapping[int, TensorView],
) -> np.ndarray:
    """Compute ``(int8_codes - zero_point) * scale`` in float32.

    Both intermediate and output precision are float32. Unrepresentable finite
    results are rejected, not quietly returned as infinities.
    """
    scale, zero = _parameters(spec, values)
    raw = _tensor(codes, spec.storage_dtype, spec.logical_shape, "codes")
    low, high = spec.code_range
    if np.any(raw < low) or np.any(raw > high):
        raise QuantizationError("codes outside storage range")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            return np.asarray((raw.astype(np.float32) - zero) * scale, dtype=np.float32)
    except FloatingPointError:
        raise QuantizationError("dequantization overflow in float32") from None


def validate_parameters(spec, values):
    if spec.scheme == "nf4/v1":
        from inferref.quantization.nf4 import nf4_parameters

        return nf4_parameters(spec, values)
    return _parameters(spec, values)


def dequantize(codes, spec, *, values):
    """Dispatch only explicit, validated quantization schemes."""
    if spec.scheme == "nf4/v1":
        from inferref.quantization.nf4 import dequantize_nf4

        return dequantize_nf4(codes, spec, values=values)
    return dequantize_affine(codes, spec, values=values)


def quantize_affine(
    source: np.ndarray, spec: QuantizationSpec, *, values: Mapping[int, TensorView],
) -> np.ndarray:
    """Return INT8 codes: clamp(round_to_even(source / scale) + zero_point).

    Source and division precision are float32. Overflow of finite division is
    deliberately saturated to the INT8 endpoint; NaN/Inf source is rejected.
    """
    scale, zero = _parameters(spec, values)
    source = _float32(source, spec.logical_shape, "source")
    with np.errstate(over="ignore", invalid="raise", divide="raise", under="ignore"):
        rounded = np.rint(source / scale)
        return np.asarray(np.clip(rounded + zero, *spec.code_range), dtype=numpy_dtype_for(spec.storage_dtype))


def linear_int8_reference(
    inputs: np.ndarray, weights: TensorView, spec: QuantizationSpec, *,
    values: Mapping[int, TensorView], bias: np.ndarray | None = None,
) -> np.ndarray:
    """Float32 ``inputs @ dequantize(weights).T + bias`` for INT8 weights.

    Weight layout is [out_features, in_features]. Inputs may have leading batch
    dimensions. This is a float32 reference, not an INT8-accumulating W8A8
    kernel; BLAS reduction order need not be bit-identical across platforms.
    Compare engines using the same codes/parameters to separate implementation
    error from the loss introduced when the original weights were quantized.
    """
    if spec.storage_dtype != "int8":
        raise QuantizationError("linear_int8_reference requires affine INT8 weights")
    return _linear_float32(inputs, weights, spec, values=values, bias=bias)


def _linear_float32(inputs, weights, spec, *, values, bias=None):
    decoded = dequantize(weights, spec, values=values)
    if decoded.ndim != 2:
        raise QuantizationError("Linear weights must have shape [out_features, in_features]")
    inputs = _float32(inputs, None, "inputs")
    if inputs.ndim < 1 or inputs.shape[-1] != decoded.shape[1]:
        raise QuantizationError("Linear input last dimension must equal in_features")
    if bias is not None:
        bias = _float32(bias, (decoded.shape[0],), "bias")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = inputs @ decoded.T
            if bias is not None:
                result = result + bias
    except FloatingPointError:
        raise QuantizationError("Linear overflow in float32") from None
    if not np.all(np.isfinite(result)):
        raise QuantizationError("Linear produced nonfinite float32 output")
    return result


def linear_int4_reference(
    inputs: np.ndarray, weights: TensorView, spec: QuantizationSpec, *,
    values: Mapping[int, TensorView],
) -> np.ndarray:
    """W4A16: float16 inputs, float32 dequant/matmul, final float16 rounding."""
    if spec.storage_dtype not in {"int4", "uint4"} or spec.scheme == "nf4/v1":
        raise QuantizationError("W4A16 requires explicit affine int4 or uint4 weights")
    return _linear_16(inputs, weights, spec, values=values)


def linear_nf4_reference(inputs, weights, spec, *, values):
    """NF4 W4A16 with float32 dequantization/accumulation and float16 output."""
    if spec.scheme != "nf4/v1":
        raise QuantizationError("NF4 Linear requires nf4/v1")
    return _linear_16(inputs, weights, spec, values=values)


def _linear_16(inputs, weights, spec, *, values):
    if not isinstance(inputs, np.ndarray) or inputs.dtype != np.dtype("float16"):
        raise QuantizationError("W4A16 inputs must be float16")
    result = _linear_float32(inputs.astype(np.float32), weights, spec, values=values)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            return result.astype(np.float16)
        except FloatingPointError:
            raise QuantizationError("W4A16 output overflow in float16") from None


def quantization_fingerprint(
    codes: TensorView, spec: QuantizationSpec, *, values: Mapping[int, TensorView],
) -> str:
    """Content digest of the descriptor, codes, and every parameter dependency.

    This weight-only digest is not a replacement for the full Alignment trace
    or testcase identity. Includes layout metadata conservatively, not paths.
    """
    validate_parameters(spec, values)
    _tensor(codes, spec.storage_dtype, spec.logical_shape, "codes")
    digest = hashlib.sha256(b"inferref-quantization-fingerprint/v1\0")
    descriptor = json.dumps(spec.to_dict(), sort_keys=True, separators=(",", ":"),
                            allow_nan=False).encode("utf-8")
    digest.update(len(descriptor).to_bytes(8, "little"))
    digest.update(descriptor)
    for tensor in (codes, *(values[value_id] for value_id in spec.dependencies)):
        blob = encode(dtype=tensor.dtype, shape=tensor.shape, stride=tensor.stride,
                      storage_offset=tensor.storage_offset, payload=tensor.payload_bytes())
        digest.update(len(blob).to_bytes(8, "little"))
        digest.update(blob)
    return digest.hexdigest()
