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
from inferref.tensor.dtype_numpy import (
    bfloat16_bytes_to_float32,
    float32_to_bfloat16_bytes,
    numpy_dtype_for,
)


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


_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


def _validate_w8a8_descriptor(spec, label: str) -> None:
    """Require the affine INT8 descriptor and the W8A8 row/reduction axes.

    ``QuantizationSpec`` stores non-negative axes only, so the frozen ``-2``
    activation axis already arrives normalized to the row axis ``0``.
    """
    if not isinstance(spec, QuantizationSpec):
        raise QuantizationError(f"{label} requires an explicit QuantizationSpec")
    if spec.scheme != "affine-int8/v1" or spec.storage_dtype != "int8":
        raise QuantizationError(f"{label} must be affine-int8/v1 with int8 storage")
    if len(spec.logical_shape) != 2:
        raise QuantizationError(f"{label} must be rank 2 [rows, K]")
    if spec.granularity == "per_axis" and spec.axis != 0:
        raise QuantizationError(f"{label} per_axis requires the row axis 0")
    if spec.granularity == "per_group" and spec.axis != 1:
        raise QuantizationError(f"{label} per_group requires the reduction axis 1")


def _w8a8_group_size(input_spec, weight_spec, k_size: int) -> int:
    """Reduction group size, ``K`` (one group) when neither side is grouped."""
    x_group = input_spec.group_size if input_spec.granularity == "per_group" else None
    w_group = weight_spec.group_size if weight_spec.granularity == "per_group" else None
    if x_group is not None and w_group is not None and x_group != w_group:
        raise QuantizationError(
            f"W8A8 per-group requires equal group sizes, got {x_group} and {w_group}"
        )
    for declared in (x_group, w_group):
        if declared is not None:
            return declared
    return max(k_size, 1)


def _w8a8_parameters(
    spec: QuantizationSpec, values: Mapping[int, TensorView], label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Unquantized float32 scale and int64 zero point in the declared shape."""
    scale = _parameter(values, spec.scale_value_id, "float32", spec.parameter_shape,
                       f"{label} scale")
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise QuantizationError(f"{label} scale must be finite and strictly positive")
    if spec.zero_point_value_id is None:
        return scale, np.zeros(spec.parameter_shape, dtype=np.int64)
    zero = _parameter(values, spec.zero_point_value_id, spec.zero_point_dtype,
                      spec.parameter_shape, f"{label} zero point")
    low, high = spec.code_range
    if np.any(zero < low) or np.any(zero > high):
        raise QuantizationError(f"{label} zero point outside int8 code range")
    return scale, zero.astype(np.int64)


def _w8a8_group_parameters(spec, scale, zero, rows: int, group: int):
    """Per-row scale and zero point that apply to one reduction group."""
    if spec.granularity == "per_group":
        return scale[:, group], zero[:, group]
    if spec.granularity == "per_axis":
        return scale, zero
    return (np.broadcast_to(scale.reshape(()), (rows,)),
            np.broadcast_to(zero.reshape(()), (rows,)))


def linear_w8a8_int32_reference(
    inputs: TensorView, input_spec: QuantizationSpec,
    weights: TensorView, weight_spec: QuantizationSpec, *,
    values: Mapping[int, TensorView], bias: np.ndarray | None = None,
) -> np.ndarray:
    """int8 x int8 -> int32 group accumulation -> float32 scaled output.

    ``inputs`` is ``[M, K]`` and ``weights`` is ``[N, K]``; the result is
    ``[M, N]`` float32. Each reduction group is summed exactly in int64 and must
    fit int32. The epilogue applies ``float32(scale_x * scale_w)`` one group at
    a time in ascending group order. There is no bias in v1.
    """
    if bias is not None:
        raise QuantizationError("linear_w8a8_int32_reference has no bias in v1")
    _validate_w8a8_descriptor(input_spec, "W8A8 input")
    _validate_w8a8_descriptor(weight_spec, "W8A8 weight")
    x_codes = _tensor(inputs, "int8", input_spec.logical_shape, "W8A8 input codes")
    w_codes = _tensor(weights, "int8", weight_spec.logical_shape, "W8A8 weight codes")
    m_size, k_size = input_spec.logical_shape
    n_size, w_k_size = weight_spec.logical_shape
    if k_size != w_k_size:
        raise QuantizationError(
            f"W8A8 reduction size mismatch: input K={k_size}, weight K={w_k_size}"
        )
    group_size = _w8a8_group_size(input_spec, weight_spec, k_size)
    x_scale, x_zero = _w8a8_parameters(input_spec, values, "W8A8 input")
    w_scale, w_zero = _w8a8_parameters(weight_spec, values, "W8A8 weight")
    y = np.zeros((m_size, n_size), dtype=np.float32)
    for group in range(-(-k_size // group_size)):
        start = group * group_size
        stop = min(k_size, start + group_size)
        xs, xz = _w8a8_group_parameters(input_spec, x_scale, x_zero, m_size, group)
        ws, wz = _w8a8_group_parameters(weight_spec, w_scale, w_zero, n_size, group)
        acc = ((x_codes[:, start:stop].astype(np.int64) - xz[:, None])
               @ (w_codes[:, start:stop].astype(np.int64) - wz[:, None]).T)
        if np.any(acc < _INT32_MIN) or np.any(acc > _INT32_MAX):
            raise QuantizationError(f"int32 accumulation overflow in group {group}")
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                scale = xs[:, None] * ws[None, :]
                y += acc.astype(np.float32) * scale
        except FloatingPointError:
            raise QuantizationError("non-finite W8A8 output") from None
    if not np.all(np.isfinite(y)):
        raise QuantizationError("non-finite W8A8 output")
    return y


def linear_int4_reference(
    inputs: np.ndarray, weights: TensorView, spec: QuantizationSpec, *,
    values: Mapping[int, TensorView],
) -> np.ndarray:
    """W4A16: float16 inputs, float32 dequant/matmul, final float16 rounding."""
    if spec.storage_dtype not in {"int4", "uint4"} or spec.scheme == "nf4/v1":
        raise QuantizationError("W4A16 requires explicit affine int4 or uint4 weights")
    return _linear_reduced_precision(inputs, weights, spec, values=values, dtype="float16")


def linear_nf4_reference(inputs, weights, spec, *, values):
    """NF4 W4A16 with float32 dequantization/accumulation and float16 output."""
    if spec.scheme != "nf4/v1":
        raise QuantizationError("NF4 Linear requires nf4/v1")
    return _linear_reduced_precision(inputs, weights, spec, values=values, dtype="float16")


def linear_int4_bf16_reference(
    inputs: np.ndarray, weights: TensorView, spec: QuantizationSpec, *,
    values: Mapping[int, TensorView],
) -> np.ndarray:
    """W4A16 bf16 I/O: float32 dequant/matmul, one nearest-even bf16 conversion."""
    if spec.storage_dtype not in {"int4", "uint4"} or spec.scheme == "nf4/v1":
        raise QuantizationError("W4A16 bf16 requires explicit affine int4 or uint4 weights")
    return _linear_reduced_precision(inputs, weights, spec, values=values, dtype="bfloat16")


def linear_nf4_bf16_reference(inputs, weights, spec, *, values):
    """NF4 W4A16 bf16 I/O with float32 dequant/accum and one nearest-even bf16 output."""
    if spec.scheme != "nf4/v1":
        raise QuantizationError("NF4 Linear requires nf4/v1")
    return _linear_reduced_precision(inputs, weights, spec, values=values, dtype="bfloat16")


def _linear_reduced_precision(inputs, weights, spec, *, values, dtype):
    """Accumulate in float32 and round to ``dtype`` exactly once.

    bfloat16 values are carried as uint16 bit patterns (see
    ``inferref.tensor.dtype_numpy``). They share float32's exponent range but
    not its finite range: a float32 value above the largest finite bfloat16
    rounds to bfloat16 infinity, so the encoded result is decoded and checked.
    """
    if not isinstance(inputs, np.ndarray) or inputs.dtype != numpy_dtype_for(dtype):
        raise QuantizationError(f"W4A16 inputs must be {dtype}")
    if dtype == "bfloat16":
        source = bfloat16_bytes_to_float32(inputs)
    else:
        source = inputs.astype(np.float32)
    result = _linear_float32(source, weights, spec, values=values)
    if dtype == "bfloat16":
        encoded = float32_to_bfloat16_bytes(result)
        if np.any(np.isinf(bfloat16_bytes_to_float32(encoded))):
            raise QuantizationError("W4A16 output overflow in bfloat16")
        return encoded
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
