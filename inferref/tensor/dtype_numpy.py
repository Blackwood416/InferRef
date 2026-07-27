"""Mapping between stable InferRef dtype names and numpy dtypes.

The only awkward case is ``bfloat16``: numpy has no native bfloat16, so it is
carried as raw ``uint16`` bit patterns and decoded on demand. This is exactly
the path an engine-side reader in C++/Rust would take, so keeping it explicit
here is deliberate.
"""

from __future__ import annotations

import numpy as np

#: InferRef dtype name -> numpy dtype used to *carry* the bytes.
#: bfloat16 is carried as uint16; see :func:`bfloat16_bytes_to_float32`.
_TO_NUMPY: dict[str, np.dtype] = {
    "bool": np.dtype(np.bool_),
    "int8": np.dtype("<i1"),
    "int16": np.dtype("<i2"),
    "int32": np.dtype("<i4"),
    "int64": np.dtype("<i8"),
    "uint8": np.dtype("<u1"),
    "uint16": np.dtype("<u2"),
    "uint32": np.dtype("<u4"),
    "uint64": np.dtype("<u8"),
    "float16": np.dtype("<f2"),
    "bfloat16": np.dtype("<u2"),
    "float32": np.dtype("<f4"),
    "float64": np.dtype("<f8"),
    "complex64": np.dtype("<c8"),
    "complex128": np.dtype("<c16"),
}

#: numpy dtype name -> InferRef dtype name. bfloat16 is intentionally absent:
#: uint16 round-trips to "uint16" and callers must name bfloat16 explicitly.
_FROM_NUMPY: dict[str, str] = {
    "bool": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "complex64": "complex64",
    "complex128": "complex128",
}


def numpy_dtype_for(name: str) -> np.dtype:
    """Return the numpy dtype used to carry InferRef dtype ``name``."""
    try:
        return _TO_NUMPY[name]
    except KeyError:
        raise ValueError(f"no numpy carrier for InferRef dtype {name!r}") from None


def inferref_dtype_for_numpy(dtype: np.dtype) -> str:
    """Return the stable InferRef name for a numpy dtype."""
    name = np.dtype(dtype).name
    try:
        return _FROM_NUMPY[name]
    except KeyError:
        raise ValueError(f"no InferRef dtype name for numpy dtype {name!r}") from None


def is_carried_as_raw(name: str) -> bool:
    """Whether ``name`` is carried in a numpy dtype that is not its real type."""
    return name == "bfloat16"


def bfloat16_bytes_to_float32(raw: np.ndarray) -> np.ndarray:
    """Decode bfloat16 bit patterns (as ``uint16``) to ``float32``.

    bfloat16 is the top 16 bits of a float32, so widening is a left shift.
    """
    u16 = np.asarray(raw, dtype="<u2")
    u32 = u16.astype(np.uint32) << np.uint32(16)
    return u32.view(np.float32)


def float32_to_bfloat16_bytes(values: np.ndarray) -> np.ndarray:
    """Encode ``float32`` to bfloat16 bit patterns using round-to-nearest-even.

    Matches PyTorch's conversion so engine-side round-trips agree.
    """
    f32 = np.ascontiguousarray(values, dtype=np.float32)
    u32 = f32.view(np.uint32)
    # Round to nearest even on the truncated low 16 bits.
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    rounded = (u32.astype(np.uint64) + rounding_bias).astype(np.uint32)
    result = (rounded >> np.uint32(16)).astype(np.uint16)
    # NaN must stay NaN rather than becoming Inf after truncation.
    nan_mask = np.isnan(f32)
    if nan_mask.any():
        result = np.where(nan_mask, np.uint16(0x7FC0), result)
    return result


def to_comparable_float(data: np.ndarray, dtype: str) -> np.ndarray:
    """Widen ``data`` to a float type suitable for numerical comparison.

    Integer and boolean tensors are promoted to ``float64`` so that the same
    metric code path handles every dtype; complex is left as-is.
    """
    if dtype == "bfloat16":
        return bfloat16_bytes_to_float32(data).astype(np.float64)
    if dtype in ("complex64", "complex128"):
        return np.asarray(data)
    return np.asarray(data).astype(np.float64)
