"""Canonical low-nibble-first INT4/UINT4 packing, independent of quantization."""

from __future__ import annotations

import numpy as np

from inferref.ir.dtypes import PACKED_DTYPES, dtype_payload_nbytes


def pack_nibbles(values: np.ndarray, dtype: str) -> bytes:
    if dtype not in PACKED_DTYPES:
        raise ValueError(f"not a packed dtype: {dtype}")
    carrier = np.dtype("int8" if dtype == "int4" else "uint8")
    if not isinstance(values, np.ndarray) or values.dtype != carrier:
        raise ValueError(f"{dtype} packing requires logical {carrier.name} codes (no implicit cast)")
    low, high = (-8, 7) if dtype == "int4" else (0, 15)
    if np.any(values < low) or np.any(values > high):
        raise ValueError(f"{dtype} codes must be in [{low}, {high}]")
    nibbles = values.reshape(-1).astype(np.uint8) & 15
    packed = np.zeros((values.size + 1) // 2, dtype=np.uint8)
    packed[:] = nibbles[::2]
    packed[:values.size // 2] |= nibbles[1::2] << 4
    return packed.tobytes()


def unpack_nibbles(payload: bytes, dtype: str, numel: int) -> np.ndarray:
    if dtype not in PACKED_DTYPES:
        raise ValueError(f"not a packed dtype: {dtype}")
    if len(payload) != dtype_payload_nbytes(dtype, numel):
        raise ValueError("packed payload length does not match logical numel")
    if numel % 2 and payload[-1] & 0xf0:
        raise ValueError("unused high nibble must be zero")
    packed = np.frombuffer(payload, dtype=np.uint8)
    values = np.empty(numel, dtype=np.uint8)
    values[::2] = packed & 15
    values[1::2] = packed[:numel // 2] >> 4
    if dtype == "int4":
        signed = values.astype(np.int8)
        signed[signed >= 8] -= 16
        return signed
    return values
