"""``.irtensor`` v0.1 binary codec (IR §20, §21).

Header layout — all little-endian, exactly the field order of IR §21::

    offset  size    field
    0       4       magic "IRTN"
    4       2       tensor_format_version (u16)
    6       2       header_size           (u16)
    8       2       dtype                 (u16)
    10      4       flags                 (u32)
    14      2       rank                  (u16)
    16      2       reserved              (u16)
    18      8       logical_numel         (u64)
    26      8       payload_nbytes        (u64)
    34      8*rank  shape[]               (i64)
    34+8r   8*rank  stride[]              (i64)
    34+16r  8       storage_offset        (i64)
            pad     -> header_size = 48 + 16*rank
    header_size     payload

The fixed part ends at byte ``42 + 16*rank``; we pad up to a multiple of 8 so
that ``payload`` always starts 8-byte aligned, giving ``header_size = 48 + 16*rank``.

Note that ``flags`` (offset 10) and ``logical_numel`` (offset 18) are *not*
naturally aligned. This is intentional — it preserves the field order the spec
declares — and is why the C++ reader parses field-by-field with ``memcpy``
rather than casting a packed struct.

Per IR §20 the payload holds the tensor's **logical value in canonical
contiguous order**. The original ``stride`` and ``storage_offset`` are retained
as metadata for layout debugging (SPEC §29) but do not describe the payload.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import numpy as np

from inferref.ir.dtypes import dtype_code, dtype_itemsize, dtype_name_from_code
from inferref.ir.version import TENSOR_FORMAT_VERSION
from inferref.tensor.dtype_numpy import (
    bfloat16_bytes_to_float32,
    numpy_dtype_for,
    to_comparable_float,
)

MAGIC = b"IRTN"

#: Fixed portion of the header, before shape/stride/storage_offset.
_FIXED = struct.Struct("<4sHHHIHHQQ")
assert _FIXED.size == 34, _FIXED.size

#: ``flags`` bit 0 — payload is canonical logical contiguous order (IR §20).
FLAG_CANONICAL_CONTIGUOUS = 1 << 0


def header_size_for_rank(rank: int) -> int:
    """Total header bytes for a tensor of ``rank`` dims (8-byte aligned payload)."""
    fixed = _FIXED.size + 16 * rank + 8  # + shape[] + stride[] + storage_offset
    return (fixed + 7) // 8 * 8


class IRTensorError(ValueError):
    """Raised when an ``.irtensor`` file is malformed or unsupported."""


@dataclass(frozen=True)
class TensorHeader:
    """Validated ``.irtensor`` metadata without loading its payload."""

    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    flags: int
    tensor_format_version: int
    header_size: int
    logical_numel: int
    payload_nbytes: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "numel": self.logical_numel,
            "nbytes": self.payload_nbytes,
        }


@dataclass
class TensorView:
    """A decoded ``.irtensor``.

    ``data`` is the canonical contiguous payload as a numpy array. For
    ``bfloat16`` — which numpy has no native dtype for — ``data`` is ``uint16``
    holding the raw bit pattern; use :meth:`as_float32` to decode it.
    """

    dtype: str
    shape: tuple[int, ...]
    #: Original reference stride; describes the *source* tensor, not ``data``.
    stride: tuple[int, ...]
    storage_offset: int
    data: np.ndarray
    flags: int = FLAG_CANONICAL_CONTIGUOUS
    tensor_format_version: int = TENSOR_FORMAT_VERSION
    #: Set when the file was loaded from disk.
    path: Path | None = field(default=None, compare=False)

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    @property
    def nbytes(self) -> int:
        return self.numel * dtype_itemsize(self.dtype)

    @property
    def is_canonical_contiguous(self) -> bool:
        return bool(self.flags & FLAG_CANONICAL_CONTIGUOUS)

    def as_float32(self) -> np.ndarray:
        """Return the payload as ``float32``, decoding ``bfloat16`` if needed."""
        if self.dtype == "bfloat16":
            return bfloat16_bytes_to_float32(self.data).reshape(self.shape)
        return self.data.astype(np.float32)

    def as_comparable(self) -> np.ndarray:
        """Return an array suitable for numerical comparison (SPEC §26)."""
        return to_comparable_float(self.data, self.dtype).reshape(self.shape)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "numel": self.numel,
            "nbytes": self.nbytes,
        }


def encode(
    *,
    dtype: str,
    shape: Sequence[int],
    stride: Sequence[int],
    storage_offset: int = 0,
    payload: bytes,
    flags: int = FLAG_CANONICAL_CONTIGUOUS,
) -> bytes:
    """Encode one ``.irtensor`` file.

    ``payload`` must already be the canonical logical contiguous bytes of the
    tensor in little-endian order.
    """
    shape = tuple(int(d) for d in shape)
    stride = tuple(int(s) for s in stride)
    if len(shape) != len(stride):
        raise IRTensorError(
            f"rank mismatch: shape has {len(shape)} dims, stride has {len(stride)}"
        )
    if any(dim < 0 for dim in shape):
        raise IRTensorError(f"negative dimension in shape {shape}")
    rank = len(shape)
    numel = 1
    for dim in shape:
        numel *= dim
    expected = numel * dtype_itemsize(dtype)
    if len(payload) != expected:
        raise IRTensorError(
            f"payload is {len(payload)} bytes, expected {expected} "
            f"({numel} elements x {dtype})"
        )

    hsize = header_size_for_rank(rank)
    out = bytearray(hsize)
    _FIXED.pack_into(
        out, 0, MAGIC, TENSOR_FORMAT_VERSION, hsize, dtype_code(dtype), flags, rank, 0, numel,
        len(payload),
    )
    off = _FIXED.size
    struct.pack_into(f"<{rank}q", out, off, *shape)
    off += 8 * rank
    struct.pack_into(f"<{rank}q", out, off, *stride)
    off += 8 * rank
    struct.pack_into("<q", out, off, int(storage_offset))
    return bytes(out) + payload


def decode(blob: bytes, *, path: Path | None = None) -> TensorView:
    """Decode an in-memory ``.irtensor``."""
    if len(blob) < _FIXED.size:
        raise IRTensorError(f"file too short to be an .irtensor ({len(blob)} bytes)")

    magic, version, hsize, dcode, flags, rank, _reserved, numel, payload_nbytes = (
        _FIXED.unpack_from(blob, 0)
    )
    if magic != MAGIC:
        raise IRTensorError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if version != TENSOR_FORMAT_VERSION:
        raise IRTensorError(
            f"unsupported .irtensor version {version}; this build reads "
            f"version {TENSOR_FORMAT_VERSION}"
        )
    expected_hsize = header_size_for_rank(rank)
    if hsize != expected_hsize:
        raise IRTensorError(
            f"header_size {hsize} does not match rank {rank} (expected {expected_hsize})"
        )
    if _reserved != 0:
        raise IRTensorError(f"reserved header field must be zero, got {_reserved}")
    if not flags & FLAG_CANONICAL_CONTIGUOUS:
        raise IRTensorError("payload is not marked canonical contiguous")
    if len(blob) < hsize + payload_nbytes:
        raise IRTensorError(
            f"truncated: need {hsize + payload_nbytes} bytes, have {len(blob)}"
        )

    dtype = dtype_name_from_code(dcode)
    off = _FIXED.size
    shape = struct.unpack_from(f"<{rank}q", blob, off)
    off += 8 * rank
    stride = struct.unpack_from(f"<{rank}q", blob, off)
    off += 8 * rank
    (storage_offset,) = struct.unpack_from("<q", blob, off)

    if any(dim < 0 for dim in shape):
        raise IRTensorError(f"negative dimension in shape {shape}")
    shape_numel = 1
    for dim in shape:
        shape_numel *= dim
    if shape_numel != numel:
        raise IRTensorError(
            f"logical_numel {numel} inconsistent with shape product {shape_numel}"
        )

    if numel * dtype_itemsize(dtype) != payload_nbytes:
        raise IRTensorError(
            f"payload_nbytes {payload_nbytes} inconsistent with {numel} x {dtype}"
        )

    raw = blob[hsize : hsize + payload_nbytes]
    array = np.frombuffer(raw, dtype=numpy_dtype_for(dtype))
    if numel:
        array = array.reshape(shape)
    else:
        array = array.reshape(shape) if shape else array

    return TensorView(
        dtype=dtype,
        shape=tuple(shape),
        stride=tuple(stride),
        storage_offset=int(storage_offset),
        data=array,
        flags=int(flags),
        tensor_format_version=int(version),
        path=path,
    )


def write(path: str | Path, **kwargs: Any) -> Path:
    """Encode and write one ``.irtensor`` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(**kwargs))
    return path


def read(path: str | Path) -> TensorView:
    """Read and decode one ``.irtensor`` from ``path``."""
    path = Path(path)
    return decode(path.read_bytes(), path=path)


def read_header(path: str | Path) -> TensorHeader:
    """Validate and read metadata without materialising the tensor payload."""

    path = Path(path)
    with path.open("rb") as stream:
        fixed = stream.read(_FIXED.size)
        if len(fixed) < _FIXED.size:
            raise IRTensorError(
                f"file too short to be an .irtensor ({len(fixed)} bytes)"
            )
        magic, version, hsize, dcode, flags, rank, reserved, numel, nbytes = (
            _FIXED.unpack(fixed)
        )
        if magic != MAGIC:
            raise IRTensorError(f"bad magic {magic!r}, expected {MAGIC!r}")
        if version != TENSOR_FORMAT_VERSION:
            raise IRTensorError(
                f"unsupported .irtensor version {version}; this build reads "
                f"version {TENSOR_FORMAT_VERSION}"
            )
        expected_hsize = header_size_for_rank(rank)
        if hsize != expected_hsize:
            raise IRTensorError(
                f"header_size {hsize} does not match rank {rank} "
                f"(expected {expected_hsize})"
            )
        if reserved != 0:
            raise IRTensorError(f"reserved header field must be zero, got {reserved}")
        if not flags & FLAG_CANONICAL_CONTIGUOUS:
            raise IRTensorError("payload is not marked canonical contiguous")
        metadata = stream.read(hsize - _FIXED.size)
        if len(metadata) < hsize - _FIXED.size:
            raise IRTensorError(f"truncated metadata header in {path}")

    shape = struct.unpack_from(f"<{rank}q", metadata, 0) if rank else ()
    stride = struct.unpack_from(f"<{rank}q", metadata, 8 * rank) if rank else ()
    (storage_offset,) = struct.unpack_from("<q", metadata, 16 * rank)
    if any(dim < 0 for dim in shape):
        raise IRTensorError(f"negative dimension in shape {shape}")
    shape_numel = 1
    for dim in shape:
        shape_numel *= dim
    if shape_numel != numel:
        raise IRTensorError(
            f"logical_numel {numel} inconsistent with shape product {shape_numel}"
        )
    dtype = dtype_name_from_code(dcode)
    if numel * dtype_itemsize(dtype) != nbytes:
        raise IRTensorError(
            f"payload_nbytes {nbytes} inconsistent with {numel} x {dtype}"
        )
    actual_size = path.stat().st_size
    if actual_size < hsize + nbytes:
        raise IRTensorError(
            f"truncated: need {hsize + nbytes} bytes, have {actual_size}"
        )
    return TensorHeader(
        dtype=dtype,
        shape=tuple(shape),
        stride=tuple(stride),
        storage_offset=int(storage_offset),
        flags=int(flags),
        tensor_format_version=int(version),
        header_size=int(hsize),
        logical_numel=int(numel),
        payload_nbytes=int(nbytes),
    )


def read_stream(stream: BinaryIO) -> TensorView:
    """Read and decode one ``.irtensor`` from an open binary stream."""
    return decode(stream.read())


def write_array(
    path: str | Path,
    array: np.ndarray,
    *,
    dtype: str | None = None,
    stride: Sequence[int] | None = None,
    storage_offset: int = 0,
) -> Path:
    """Convenience: write a numpy array as an ``.irtensor``.

    Used by engine-side Python tooling and tests. ``stride`` defaults to the
    canonical contiguous stride of ``array``.
    """
    from inferref.tensor.dtype_numpy import inferref_dtype_for_numpy

    resolved = dtype or inferref_dtype_for_numpy(array.dtype)
    contiguous = np.ascontiguousarray(array, dtype=numpy_dtype_for(resolved))
    if stride is None:
        stride = contiguous_stride(contiguous.shape)
    return write(
        path,
        dtype=resolved,
        shape=contiguous.shape,
        stride=stride,
        storage_offset=storage_offset,
        payload=contiguous.tobytes(),
    )


def contiguous_stride(shape: Sequence[int]) -> tuple[int, ...]:
    """Row-major contiguous stride, in elements, for ``shape``."""
    stride: list[int] = [1] * len(shape)
    acc = 1
    for i in range(len(shape) - 1, -1, -1):
        stride[i] = acc
        acc *= int(shape[i])
    return tuple(stride)
