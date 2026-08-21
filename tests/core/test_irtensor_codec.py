"""``.irtensor`` v0.1 codec tests.

Covers the binary contract: every dtype round-trips, headers stay 8-byte
aligned, and malformed files are rejected rather than silently misread.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from inferref.ir.dtypes import (
    DTYPE_CODES,
    dtype_code,
    dtype_itemsize,
    dtype_name_from_code,
)
from inferref.tensor import codec
from inferref.tensor.dtype_numpy import (
    bfloat16_bytes_to_float32,
    float32_to_bfloat16_bytes,
    numpy_dtype_for,
)


def _sample(name: str, count: int = 6) -> np.ndarray:
    """A deterministic array of ``count`` elements in InferRef dtype ``name``."""
    if name == "bool":
        return np.array([True, False] * (count // 2), dtype=np.bool_)
    if name == "bfloat16":
        # Carried as raw uint16 bit patterns.
        return float32_to_bfloat16_bytes(np.arange(count, dtype=np.float32) * 1.5)
    dtype = numpy_dtype_for(name)
    if dtype.kind == "c":
        return (np.arange(count) + 1j * np.arange(count)).astype(dtype)
    if dtype.kind in "iu":
        return np.arange(count).astype(dtype)
    return (np.arange(count) * 0.5).astype(dtype)


@pytest.mark.parametrize("name", sorted(DTYPE_CODES))
def test_roundtrip_every_dtype(name: str) -> None:
    array = _sample(name)
    shape = (2, 3)
    stride = codec.contiguous_stride(shape)

    blob = codec.encode(
        dtype=name,
        shape=shape,
        stride=stride,
        storage_offset=0,
        payload=array.tobytes(),
    )
    view = codec.decode(blob)

    assert view.dtype == name
    assert view.shape == shape
    assert view.stride == stride
    assert view.numel == 6
    assert view.nbytes == 6 * dtype_itemsize(name)
    assert np.array_equal(view.data, array.reshape(shape))


@pytest.mark.parametrize("rank", range(7))
def test_header_size_is_8_byte_aligned(rank: int) -> None:
    """Payload must start 8-byte aligned so C++ readers can memcpy safely."""
    size = codec.header_size_for_rank(rank)
    assert size == 48 + 16 * rank
    assert size % 8 == 0


def test_header_field_offsets_match_spec() -> None:
    """Field order follows Trace IR §21 exactly."""
    blob = codec.encode(
        dtype="float32", shape=(2, 3), stride=(3, 1), storage_offset=7,
        payload=np.zeros(6, dtype="<f4").tobytes(),
    )
    assert blob[0:4] == b"IRTN"
    assert struct.unpack_from("<H", blob, 4)[0] == 1            # tensor_format_version
    assert struct.unpack_from("<H", blob, 6)[0] == 80           # header_size (rank 2)
    assert struct.unpack_from("<H", blob, 8)[0] == dtype_code("float32")
    assert struct.unpack_from("<I", blob, 10)[0] == codec.FLAG_CANONICAL_CONTIGUOUS
    assert struct.unpack_from("<H", blob, 14)[0] == 2           # rank
    assert struct.unpack_from("<H", blob, 16)[0] == 0           # reserved
    assert struct.unpack_from("<Q", blob, 18)[0] == 6           # logical_numel
    assert struct.unpack_from("<Q", blob, 26)[0] == 24          # payload_nbytes
    assert struct.unpack_from("<2q", blob, 34) == (2, 3)        # shape
    assert struct.unpack_from("<2q", blob, 50) == (3, 1)        # stride
    assert struct.unpack_from("<q", blob, 66)[0] == 7           # storage_offset


def test_dtype_codes_are_stable() -> None:
    """Codes are frozen at format 0.1 and shared with the C++ reader."""
    expected = {
        "bool": 1, "int8": 2, "int16": 3, "int32": 4, "int64": 5,
        "uint8": 6, "uint16": 7, "uint32": 8, "uint64": 9,
        "float16": 10, "bfloat16": 11, "float32": 12, "float64": 13,
        "complex64": 14, "complex128": 15,
    }
    assert DTYPE_CODES == expected
    for name, code in expected.items():
        assert dtype_name_from_code(code) == name


def test_bfloat16_decodes_without_torch() -> None:
    """numpy has no bfloat16; the reader widens it in software."""
    values = np.array([1.5, 2.5, -0.75, 0.0], dtype=np.float32)
    bits = float32_to_bfloat16_bytes(values)
    view = codec.decode(
        codec.encode(
            dtype="bfloat16", shape=(4,), stride=(1,), payload=bits.tobytes()
        )
    )
    assert view.dtype == "bfloat16"
    assert np.allclose(view.as_float32(), values)
    assert np.allclose(bfloat16_bytes_to_float32(view.data), values)


def test_float16_and_bfloat16_ieee_edge_vectors() -> None:
    values = np.array(
        [
            0.0,
            -0.0,
            np.ldexp(1.0, -24),
            np.ldexp(1023.0, -24),
            np.ldexp(1.0, -14),
            65504.0,
            np.inf,
            -np.inf,
            np.nan,
            1.0 + np.ldexp(1.0, -11),
            1.0 + np.ldexp(3.0, -11),
        ],
        dtype=np.float32,
    )
    half = values.astype("<f2").view("<u2")
    assert half[:8].tolist() == [
        0x0000,
        0x8000,
        0x0001,
        0x03FF,
        0x0400,
        0x7BFF,
        0x7C00,
        0xFC00,
    ]
    assert half[8] & 0x7FFF > 0x7C00
    assert half[9:].tolist() == [0x3C00, 0x3C02]

    bfloat = float32_to_bfloat16_bytes(values)
    assert bfloat[0] == 0x0000
    assert bfloat[1] == 0x8000
    assert bfloat[6] == 0x7F80
    assert bfloat[7] == 0xFF80
    assert bfloat[8] == 0x7FC0


def test_stride_is_metadata_not_payload_layout() -> None:
    """IR §20: the payload is canonical contiguous whatever the stride says."""
    payload = np.arange(6, dtype="<f4")
    view = codec.decode(
        codec.encode(
            dtype="float32",
            shape=(2, 3),
            stride=(1, 2),          # a transposed view's stride
            storage_offset=4,
            payload=payload.tobytes(),
        )
    )
    assert view.stride == (1, 2)
    assert view.storage_offset == 4
    # Payload order is unaffected by the recorded stride.
    assert np.array_equal(view.data.reshape(-1), payload)


def test_write_and_read_file(tmp_path) -> None:
    path = tmp_path / "t.irtensor"
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    codec.write_array(path, array)
    view = codec.read(path)
    assert view.shape == (3, 4)
    assert view.stride == (4, 1)
    assert np.array_equal(view.data, array)
    assert view.path == path


def test_zero_element_tensor_roundtrips() -> None:
    view = codec.decode(
        codec.encode(dtype="float32", shape=(0, 3), stride=(3, 1), payload=b"")
    )
    assert view.numel == 0
    assert view.shape == (0, 3)
    assert view.data.size == 0


def test_scalar_rank0_tensor_roundtrips() -> None:
    view = codec.decode(
        codec.encode(
            dtype="float32", shape=(), stride=(), payload=np.float32(2.5).tobytes()
        )
    )
    assert view.rank == 0
    assert view.numel == 1
    assert float(view.data.reshape(-1)[0]) == 2.5


def test_rejects_bad_magic() -> None:
    blob = bytearray(codec.encode(dtype="float32", shape=(1,), stride=(1,),
                                  payload=np.zeros(1, "<f4").tobytes()))
    blob[0:4] = b"XXXX"
    with pytest.raises(codec.IRTensorError, match="bad magic"):
        codec.decode(bytes(blob))


def test_rejects_truncated_payload() -> None:
    blob = codec.encode(dtype="float32", shape=(4,), stride=(1,),
                        payload=np.zeros(4, "<f4").tobytes())
    with pytest.raises(codec.IRTensorError, match="truncated"):
        codec.decode(blob[:-4])


def test_rejects_payload_size_mismatch() -> None:
    with pytest.raises(codec.IRTensorError, match="payload is"):
        codec.encode(dtype="float32", shape=(4,), stride=(1,), payload=b"\x00" * 8)


def test_rejects_rank_mismatch() -> None:
    with pytest.raises(codec.IRTensorError, match="rank mismatch"):
        codec.encode(dtype="float32", shape=(2, 2), stride=(1,), payload=b"\x00" * 16)


def test_rejects_negative_dimensions_at_encode_and_decode() -> None:
    with pytest.raises(codec.IRTensorError, match="negative dimension"):
        codec.encode(dtype="float32", shape=(-1, -2), stride=(2, 1), payload=b"\x00" * 8)

    blob = bytearray(
        codec.encode(
            dtype="float32",
            shape=(1, 2),
            stride=(2, 1),
            payload=np.zeros(2, dtype="<f4").tobytes(),
        )
    )
    struct.pack_into("<q", blob, 34, -1)
    with pytest.raises(codec.IRTensorError, match="negative dimension"):
        codec.decode(bytes(blob))


def test_decode_rejects_shape_numel_mismatch() -> None:
    blob = bytearray(
        codec.encode(
            dtype="float32",
            shape=(2, 2),
            stride=(2, 1),
            payload=np.zeros(4, dtype="<f4").tobytes(),
        )
    )
    struct.pack_into("<Q", blob, 18, 3)
    with pytest.raises(codec.IRTensorError, match="shape product"):
        codec.decode(bytes(blob))


def test_write_array_normalises_big_endian_to_little_endian(tmp_path) -> None:
    values = np.array([1.0, 2.0], dtype=">f4")
    path = codec.write_array(tmp_path / "big-endian.irtensor", values)
    decoded = codec.read(path)

    assert decoded.dtype == "float32"
    assert np.array_equal(decoded.data, np.array([1.0, 2.0], dtype="<f4"))


def test_rejects_unknown_dtype_code() -> None:
    blob = bytearray(codec.encode(dtype="float32", shape=(1,), stride=(1,),
                                  payload=np.zeros(1, "<f4").tobytes()))
    struct.pack_into("<H", blob, 8, 999)
    with pytest.raises(ValueError, match="unknown InferRef dtype code"):
        codec.decode(bytes(blob))
