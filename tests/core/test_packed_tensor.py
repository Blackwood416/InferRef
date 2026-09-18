from __future__ import annotations

import struct

import numpy as np
import pytest

from inferref.ir.dtypes import (
    dtype_bits,
    dtype_code,
    dtype_itemsize,
    dtype_payload_nbytes,
)
from inferref.ir.tensor_value import TensorValueRecord
from inferref.tensor import codec, pack_nibbles, unpack_nibbles


@pytest.mark.parametrize("dtype,carrier", [("int4", "int8"), ("uint4", "uint8")])
def test_all_byte_pairs(dtype, carrier, tmp_path):
    expected = [value for byte in range(256) for value in (byte % 16, byte // 16)]
    if dtype == "int4":
        expected = [value if value < 8 else value - 16 for value in expected]
    codes = np.array(expected, dtype=carrier).reshape(16, 32)
    payload = bytes(range(256))
    assert pack_nibbles(codes, dtype) == payload
    assert unpack_nibbles(payload, dtype, 512).tolist() == expected
    path = codec.write_array(tmp_path / "packed.irtensor", codes, dtype=dtype)
    tensor = codec.read(path)
    assert tensor.tensor_format_version == 2
    assert tensor.nbytes == 256
    assert tensor.data.tolist() == codes.tolist()
    assert tensor.as_float32().tolist() == codes.tolist()
    assert tensor.payload_bytes() == payload
    assert codec.read_header(path).payload_nbytes == 256
    assert tensor.dtype == dtype
    assert dtype_code(dtype) == (18 if dtype == "int4" else 19)
    assert dtype_bits(dtype) == 4
    with pytest.raises(ValueError, match="sub-byte"):
        dtype_itemsize(dtype)


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (1,), (3,), (3, 3)])
@pytest.mark.parametrize("dtype", ["int4", "uint4"])
def test_scalar_empty_odd_shapes(dtype, shape, tmp_path):
    codes = np.ones(shape, dtype="int8" if dtype == "int4" else "uint8")
    path = codec.write_array(tmp_path / "p.irtensor", codes, dtype=dtype)
    tensor = codec.read(path)
    assert tensor.data.shape == shape
    assert tensor.nbytes == (codes.size + 1) // 2
    assert TensorValueRecord(dtype=dtype, shape=shape).logical_nbytes == tensor.nbytes
    assert dtype_payload_nbytes(dtype, codes.size) == tensor.nbytes
    assert np.array_equal(tensor.data, codes)
    if codes.size % 2:
        assert path.read_bytes()[-1] & 0xf0 == 0


@pytest.mark.parametrize("dtype,values", [
    ("int4", np.array([-9], dtype=np.int8)), ("int4", np.array([8], dtype=np.int8)),
    ("uint4", np.array([16], dtype=np.uint8)), ("int4", np.array([1.5], dtype=np.float32)),
    ("uint4", np.array([-1], dtype=np.int8)),
])
def test_no_truncation_or_implicit_cast(dtype, values, tmp_path):
    with pytest.raises(ValueError):
        codec.write_array(tmp_path / "bad.irtensor", values, dtype=dtype)
    assert not (tmp_path / "bad.irtensor").exists()


@pytest.mark.parametrize("fault", ["padding", "short", "length", "legacy_version", "future_version"])
def test_malformed_wire_rejected_by_both_readers(fault, tmp_path):
    blob = bytearray(codec.encode(dtype="int4", shape=(3,), stride=(1,), payload=b"\x87\x0f"))
    if fault == "padding":
        blob[-1] |= 0x10
    elif fault == "short":
        blob.pop()
    elif fault == "length":
        struct.pack_into("<Q", blob, 26, 3)
        blob.append(0)
    else:
        struct.pack_into("<H", blob, 4, 1 if fault == "legacy_version" else 3)
    path = tmp_path / "bad.irtensor"
    path.write_bytes(blob)
    with pytest.raises(ValueError):
        codec.decode(blob)
    with pytest.raises(ValueError):
        codec.read_header(path)


def test_legacy_writers_keep_version_one(tmp_path):
    path = codec.write_array(tmp_path / "i8.irtensor", np.array([1], dtype=np.int8))
    assert codec.read(path).tensor_format_version == 1
    blob = bytearray(path.read_bytes())
    struct.pack_into("<H", blob, 4, 2)
    with pytest.raises(ValueError, match="requires tensor_format_version 1"):
        codec.decode(blob)


def test_noncontiguous_codes_and_layout(tmp_path):
    data = np.arange(-8, 8, dtype=np.int8).reshape(4, 4).T
    path = codec.write_array(tmp_path / "view.irtensor", data, dtype="int4", stride=(1, 4), storage_offset=3)
    view = codec.read(path)
    assert np.array_equal(view.data, data)
    assert view.stride == (1, 4)
    assert view.storage_offset == 3
