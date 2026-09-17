"""Native FP8 contract; deliberately requires numpy only, not Torch/ml_dtypes."""

from __future__ import annotations

import math
import struct
import subprocess
import sys

import numpy as np
import pytest

from inferref.agent.protocol import AdapterCapabilities
from inferref.compare.compare import compare_tensors
from inferref.compare.tolerance import TolerancePolicy
from inferref.ir.dtypes import FLOAT8_DTYPES
from inferref.tensor import codec, float8_bytes_to_float32
from inferref.tensor.dtype_numpy import inferref_dtype_for_numpy, is_carried_as_raw


def expected_values(dtype):
    # Independent exponent-bin construction from ONNX's FP8 encoding tables:
    # subnormals followed by evenly spaced values in each exponent bin.
    if dtype == "float8_e4m3fn":
        positive = [i / 512 for i in range(8)]
        for exponent in range(-6, 9):
            positive.extend((1 + i / 8) * 2.0**exponent for i in range(8))
        positive[-1] = math.nan
    else:
        positive = [i / 65536 for i in range(4)]
        for exponent in range(-14, 16):
            positive.extend((1 + i / 4) * 2.0**exponent for i in range(4))
        positive.extend([math.inf, math.nan, math.nan, math.nan])
    return np.array(positive + [-x for x in positive], dtype=np.float32)


def view(dtype, bits):
    raw = np.asarray(bits, dtype=np.uint8)
    return codec.decode(codec.encode(dtype=dtype, shape=raw.shape,
                                    stride=codec.contiguous_stride(raw.shape),
                                    payload=raw.tobytes()))


@pytest.mark.parametrize("dtype", sorted(FLOAT8_DTYPES))
def test_all_256_encodings(dtype, tmp_path):
    raw = np.arange(256, dtype=np.uint8).reshape(16, 16)
    path = codec.write_array(tmp_path / "fp8.irtensor", raw, dtype=dtype)
    tensor = codec.read(path)
    expected = expected_values(dtype).reshape(16, 16)
    np.testing.assert_array_equal(tensor.as_float32(), expected)
    np.testing.assert_array_equal(tensor.as_comparable(), expected.astype(np.float64))
    np.testing.assert_array_equal(np.signbit(tensor.as_float32()), np.signbit(expected))
    np.testing.assert_array_equal(tensor.data, raw)
    assert tensor.as_float32().dtype == np.float32
    assert tensor.as_comparable().dtype == np.float64
    assert codec.read_header(path).payload_nbytes == 256
    assert tensor.nbytes == 256
    assert tensor.tensor_format_version == 1
    assert is_carried_as_raw(dtype)


@pytest.mark.parametrize("dtype", sorted(FLOAT8_DTYPES))
@pytest.mark.parametrize("shape", [(), (0, 3), (3, 0, 2)])
def test_scalar_and_empty(dtype, shape, tmp_path):
    raw = np.zeros(shape, dtype=np.uint8)
    tensor = view(dtype, raw)
    assert tensor.as_float32().shape == shape
    assert tensor.data.shape == shape
    path = codec.write_array(tmp_path / "shape.irtensor", raw, dtype=dtype)
    assert codec.read(path).shape == shape


@pytest.mark.parametrize("dtype,one,step", [("float8_e4m3fn", 0x38, .125),
                                          ("float8_e5m2", 0x3c, .25)])
def test_comparison_uses_values_not_carrier_bytes(dtype, one, step):
    a, b = view(dtype, [one]), view(dtype, [one + 1])
    exact = TolerancePolicy()
    assert exact.for_dtype(dtype) == (0, 0)
    result = compare_tensors("fp8", a, b, policy=exact)
    assert result.status == "fail"
    assert result.metrics.max_abs_error == step
    assert compare_tensors("fp8", a, b,
                           policy=TolerancePolicy(override_atol=step)).passed
    assert compare_tensors("zero", view(dtype, [0]), view(dtype, [128]), policy=exact).passed
    assert compare_tensors("nan", view(dtype, [127]), view(dtype, [255]), policy=exact).passed


def test_e5m2_infinity_and_dtype_mismatch():
    a = view("float8_e5m2", [0x7c])
    assert not compare_tensors("inf", a, view("float8_e5m2", [0xfc]),
                               policy=TolerancePolicy()).passed
    assert not compare_tensors("dtype", a, view("float8_e4m3fn", [0x7c]),
                               policy=TolerancePolicy()).passed


@pytest.mark.parametrize("dtype", sorted(FLOAT8_DTYPES))
def test_adapter_must_explicitly_declare_fp8(dtype):
    caps = AdapterCapabilities(device_types=("cpu",), dtypes=("uint8",), max_rank=8)
    assert caps.incompatibilities({"dtypes": [dtype]}) == [{"kind": "dtype", "required": dtype}]
    caps = AdapterCapabilities(device_types=("cpu",), dtypes=(dtype,), max_rank=8)
    assert not caps.incompatibilities({"dtypes": [dtype]})


def test_no_implicit_numeric_cast_or_dtype_guessing(tmp_path):
    assert inferref_dtype_for_numpy(np.dtype("uint8")) == "uint8"
    for dtype in FLOAT8_DTYPES:
        with pytest.raises(codec.IRTensorError, match="raw uint8"):
            codec.write_array(tmp_path / "bad.irtensor", np.array([1.5]), dtype=dtype)
        with pytest.raises(ValueError, match="raw uint8"):
            float8_bytes_to_float32(np.array([1.5]), dtype)
    with pytest.raises(ValueError, match="unsupported FP8"):
        float8_bytes_to_float32(np.array([0], dtype=np.uint8), "float8_e4m3fnuz")


def test_decoder_does_not_import_torch():
    script = '''
import sys
import numpy as np
from inferref.tensor import float8_bytes_to_float32
assert "torch" not in sys.modules
assert float8_bytes_to_float32(np.array([56], dtype=np.uint8), "float8_e4m3fn")[0] == 1
assert "torch" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("dtype", sorted(FLOAT8_DTYPES))
def test_fp8_malformed_payload_rejected(dtype, tmp_path):
    blob = codec.encode(dtype=dtype, shape=(2,), stride=(1,), payload=b"\0\0")
    with pytest.raises(codec.IRTensorError, match="truncated"):
        codec.decode(blob[:-1])
    path = tmp_path / "truncated.irtensor"
    path.write_bytes(blob[:-1])
    with pytest.raises(codec.IRTensorError, match="truncated"):
        codec.read_header(path)
    malformed = bytearray(blob)
    struct.pack_into("<Q", malformed, 26, 1)
    with pytest.raises(codec.IRTensorError, match="payload_nbytes"):
        codec.decode(bytes(malformed))
    path.write_bytes(malformed)
    with pytest.raises(codec.IRTensorError, match="payload_nbytes"):
        codec.read_header(path)
