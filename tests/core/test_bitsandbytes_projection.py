"""Numpy-only nested scale decoding, independent of the bnb runtime."""

from copy import deepcopy

import numpy as np
import pytest

from inferref.quantization import QuantizationError, import_bitsandbytes_nf4_projection
from inferref.quantization.bitsandbytes import QUANT_STATE_PROFILE
from inferref.quantization.nf4 import NF4_CODEBOOK


def state(dtype="bfloat16", nested=True):
    count = 258
    inner = {"shape": None, "dtype": "float32", "quant_type": None, "blocksize": 256,
             "absmax": np.array([.25, .5], np.float32),
             "code": np.arange(256, dtype=np.float32) / 256,
             "nested": False, "offset": None, "state2": None}
    return {"shape": (count * 64 - 3,), "dtype": dtype, "quant_type": "nf4", "blocksize": 64,
            "absmax": np.arange(count, dtype=np.uint16).astype(np.uint8) if nested else np.ones(count, np.float32),
            "code": np.array(NF4_CODEBOOK, np.float32), "nested": nested,
            "offset": np.array(.0625, np.float32) if nested else None, "state2": inner if nested else None}


def run(qs, profile=QUANT_STATE_PROFILE):
    return import_bitsandbytes_nf4_projection(np.full((qs["shape"][0] + 1) // 2, 255, np.uint8), qs,
                                              profile=profile, scale_value_id=10, codebook_value_id=11)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("nested", [True, False])
def test_nested_boundary_tail_source_rounding_and_copies(dtype, nested):
    qs = state(dtype, nested)
    item = run(qs)
    expected = np.array([((i % 256) / 256 * (.25 if i < 256 else .5) + .0625)
                         if nested else 1 for i in range(258)], np.float32)
    np.testing.assert_array_equal(item.quantized.parameters[10].data, expected)
    source = item.dequantize_source()
    assert source.dtype == dtype and source.shape == qs["shape"]
    rounded = expected.copy()
    if dtype == "bfloat16":
        # Above .25 the BF16 spacing is 1/512; include ties-to-even explicitly.
        mask = rounded >= .25
        rounded[mask] = np.rint(rounded[mask] * 512) / 512
    np.testing.assert_array_equal(source.as_float32(), rounded[np.arange(qs["shape"][0]) // 64])
    assert item.quantized.codes.payload_bytes()[-1] == 15
    qs["absmax"][:] = 0
    qs["code"][:] = 0
    if nested:
        qs["state2"]["absmax"][:] = 0
    np.testing.assert_array_equal(item.quantized.parameters[10].data, expected)


@pytest.mark.parametrize("key,value", [
    ("nested", 1), ("dtype", "float64"), ("blocksize", True), ("blocksize", 128),
    ("offset", np.array(np.nan, np.float32)), ("offset", np.array(0, np.float64)),
    ("offset", np.zeros(1, np.float32)), ("absmax", np.zeros(258, np.float32)),
    ("absmax", np.zeros(257, np.uint8)), ("code", np.zeros(16, np.float32)),
    ("state2", None), ("unexpected", 0),
])
def test_invalid_outer_state(key, value):
    qs = state()
    qs[key] = value
    with pytest.raises(QuantizationError):
        run(qs)


@pytest.mark.parametrize("key,value", [
    ("nested", True), ("state2", {}), ("blocksize", 64), ("dtype", "float16"),
    ("shape", (258,)), ("quant_type", "nf4"), ("offset", np.array(0, np.float32)),
    ("code", np.zeros(255, np.float32)), ("code", np.full(256, np.inf, np.float32)),
    ("absmax", np.array([-1, 1], np.float32)), ("absmax", np.ones(3, np.float32)),
])
def test_invalid_inner_state(key, value):
    qs = state()
    qs["state2"][key] = value
    with pytest.raises(QuantizationError):
        run(qs)


def test_negative_effective_scales_unknown_profiles_and_missing_fields():
    qs = state()
    qs["offset"] = np.array(-1, np.float32)
    with pytest.raises(QuantizationError, match="nonnegative"):
        run(qs)
    with pytest.raises(QuantizationError, match="profile"):
        run(state(), "auto")
    qs = state()
    del qs["state2"]["offset"]
    with pytest.raises(QuantizationError, match="fields"):
        run(qs)
    qs = state(nested=False)
    qs["state2"] = deepcopy(state()["state2"])
    with pytest.raises(QuantizationError, match="non-nested"):
        run(qs)
