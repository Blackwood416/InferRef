"""Native FP8 capture from Torch, without relying on the Torch/NumPy bridge."""
import json

import numpy as np
import pytest
import torch

import inferref
from inferref.compare.compare import compare_testcase, compare_traces
from inferref.frontend.pytorch.capture import logical_bytes
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.tensor import codec, float8_bytes_to_float32
from inferref.testcase.extract import extract_operator
from inferref.testcase.validate import validate_testcase


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
def test_all_fp8_bits_agree_with_torch(dtype):
    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        pytest.skip(f"installed Torch does not expose {dtype}")
    values = torch.arange(256, dtype=torch.uint8).view(torch_dtype)
    # list(), not numpy(), keeps this oracle independent of the NumPy ABI.
    expected = np.array(values.float().tolist(), dtype=np.float32)
    actual = float8_bytes_to_float32(np.arange(256, dtype=np.uint8), dtype)
    np.testing.assert_array_equal(actual, expected)
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(np.signbit(actual[finite]), np.signbit(expected[finite]))
    assert logical_bytes(values) == bytes(range(256))


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "offset", "scalar", "empty"])
def test_fp8_trace_capture_roundtrip(dtype, layout, tmp_path):
    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        pytest.skip(f"installed Torch does not expose {dtype}")
    x = torch.tensor([0., 1., -2., .5, 4., -0.]).to(torch_dtype).reshape(2, 3)
    if layout == "transposed":
        x = x.T
    elif layout == "offset":
        x = x.flatten()[1::2]
    elif layout == "scalar":
        x = x[0, 1]
    elif layout == "empty":
        x = x[:0]
    root = tmp_path / "trace"
    with torch.no_grad(), inferref.trace(output=root, capture_tensors="all") as session:
        session.mark_input("x", x)
        y = x.clone()
        session.mark_output("y", y)
    package = TracePackage.load(root)
    assert not [issue for issue in validate_package(package) if issue.severity == "error"]
    records = [value for value in package.graph.values if value.dtype == dtype]
    assert records
    for value in records:
        assert value.capture.mode == "full"
        tensor = codec.read(package.tensor_payload_path(value.capture.payload))
        assert tensor.dtype == dtype
        assert tensor.shape == tuple(x.shape)
        assert tensor.data.tobytes() == logical_bytes(x)
    assert compare_traces(root, root).status == "pass"
    clone = next(op for op in package.graph.operators if "clone" in op.canonical_name)
    case = tmp_path / "case"
    extracted = extract_operator(package, clone.id, case)
    assert extracted.reproducible
    assert validate_testcase(case).valid
    manifest = json.loads((case / "testcase.json").read_text(encoding="utf-8"))
    assert manifest["requirements"]["dtypes"] == [dtype]
    assert compare_testcase(case, case / "reference").status == "pass"
