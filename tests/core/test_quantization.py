"""Standalone affine INT8 semantics: no Torch and no implicit wire integration."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from inferref.agent.executor import execute_adapter
from inferref.agent.protocol import EngineAdapter
from inferref.compare.compare import compare_tensors, compare_testcase
from inferref.compare.tolerance import TolerancePolicy
from inferref.ir.tensor_value import TensorValueRecord
from inferref.quantization import (
    QuantizationError,
    QuantizationSpec,
    dequantize_affine,
    linear_int8_reference,
    quantization_fingerprint,
    quantize_affine,
)
from inferref.tensor import codec
from inferref.testcase.validate import TestcaseValidationError as ValidationError
from inferref.testcase.validate import validate_testcase


def tensor(data, dtype="float32"):
    array = np.asarray(data, dtype=dtype)
    return codec.decode(codec.encode(dtype=dtype, shape=array.shape,
                                    stride=codec.contiguous_stride(array.shape),
                                    payload=array.tobytes()))


def test_descriptor_roundtrip_is_explicit_and_immutable():
    spec = QuantizationSpec(logical_shape=(2, 5), scale_value_id=10,
                            zero_point_value_id=11, granularity="per_group",
                            axis=1, group_size=2)
    assert spec.parameter_shape == (2, 3)
    assert spec.dependencies == (10, 11)
    document = json.loads(json.dumps(spec.to_dict()))
    assert QuantizationSpec.from_dict(document) == spec
    assert document["format_version"] == "1.0"
    assert document["rounding"] == "nearest_even"
    assert document["saturation"] == "clamp"
    with pytest.raises(FrozenInstanceError):
        spec.axis = 0
    symmetric = QuantizationSpec(logical_shape=(), scale_value_id=0)
    assert symmetric.dependencies == (0,)
    assert symmetric.to_dict()["zero_point_value_id"] is None


@pytest.mark.parametrize("field,value", [
    ("scheme", "nf4/v1"), ("storage_dtype", "uint8"), ("compute_dtype", "float16"),
    ("rounding", "nearest_away"), ("saturation", "wrap"), ("granularity", "blockwise"),
    ("logical_shape", [2]), ("logical_shape", (True,)), ("logical_shape", (-1,)),
    ("logical_shape", (2**63,)), ("scale_value_id", -1), ("scale_value_id", True),
    ("scale_value_id", 1.0), ("zero_point_value_id", True), ("zero_point_value_id", -1),
    ("zero_point_value_id", 10), ("axis", 0), ("group_size", 2),
])
def test_descriptor_rejects_ambiguous_or_unsupported_fields(field, value):
    kwargs = {"logical_shape": (2,), "scale_value_id": 10, field: value}
    with pytest.raises(QuantizationError):
        QuantizationSpec(**kwargs)


@pytest.mark.parametrize("granularity,axis,group_size", [
    ("per_axis", None, None), ("per_axis", -1, None), ("per_axis", 2, None),
    ("per_axis", True, None), ("per_axis", 0, 2), ("per_group", 0, None),
    ("per_group", 0, 0), ("per_group", 0, True), ("per_group", 0, 2**63),
])
def test_descriptor_axis_and_group_constraints(granularity, axis, group_size):
    with pytest.raises(QuantizationError):
        QuantizationSpec(logical_shape=(2, 3), scale_value_id=1,
                         granularity=granularity, axis=axis, group_size=group_size)


@pytest.mark.parametrize("field,value", [
    ("format", "inferref-trace"), ("format_version", "1.1"),
    ("format_version", 1), ("logical_shape", "2"), ("scale_value_id", "1"),
    ("extra", {}),
])
def test_json_rejects_unknown_semantics(field, value):
    data = QuantizationSpec(logical_shape=(2,), scale_value_id=1).to_dict()
    data[field] = value
    with pytest.raises(QuantizationError):
        QuantizationSpec.from_dict(data)


def test_json_requires_every_field_including_nulls():
    data = QuantizationSpec(logical_shape=(2,), scale_value_id=1).to_dict()
    for field in data:
        with pytest.raises(QuantizationError, match="missing"):
            QuantizationSpec.from_dict({k: v for k, v in data.items() if k != field})
    for invalid in (None, [], "", 1):
        with pytest.raises(QuantizationError, match="object"):
            QuantizationSpec.from_dict(invalid)


def test_per_tensor_symmetric_and_asymmetric_exhaust_int8():
    raw = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    spec = QuantizationSpec(logical_shape=(256,), scale_value_id=1)
    params = {1: tensor(.25)}
    result = dequantize_affine(tensor(raw, "int8"), spec, values=params)
    np.testing.assert_array_equal(result, raw.astype(np.float32) * .25)
    np.testing.assert_array_equal(quantize_affine(result, spec, values=params), raw)
    spec = replace(spec, zero_point_value_id=2)
    params[2] = tensor(-128, "int8")
    result = dequantize_affine(tensor(raw, "int8"), spec, values=params)
    np.testing.assert_array_equal(result, np.arange(256, dtype=np.float32) * .25)
    np.testing.assert_array_equal(quantize_affine(result, spec, values=params), raw)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_per_axis_parameter_broadcasting(axis):
    shape = (2, 3, 4)
    raw = np.arange(24, dtype=np.int8).reshape(shape)
    scales = np.arange(1, shape[axis] + 1, dtype=np.float32) * .125
    zeros = np.arange(shape[axis], dtype=np.int8)
    spec = QuantizationSpec(logical_shape=shape, scale_value_id=1, zero_point_value_id=2,
                            granularity="per_axis", axis=axis)
    params = {1: tensor(scales), 2: tensor(zeros, "int8")}
    expected = np.empty(shape, dtype=np.float32)
    for index in np.ndindex(shape):
        expected[index] = (int(raw[index]) - int(zeros[index[axis]])) * scales[index[axis]]
    actual = dequantize_affine(tensor(raw, "int8"), spec, values=params)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(quantize_affine(actual, spec, values=params), raw)


@pytest.mark.parametrize("axis,group_size", [(0, 2), (1, 2), (1, 10**12), (1, 1)])
def test_per_group_including_tail_and_huge_group(axis, group_size):
    shape = (3, 5)
    spec = QuantizationSpec(logical_shape=shape, scale_value_id=1, zero_point_value_id=2,
                            granularity="per_group", axis=axis, group_size=group_size)
    count = np.prod(spec.parameter_shape)
    scales = (np.arange(1, count + 1, dtype=np.float32) * .125).reshape(spec.parameter_shape)
    zeros = np.arange(count, dtype=np.int8).reshape(spec.parameter_shape)
    raw = np.arange(15, dtype=np.int8).reshape(shape)
    expected = np.empty(shape, dtype=np.float32)
    for index in np.ndindex(shape):
        parameter_index = list(index)
        parameter_index[axis] //= group_size
        parameter_index = tuple(parameter_index)
        expected[index] = (int(raw[index]) - int(zeros[parameter_index])) * scales[parameter_index]
    params = {1: tensor(scales), 2: tensor(zeros, "int8")}
    actual = dequantize_affine(tensor(raw, "int8"), spec, values=params)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(quantize_affine(actual, spec, values=params), raw)


@pytest.mark.parametrize("shape,granularity,axis,group_size", [
    ((), "per_tensor", None, None), ((0, 3), "per_tensor", None, None),
    ((0, 3), "per_axis", 0, None), ((0, 3), "per_group", 0, 2),
    ((0, 3), "per_group", 1, 2),
])
def test_scalar_and_empty_shapes(shape, granularity, axis, group_size):
    spec = QuantizationSpec(logical_shape=shape, scale_value_id=1,
                            granularity=granularity, axis=axis, group_size=group_size)
    params = {1: tensor(np.ones(spec.parameter_shape, dtype=np.float32))}
    raw = np.zeros(shape, dtype=np.int8)
    decoded = dequantize_affine(tensor(raw, "int8"), spec, values=params)
    encoded = quantize_affine(np.zeros(shape, dtype=np.float32), spec, values=params)
    assert isinstance(decoded, np.ndarray) and decoded.shape == shape
    assert isinstance(encoded, np.ndarray) and encoded.shape == shape
    assert decoded.dtype == np.float32 and encoded.dtype == np.int8


def test_round_to_even_then_add_zero_point_and_saturate():
    source = np.array([-1e20, -2.5, -1.5, -.5, .5, 1.5, 2.5, 1e20], dtype=np.float32)
    spec = QuantizationSpec(logical_shape=source.shape, scale_value_id=1, zero_point_value_id=2)
    params = {1: tensor(1.), 2: tensor(1, "int8")}
    np.testing.assert_array_equal(quantize_affine(source, spec, values=params),
                                  [-128, -1, -1, 1, 1, 3, 3, 127])
    params[1] = tensor(np.nextafter(np.float32(0), np.float32(1)))
    saturated = quantize_affine(source, spec, values=params)
    np.testing.assert_array_equal(saturated, [-128] * 4 + [127] * 4)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_scales_rejected(scale):
    spec = QuantizationSpec(logical_shape=(2,), scale_value_id=1)
    for function, source in ((dequantize_affine, tensor([1, 2], "int8")),
                             (quantize_affine, np.array([1, 2], dtype=np.float32))):
        with pytest.raises(QuantizationError, match="strictly positive"):
            function(source, spec, values={1: tensor(scale)})


def test_missing_and_wrong_parameter_types_shapes():
    spec = QuantizationSpec(logical_shape=(2,), scale_value_id=1, zero_point_value_id=2)
    raw = tensor([1, 2], "int8")
    for params, message in [
        ({}, "missing scale"), ({1: tensor(1.)}, "missing zero point"),
        ({1: tensor([1.]), 2: tensor(0, "int8")}, "scale must have"),
        ({1: tensor(1., "float64"), 2: tensor(0, "int8")}, "scale must have"),
        ({1: tensor(1.), 2: tensor(128, "int16")}, "zero point must have"),
        ({1: tensor(1.), 2: tensor([0], "int8")}, "zero point must have"),
        ({1: spec, 2: tensor(0, "int8")}, "unquantized TensorView"),
    ]:
        with pytest.raises(QuantizationError, match=message):
            dequantize_affine(raw, spec, values=params)


def test_malformed_codes_and_nonfinite_numeric_source():
    spec = QuantizationSpec(logical_shape=(2,), scale_value_id=1)
    params = {1: tensor(1.)}
    for codes in (tensor([1, 2], "uint8"), tensor([1], "int8")):
        with pytest.raises(QuantizationError, match="codes must have"):
            dequantize_affine(codes, spec, values=params)
    codes = tensor([1, 2], "int8")
    codes.data = np.array([1., 2.], dtype=np.float32)
    with pytest.raises(QuantizationError, match="carrier"):
        dequantize_affine(codes, spec, values=params)
    for data in (np.array([1., np.nan], dtype=np.float32),
                 np.array([1., np.inf], dtype=np.float32), np.array([1., 2.]),
                 np.array([1.], dtype=np.float32)):
        with pytest.raises(QuantizationError):
            quantize_affine(data, spec, values=params)


def test_dequantization_overflow_is_explicit():
    spec = QuantizationSpec(logical_shape=(1,), scale_value_id=1)
    with pytest.raises(QuantizationError, match="overflow"):
        dequantize_affine(tensor([127], "int8"), spec,
                          values={1: tensor(np.finfo(np.float32).max)})


def test_float32_underflow_is_independent_of_callers_numpy_error_mode():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    spec = QuantizationSpec(logical_shape=(1,), scale_value_id=1)
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(
            quantize_affine(np.array([tiny], np.float32), spec,
                            values={1: tensor(np.finfo(np.float32).max)}), [0])
        np.testing.assert_array_equal(
            dequantize_affine(tensor([1], "int8"), spec, values={1: tensor(tiny)}), [tiny])
        output = linear_int8_reference(np.array([.5], np.float32), tensor([[1]], "int8"),
                                       replace(spec, logical_shape=(1, 1)),
                                       values={1: tensor(tiny)})
        np.testing.assert_array_equal(output, [0.])


def test_linear_reference_uses_same_quantized_weights_and_reports_injected_error():
    # Expected weights after dequantizing are [[1, 2, 3], [-2, 0, 2]].
    spec = QuantizationSpec(logical_shape=(2, 3), scale_value_id=1,
                            zero_point_value_id=2, granularity="per_axis", axis=0)
    params = {1: tensor([.5, 2.]), 2: tensor([0, 1], "int8")}
    weights = tensor([[2, 4, 6], [0, 1, 2]], "int8")
    inputs = np.array([[[1., 2., 3.], [-1., 0., 1.]]], dtype=np.float32)
    bias = np.array([.5, -1.], dtype=np.float32)
    expected = np.array([[[14.5, 3.], [2.5, 3.]]], dtype=np.float32)
    actual = linear_int8_reference(inputs, weights, spec, values=params, bias=bias)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    good = compare_tensors("linear", tensor(expected), tensor(actual), policy=TolerancePolicy())
    assert good.passed
    broken = actual.copy()
    broken[0, 0, 1] += .5
    bad = compare_tensors("linear", tensor(expected), tensor(broken), policy=TolerancePolicy())
    assert not bad.passed and bad.metrics.first_mismatch_index == (0, 0, 1)


def test_linear_shapes_precision_and_overflow():
    spec = QuantizationSpec(logical_shape=(2, 3), scale_value_id=1)
    params = {1: tensor(1.)}
    weights = tensor([[1, 2, 3], [4, 5, 6]], "int8")
    for inputs, bias in ((np.ones((2, 4), np.float32), None),
                         (np.ones((3,), np.float64), None),
                         (np.ones((3,), np.float32), np.ones((1,), np.float32)),
                         (np.ones((3,), np.float32), np.ones((2,), np.float64))):
        with pytest.raises(QuantizationError):
            linear_int8_reference(inputs, weights, spec, values=params, bias=bias)
    with pytest.raises(QuantizationError, match="overflow|nonfinite"):
        linear_int8_reference(np.full((3,), np.finfo(np.float32).max, np.float32),
                              weights, spec, values=params)
    empty_spec = replace(spec, logical_shape=(2, 0))
    out = linear_int8_reference(np.zeros((4, 0), np.float32),
                               tensor(np.zeros((2, 0), np.int8), "int8"),
                               empty_spec, values=params)
    np.testing.assert_array_equal(out, np.zeros((4, 2), np.float32))


def test_fingerprint_includes_parameters_descriptor_and_codes_but_not_paths(tmp_path):
    spec = QuantizationSpec(logical_shape=(2,), scale_value_id=1, zero_point_value_id=2)
    params = {1: tensor(.5), 2: tensor(0, "int8")}
    raw = tensor([1, 2], "int8")
    first = quantization_fingerprint(raw, spec, values=params)
    raw.path = tmp_path / "relocated.irtensor"
    params[1].path = tmp_path / "scale.irtensor"
    assert quantization_fingerprint(raw, spec, values=params) == first
    assert quantization_fingerprint(raw, spec, values=dict(reversed(list(params.items())))) == first
    assert quantization_fingerprint(raw, spec, values={**params, 1: tensor(.25)}) != first
    assert quantization_fingerprint(raw, spec, values={**params, 2: tensor(1, "int8")}) != first
    assert quantization_fingerprint(tensor([1, 3], "int8"), spec, values=params) != first
    assert quantization_fingerprint(raw, replace(spec, zero_point_value_id=None),
                                    values=params) != first


def test_descriptor_import_is_stdlib_only():
    code = '''
import sys
from inferref.quantization import QuantizationSpec
assert QuantizationSpec(logical_shape=(2,), scale_value_id=1).parameter_shape == ()
assert 'numpy' not in sys.modules
assert 'torch' not in sys.modules
from inferref.quantization import quantize_affine
assert 'numpy' in sys.modules
assert 'torch' not in sys.modules
'''
    subprocess.run([sys.executable, "-c", code], check=True)


def test_embedded_trace_quantization_is_rejected_before_silent_lowering():
    spec = QuantizationSpec(logical_shape=(2,), scale_value_id=1).to_dict()
    decoded = TensorValueRecord.from_dict({"id": 0, "dtype": "int8", "shape": [2],
                                          "stride": [1], "quantization": spec})
    assert decoded.quantization.to_dict() == spec
    from inferref.ir.graph import Graph
    from inferref.ir.package import TracePackage
    from inferref.quantization.artifacts import validate_trace_metadata

    with pytest.raises(QuantizationError, match="requires trace format_version 1.0"):
        validate_trace_metadata(TracePackage(graph=Graph(values=[decoded])))
    value = TensorValueRecord(id=0, dtype="int8", shape=(2,), stride=(1,))
    value.unknown["quantization"] = spec
    with pytest.raises(QuantizationError, match="embedded quantization"):
        value.to_dict()
    from inferref.testcase.extract import _tensor_metadata

    with pytest.raises(QuantizationError, match="embedded quantization"):
        _tensor_metadata(value)


@pytest.mark.parametrize("collection", ["inputs", "outputs", "values"])
def test_embedded_testcase_quantization_is_invalid(collection, tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "fixtures/scenarios/kv-chain/cases/prefill"
    target = tmp_path / "case"
    shutil.copytree(source, target)
    assert validate_testcase(target).valid
    path = target / "testcase.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[collection][0]["quantization"] = QuantizationSpec(
        logical_shape=(2,), scale_value_id=1).to_dict()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    validation = validate_testcase(target)
    assert not validation.valid
    assert any(issue.code == "quantization_unsupported" for issue in validation.errors)
    with pytest.raises(ValidationError, match="quantization_unsupported"):
        compare_testcase(target, target / "reference")

    def forbidden_launch(*args, **kwargs):
        pytest.fail("an invalid quantized testcase must not launch an engine")

    monkeypatch.setattr("inferref.agent.executor.subprocess.Popen", forbidden_launch)
    adapter = EngineAdapter(name="never-launch", command=(sys.executable, "-c", "pass"))
    with pytest.raises(ValidationError, match="quantization_unsupported"):
        execute_adapter(target, adapter, tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_documented_example_is_torch_free_and_deterministic():
    script = Path(__file__).resolve().parents[2] / "examples/quantization/int8_reference.py"
    result = subprocess.run([sys.executable, str(script)], check=True,
                            text=True, capture_output=True)
    document = json.loads(result.stdout)
    assert document["linear_output"] == [[9.5, 10.0]]
    assert document["decoded_weights"] == [[1., 2., 1.5], [-2., 0., 4.]]
    assert document["example_scope"] == "standalone_reference_api"
    assert len(document["fingerprint"]) == 64
