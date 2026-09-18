from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inferref.agent.executor import execute_adapter
from inferref.agent.protocol import EngineAdapter
from inferref.alignment.hashing import normalized_adapter_sha256, trace_metadata_sha256
from inferref.alignment.planner import plan_alignment
from inferref.alignment.runner import AlignmentRunner
from inferref.alignment.store import AlignmentStore
from inferref.cli.main import main
from inferref.ir.validate import validate_package
from inferref.quantization import (
    QuantizationError,
    QuantizationSpec,
    dequantize_affine,
    linear_int4_reference,
    quantization_fingerprint,
    quantize_affine,
)
from inferref.suite import run_suite
from inferref.tensor import codec
from inferref.testcase.extract import extract_operator
from inferref.testcase.validate import validate_testcase

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/quantization"


def view(data, dtype, tmp_path, label):
    return codec.read(codec.write_array(tmp_path / f"{label}.irtensor", np.asarray(data), dtype=dtype))


@pytest.mark.parametrize("unsigned", [False, True])
def test_affine_four_bit_all_codes_and_rounding(unsigned, tmp_path):
    dtype = "uint4" if unsigned else "int4"
    carrier = "uint8" if unsigned else "int8"
    codes = np.arange(0 if unsigned else -8, 16 if unsigned else 8, dtype=carrier)
    spec = QuantizationSpec(logical_shape=(16,), scale_value_id=10, zero_point_value_id=11,
                            scheme=f"affine-{dtype}/v1", storage_dtype=dtype)
    zero = 8 if unsigned else -1
    params = {10: view(np.array(.5, dtype=np.float32), "float32", tmp_path, "scale"),
              11: view(np.array(zero, dtype=carrier), carrier, tmp_path, "zero")}
    raw = view(codes, dtype, tmp_path, "raw")
    decoded = dequantize_affine(raw, spec, values=params)
    assert np.array_equal(decoded, (codes.astype(np.float32) - zero) * .5)
    assert np.array_equal(quantize_affine(decoded, spec, values=params), codes)
    first = quantization_fingerprint(raw, spec, values=params)
    params[10] = view(np.array(.25, dtype=np.float32), "float32", tmp_path, "scale2")
    assert quantization_fingerprint(raw, spec, values=params) != first
    params[11] = view(np.array(16 if unsigned else -9, dtype=carrier), carrier, tmp_path, "badzero")
    with pytest.raises(QuantizationError, match="zero point"):
        dequantize_affine(raw, spec, values=params)


@pytest.mark.parametrize("unsigned", [False, True])
def test_w4a16_complete_workflow(unsigned, tmp_path, capsys):
    package = runpy.run_path(str(EXAMPLE / "int8_trace.py"))["build_trace"](tmp_path / "trace", packed=True, unsigned=unsigned)
    assert not [i for i in validate_package(package) if i.severity == "error"]
    root = tmp_path / "case"
    assert main(["testcase", "extract", str(package.root), "--op", "0", "--output", str(root),
                 "--input-names", "x,weight", "--output-names", "y", "--contract", "linear/affine-w4a16/v1"]) == 0
    capsys.readouterr()
    result = validate_testcase(root)
    assert result.valid, result.to_dict()
    raw = codec.read(root / "inputs/weight.irtensor")
    assert raw.nbytes == 3 and raw.tensor_format_version == 2
    engine = EngineAdapter.load(EXAMPLE / "int4_numpy.adapter.json")
    execution = execute_adapter(root, engine, tmp_path / "runs")
    assert execution["status"] == "pass", execution
    assert execution["comparison"]["comparisons"][0]["comparison_domain"] == "float16_kernel_output"
    bug = execute_adapter(root, replace(engine, command=(*engine.command, "--inject-bug")), tmp_path / "runs")
    assert bug["status"] == "mismatch", bug
    old = EngineAdapter.load(EXAMPLE / "int8_numpy.adapter.json")
    assert execute_adapter(root, old, tmp_path / "old-runs")["status"] == "unsupported"
    assert not (tmp_path / "old-runs").exists()
    store = AlignmentStore.initialize(tmp_path / "alignment", trace_path=package.root,
                                      adapter_path=EXAMPLE / "int4_numpy.adapter.json",
                                      trace_metadata_digest=trace_metadata_sha256(package.root),
                                      adapter_digest=normalized_adapter_sha256(engine))
    plan = plan_alignment(package, adapter=engine)
    runner = AlignmentRunner(store, package, engine)
    run = runner.run(limit=1, engine_revision="w4a16-test")
    assert run[0].status == "pass"
    assert run[0].boundary == plan.candidates[0].boundary
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"format": "inferref-suite", "format_version": "0.1", "name": "w4",
                                "cases": [{"id": "linear", "testcase": "case", "tags": []}]}), encoding="utf-8")
    assert run_suite(suite, engine, tmp_path / "suite-runs")["counts"]["pass"] == 1


def test_w4a16_input_type_and_overflow(tmp_path):
    spec = QuantizationSpec(logical_shape=(1, 1), scale_value_id=10, scheme="affine-int4/v1", storage_dtype="int4")
    raw = view(np.array([[7]], dtype=np.int8), "int4", tmp_path, "raw")
    params = {10: view(np.array(1, dtype=np.float32), "float32", tmp_path, "scale")}
    with pytest.raises(QuantizationError, match="float16"):
        linear_int4_reference(np.array([1], dtype=np.float32), raw, spec, values=params)
    with pytest.raises(QuantizationError, match="overflow"):
        linear_int4_reference(np.array([65504], dtype=np.float16), raw, spec, values=params)


def test_four_bit_parameter_corruption_does_not_replace_existing_case(tmp_path):
    package = runpy.run_path(str(EXAMPLE / "int8_trace.py"))["build_trace"](tmp_path / "trace", packed=True)
    root = extract_operator(package, 0, tmp_path / "case").path
    original = (root / "testcase.json").read_bytes()
    package.graph.value(1).quantization = replace(package.graph.value(1).quantization, scale_value_id=999)
    with pytest.raises(ValueError):
        extract_operator(package, 0, root, force=True)
    assert (root / "testcase.json").read_bytes() == original
