from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inferref.agent.executor import execute_adapter
from inferref.agent.protocol import EngineAdapter
from inferref.alignment.hashing import (
    normalized_adapter_sha256,
    trace_metadata_sha256,
    tree_sha256,
)
from inferref.alignment.planner import plan_alignment
from inferref.alignment.runner import AlignmentRunner
from inferref.alignment.store import AlignmentStore
from inferref.compare.compare import compare_traces
from inferref.ir.validate import validate_package
from inferref.quantization import (
    QuantizationError,
    QuantizationSpec,
    dequantize_affine,
    dequantize_nf4,
    import_bitsandbytes_nf4,
    quantization_fingerprint,
)
from inferref.quantization.nf4 import BITSANDBYTES_PROFILE
from inferref.suite import run_suite
from inferref.tensor import codec
from inferref.testcase.extract import extract_operator
from inferref.testcase.validate import validate_testcase

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/quantization"
# Independent normative vector from bitsandbytes 0.48.2 get_4bit_type("nf4").
# Keep separate from the production table so accidental codebook drift fails.
EXPECTED_NF4 = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]


def state(shape=(16,)):
    count = int(np.prod(shape))
    return {"shape": shape, "dtype": "float16", "quant_type": "nf4", "blocksize": 64,
            "absmax": np.ones((count + 63) // 64, dtype=np.float32),
            "code": np.array(EXPECTED_NF4, dtype=np.float32),
            "nested": False, "offset": None, "state2": None}


def imported(raw, qs):
    return import_bitsandbytes_nf4(raw, qs, profile=BITSANDBYTES_PROFILE, scale_value_id=10, codebook_value_id=11)


def test_all_codebook_entries_and_source_nibble_order():
    # Source high nibble first; canonical wire reverses each pair.
    raw = np.array([0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef], dtype=np.uint8)
    item = imported(raw, state())
    assert item.codes.data.tolist() == list(range(16))
    assert item.codes.payload_bytes() == bytes([0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe])
    assert np.array_equal(dequantize_nf4(item.codes, item.spec, values=item.parameters), np.asarray(EXPECTED_NF4, dtype=np.float32))
    assert QuantizationSpec.from_dict(item.spec.to_dict()) == item.spec
    assert item.spec.dependencies == (10, 11)
    with pytest.raises(QuantizationError, match="not affine"):
        dequantize_affine(item.codes, item.spec, values=item.parameters)


@pytest.mark.parametrize("shape", [(), (0,), (3,), (65,), (3, 23)])
def test_blocks_tails_zero_scale_and_scalar(shape):
    qs = state(shape)
    count = int(np.prod(shape))
    # Every used nibble is code 15 -> +1; importer discards unspecified tail.
    raw = np.full((count + 1) // 2, 255, dtype=np.uint8)
    qs["absmax"] = np.arange(qs["absmax"].size, dtype=np.float32)
    item = imported(raw, qs)
    output = dequantize_nf4(item.codes, item.spec, values=item.parameters)
    assert np.array_equal(output, np.array([i // 64 for i in range(count)], dtype=np.float32).reshape(shape))
    if count % 2:
        assert item.codes.payload_bytes()[-1] == 15


@pytest.mark.parametrize("change", [
    {"nested": True}, {"state2": {}}, {"offset": 0.0}, {"quant_type": "fp4"},
    {"dtype": "bfloat16"}, {"blocksize": 128}, {"blocksize": True}, {"unexpected": 1},
    {"shape": [True]}, {"code": np.arange(16, dtype=np.float32)},
    {"absmax": np.array([-1], dtype=np.float32)}, {"absmax": np.array([np.nan], dtype=np.float32)},
    {"absmax": np.ones(1, dtype=np.float16)}, {"absmax": np.ones(2, dtype=np.float32)},
])
def test_importer_fails_closed(change):
    qs = state()
    qs.update(change)
    with pytest.raises(QuantizationError):
        imported(np.zeros(8, dtype=np.uint8), qs)


def test_importer_requires_explicit_complete_profile():
    raw = np.zeros(8, dtype=np.uint8)
    with pytest.raises(QuantizationError):
        import_bitsandbytes_nf4(raw, state(), profile="auto", scale_value_id=10, codebook_value_id=11)
    qs = state()
    del qs["code"]
    with pytest.raises(QuantizationError):
        imported(raw, qs)
    with pytest.raises(QuantizationError):
        imported(raw.astype(np.float32), state())
    with pytest.raises(QuantizationError):
        imported(raw[:-1], state())


def test_nf4_artifact_execution_and_evidence(tmp_path):
    package = runpy.run_path(str(EXAMPLE / "nf4_trace.py"))["build_trace"](tmp_path / "trace")
    assert not [i for i in validate_package(package) if i.severity == "error"]
    root = extract_operator(package, 0, tmp_path / "case", input_names=("x", "weight"), output_names=("y",),
                            contracts=("linear/nf4-w4a16/v1",)).path
    result = validate_testcase(root)
    assert result.valid, result.to_dict()
    assert {item["value_id"] for item in result.manifest["quantization_parameters"]} == {10, 11}
    engine = EngineAdapter.load(EXAMPLE / "nf4_numpy.adapter.json")
    assert execute_adapter(root, engine, tmp_path / "runs")["status"] == "pass"
    assert execute_adapter(root, replace(engine, command=(*engine.command, "--inject-bug")), tmp_path / "runs")["status"] == "mismatch"
    affine = EngineAdapter.load(EXAMPLE / "int4_numpy.adapter.json")
    assert execute_adapter(root, affine, tmp_path / "rejected")["status"] == "unsupported"
    assert not (tmp_path / "rejected").exists()
    plan = plan_alignment(package, adapter=engine)
    store = AlignmentStore.initialize(tmp_path / "alignment", trace_path=package.root,
                                      adapter_path=EXAMPLE / "nf4_numpy.adapter.json",
                                      trace_metadata_digest=trace_metadata_sha256(package.root),
                                      adapter_digest=normalized_adapter_sha256(engine))
    results = AlignmentRunner(store, package, engine).run(limit=1, engine_revision="nf4-test")
    assert results[0].status == "pass"
    assert results[0].boundary == plan.candidates[0].boundary
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"format": "inferref-suite", "format_version": "0.1", "name": "nf4",
                                "cases": [{"id": "linear", "testcase": "case", "tags": []}]}), encoding="utf-8")
    assert run_suite(suite, engine, tmp_path / "suite-runs")["counts"]["pass"] == 1
    first = tree_sha256(root)
    codec.write_array(root / "quantization_parameters/v11.irtensor", np.zeros(16, dtype=np.float32))
    assert tree_sha256(root) != first
    assert not validate_testcase(root).valid


def test_nf4_comparison_uses_each_side_scale(tmp_path):
    package = runpy.run_path(str(EXAMPLE / "nf4_trace.py"))["build_trace"](tmp_path / "trace")
    package.graph.operators[0].result = package.graph.operators[0].positional_args[1]
    package.save(package.root)
    original = package.root
    package.save(tmp_path / "actual")
    codec.write_array(package.root / "tensors/v10.irtensor", np.full(2, 2, dtype=np.float32))
    assert compare_traces(original, package.root).status == "fail"
    assert compare_traces(original, package.root, quantization_mode="raw_codes").status == "pass"
    codes = codec.read(package.root / "tensors/v1.irtensor")
    spec = package.graph.value(1).quantization
    parameters = {i: codec.read(package.root / f"tensors/v{i}.irtensor") for i in spec.dependencies}
    assert len(quantization_fingerprint(codes, spec, values=parameters)) == 64


def test_nf4_schema_never_infers_codebook_or_nested_parameters():
    spec = imported(np.zeros(8, dtype=np.uint8), state()).spec
    for changes in ({"codebook": "unknown"}, {"codebook_value_id": 10}, {"granularity": "per_group"},
                    {"zero_point_value_id": 12}, {"rounding": "nearest_even"}, {"axis": 0}):
        with pytest.raises(QuantizationError):
            replace(spec, **changes)


def test_dedup_does_not_merge_nf4_and_affine_uint4(tmp_path):
    from inferref.testcase.dedup import operator_signature

    package = runpy.run_path(str(EXAMPLE / "nf4_trace.py"))["build_trace"](tmp_path / "trace")
    first = operator_signature(package.graph, package.graph.op(0))
    package.graph.value(1).quantization = QuantizationSpec(
        logical_shape=(3, 23), scale_value_id=10, scheme="affine-uint4/v1", storage_dtype="uint4")
    assert operator_signature(package.graph, package.graph.op(0)) != first
