"""Quantized artifacts must work end-to-end and fail closed at every boundary."""

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
    HashingError,
    normalized_adapter_sha256,
    trace_metadata_sha256,
    tree_sha256,
)
from inferref.alignment.planner import boundary_requirements, plan_alignment
from inferref.alignment.runner import AlignmentRunner
from inferref.alignment.store import AlignmentStore
from inferref.cli.main import main
from inferref.compare.compare import compare_testcase, compare_traces
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.quantization import QuantizationError
from inferref.suite import run_suite
from inferref.tensor import codec
from inferref.testcase.extract import extract_operator
from inferref.testcase.validate import validate_testcase

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/quantization"


def trace(tmp_path):
    return runpy.run_path(str(EXAMPLE / "int8_trace.py"))["build_trace"](tmp_path / "trace")


def case(package, root, *, contract=False):
    return extract_operator(package, 0, root, input_names=("x", "weight"), output_names=("y",),
                            contracts=("linear/affine-int8/v1",) if contract else None)


def adapter():
    return EngineAdapter.load(EXAMPLE / "int8_numpy.adapter.json")


def edit_json(path, change):
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_trace_and_standalone_parameter_roundtrip(tmp_path):
    package = trace(tmp_path)
    assert not [issue for issue in validate_package(package) if issue.severity == "error"]
    loaded = TracePackage.load(package.root)
    assert loaded.graph.value(1).quantization == package.graph.value(1).quantization
    extracted = case(loaded, tmp_path / "case", contract=True)
    assert extracted.reproducible
    result = validate_testcase(extracted.path)
    assert result.valid, result.to_dict()
    manifest = result.manifest
    assert manifest["format_version"] == "0.4"
    assert [item["name"] for item in manifest["inputs"]] == ["x", "weight"]
    assert [item["value_id"] for item in manifest["quantization_parameters"]] == [10]
    assert {item["id"] for item in manifest["values"]} == {0, 1, 2, 10}
    assert manifest["requirements"]["features"] == ["quantization/affine-int8/v1/per_group"]


def test_planner_validates_parameters_without_materializing_weights(tmp_path, monkeypatch):
    import inferref.quantization.trace as quantized_trace

    package = trace(tmp_path)
    original = quantized_trace.trace_tensor
    reads = []

    def parameter_only(package, value_id, **kwargs):
        assert value_id == 10, "planning must not load/dequantize the full weight"
        reads.append(value_id)
        return original(package, value_id, **kwargs)

    monkeypatch.setattr(quantized_trace, "trace_tensor", parameter_only)
    plan = plan_alignment(package, adapter=adapter())
    assert reads
    assert plan.candidates
    assert all(c.runnability.eligible for c in plan.candidates)


@pytest.mark.parametrize("version", ["0.1", "0.9", "1.1", "2.0"])
def test_trace_version_gate(tmp_path, version):
    package = trace(tmp_path)
    package.manifest.format_version = version
    with pytest.raises(ValueError):
        package.save(tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    edit_json(package.root / "manifest.json", lambda m: m.update(format_version=version))
    with pytest.raises(ValueError):
        TracePackage.load(package.root)


@pytest.mark.parametrize("mutation", ["missing", "cycle", "duplicate", "shape", "null"])
def test_invalid_graph_dependencies_rejected_before_copy(tmp_path, mutation):
    package = trace(tmp_path)
    record = package.graph.value(1)
    if mutation == "missing":
        record.quantization = replace(record.quantization, scale_value_id=99)
    elif mutation == "cycle":
        record.quantization = replace(record.quantization, scale_value_id=1)
    elif mutation == "duplicate":
        package.graph.values.append(package.graph.value(10))
    elif mutation == "shape":
        package.graph.value(10).shape = (1,)
    else:
        edit_json(package.root / "graph.json", lambda g: g["values"][1].update(quantization=None))
        with pytest.raises(QuantizationError):
            TracePackage.load(package.root)
        return
    with pytest.raises(QuantizationError):
        case(package, tmp_path / "case")
    assert not (tmp_path / "case").exists()


@pytest.mark.parametrize("mutation", ["version", "descriptor", "duplicate", "missing", "negative", "escape", "shape"])
def test_invalid_testcase_cannot_launch(tmp_path, monkeypatch, mutation):
    package = trace(tmp_path)
    root = case(package, tmp_path / "case").path
    path = root / "testcase.json"
    if mutation == "negative":
        codec.write_array(root / "quantization_parameters/v10.irtensor", np.full((2, 2), -1, dtype=np.float32))
    else:
        def corrupt(m):
            if mutation == "version":
                m["format_version"] = "0.3"
            elif mutation == "descriptor":
                m["inputs"][1]["quantization"]["group_size"] = 3
            elif mutation == "duplicate":
                m["quantization_parameters"] *= 2
            elif mutation == "missing":
                m["quantization_parameters"] = []
            elif mutation == "escape":
                m["quantization_parameters"][0]["payload"] = "../outside.irtensor"
            else:
                m["quantization_parameters"][0]["shape"] = [4]
        edit_json(path, corrupt)
    assert not validate_testcase(root).valid
    monkeypatch.setattr("inferref.agent.executor.subprocess.Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError):
        execute_adapter(root, adapter(), tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("legacy", [True, False])
def test_undeclared_capabilities_fail_closed(tmp_path, monkeypatch, legacy):
    package = trace(tmp_path)
    root = case(package, tmp_path / "case").path
    engine = adapter()
    engine = replace(engine, capabilities=None if legacy else replace(engine.capabilities, features=()))
    monkeypatch.setattr("inferref.agent.executor.subprocess.Popen", lambda *a, **k: pytest.fail("must not launch"))
    result = execute_adapter(root, engine, tmp_path / "runs")
    assert result["status"] == "unsupported"
    assert not (tmp_path / "runs").exists()
    plan = plan_alignment(package, adapter=engine, include_unsupported=True)
    assert all(c.runnability.status == "unsupported" for c in plan.candidates)


def test_adapter_pass_and_injected_bug(tmp_path):
    package = trace(tmp_path)
    root = case(package, tmp_path / "case", contract=True).path
    result = execute_adapter(root, adapter(), tmp_path / "runs")
    assert result["status"] == "pass", result
    engine = adapter()
    result = execute_adapter(root, replace(engine, command=(*engine.command, "--inject-bug")), tmp_path / "runs")
    assert result["status"] == "mismatch", result
    assert result["comparison"]["first_failure"]["operator"] == "inferref.linear_affine_int8.default"


def test_trace_hash_and_comparison_bind_own_parameters(tmp_path):
    package = trace(tmp_path)
    # Expose weight as an op output to exercise trace comparison of q-values.
    package.graph.operators[0].result = package.graph.operators[0].positional_args[1]
    package.save(package.root)
    package.save(tmp_path / "actual")
    reference = tmp_path / "trace"
    before = trace_metadata_sha256(package.root)
    codec.write_array(package.root / "tensors/v10.irtensor", np.full((2, 2), 2, dtype=np.float32))
    assert trace_metadata_sha256(package.root) != before
    report = compare_traces(reference, package.root)
    assert report.status == "fail"
    assert report.first_failure.comparison_domain == "dequantized"
    raw = compare_traces(reference, package.root, quantization_mode="raw_codes")
    assert raw.status == "pass"
    assert raw.comparisons[0].to_dict()["comparison_domain"] == "raw_codes"
    codec.write_array(package.root / "tensors/v10.irtensor", np.full((2, 2), -1, dtype=np.float32))
    assert compare_traces(reference, package.root).status == "error"


def test_testcase_hash_binds_scale_bytes(tmp_path):
    root = case(trace(tmp_path), tmp_path / "case").path
    before = tree_sha256(root)
    codec.write_array(root / "quantization_parameters/v10.irtensor", np.full((2, 2), 2, dtype=np.float32))
    assert tree_sha256(root) != before


def test_quantized_output_requires_float32_engine_output(tmp_path):
    package = trace(tmp_path)
    package.graph.operators[0].result = package.graph.operators[0].positional_args[1]
    package.graph.value(2).producer = None
    package.graph.value(1).producer = 0
    package.graph.reindex()
    root = case(package, tmp_path / "case").path
    output = tmp_path / "engine"
    codec.write_array(output / "y.irtensor", np.array([[1, 2, 1.5], [-2, 0, 4]], dtype=np.float32))
    report = compare_testcase(root, output)
    assert report.status == "pass", report.to_dict()
    codec.write_array(output / "y.irtensor", np.array([[2, 4, 6], [-2, 0, 2]], dtype=np.int8))
    assert compare_testcase(root, output).status == "fail"


def test_alignment_plan_run_and_stale_scale(tmp_path):
    package = trace(tmp_path)
    engine = adapter()
    store = AlignmentStore.initialize(tmp_path / "alignment", trace_path=package.root,
                                      adapter_path=EXAMPLE / "int8_numpy.adapter.json",
                                      trace_metadata_digest=trace_metadata_sha256(package.root),
                                      adapter_digest=normalized_adapter_sha256(engine))
    runner = AlignmentRunner(store, package, engine)
    plan = plan_alignment(package, session=store.snapshot, adapter=engine)
    candidate = plan.candidates[0]
    root = case(package, tmp_path / "case").path
    assert boundary_requirements(package, runner.inventory.resolve(candidate.boundary)) == validate_testcase(root).manifest["requirements"]
    result = runner.run(limit=1, engine_revision="int8-test")
    assert result[0].status == "pass", result
    assert result[0].boundary == candidate.boundary
    codec.write_array(package.root / "tensors/v10.irtensor", np.full((2, 2), 2, dtype=np.float32))
    with pytest.raises(ValueError, match="digest mismatch"):
        AlignmentRunner(store, package, engine)
    with pytest.raises(ValueError, match="digest mismatch"):
        runner.run_candidate(candidate.boundary, engine_revision="int8-test")


def test_suite_and_agent_cli_smoke(tmp_path, capsys):
    package = trace(tmp_path)
    root = case(package, tmp_path / "case", contract=True).path
    assert main(["agent", "context", str(package.root), "--json"]) == 0
    assert main(["agent", "run", str(root), "--adapter", str(EXAMPLE / "int8_numpy.adapter.json"),
                 "--runs-dir", str(tmp_path / "agent-runs"), "--json"]) == 0
    capsys.readouterr()
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"format": "inferref-suite", "format_version": "0.1", "name": "int8",
                                "cases": [{"id": "linear", "testcase": "case", "tags": ["quantization"]}]}), encoding="utf-8")
    result = run_suite(suite, adapter(), tmp_path / "suite-runs")
    assert result["counts"]["pass"] == 1, result


def test_relocated_trace_identity_and_parameter_bytes(tmp_path):
    package = trace(tmp_path)
    original = trace_metadata_sha256(package.root)
    package.save(tmp_path / "relocated")
    assert trace_metadata_sha256(package.root) == original
    root = case(package, tmp_path / "case").path
    assert execute_adapter(root, adapter(), tmp_path / "runs")["status"] == "pass"
    codec.write_array(package.root / "tensors/v1.irtensor", np.ones((2, 3), dtype=np.int8))
    assert trace_metadata_sha256(package.root) != original


def test_quantization_payload_symlink_is_not_hashable(tmp_path):
    package = trace(tmp_path)
    target = package.root / "tensors/v10.irtensor"
    link = package.root / "tensors/linked.irtensor"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    package.graph.value(10).capture.payload = "tensors/linked.irtensor"
    package.save(package.root)
    with pytest.raises(HashingError, match="linked quantization"):
        trace_metadata_sha256(package.root)


def test_missing_parameter_payload_blocks_execution(tmp_path, monkeypatch):
    package = trace(tmp_path)
    package.graph.value(10).capture.mode = "metadata"
    package.graph.value(10).capture.payload = None
    extracted = case(package, tmp_path / "case")
    assert not extracted.reproducible
    assert extracted.missing_payloads == [10]
    monkeypatch.setattr("inferref.agent.executor.subprocess.Popen", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError):
        execute_adapter(extracted.path, adapter(), tmp_path / "runs")
    plan = plan_alignment(package, adapter=adapter())
    assert all(c.runnability.status == "blocked" for c in plan.candidates)
    assert all(not c.runnability.eligible for c in plan.candidates)


def test_trace_cli_comparison_domains(tmp_path, capsys):
    package = trace(tmp_path)
    package.graph.operators[0].result = package.graph.operators[0].positional_args[1]
    package.save(package.root)
    original = package.root
    package.save(tmp_path / "actual")
    codec.write_array(package.root / "tensors/v10.irtensor", np.ones((2, 2), dtype=np.float32))
    assert main(["compare", str(original), str(package.root), "--json"]) != 0
    report = json.loads(capsys.readouterr().out)
    assert report["first_failure"]["comparison_domain"] == "dequantized"
    assert main(["compare", str(original), str(package.root), "--quantization-mode", "raw_codes", "--json"]) == 0


def test_engine_manifest_cannot_hide_quantization(tmp_path):
    root = case(trace(tmp_path), tmp_path / "case").path
    output = tmp_path / "engine"
    codec.write_array(output / "y.irtensor", np.array([[9.5, 10]], dtype=np.float32))
    (output / "manifest.json").write_text(json.dumps({"outputs": [
        {"name": "y", "payload": "y.irtensor", "quantization": {}}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="trace-vs-trace"):
        compare_testcase(root, output)
