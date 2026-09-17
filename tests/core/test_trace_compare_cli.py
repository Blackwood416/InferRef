"""Torch-free smoke coverage of both trace comparison entry points."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inferref.cli.main import EXIT_FAIL, EXIT_OK, EXIT_USAGE, main
from inferref.compare.compare import compare_traces
from inferref.compare.tolerance import TolerancePolicy
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef
from inferref.tensor.codec import write_array


def _trace(root: Path, *, delta: float = 0, missing: bool = False) -> Path:
    graph = Graph()
    for index in range(2):
        payload = f"payloads/v{index}.irtensor"
        write_array(root / payload, np.array([index + delta], dtype=np.float32))
        graph.values.append(TensorValueRecord(
            id=index, shape=(1,), stride=(1,), dtype="float32", producer=index,
            capture=CaptureInfo(mode="full", payload=payload),
        ))
        if not missing or index != 0:
            graph.operators.append(OperatorRecord(
                id=index, execution_index=index, op="identity", result=TensorRef(index),
            ))
    graph.reindex()
    TracePackage(manifest=Manifest(), graph=graph, root=root).save(root)
    return root


@pytest.mark.parametrize("json_flag", [None, "--json", "--json-summary"])
def test_trace_compare_cli_happy_path(tmp_path: Path, capsys, json_flag) -> None:
    ref, actual = _trace(tmp_path / "ref"), _trace(tmp_path / "actual")
    assert compare_traces(ref, actual).status == "pass"
    args = ["compare", str(ref), str(actual)] + ([json_flag] if json_flag else [])
    assert main(args) == EXIT_OK
    output = capsys.readouterr().out
    if json_flag == "--json":
        assert json.loads(output)["status"] == "pass"
    elif json_flag:
        assert isinstance(json.loads(output), dict)
    else:
        assert "PASS" in output.upper()


def test_trace_first_failure_and_missing_counterpart(tmp_path: Path, capsys) -> None:
    ref = _trace(tmp_path / "ref")
    actual = _trace(tmp_path / "actual", delta=0.25)
    report = compare_traces(ref, actual, first_failure=True)
    assert report.status == "fail"
    assert len(report.comparisons) == 1
    assert report.first_failure.execution_index == 0
    assert len(compare_traces(ref, actual).comparisons) == 2
    assert main(["compare", str(ref), str(actual), "--first-failure", "--json"]) == EXIT_FAIL
    assert json.loads(capsys.readouterr().out)["status"] == "fail"
    missing = _trace(tmp_path / "missing", missing=True)
    assert compare_traces(ref, missing).comparisons[0].status == "missing"


def test_trace_numeric_config_is_applied_and_scalar_flags_win(tmp_path: Path, capsys) -> None:
    ref, actual = _trace(tmp_path / "ref"), _trace(tmp_path / "actual", delta=0.25)
    config = {"atol": 0.5, "rtol": 0}
    assert compare_traces(ref, actual, comparison_config=config).status == "pass"
    assert compare_traces(ref, actual, comparison_config={"per_dtype": {
        "float32": {"atol": 0.5, "rtol": 0},
    }}).status == "pass"
    assert compare_traces(ref, actual, comparison_config=config,
                          policy=TolerancePolicy(override_atol=0)).status == "fail"
    args = ["compare", str(ref), str(actual), "--comparator", "tensor/numeric/v1",
            "--comparison-config", json.dumps(config), "--json"]
    assert main(args) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert main([*args, "--atol", "0"]) == EXIT_FAIL
    assert json.loads(capsys.readouterr().out)["status"] == "fail"


@pytest.mark.parametrize("options", [
    ["--comparator", "custom/plugin/v1"],
    ["--comparison-config", '{"unknown":1}'],
    ["--comparison-config", '{"atol":-1}'],
    ["--comparison-config", '{"strict_layout":"yes"}'],
    ["--comparison-config", "[]"],
])
def test_trace_invalid_options_are_structured_usage_errors(tmp_path: Path, capsys, options) -> None:
    ref = _trace(tmp_path / "ref")
    assert main(["compare", str(ref), str(ref), *options, "--json"]) == EXIT_USAGE
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "error"
    assert "Traceback" not in output.err


def test_compare_missing_input_is_a_structured_usage_error(tmp_path: Path, capsys) -> None:
    assert main(["compare", str(tmp_path / "absent"), str(tmp_path / "actual"), "--json"]) == EXIT_USAGE
    assert json.loads(capsys.readouterr().out)["status"] == "error"
