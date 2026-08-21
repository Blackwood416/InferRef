"""Tests for Comparison Spec v0.1 and Effective Policy Resolution (SPEC §6, §8, §9, §10, §11, §12).

Tests:
1. ComparisonSpec dataclass and schema validation (wire format, unknown comparator rejection, config validation).
2. Conditional format versioning (testcase 0.2 vs 0.3, suite 0.2 vs 0.3, comparison_requires_0_3, Suite.to_dict preservation).
3. Two-Axis tolerance resolution and effective_comparison with fine-grained sources tracking.
4. Pre-flight validation before engine execution.
5. End-to-end propagation through testcase, suite, scenario, agent protocol, and MCP server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inferref.agent.executor import execute_adapter
from inferref.agent.protocol import (
    ENGINE_ADAPTER_FORMAT,
    ENGINE_ADAPTER_VERSION,
    AdapterCapabilities,
    AgentProtocolError,
    EngineAdapter,
)
from inferref.agent.service import capabilities, compare_outputs, run_engine, run_scenario
from inferref.comparators.numeric import NUMERIC_COMPARATOR_ID
from inferref.comparators.protocol import Artifact, ArtifactSet, ComparatorPlugin, ComparatorResult
from inferref.comparators.registry import _reset_registry, register_builtin_comparator
from inferref.compare.compare import compare_testcase
from inferref.compare.tolerance import DEFAULT_TOLERANCES, TolerancePolicy
from inferref.comparison import (
    COMPARISON_SPEC_FORMAT,
    COMPARISON_SPEC_VERSION,
    ComparisonSpec,
    ComparisonSpecValidationError,
    EffectiveComparison,
    OutputComparisonSpec,
    resolve_comparison_policy,
    validate_comparison_spec,
)
from inferref.scenario.run import run_scenario as run_scenario_func
from inferref.scenario.schema import load_scenario
from inferref.suite.run import run_suite
from inferref.suite.schema import Suite, SuiteCase, SuiteError, load_suite, validate_suite
from inferref.tensor import codec
from inferref.testcase.requirements import derive_requirements
from inferref.testcase.validate import validate_testcase


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _reset_registry()
    yield
    _reset_registry()


def _write_irtensor(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    codec.write_array(path, array)
    return path


def _create_minimal_testcase(root: Path, *, format_version: str = "0.2", comparison: dict[str, Any] | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    in_path = _write_irtensor(root / "inputs" / "x.irtensor", np.array([1.0, 2.0], dtype=np.float32))
    ref_path = _write_irtensor(root / "reference" / "y.irtensor", np.array([2.0, 4.0], dtype=np.float32))

    in_meta = codec.read(in_path).to_metadata()
    ref_meta = codec.read(ref_path).to_metadata()

    manifest: dict[str, Any] = {
        "format": "inferref-testcase",
        "format_version": format_version,
        "name": "test_op",
        "origin": {"trace": "trace", "operator_id": 1},
        "reproducible": True,
        "inputs": [{"name": "x", "value_id": 1, "payload": "inputs/x.irtensor", **in_meta}],
        "outputs": [{"name": "y", "value_id": 2, "payload": "reference/y.irtensor", **ref_meta}],
        "nodes": [
            {
                "id": 1,
                "operator": "aten::mul",
                "positional_args": [{"kind": "tensor", "value_id": 1}],
                "keyword_args": {},
                "result": {"kind": "tensor", "value_id": 2},
            }
        ],
        "values": [
            {"id": 1, "storage_id": 11, **in_meta},
            {"id": 2, "storage_id": 12, **ref_meta},
        ],
    }
    if comparison is not None:
        manifest["comparison"] = comparison
    if format_version != "0.1":
        manifest["requirements"] = derive_requirements(manifest)

    (root / "testcase.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _make_adapter(
    root: Path,
    script: Path,
    *,
    command: list[str] | None = None,
) -> Path:
    payload = {
        "format": ENGINE_ADAPTER_FORMAT,
        "format_version": ENGINE_ADAPTER_VERSION,
        "name": "test-engine",
        "command": command or ["{python}", str(script), "{testcase}", "{output}"],
        "timeout_seconds": 30.0,
        "max_output_chars": 65_536,
        "max_artifact_bytes": 1_073_741_824,
        "max_artifact_files": 10_000,
        "target_device": "cpu",
        "capabilities": {
            "device_types": ["cpu"],
            "dtypes": ["float32", "float16", "bfloat16", "int64"],
            "max_rank": 8,
            "features": [
                "multiple_outputs",
                "strided_inputs",
                "alias_effects",
                "mutation_effects",
            ],
        },
    }
    path = root / "adapter.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# -- Custom Test Comparator Plugin --

class MockVisionDetectorComparator:
    id = "vision/object-detection/v1"
    version = "1.0.0"
    description = "Mock Object Detection Comparator"

    def validate_config(self, config: dict[str, Any] | None = None) -> None:
        if config and "min_iou" in config and not (0.0 <= config["min_iou"] <= 1.0):
            raise ValueError("min_iou must be between 0.0 and 1.0")

    def compare(
        self,
        reference: ArtifactSet,
        actual: ArtifactSet,
        config: dict[str, Any] | None = None,
    ) -> ComparatorResult:
        min_iou = (config or {}).get("min_iou", 0.5)
        return ComparatorResult(
            comparator=self.id,
            status="pass",
            metrics={"mean_iou": 0.85, "min_iou_threshold": min_iou},
        )


# -- 1. ComparisonSpec Dataclass & Validation --

def test_comparison_spec_defaults() -> None:
    spec = ComparisonSpec()
    assert spec.format == COMPARISON_SPEC_FORMAT
    assert spec.format_version == COMPARISON_SPEC_VERSION
    assert spec.comparator is None
    assert spec.to_dict()["comparator"] == NUMERIC_COMPARATOR_ID
    assert spec.config == {}
    assert spec.outputs == {}
    assert spec.tolerances is None
    assert spec.tensor_layout is None
    assert spec.output_roles == []


def test_comparison_spec_wire_roundtrip() -> None:
    wire = {
        "format": "inferref-comparison",
        "format_version": "0.1",
        "comparator": "tensor/numeric/v1",
        "config": {
            "atol": 0.001,
            "rtol": 0.01,
            "strict_layout": True,
            "ignore_stride": False,
            "per_dtype": {"float32": {"atol": 1e-4, "rtol": 1e-4}},
        },
        "outputs": {
            "logits": {
                "comparator": "tensor/numeric/v1",
                "config": {"atol": 1e-5},
            }
        },
    }
    spec = ComparisonSpec.from_dict(wire)
    assert spec.format == "inferref-comparison"
    assert spec.format_version == "0.1"
    assert spec.comparator == "tensor/numeric/v1"
    assert spec.config["atol"] == 0.001
    assert spec.config["strict_layout"] is True
    assert "logits" in spec.outputs
    assert spec.outputs["logits"].config["atol"] == 1e-5

    dumped = spec.to_dict()
    assert dumped == wire


def test_comparison_spec_rejects_unknown_comparator() -> None:
    spec = ComparisonSpec(comparator="unknown/nonexistent/v1")
    with pytest.raises(ComparisonSpecValidationError, match="unknown comparator 'unknown/nonexistent/v1'"):
        spec.validate(check_registry=True)


def test_comparison_spec_validates_config_before_engine() -> None:
    register_builtin_comparator(MockVisionDetectorComparator())

    # Valid config
    valid_spec = ComparisonSpec(
        comparator="vision/object-detection/v1",
        config={"min_iou": 0.7},
    )
    valid_spec.validate(check_registry=True)

    # Invalid config (min_iou out of bounds)
    invalid_spec = ComparisonSpec(
        comparator="vision/object-detection/v1",
        config={"min_iou": 1.5},
    )
    with pytest.raises(ComparisonSpecValidationError, match="invalid_comparison_config: min_iou must be between 0.0 and 1.0"):
        invalid_spec.validate(check_registry=True)


def test_comparison_spec_validates_per_output_comparator_and_config() -> None:
    register_builtin_comparator(MockVisionDetectorComparator())

    # Invalid per-output comparator
    spec_bad_role_comp = ComparisonSpec(
        comparator="tensor/numeric/v1",
        outputs={"boxes": OutputComparisonSpec(comparator="fake/comparator/v1")},
    )
    with pytest.raises(ComparisonSpecValidationError, match="unknown comparator 'fake/comparator/v1' for output role 'boxes'"):
        spec_bad_role_comp.validate(check_registry=True)

    # Invalid per-output config
    spec_bad_role_cfg = ComparisonSpec(
        comparator="tensor/numeric/v1",
        outputs={"boxes": OutputComparisonSpec(comparator="vision/object-detection/v1", config={"min_iou": -0.5})},
    )
    with pytest.raises(ComparisonSpecValidationError, match="invalid_comparison_config for output role 'boxes'"):
        spec_bad_role_cfg.validate(check_registry=True)


# -- 2. Conditional Format Versioning --

def test_testcase_validation_rejects_comparison_in_v0_2(tmp_path: Path) -> None:
    tc_dir = tmp_path / "tc_v02_with_comp"
    _create_minimal_testcase(
        tc_dir,
        format_version="0.2",
        comparison={"format": "inferref-comparison", "format_version": "0.1", "comparator": "tensor/numeric/v1"},
    )
    val = validate_testcase(tc_dir)
    assert not val.valid
    errors = [issue for issue in val.errors if issue.code == "comparison_requires_0_3"]
    assert len(errors) == 1
    assert "comparison requires format_version 0.3" in errors[0].message


def test_testcase_validation_accepts_comparison_in_v0_3(tmp_path: Path) -> None:
    tc_dir = tmp_path / "tc_v03_with_comp"
    _create_minimal_testcase(
        tc_dir,
        format_version="0.3",
        comparison={"format": "inferref-comparison", "format_version": "0.1", "comparator": "tensor/numeric/v1"},
    )
    val = validate_testcase(tc_dir)
    assert val.valid


def test_testcase_validation_accepts_v0_3_without_comparison(tmp_path: Path) -> None:
    tc_dir = tmp_path / "tc_v03_no_comp"
    _create_minimal_testcase(tc_dir, format_version="0.3", comparison=None)
    val = validate_testcase(tc_dir)
    assert val.valid


def test_suite_format_version_preservation_and_comparison_versioning(tmp_path: Path) -> None:
    tc_dir = tmp_path / "tc"
    _create_minimal_testcase(tc_dir, format_version="0.3", comparison={"comparator": "tensor/numeric/v1"})

    # Suite v0.2 with comparison in case must fail load_suite
    suite_file_v02 = tmp_path / "suite_v02.json"
    suite_v02_data = {
        "format": "inferref-suite",
        "format_version": "0.2",
        "name": "v02_suite",
        "cases": [
            {
                "id": "case_1",
                "testcase": "tc",
                "comparison": {"comparator": "tensor/numeric/v1"},
            }
        ],
    }
    suite_file_v02.write_text(json.dumps(suite_v02_data), encoding="utf-8")
    with pytest.raises(SuiteError, match="comparison_requires_0_3"):
        load_suite(suite_file_v02)

    # Suite v0.3 with comparison succeeds
    suite_file_v03 = tmp_path / "suite_v03.json"
    suite_v03_data = {
        "format": "inferref-suite",
        "format_version": "0.3",
        "name": "v03_suite",
        "cases": [
            {
                "id": "case_1",
                "testcase": "tc",
                "comparison": {"comparator": "tensor/numeric/v1", "config": {"atol": 1e-4}},
            }
        ],
    }
    suite_file_v03.write_text(json.dumps(suite_v03_data), encoding="utf-8")
    suite = load_suite(suite_file_v03)
    assert suite.format_version == "0.3"
    assert suite.cases[0].comparison is not None
    assert suite.cases[0].comparison.config["atol"] == 1e-4

    # Suite.to_dict preserves source format_version
    dict_repr = suite.to_dict()
    assert dict_repr["format_version"] == "0.3"
    assert dict_repr["cases"][0]["comparison"]["config"]["atol"] == 1e-4


# -- 3. Two-Axis Tolerance Resolution & Effective Comparison Generator --

def test_two_axis_resolution_precedence() -> None:
    # Testcase spec: defines per_dtype (Axis A) and atol (Axis B)
    tc_spec = ComparisonSpec(
        comparator="tensor/numeric/v1",
        config={
            "atol": 0.01,
            "rtol": 0.02,
            "strict_layout": False,
            "per_dtype": {"float32": {"atol": 1e-4, "rtol": 1e-4}},
        },
    )

    # Suite spec: overrides atol (Axis B) and per_dtype (Axis A)
    suite_spec = ComparisonSpec(
        comparator="tensor/numeric/v1",
        config={
            "atol": 0.005,
            "strict_layout": True,
            "per_dtype": {"float32": {"atol": 1e-5, "rtol": 1e-5}},
        },
    )

    # CLI overrides: CLI atol overrides suite and testcase
    effective = resolve_comparison_policy(
        testcase_spec=tc_spec,
        suite_spec=suite_spec,
        cli_atol=0.001,
    )

    # Axis A: suite per_dtype replaces testcase per_dtype
    assert effective.config["per_dtype"]["float32"]["atol"] == 1e-5
    assert effective.sources["config.per_dtype"] == "suite"

    # Axis B: CLI atol overrides suite atol and testcase atol
    assert effective.config["atol"] == 0.001
    assert effective.sources["config.atol"] == "cli"

    # rtol comes from testcase (suite didn't specify rtol, CLI didn't override)
    assert effective.config["rtol"] == 0.02
    assert effective.sources["config.rtol"] == "testcase"

    # strict_layout comes from suite
    assert effective.config["strict_layout"] is True
    assert effective.sources["config.strict_layout"] == "suite"

    # ignore_stride defaults to False
    assert effective.config["ignore_stride"] is False
    assert effective.sources["config.ignore_stride"] == "default"


def test_two_axis_resolution_fine_grained_sources(tmp_path: Path) -> None:
    # Custom CLI tolerance file
    tol_file = tmp_path / "custom_tol.json"
    tol_file.write_text(json.dumps({"float16": {"atol": 0.005, "rtol": 0.005}}), encoding="utf-8")

    tc_spec = ComparisonSpec(
        comparator="tensor/numeric/v1",
        config={"rtol": 0.05},
    )

    effective = resolve_comparison_policy(
        testcase_spec=tc_spec,
        cli_tolerance=tol_file,
        cli_atol=0.002,
    )

    assert effective.sources["comparator"] == "testcase"
    assert effective.sources["config.per_dtype"] == "cli"
    assert effective.sources["config.atol"] == "cli"
    assert effective.sources["config.rtol"] == "testcase"
    assert effective.sources["config.strict_layout"] == "default"
    assert effective.sources["config.ignore_stride"] == "default"


def test_per_output_resolution_and_sources() -> None:
    tc_spec = ComparisonSpec(
        comparator="tensor/numeric/v1",
        outputs={
            "boxes": OutputComparisonSpec(comparator="vision/object-detection/v1", config={"min_iou": 0.5}),
            "scores": OutputComparisonSpec(config={"atol": 1e-3}),
        },
    )
    suite_spec = ComparisonSpec(
        outputs={
            "boxes": OutputComparisonSpec(config={"min_iou": 0.75}),
        }
    )

    effective = resolve_comparison_policy(
        testcase_spec=tc_spec,
        suite_spec=suite_spec,
    )

    assert "boxes" in effective.per_output
    assert effective.per_output["boxes"]["comparator"] == "vision/object-detection/v1"
    assert effective.per_output["boxes"]["config"]["min_iou"] == 0.75
    assert effective.per_output["boxes"]["sources"]["comparator"] == "testcase"
    assert effective.per_output["boxes"]["sources"]["config.min_iou"] == "suite"

    assert "scores" in effective.per_output
    assert effective.per_output["scores"]["comparator"] == "tensor/numeric/v1"
    assert effective.per_output["scores"]["config"]["atol"] == 1e-3
    assert effective.per_output["scores"]["sources"]["config.atol"] == "testcase"


# -- 4. Pre-Flight Validation Before Engine Launch --

def test_preflight_validation_rejects_invalid_policy_before_engine(tmp_path: Path) -> None:
    tc_dir = tmp_path / "tc_valid"
    _create_minimal_testcase(tc_dir, format_version="0.3")

    script = tmp_path / "engine.py"
    script.write_text("import sys; sys.exit(0)", encoding="utf-8")
    adapter_path = _make_adapter(tmp_path, script)
    adapter = EngineAdapter.load(adapter_path)

    runs_dir = tmp_path / "runs"
    with pytest.raises(AgentProtocolError, match="invalid comparison policy: unknown comparator 'nonexistent/comparator/v1'"):
        execute_adapter(tc_dir, adapter, runs_dir, comparator="nonexistent/comparator/v1")


# -- 5. Pipeline Wiring & End-to-End Propagation --

def test_run_engine_records_effective_comparison(tmp_path: Path) -> None:
    tc_dir = tmp_path / "tc"
    _create_minimal_testcase(
        tc_dir,
        format_version="0.3",
        comparison={
            "format": "inferref-comparison",
            "format_version": "0.1",
            "comparator": "tensor/numeric/v1",
            "config": {"atol": 0.05, "rtol": 0.05},
        },
    )

    adapter_script = tmp_path / "mock_engine.py"
    adapter_script.write_text(
        """
import shutil, sys
from pathlib import Path
tc = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
shutil.copyfile(tc / "reference" / "y.irtensor", out / "y.irtensor")
""",
        encoding="utf-8",
    )

    adapter_file = _make_adapter(tmp_path, adapter_script)

    runs_dir = tmp_path / "runs"
    response = run_engine(
        testcase=tc_dir,
        adapter_path=adapter_file,
        runs_root=runs_dir,
        atol=0.001,
    )

    assert response.status == "pass"
    data = response.data
    assert "effective_comparison" in data
    eff = data["effective_comparison"]
    assert eff["comparator"] == "tensor/numeric/v1"
    assert eff["config"]["atol"] == 0.001
    assert eff["sources"]["config.atol"] == "cli"
    assert eff["sources"]["config.rtol"] == "testcase"

    # Verify inferref-run.json contains effective_comparison
    run_record_file = Path(data["output"]) / "inferref-run.json"
    assert run_record_file.is_file()
    saved_record = json.loads(run_record_file.read_text(encoding="utf-8"))
    assert "effective_comparison" in saved_record
    assert saved_record["effective_comparison"]["config"]["atol"] == 0.001


def test_suite_run_propagates_comparison(tmp_path: Path) -> None:
    tc_dir = tmp_path / "tc"
    _create_minimal_testcase(tc_dir, format_version="0.3")

    adapter_script = tmp_path / "engine.py"
    adapter_script.write_text(
        """
import shutil, sys
from pathlib import Path
tc = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
shutil.copyfile(tc / "reference" / "y.irtensor", out / "y.irtensor")
""",
        encoding="utf-8",
    )
    adapter_file = _make_adapter(tmp_path, adapter_script)

    suite_file = tmp_path / "suite.json"
    suite_file.write_text(
        json.dumps(
            {
                "format": "inferref-suite",
                "format_version": "0.3",
                "name": "prop_suite",
                "cases": [
                    {
                        "id": "c1",
                        "testcase": "tc",
                        "comparison": {
                            "comparator": "tensor/numeric/v1",
                            "config": {"atol": 0.004, "rtol": 0.004},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    runs_dir = tmp_path / "suite_runs"
    report = run_suite(suite_file, adapter_file, runs_dir)
    assert report["status"] == "pass"
    case_run = report["cases"][0]["run"]
    assert "effective_comparison" in case_run
    assert case_run["effective_comparison"]["config"]["atol"] == 0.004
    assert case_run["effective_comparison"]["sources"]["config.atol"] == "suite"


def test_agent_capabilities_reports_reads_writes_formats() -> None:
    cap = capabilities()
    assert cap.status == "ok"
    formats = cap.data["formats"]

    assert "testcase" in formats
    assert formats["testcase"]["writes"] == ["0.2", "0.3"]
    assert formats["testcase"]["reads"] == ["0.1", "0.2", "0.3"]

    assert "suite" in formats
    assert formats["suite"]["reads"] == ["0.1", "0.2", "0.3"]


def test_scenario_run_propagates_comparison(tmp_path: Path) -> None:
    import shutil
    repo_root = Path(__file__).resolve().parents[2]
    scenario_fixture = repo_root / "tests" / "fixtures" / "scenarios" / "kv-chain"
    copy_engine = repo_root / "tests" / "fixtures" / "adapters" / "copy_engine.py"

    scenario_dir = tmp_path / "scenario"
    shutil.copytree(scenario_fixture, scenario_dir)
    manifest = json.loads((scenario_dir / "scenario.json").read_text(encoding="utf-8"))
    manifest["steps"][0]["comparison"] = {
        "comparator": "tensor/numeric/v1",
        "config": {"atol": 0.007},
    }
    (scenario_dir / "scenario.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    adapter_file = _make_adapter(tmp_path, copy_engine)
    runs_dir = tmp_path / "scenario_runs"
    report = run_scenario_func(scenario_dir, adapter_file, runs_dir)
    assert report["status"] == "pass"
    step_result = report["steps"][0]
    assert "effective_comparison" in step_result["run"]
    assert step_result["run"]["effective_comparison"]["config"]["atol"] == 0.007


def test_custom_comparator_plugin_execution_in_executor(tmp_path: Path) -> None:
    register_builtin_comparator(MockVisionDetectorComparator())

    tc_dir = tmp_path / "tc_custom"
    _create_minimal_testcase(
        tc_dir,
        format_version="0.3",
        comparison={
            "format": "inferref-comparison",
            "format_version": "0.1",
            "comparator": "vision/object-detection/v1",
            "config": {"min_iou": 0.6},
        },
    )

    adapter_script = tmp_path / "engine_custom.py"
    adapter_script.write_text(
        """
import shutil, sys
from pathlib import Path
tc = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
shutil.copyfile(tc / "reference" / "y.irtensor", out / "y.irtensor")
""",
        encoding="utf-8",
    )
    adapter_file = _make_adapter(tmp_path, adapter_script)
    adapter = EngineAdapter.load(adapter_file)

    runs_dir = tmp_path / "custom_runs"
    result = execute_adapter(tc_dir, adapter, runs_dir)
    assert result["status"] == "pass"
    assert "comparison" in result
    assert result["comparison"]["comparator"]["comparator"] == "vision/object-detection/v1"
    assert result["comparison"]["comparator"]["metrics"]["mean_iou"] == 0.85
    assert result["comparison"]["comparator"]["metrics"]["min_iou_threshold"] == 0.6
    assert result["effective_comparison"]["comparator"] == "vision/object-detection/v1"


def test_cli_compare_with_comparator_flag(tmp_path: Path) -> None:
    from inferref.cli.main import main

    tc_dir = tmp_path / "tc_cli"
    _create_minimal_testcase(tc_dir, format_version="0.3")

    engine_out = tmp_path / "engine_out"
    engine_out.mkdir(parents=True, exist_ok=True)
    _write_irtensor(engine_out / "y.irtensor", np.array([2.0, 4.0], dtype=np.float32))

    rc = main(["compare", str(tc_dir), str(engine_out), "--comparator", "tensor/numeric/v1", "--atol", "0.01", "--json"])
    assert rc == 0


def test_mcp_server_tools_support_comparator(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from inferref.agent.mcp_server import create_server

    server = create_server(read_roots=[tmp_path], write_roots=[tmp_path])
    assert server is not None

    tc_dir = tmp_path / "tc_mcp"
    _create_minimal_testcase(tc_dir, format_version="0.3")

    engine_out = tmp_path / "engine_out_mcp"
    engine_out.mkdir(parents=True, exist_ok=True)
    _write_irtensor(engine_out / "y.irtensor", np.array([2.0, 4.0], dtype=np.float32))

    # Test compare_outputs tool logic
    resp = compare_outputs(tc_dir, engine_out, comparator="tensor/numeric/v1", atol=0.001)
    assert resp.status == "pass"
    assert "effective_comparison" in resp.data
    assert resp.data["effective_comparison"]["config"]["atol"] == 0.001


def test_cli_comparison_config_flag(tmp_path: Path) -> None:
    from inferref.cli.main import main

    tc_dir = tmp_path / "tc_cli_cfg"
    _create_minimal_testcase(tc_dir, format_version="0.3")

    engine_out = tmp_path / "engine_out_cfg"
    engine_out.mkdir(parents=True, exist_ok=True)
    _write_irtensor(engine_out / "y.irtensor", np.array([2.0, 4.0], dtype=np.float32))

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"atol": 0.005, "rtol": 0.005}), encoding="utf-8")

    # 1. Via config file path
    rc = main(["compare", str(tc_dir), str(engine_out), "--comparison-config", str(cfg_file), "--json"])
    assert rc == 0

    # 2. Via inline JSON string
    rc_inline = main(["compare", str(tc_dir), str(engine_out), "--comparison-config", '{"atol": 0.005}', "--json"])
    assert rc_inline == 0


def test_format_version_0_3_without_comparison_warning(tmp_path: Path) -> None:
    tc_dir = tmp_path / "tc_warn"
    _create_minimal_testcase(tc_dir, format_version="0.3", comparison=None)

    result = validate_testcase(tc_dir)
    assert not result.errors
    warn = next((i for i in result.issues if i.code == "format_version_0_3_without_comparison"), None)
    assert warn is not None
    assert warn.severity == "warning"


def test_deduplicate_comparator_unknown_error(tmp_path: Path) -> None:
    tc_dir = tmp_path / "tc_dedup"
    _create_minimal_testcase(
        tc_dir,
        format_version="0.3",
        comparison={
            "format": "inferref-comparison",
            "format_version": "0.1",
            "comparator": "nonexistent/comparator/v999",
            "outputs": {
                "y": {"config": {"foo": "bar"}}
            }
        }
    )

    result = validate_testcase(tc_dir)
    assert result.errors
    comp_errors = [e for e in result.errors if e.code == "comparator_unknown"]
    # Should only report top-level comparator unknown once, not for output role 'y'
    assert len(comp_errors) == 1
    assert comp_errors[0].where == "comparison.comparator"


def test_strict_comparator_in_execute_adapter_preflight(tmp_path: Path) -> None:
    class StrictDetectorComparator:
        id = "vision/strict-detection/v1"
        version = "1.0.0"
        description = "Strict Detector"

        def validate_config(self, config: dict[str, Any] | None = None) -> None:
            allowed = {"min_iou", "score_threshold"}
            if config:
                unknown = set(config.keys()) - allowed
                if unknown:
                    raise ValueError(f"unknown object detection comparator config key(s): {sorted(unknown)}")
                if "min_iou" in config and not (0.0 <= config["min_iou"] <= 1.0):
                    raise ValueError("min_iou must be between 0.0 and 1.0")

        def compare(
            self,
            reference: ArtifactSet,
            actual: ArtifactSet,
            config: dict[str, Any] | None = None,
        ) -> ComparatorResult:
            return ComparatorResult(
                comparator=self.id,
                status="pass",
                metrics={"mean_iou": 0.95},
            )

    register_builtin_comparator(StrictDetectorComparator())

    tc_dir = tmp_path / "tc_strict"
    _create_minimal_testcase(
        tc_dir,
        format_version="0.3",
        comparison={
            "format": "inferref-comparison",
            "format_version": "0.1",
            "comparator": "vision/strict-detection/v1",
            "config": {"min_iou": 0.8},
        },
    )

    adapter_script = tmp_path / "engine_strict.py"
    adapter_script.write_text(
        """
import shutil, sys
from pathlib import Path
tc = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
shutil.copyfile(tc / "reference" / "y.irtensor", out / "y.irtensor")
""",
        encoding="utf-8",
    )
    adapter_file = _make_adapter(tmp_path, adapter_script)
    adapter = EngineAdapter.load(adapter_file)

    runs_dir = tmp_path / "strict_runs"
    # execute_adapter must succeed pre-flight validation and execution
    result = execute_adapter(tc_dir, adapter, runs_dir)
    assert result["status"] == "pass"
    assert result["effective_comparison"]["comparator"] == "vision/strict-detection/v1"


def test_suite_comparator_overrides_testcase_comparator_to_numeric() -> None:
    tc_spec = ComparisonSpec(
        format_version="0.1",
        comparator="vision/object-detection/v1",
        config={"min_iou": 0.5},
    )
    suite_spec = ComparisonSpec(
        format_version="0.1",
        comparator="tensor/numeric/v1",
        config={"atol": 1e-4},
    )

    resolved = resolve_comparison_policy(
        testcase_spec=tc_spec,
        suite_spec=suite_spec,
    )
    assert resolved.comparator == "tensor/numeric/v1"
    assert resolved.sources["comparator"] == "suite"


def test_cli_strict_layout_false_overrides_testcase_true() -> None:
    tc_spec = ComparisonSpec(
        format_version="0.1",
        config={"strict_layout": True},
    )
    resolved = resolve_comparison_policy(
        testcase_spec=tc_spec,
        cli_strict_layout=False,
    )
    assert resolved.config["strict_layout"] is False
    assert resolved.sources["config.strict_layout"] == "cli"
