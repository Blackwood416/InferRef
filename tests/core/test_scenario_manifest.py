"""Scenario v0.1 manifest schema and validation rules (SPEC §4, §5)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from inferref.cli.main import EXIT_FAIL, EXIT_OK, main
from inferref.scenario import ScenarioError, load_scenario, validate_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "scenarios" / "kv-chain"


def _fixture(tmp_path: Path) -> Path:
    scenario = tmp_path / "scenario"
    shutil.copytree(FIXTURE, scenario)
    return scenario


def _manifest(path: Path) -> dict:
    return json.loads((path / "scenario.json").read_text(encoding="utf-8"))


def _write(path: Path, data: dict) -> None:
    (path / "scenario.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _codes(report: dict) -> list[str]:
    return [issue["code"] for issue in report["issues"]]


def test_fixture_scenario_loads_and_validates(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    loaded = load_scenario(scenario)
    assert loaded.id == "kv-chain"
    assert [step.id for step in loaded.steps] == ["prefill", "decode-0", "decode-1"]
    assert {item.name for item in loaded.inputs} == {
        "prefill_kv",
        "prefill_tokens",
        "decode_tokens_0",
        "decode_tokens_1",
    }
    report = validate_scenario(scenario)
    assert report["status"] == "pass"
    assert report["schema_valid"] is True
    assert report["runnable"] is True
    assert report["non_runnable_steps"] == []
    assert report["issues"] == []


def test_validate_rejects_wrong_format_and_version(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["format"] = "inferref-suite"
    data["format_version"] = "0.2"
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert report["schema_valid"] is False
    assert report["status"] == "fail"
    assert "scenario_invalid_manifest" in _codes(report)


@pytest.mark.parametrize("bad_id", ["../escape", "case.", "CON", "a b", ""])
def test_validate_rejects_bad_scenario_id(tmp_path: Path, bad_id: str) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["id"] = bad_id
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert report["schema_valid"] is False
    assert "scenario_invalid_id" in _codes(report)


def test_validate_rejects_empty_inputs(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["inputs"] = {}
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert report["schema_valid"] is False
    assert "scenario_invalid_manifest" in _codes(report)


def test_validate_rejects_non_tensor_value_kind(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["inputs"]["prefill_kv"] = {"kind": "scalar"}
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert report["schema_valid"] is False
    assert "scenario_invalid_manifest" in _codes(report)


def test_validate_rejects_bad_and_duplicate_step_ids(tmp_path: Path) -> None:
    bad = _fixture(tmp_path / "bad")
    data = _manifest(bad)
    data["steps"][0]["id"] = "bad id"
    _write(bad, data)
    report = validate_scenario(bad)
    assert "scenario_bad_step_id" in _codes(report)

    duplicate = _fixture(tmp_path / "duplicate")
    data = _manifest(duplicate)
    data["steps"][1]["id"] = "prefill"
    _write(duplicate, data)
    report = validate_scenario(duplicate)
    assert "scenario_duplicate_step_id" in _codes(report)


def test_validate_rejects_testcase_path_escape(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["steps"][0]["testcase"] = "../outside"
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert "scenario_path_escape" in _codes(report)


def test_validate_rejects_missing_testcase_directory(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["steps"][1]["testcase"] = "cases/does-not-exist"
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert "scenario_testcase_invalid" in _codes(report)


def test_validate_rejects_unknown_role(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["steps"][0]["bindings"]["inputs"]["tokens"] = "scenario.inputs.prefill_tokens"
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert "scenario_role_unknown" in _codes(report)


def test_validate_rejects_undeclared_source_and_destination(
    tmp_path: Path,
) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["steps"][0]["bindings"]["inputs"]["cache"] = "scenario.inputs.missing"
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert "scenario_source_undeclared" in _codes(report)

    data = _manifest(scenario)
    data["steps"][2]["bindings"]["outputs"]["cache_out"] = "scenario.outputs.nope"
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert "scenario_destination_undeclared" in _codes(report)


def test_validate_rejects_uninitialized_state_read(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["state"]["kv"] = {"kind": "tensor"}
    data["steps"][0]["bindings"]["inputs"]["cache"] = "state.kv"
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert "scenario_state_uninitialized" in _codes(report)


def test_validate_rejects_double_state_write_in_one_step(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["steps"][0]["bindings"]["outputs"]["logits"] = "state.kv"
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert "scenario_state_written_twice" in _codes(report)


def test_validate_rejects_unwritten_scenario_output(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["steps"][0]["bindings"]["outputs"].pop("logits")
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert "scenario_output_unwritten" in _codes(report)


def test_validate_rejects_bad_state_init(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["state"]["kv"] = {"kind": "tensor", "init": "scenario.outputs.logits"}
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert "scenario_invalid_manifest" in _codes(report)

    data = _manifest(scenario)
    data["state"]["kv"] = {"kind": "tensor", "init": "scenario.inputs.nope"}
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert "scenario_invalid_manifest" in _codes(report)


def test_validate_rejects_invalid_reference_syntax(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["steps"][0]["bindings"]["inputs"]["cache"] = "kv.preload"
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert "scenario_reference_invalid" in _codes(report)


def test_validate_flags_non_reproducible_steps(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    testcase = scenario / "cases" / "decode-0" / "testcase.json"
    data = json.loads(testcase.read_text(encoding="utf-8"))
    data["reproducible"] = False
    testcase.write_text(json.dumps(data), encoding="utf-8")

    strict = validate_scenario(scenario)
    assert strict["schema_valid"] is True
    assert strict["runnable"] is False
    assert strict["status"] == "fail"
    assert strict["non_runnable_steps"] == ["decode-0"]

    inventory = validate_scenario(scenario, allow_nonreproducible=True)
    assert inventory["runnable"] is False
    assert inventory["allow_nonreproducible"] is True


def test_validate_schema_failure_is_not_runnable(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["steps"][0]["testcase"] = "../outside"
    _write(scenario, data)
    report = validate_scenario(scenario)
    assert report["schema_valid"] is False
    assert report["runnable"] is False
    assert report["status"] == "fail"
    assert report["error"]


def test_load_scenario_raises_on_invalid(tmp_path: Path) -> None:
    scenario = _fixture(tmp_path)
    data = _manifest(scenario)
    data["format"] = "wrong"
    _write(scenario, data)
    with pytest.raises(ScenarioError) as excinfo:
        load_scenario(scenario)
    assert "invalid scenario" in str(excinfo.value)
    assert excinfo.value.issues


def test_cli_scenario_validate_exit_codes(tmp_path: Path, capsys) -> None:
    scenario = _fixture(tmp_path)
    assert main(["scenario", "validate", str(scenario), "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["status"] == "pass"

    testcase = scenario / "cases" / "decode-0" / "testcase.json"
    data = json.loads(testcase.read_text(encoding="utf-8"))
    data["reproducible"] = False
    testcase.write_text(json.dumps(data), encoding="utf-8")
    assert main(["scenario", "validate", str(scenario), "--json"]) == EXIT_FAIL
    capsys.readouterr()
    assert (
        main(
            [
                "scenario",
                "validate",
                str(scenario),
                "--allow-nonreproducible",
                "--json",
            ]
        )
        == EXIT_OK
    )

    manifest = _manifest(scenario)
    manifest["format"] = "not-a-scenario"
    _write(scenario, manifest)
    assert main(["scenario", "validate", str(scenario), "--json"]) == EXIT_FAIL
