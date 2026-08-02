"""Framework-neutral Agent protocol and engine-adapter tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from inferref.agent import capabilities, compare_outputs, context, run_engine
from inferref.agent.protocol import ENGINE_ADAPTER_FORMAT, ENGINE_ADAPTER_VERSION
from inferref.cli.main import EXIT_OK, main
from inferref.tensor import codec

ENGINE_SOURCE = r"""from __future__ import annotations
import json
import os
import sys
from pathlib import Path
import numpy as np
from inferref.tensor import codec

testcase = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
manifest = json.loads((testcase / "testcase.json").read_text(encoding="utf-8"))
entry = manifest["inputs"][0]
value = codec.read(testcase / entry["payload"]).as_comparable()
if os.environ.get("INFERREF_ENGINE_MODE") == "error":
    print("deliberate adapter failure", file=sys.stderr)
    raise SystemExit(17)
result = value + (2.0 if os.environ.get("INFERREF_ENGINE_MODE") == "mismatch" else 1.0)
codec.write_array(output / "out.irtensor", np.asarray(result, dtype=np.float32))
"""


def _make_testcase(root: Path) -> Path:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    codec.write_array(root / "inputs" / "x.irtensor", values)
    codec.write_array(root / "reference" / "out.irtensor", values + 1.0)
    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.1",
        "name": "agent-add-one",
        "reproducible": True,
        "origin": {"kind": "test"},
        "inputs": [{"name": "x", "value_id": 1, "payload": "inputs/x.irtensor"}],
        "outputs": [
            {"name": "out", "value_id": 2, "payload": "reference/out.irtensor"}
        ],
        "nodes": [],
        "values": [],
    }
    (root / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _make_adapter(
    root: Path,
    script: Path,
    *,
    mode: str | None = None,
    command: list[str] | None = None,
) -> Path:
    payload = {
        "format": ENGINE_ADAPTER_FORMAT,
        "format_version": ENGINE_ADAPTER_VERSION,
        "name": "test-engine",
        "command": command or ["{python}", str(script), "{testcase}", "{output}"],
        "timeout_seconds": 30,
    }
    if mode is not None:
        payload["environment"] = {"INFERREF_ENGINE_MODE": mode}
    path = root / "adapter.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, mode: str | None = None) -> tuple[Path, Path]:
    testcase = _make_testcase(tmp_path / "testcase")
    script = tmp_path / "engine.py"
    script.write_text(ENGINE_SOURCE, encoding="utf-8")
    return testcase, _make_adapter(tmp_path, script, mode=mode)


def test_capabilities_are_versioned_and_actionable() -> None:
    payload = capabilities().to_dict()
    assert payload["protocol"] == {
        "format": "inferref-agent-response",
        "version": "0.1",
    }
    assert payload["data"]["inferref_version"] == "0.3.0"
    assert "inferref_run_engine" in payload["data"]["mcp_tools"]
    assert payload["next_actions"][0]["operation"] == "context"


def test_testcase_context_is_compact_and_actionable(tmp_path: Path) -> None:
    testcase = _make_testcase(tmp_path / "testcase")
    response = context(testcase)
    assert response.status == "ok"
    assert response.data["artifact"] == "testcase"
    assert response.data["reproducible"] is True
    assert response.next_actions[0]["operation"] == "run_engine"


def test_engine_adapter_passes_and_never_reuses_output_directory(
    tmp_path: Path,
) -> None:
    testcase, adapter = _fixture(tmp_path)
    first = run_engine(testcase, adapter, tmp_path / "runs")
    second = run_engine(testcase, adapter, tmp_path / "runs")

    assert first.status == second.status == "pass"
    assert first.data["output"] != second.data["output"]
    assert first.data["execution"]["command"][0] == str(Path(sys.executable).resolve())
    assert Path(first.data["output"], "inferref-run.json").is_file()
    assert first.data["comparison"]["status"] == "pass"


def test_engine_mismatch_is_distinct_from_execution_error(tmp_path: Path) -> None:
    testcase, adapter = _fixture(tmp_path, mode="mismatch")
    response = run_engine(testcase, adapter, tmp_path / "runs")

    assert response.status == "fail"
    assert response.data["status"] == "mismatch"
    assert response.data["comparison"]["first_failure"]["name"] == "out"
    assert response.data["adapter"]["environment_keys"] == ["INFERREF_ENGINE_MODE"]
    assert "environment" not in response.data["adapter"]
    assert response.next_actions[0]["operation"] == "modify_engine"


def test_engine_process_error_is_structured_and_persisted(tmp_path: Path) -> None:
    testcase, adapter = _fixture(tmp_path, mode="error")
    response = run_engine(testcase, adapter, tmp_path / "runs")

    assert response.status == "error"
    assert response.data["status"] == "execution_error"
    assert response.data["execution"]["exit_code"] == 17
    assert "deliberate adapter failure" in response.data["execution"]["stderr"]
    assert Path(response.data["output"], "inferref-run.json").is_file()


def test_invalid_adapter_command_is_rejected_without_running(tmp_path: Path) -> None:
    testcase, _ = _fixture(tmp_path)
    adapter = _make_adapter(
        tmp_path,
        tmp_path / "engine.py",
        command=["{python}", "engine.py"],
    )
    response = run_engine(testcase, adapter, tmp_path / "runs")
    assert response.status == "error"
    assert response.diagnostics[0]["code"] == "adapter_failed"
    assert "testcase" in response.diagnostics[0]["message"]
    assert not (tmp_path / "runs").exists()


def test_malformed_adapter_types_return_protocol_error(tmp_path: Path) -> None:
    testcase = _make_testcase(tmp_path / "testcase")
    adapter = tmp_path / "adapter.json"
    adapter.write_text("[]", encoding="utf-8")
    response = run_engine(testcase, adapter, tmp_path / "runs")
    assert response.status == "error"
    assert "JSON object" in response.diagnostics[0]["message"]


def test_context_rejects_unknown_testcase_version(tmp_path: Path) -> None:
    testcase = _make_testcase(tmp_path / "testcase")
    manifest_path = testcase / "testcase.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = "9.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    response = context(testcase)
    assert response.status == "error"
    assert "unsupported testcase format_version" in response.diagnostics[0]["message"]


def test_compare_outputs_returns_agent_envelope(tmp_path: Path) -> None:
    testcase = _make_testcase(tmp_path / "testcase")
    codec.write_array(
        tmp_path / "engine" / "out.irtensor",
        np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0,
    )
    response = compare_outputs(testcase, tmp_path / "engine")
    assert response.status == "pass"
    assert response.data["status"] == "pass"


def test_agent_cli_emits_same_json_contract(tmp_path: Path, capsys) -> None:
    testcase = _make_testcase(tmp_path / "testcase")
    assert main(["agent", "context", str(testcase), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "context"
    assert payload["data"]["artifact"] == "testcase"
