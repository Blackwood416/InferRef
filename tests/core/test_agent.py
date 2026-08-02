"""Framework-neutral Agent protocol and engine-adapter tests."""

from __future__ import annotations

import json
import sys
import time
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
    input_path = codec.write_array(root / "inputs" / "x.irtensor", values)
    output_path = codec.write_array(
        root / "reference" / "out.irtensor", values + 1.0
    )
    input_metadata = codec.read(input_path).to_metadata()
    output_metadata = codec.read(output_path).to_metadata()
    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.1",
        "name": "agent-add-one",
        "reproducible": True,
        "origin": {"kind": "test"},
        "inputs": [
            {
                "name": "x",
                "value_id": 1,
                "payload": "inputs/x.irtensor",
                **input_metadata,
            }
        ],
        "outputs": [
            {
                "name": "out",
                "value_id": 2,
                "payload": "reference/out.irtensor",
                **output_metadata,
            }
        ],
        "nodes": [],
        "values": [
            {"id": 1, **input_metadata},
            {"id": 2, **output_metadata},
        ],
    }
    (root / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _make_adapter(
    root: Path,
    script: Path,
    *,
    mode: str | None = None,
    command: list[str] | None = None,
    timeout_seconds: float = 30,
    max_output_chars: int = 65_536,
    max_artifact_bytes: int = 1_073_741_824,
) -> Path:
    payload = {
        "format": ENGINE_ADAPTER_FORMAT,
        "format_version": ENGINE_ADAPTER_VERSION,
        "name": "test-engine",
        "command": command or ["{python}", str(script), "{testcase}", "{output}"],
        "timeout_seconds": timeout_seconds,
        "max_output_chars": max_output_chars,
        "max_artifact_bytes": max_artifact_bytes,
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
    assert first.data["execution"]["command"][0] == str(Path(sys.executable).absolute())
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


def test_stdout_flood_hits_hard_limit_without_unbounded_capture(tmp_path: Path) -> None:
    testcase = _make_testcase(tmp_path / "testcase")
    script = tmp_path / "flood.py"
    script.write_text(
        "import os\nwhile True:\n    os.write(1, b'x' * 65536)\n",
        encoding="utf-8",
    )
    adapter = _make_adapter(
        tmp_path,
        script,
        command=["{python}", str(script), "{testcase}", "{output}"],
        max_output_chars=4096,
    )

    response = run_engine(testcase, adapter, tmp_path / "runs")

    assert response.status == "error"
    assert response.data["status"] == "output_limit"
    execution = response.data["execution"]
    assert execution["stdout_bytes"] > 4096
    assert len(execution["stdout"].encode()) < 4300
    assert (
        Path(response.data["output"], execution["stdout_path"]).stat().st_size
        == 4096
    )


def test_timeout_terminates_descendant_process(tmp_path: Path) -> None:
    testcase = _make_testcase(tmp_path / "testcase")
    marker = tmp_path / "descendant-survived"
    script = tmp_path / "spawn_child.py"
    script.write_text(
        """import subprocess, sys, time
marker = sys.argv[3]
code = ("import pathlib,sys,time; time.sleep(1.5); "
        "pathlib.Path(sys.argv[1]).write_text('alive')")
subprocess.Popen([sys.executable, "-c", code, marker],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(60)
""",
        encoding="utf-8",
    )
    adapter = _make_adapter(
        tmp_path,
        script,
        command=[
            "{python}",
            str(script),
            "{testcase}",
            "{output}",
            str(marker),
        ],
        timeout_seconds=0.25,
    )

    response = run_engine(testcase, adapter, tmp_path / "runs")
    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)

    assert response.status == "error"
    assert response.data["status"] == "timeout"
    assert not marker.exists()


def test_normal_exit_also_cleans_up_descendants(tmp_path: Path) -> None:
    testcase = _make_testcase(tmp_path / "testcase")
    marker = tmp_path / "detached-descendant-survived"
    script = tmp_path / "exit_with_child.py"
    script.write_text(
        """import json, pathlib, subprocess, sys
import numpy as np
from inferref.tensor import codec
marker = sys.argv[3]
code = ("import pathlib,sys,time; time.sleep(1.5); "
        "pathlib.Path(sys.argv[1]).write_text('alive')")
subprocess.Popen([sys.executable, "-c", code, marker],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
testcase = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
manifest = json.loads((testcase / "testcase.json").read_text())
value = codec.read(testcase / manifest["inputs"][0]["payload"]).as_comparable()
codec.write_array(output / "out.irtensor", np.asarray(value + 1, dtype=np.float32))
""",
        encoding="utf-8",
    )
    adapter = _make_adapter(
        tmp_path,
        script,
        command=[
            "{python}",
            str(script),
            "{testcase}",
            "{output}",
            str(marker),
        ],
    )

    response = run_engine(testcase, adapter, tmp_path / "runs")
    deadline = time.monotonic() + 2.5
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.05)

    assert response.status == "pass"
    assert not marker.exists()
    expected_strategy = (
        "windows_job_object" if sys.platform == "win32" else "posix_process_group"
    )
    assert response.data["execution"]["process_tree_strategy"] == expected_strategy


def test_artifact_growth_is_detected_during_execution(tmp_path: Path) -> None:
    testcase = _make_testcase(tmp_path / "testcase")
    script = tmp_path / "large_artifact.py"
    script.write_text(
        """import pathlib, sys, time
output = pathlib.Path(sys.argv[2])
with (output / "huge.bin").open("wb") as stream:
    while True:
        stream.write(b'x' * 65536)
        stream.flush()
        time.sleep(0.005)
""",
        encoding="utf-8",
    )
    adapter = _make_adapter(
        tmp_path,
        script,
        command=["{python}", str(script), "{testcase}", "{output}"],
        max_artifact_bytes=131_072,
    )

    response = run_engine(testcase, adapter, tmp_path / "runs")

    assert response.status == "error"
    assert response.data["status"] == "artifact_limit"
    assert response.data["execution"]["artifact_bytes"] > 131_072


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
