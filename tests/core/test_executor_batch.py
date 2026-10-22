"""Executor batch helper: one adapter process for an ordered testcase batch."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from inferref.agent.executor import execute_adapter_batch
from inferref.agent.protocol import (
    ADAPTER_FEATURES,
    ENGINE_ADAPTER_FORMAT,
    ENGINE_ADAPTER_VERSION,
    AdapterCapabilities,
    AgentProtocolError,
    EngineAdapter,
)

# A stand-in runtime bridge: it counts its own launches, honours one
# ``--batch-manifest`` argument, and publishes batch-results.json the way
# cpp/include/inferref/bridge.hpp does. Behaviour knobs arrive as environment
# variables so the test never has to edit the script.
FAKE_BRIDGE = """\
import json
import os
import sys
from pathlib import Path


def batch_manifest(argv):
    if "--batch-manifest" not in argv:
        raise SystemExit("missing --batch-manifest")
    return Path(argv[argv.index("--batch-manifest") + 1])


def main():
    manifest_path = batch_manifest(sys.argv[1:])
    counter = Path(os.environ["FAKE_LAUNCH_COUNTER"])
    launches = int(counter.read_text(encoding="utf-8")) if counter.is_file() else 0
    counter.write_text(str(launches + 1), encoding="utf-8")
    dump = os.environ.get("FAKE_ENV_DUMP")
    if dump:
        keys = (
            "INFERREF_TESTCASE",
            "INFERREF_OUTPUT",
            "INFERREF_BATCH_MANIFEST",
            "INFERREF_BATCH_OUTPUT",
        )
        Path(dump).write_text(
            json.dumps({key: os.environ.get(key) for key in keys}),
            encoding="utf-8",
        )
    results_path = manifest_path.parent / "batch-results.json"
    override = os.environ.get("FAKE_RESULTS_OVERRIDE")
    if override:
        results_path.write_text(
            Path(override).read_text(encoding="utf-8"), encoding="utf-8"
        )
        raise SystemExit(int(os.environ.get("FAKE_EXIT_CODE", "0")))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failing = {value for value in os.environ.get("FAKE_FAIL_IDS", "").split(",") if value}
    items = []
    passed = 0
    failed = 0
    for item in manifest["items"]:
        output_dir = Path(item["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        if item["id"] in failing:
            failed += 1
            exit_code = 4
            error = "fake engine rejected the item"
        else:
            passed += 1
            exit_code = 0
            error = None
            (output_dir / "engine-output.json").write_text("{}", encoding="utf-8")
        items.append(
            {
                "id": item["id"],
                "status": "passed" if exit_code == 0 else "failed",
                "exit_code": exit_code,
                "testcase_dir": item["testcase_dir"],
                "output_dir": item["output_dir"],
                "error": error,
            }
        )
    results_path.write_text(
        json.dumps(
            {
                "format": "inferref-batch-results",
                "format_version": "1.0",
                "items": items,
                "counts": {"total": len(items), "passed": passed, "failed": failed},
            }
        ),
        encoding="utf-8",
    )
    raise SystemExit(0 if failed == 0 else 1)


main()
"""


def _fake_bridge(tmp_path: Path) -> Path:
    script = tmp_path / "fake_bridge.py"
    script.write_text(FAKE_BRIDGE, encoding="utf-8")
    return script


def _adapter_document(
    tmp_path: Path,
    script: Path,
    *,
    features: Sequence[str] = ("batch_execution",),
    environment: dict[str, str] | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, object]:
    return {
        "format": ENGINE_ADAPTER_FORMAT,
        "format_version": ENGINE_ADAPTER_VERSION,
        "name": "fake-bridge",
        "target_device": "cpu",
        "capabilities": {
            "device_types": ["cpu"],
            "dtypes": ["float32"],
            "max_rank": 2,
            "features": list(features),
        },
        "command": list(command) if command is not None else ["{python}", str(script)],
        "environment": dict(environment or {}),
    }


def _adapter(
    tmp_path: Path,
    script: Path,
    *,
    features: Sequence[str] = ("batch_execution",),
    environment: dict[str, str] | None = None,
) -> EngineAdapter:
    path = tmp_path / "adapter.json"
    path.write_text(
        json.dumps(
            _adapter_document(
                tmp_path, script, features=features, environment=environment
            )
        ),
        encoding="utf-8",
    )
    return EngineAdapter.load(path)


def _counting(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    counter = tmp_path / "launches.txt"
    return counter, {"FAKE_LAUNCH_COUNTER": str(counter)}


def _launch_count(counter: Path) -> int:
    return int(counter.read_text(encoding="utf-8")) if counter.is_file() else 0


def _cases(root: Path, ids: Sequence[str]) -> list[tuple[str, Path]]:
    root.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, Path]] = []
    for item_id in ids:
        case = root / item_id
        case.mkdir()
        items.append((item_id, case))
    return items


def _results_document(items: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "format": "inferref-batch-results",
        "format_version": "1.0",
        "items": list(items),
        "counts": {"total": len(items), "passed": 0, "failed": 0},
    }


def _result_item(
    item_id: str, *, status: str = "passed", exit_code: int = 0, output_dir: str = "."
) -> dict[str, object]:
    return {
        "id": item_id,
        "status": status,
        "exit_code": exit_code,
        "output_dir": output_dir,
        "error": None,
    }


def test_batch_execution_feature_token_is_known() -> None:
    assert "batch_execution" in ADAPTER_FEATURES


def test_three_items_launch_the_adapter_once(tmp_path: Path) -> None:
    script = _fake_bridge(tmp_path)
    counter, environment = _counting(tmp_path)
    adapter = _adapter(tmp_path, script, environment=environment)
    cases = _cases(tmp_path / "cases", ["step-0", "step-1", "step-2"])

    result = execute_adapter_batch(cases, adapter, tmp_path / "runs")

    assert result["status"] == "pass"
    assert result["execution"]["exit_code"] == 0
    assert _launch_count(counter) == 1
    assert [item["id"] for item in result["results"]] == ["step-0", "step-1", "step-2"]
    assert [item["status"] for item in result["results"]] == [
        "passed",
        "passed",
        "passed",
    ]
    assert [item["exit_code"] for item in result["results"]] == [0, 0, 0]
    assert result["counts"] == {"total": 3, "passed": 3, "failed": 0}

    manifest_path = Path(result["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "inferref-batch"
    assert manifest["format_version"] == "1.0"
    assert [item["id"] for item in manifest["items"]] == [
        "step-0",
        "step-1",
        "step-2",
    ]
    assert [item["testcase_dir"] for item in manifest["items"]] == [
        case.resolve().as_posix() for _, case in cases
    ]
    assert [Path(item["output_dir"]) for item in result["results"]] == [
        Path(item["output_dir"]) for item in manifest["items"]
    ]
    assert Path(result["batch_output"]) == manifest_path.parent
    assert (manifest_path.parent / "batch-results.json").is_file()

    # One process serves every item, so the adapter's per-testcase budgets are
    # multiplied by the item count.
    assert result["execution"]["timeout_seconds"] == adapter.timeout_seconds * 3
    assert result["execution"]["max_artifact_files"] == adapter.max_artifact_files * 3


def test_explicit_batch_timeout_replaces_the_scaled_budget(tmp_path: Path) -> None:
    script = _fake_bridge(tmp_path)
    counter, environment = _counting(tmp_path)
    adapter = _adapter(tmp_path, script, environment=environment)
    cases = _cases(tmp_path / "cases", ["step-0", "step-1"])

    result = execute_adapter_batch(
        cases, adapter, tmp_path / "runs", timeout_seconds=42.0
    )

    assert result["status"] == "pass"
    assert result["execution"]["timeout_seconds"] == 42.0
    assert _launch_count(counter) == 1


def test_one_failing_item_is_recorded_without_aborting_the_batch(
    tmp_path: Path,
) -> None:
    script = _fake_bridge(tmp_path)
    counter, environment = _counting(tmp_path)
    environment["FAKE_FAIL_IDS"] = "step-1"
    adapter = _adapter(tmp_path, script, environment=environment)
    cases = _cases(tmp_path / "cases", ["step-0", "step-1", "step-2"])

    result = execute_adapter_batch(cases, adapter, tmp_path / "runs")

    assert _launch_count(counter) == 1
    assert result["status"] == "failed"
    assert result["execution"]["exit_code"] == 1
    assert [item["status"] for item in result["results"]] == [
        "passed",
        "failed",
        "passed",
    ]
    assert result["counts"] == {"total": 3, "passed": 2, "failed": 1}
    failed = result["results"][1]
    assert failed["exit_code"] == 4
    assert failed["error"] == "fake engine rejected the item"


def test_non_zero_exit_with_a_passing_report_is_an_execution_error(
    tmp_path: Path,
) -> None:
    script = _fake_bridge(tmp_path)
    counter, environment = _counting(tmp_path)
    request = _cases(tmp_path / "cases", ["step-0"])
    override = tmp_path / "results.json"
    override.write_text(
        json.dumps(_results_document([_result_item("step-0")])), encoding="utf-8"
    )
    environment["FAKE_RESULTS_OVERRIDE"] = str(override)
    environment["FAKE_EXIT_CODE"] = "1"
    adapter = _adapter(tmp_path, script, environment=environment)

    result = execute_adapter_batch(request, adapter, tmp_path / "runs")

    assert [item["status"] for item in result["results"]] == ["passed"]
    assert result["execution"]["exit_code"] == 1
    assert result["status"] == "execution_error"
    assert _launch_count(counter) == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{not json", "not valid JSON"),
        (
            {"format": "inferref-other", "format_version": "1.0", "items": []},
            "format must be",
        ),
        ({"format": "inferref-batch-results", "items": []}, "format_version"),
        (
            _results_document([_result_item("step-0")]),
            "missing item id",
        ),
        (
            _results_document(
                [_result_item("step-0"), _result_item("step-0"), _result_item("step-1")]
            ),
            "duplicate batch result id",
        ),
        (
            _results_document([_result_item("step-0"), _result_item("step-9")]),
            "not requested",
        ),
        (
            _results_document([_result_item("step-0", status="passed", exit_code=3)]),
            "contradicts exit_code",
        ),
    ],
)
def test_malformed_batch_results_are_reported(
    tmp_path: Path, payload: object, message: str
) -> None:
    script = _fake_bridge(tmp_path)
    counter, environment = _counting(tmp_path)
    cases = _cases(tmp_path / "cases", ["step-0", "step-1"])
    override = tmp_path / "results.json"
    override.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )
    environment["FAKE_RESULTS_OVERRIDE"] = str(override)
    adapter = _adapter(tmp_path, script, environment=environment)

    result = execute_adapter_batch(cases, adapter, tmp_path / "runs")

    assert _launch_count(counter) == 1
    assert result["status"] == "results_error"
    assert message in result["error"]
    assert result["results"] == []


def test_missing_batch_results_file_is_reported(tmp_path: Path) -> None:
    script = _fake_bridge(tmp_path)
    counter, environment = _counting(tmp_path)
    cases = _cases(tmp_path / "cases", ["step-0"])
    # An adapter that exits cleanly without publishing a report leaves the
    # caller nothing to map, so the batch must not look like a pass.
    adapter_document = _adapter_document(tmp_path, script, environment=environment)
    adapter_document["command"] = ["{python}", "-c", "pass"]
    path = tmp_path / "adapter.json"
    path.write_text(json.dumps(adapter_document), encoding="utf-8")
    adapter = EngineAdapter.load(path)

    result = execute_adapter_batch(cases, adapter, tmp_path / "runs")

    assert result["execution"]["exit_code"] == 0
    assert result["status"] == "results_error"
    assert "cannot read batch results" in result["error"]
    assert _launch_count(counter) == 0


def test_per_testcase_command_template_is_rejected_for_batch(tmp_path: Path) -> None:
    script = _fake_bridge(tmp_path)
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text(
        json.dumps(
            _adapter_document(
                tmp_path,
                script,
                command=[
                    "{python}",
                    str(script),
                    "--testcase",
                    "{testcase}",
                    "--output",
                    "{output}",
                ],
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(AgentProtocolError, match="batch_execution"):
        EngineAdapter.load(adapter_path)

    # A directly constructed adapter bypasses load-time validation, so the
    # helper still refuses to guess how to run a batch with that template.
    counter = tmp_path / "launches.txt"
    adapter = EngineAdapter(
        name="hand-built",
        command=("{python}", str(script), "{testcase}", "{output}"),
        capabilities=AdapterCapabilities(
            device_types=("cpu",),
            dtypes=("float32",),
            max_rank=2,
            features=("batch_execution",),
        ),
        environment={"FAKE_LAUNCH_COUNTER": str(counter)},
    )

    result = execute_adapter_batch(
        _cases(tmp_path / "cases", ["step-0", "step-1"]), adapter, tmp_path / "runs"
    )

    assert result["status"] == "unsupported"
    assert result["execution"] is None
    assert result["results"] == []
    assert _launch_count(counter) == 0
    assert "{testcase}" in result["unsupported"][0]
    assert "--batch-manifest" in result["unsupported"][0]


def test_batch_adapter_keeps_testcase_placeholders_only_in_environment(
    tmp_path: Path,
) -> None:
    script = _fake_bridge(tmp_path)
    counter, environment = _counting(tmp_path)
    dump = tmp_path / "environment.json"
    environment["FAKE_ENV_DUMP"] = str(dump)
    environment["INFERREF_TESTCASE"] = "{testcase}"
    environment["INFERREF_OUTPUT"] = "{output}"
    adapter = _adapter(tmp_path, script, environment=environment)
    cases = _cases(tmp_path / "cases", ["step-0"])

    result = execute_adapter_batch(cases, adapter, tmp_path / "runs")

    assert result["status"] == "pass"
    assert _launch_count(counter) == 1
    observed = json.loads(dump.read_text(encoding="utf-8"))
    assert observed["INFERREF_TESTCASE"] == result["manifest"]
    assert observed["INFERREF_OUTPUT"] == result["batch_output"]
    assert observed["INFERREF_BATCH_MANIFEST"] == result["manifest"]
    assert observed["INFERREF_BATCH_OUTPUT"] == result["batch_output"]


def test_batch_adapter_without_placeholders_loads(tmp_path: Path) -> None:
    script = _fake_bridge(tmp_path)
    path = tmp_path / "adapter.json"
    path.write_text(json.dumps(_adapter_document(tmp_path, script)), encoding="utf-8")

    adapter = EngineAdapter.load(path)

    assert adapter.supports_batch_execution
    assert adapter.command == ("{python}", str(script))


def test_sequential_adapters_still_need_testcase_and_output_placeholders(
    tmp_path: Path,
) -> None:
    script = _fake_bridge(tmp_path)
    path = tmp_path / "adapter.json"
    path.write_text(
        json.dumps(_adapter_document(tmp_path, script, features=())), encoding="utf-8"
    )

    with pytest.raises(AgentProtocolError, match="must reference placeholder"):
        EngineAdapter.load(path)


def test_batch_requests_are_validated_before_launch(tmp_path: Path) -> None:
    script = _fake_bridge(tmp_path)
    counter, environment = _counting(tmp_path)
    adapter = _adapter(tmp_path, script, environment=environment)
    cases = _cases(tmp_path / "cases", ["step-0", "step-1"])

    with pytest.raises(AgentProtocolError, match="at least one item"):
        execute_adapter_batch([], adapter, tmp_path / "runs")
    with pytest.raises(AgentProtocolError, match="duplicate batch item id"):
        execute_adapter_batch(
            [("step-0", cases[0][1]), ("step-0", cases[1][1])],
            adapter,
            tmp_path / "runs",
        )
    with pytest.raises(AgentProtocolError, match="does not exist"):
        execute_adapter_batch(
            [("step-0", tmp_path / "missing")], adapter, tmp_path / "runs"
        )
    with pytest.raises(AgentProtocolError, match="must be an identifier"):
        execute_adapter_batch([("../step-0", cases[0][1])], adapter, tmp_path / "runs")
    assert _launch_count(counter) == 0
