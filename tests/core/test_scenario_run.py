"""Scenario executor, replay modes, and v0.1 run report (SPEC §6, §7)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from inferref.cli.main import EXIT_FAIL, EXIT_OK, main
from inferref.scenario import run_scenario
from inferref.tensor import codec

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "scenarios" / "kv-chain"
COPY_ENGINE = REPO_ROOT / "tests" / "fixtures" / "adapters" / "copy_engine.py"


def _adapter(
    tmp_path: Path,
    *,
    dtypes: list[str] | None = None,
    features: list[str] | None = None,
    corruption: str | None = None,
) -> Path:
    payload: dict = {
        "format": "inferref-engine-adapter",
        "format_version": "0.2",
        "name": "kv-copy",
        "target_device": "cpu",
        "capabilities": {
            "device_types": ["cpu"],
            "dtypes": dtypes or ["float32"],
            "max_rank": 8,
            "features": features if features is not None else ["multiple_outputs"],
        },
        "command": ["{python}", str(COPY_ENGINE), "{testcase}", "{output}"],
    }
    if corruption is not None:
        payload["environment"] = {"INFERREF_ENGINE_STATE_CORRUPTION": corruption}
    path = tmp_path / "adapter.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_root(report: dict) -> Path:
    return Path(report["steps"][0]["run"]["testcase"]).parent.parent.parent


def test_reference_state_run_passes_with_expected_layout(tmp_path: Path) -> None:
    report = run_scenario(SCENARIO_FIXTURE, _adapter(tmp_path), tmp_path / "runs")
    assert report["format"] == "inferref-scenario-run"
    assert report["format_version"] == "0.1"
    assert report["status"] == "pass"
    assert report["accepted"] is True
    assert report["scenario"]["id"] == "kv-chain"
    assert report["adapter"] == {"id": "kv-copy", "name": "kv-copy"}
    assert report["state_mode"] == "reference"
    assert report["compare_state"] is False
    assert report["outputs"] == ["logits"]
    assert [step["status"] for step in report["steps"]] == ["pass", "pass", "pass"]
    assert all(
        step["state_status"] == "not_applicable" for step in report["steps"]
    )
    assert all("run_id" in step["run"] for step in report["steps"])

    run_root = _run_root(report)
    assert run_root.name.startswith("scenario-")
    assert (run_root / "inferref-scenario-run.json").is_file()
    assert (run_root / "state" / "kv" / "kv.irtensor").is_file()
    assert (run_root / "outputs" / "logits.irtensor").is_file()
    for step in report["steps"]:
        step_root = run_root / "steps" / step["id"]
        assert (step_root / "testcase" / "testcase.json").is_file()
        assert (step_root / "inferref-run.json").is_file()
        assert (step_root / "output").is_dir()


def test_engine_state_run_passes_and_compares_state(tmp_path: Path) -> None:
    report = run_scenario(
        SCENARIO_FIXTURE,
        _adapter(tmp_path),
        tmp_path / "runs",
        state_mode="engine",
        compare_state=True,
    )
    assert report["status"] == "pass"
    assert report["state_mode"] == "engine"
    assert report["compare_state"] is True
    assert [step["state_status"] for step in report["steps"]] == ["ok", "ok", "ok"]


@pytest.mark.parametrize(
    ("corruption", "expected"),
    [
        ("shape", "state_shape_mismatch"),
        ("dtype", "state_dtype_mismatch"),
        ("value", "state_mismatch"),
    ],
)
def test_engine_state_corruption_is_localized_with_compare_state(
    tmp_path: Path, corruption: str, expected: str
) -> None:
    report = run_scenario(
        SCENARIO_FIXTURE,
        _adapter(tmp_path, corruption=corruption),
        tmp_path / "runs",
        state_mode="engine",
        compare_state=True,
    )
    assert report["status"] == "fail"
    assert report["accepted"] is False
    assert len(report["steps"]) == 1
    step = report["steps"][0]
    assert step["id"] == "prefill"
    assert step["state_status"] == expected
    assert step["status"] == "mismatch"


def test_engine_state_value_corruption_surfaces_downstream_without_compare(
    tmp_path: Path,
) -> None:
    report = run_scenario(
        SCENARIO_FIXTURE,
        _adapter(tmp_path, corruption="value"),
        tmp_path / "runs",
        state_mode="engine",
        compare_state=False,
    )
    assert report["status"] == "fail"
    statuses = [step["status"] for step in report["steps"]]
    assert statuses == ["mismatch", "mismatch", "mismatch"]
    assert all(
        step["state_status"] == "not_compared" for step in report["steps"]
    )


def test_engine_state_missing_state_output_stops_chain(tmp_path: Path) -> None:
    report = run_scenario(
        SCENARIO_FIXTURE,
        _adapter(tmp_path, corruption="missing-state"),
        tmp_path / "runs",
        state_mode="engine",
        compare_state=True,
    )
    assert report["status"] == "error"
    assert report["accepted"] is False
    assert len(report["steps"]) == 1
    step = report["steps"][0]
    assert step["id"] == "prefill"
    assert step["status"] == "error"
    assert step["state_status"] == "state_missing"
    assert step["run"]["scenario_error"]["code"] == "scenario_state_missing"

    without_compare = run_scenario(
        SCENARIO_FIXTURE,
        _adapter(tmp_path, corruption="missing-state"),
        tmp_path / "runs-without-compare",
        state_mode="engine",
        compare_state=False,
    )
    assert without_compare["status"] == "error"
    assert len(without_compare["steps"]) == 1
    assert without_compare["steps"][0]["state_status"] == "state_missing"


def test_engine_state_unsupported_step_stops_chain(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, features=[])
    report = run_scenario(
        SCENARIO_FIXTURE,
        adapter,
        tmp_path / "runs",
        state_mode="engine",
        allow_unsupported=True,
    )
    assert report["status"] == "unsupported"
    assert report["accepted"] is True
    assert len(report["steps"]) == 1
    assert report["steps"][0]["status"] == "unsupported"

    strict = run_scenario(
        SCENARIO_FIXTURE, adapter, tmp_path / "strict", state_mode="engine"
    )
    assert strict["status"] == "fail"
    assert strict["accepted"] is False


def test_unbound_input_falls_back_to_embedded_payload(tmp_path: Path) -> None:
    report = run_scenario(SCENARIO_FIXTURE, _adapter(tmp_path), tmp_path / "runs")
    assert report["status"] == "pass"
    effective = (
        _run_root(report) / "steps" / "prefill" / "testcase"
    )
    manifest = json.loads(
        (effective / "testcase.json").read_text(encoding="utf-8")
    )
    scale = next(item for item in manifest["inputs"] if item["name"] == "scale")
    assert scale["payload"] == "inputs/scale.irtensor"
    assert (effective / scale["payload"]).is_file()
    embedded = codec.read(SCENARIO_FIXTURE / "cases" / "prefill" / "inputs" / "scale.irtensor")
    used = codec.read(effective / scale["payload"])
    assert used.shape == embedded.shape
    assert used.dtype == embedded.dtype


def test_effective_testcase_metadata_is_patched_for_bound_state(
    tmp_path: Path,
) -> None:
    report = run_scenario(SCENARIO_FIXTURE, _adapter(tmp_path), tmp_path / "runs")
    effective = _run_root(report) / "steps" / "decode-0" / "testcase"
    manifest = json.loads(
        (effective / "testcase.json").read_text(encoding="utf-8")
    )
    cache = next(item for item in manifest["inputs"] if item["name"] == "cache")
    assert cache["payload"] == "inputs/cache.irtensor"
    assert cache["shape"] == [1, 2, 6, 8]
    header = codec.read_header(effective / cache["payload"])
    assert list(header.shape) == [1, 2, 6, 8]
    assert header.dtype == "float32"


def test_unsupported_steps_aggregate_with_allow_unsupported(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, dtypes=["float16"])
    strict = run_scenario(SCENARIO_FIXTURE, adapter, tmp_path / "strict")
    assert strict["status"] == "fail"
    assert strict["accepted"] is False
    assert all(step["status"] == "unsupported" for step in strict["steps"])

    inventory = run_scenario(
        SCENARIO_FIXTURE, adapter, tmp_path / "inventory", allow_unsupported=True
    )
    assert inventory["status"] == "unsupported"
    assert inventory["accepted"] is True


def test_partial_status_for_mixed_pass_unsupported(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, features=[])
    report = run_scenario(
        SCENARIO_FIXTURE, adapter, tmp_path / "runs", allow_unsupported=True
    )
    assert report["status"] == "partial"
    assert report["accepted"] is True
    assert [step["status"] for step in report["steps"]] == [
        "unsupported",
        "pass",
        "pass",
    ]

    strict = run_scenario(SCENARIO_FIXTURE, adapter, tmp_path / "strict")
    assert strict["status"] == "fail"
    assert strict["accepted"] is False


def test_fail_fast_raises_on_infrastructure_exception(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    data = json.loads(adapter.read_text(encoding="utf-8"))
    data["cwd"] = "missing-directory"
    adapter.write_text(json.dumps(data), encoding="utf-8")

    report = run_scenario(SCENARIO_FIXTURE, adapter, tmp_path / "runs")
    assert report["status"] == "error"
    assert report["steps"][0]["status"] == "error"
    assert report["steps"][0]["run"]["status"] == "infrastructure_error"

    with pytest.raises(Exception, match="working directory"):
        run_scenario(SCENARIO_FIXTURE, adapter, tmp_path / "runs", fail_fast=True)


def test_scenario_run_rejects_bad_state_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="state_mode"):
        run_scenario(
            SCENARIO_FIXTURE,
            _adapter(tmp_path),
            tmp_path / "runs",
            state_mode="session",
        )


def test_cli_scenario_run_exit_codes(tmp_path: Path, capsys) -> None:
    adapter = _adapter(tmp_path)
    assert (
        main(
            [
                "scenario",
                "run",
                str(SCENARIO_FIXTURE),
                "--adapter",
                str(adapter),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--json",
            ]
        )
        == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"
    assert payload["format"] == "inferref-scenario-run"

    corrupt = _adapter(tmp_path, corruption="value")
    assert (
        main(
            [
                "scenario",
                "run",
                str(SCENARIO_FIXTURE),
                "--adapter",
                str(corrupt),
                "--runs-dir",
                str(tmp_path / "failing-runs"),
                "--state-mode",
                "engine",
                "--compare-state",
                "--json",
            ]
        )
        == EXIT_FAIL
    )


def test_scenario_validation_gate_blocks_invalid_manifests(tmp_path: Path) -> None:
    import shutil

    scenario = tmp_path / "broken"
    shutil.copytree(SCENARIO_FIXTURE, scenario)
    data = json.loads((scenario / "scenario.json").read_text(encoding="utf-8"))
    data["steps"][0]["testcase"] = "../outside"
    (scenario / "scenario.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        run_scenario(scenario, _adapter(tmp_path), tmp_path / "runs")


def test_scenario_run_never_imports_torch(tmp_path: Path) -> None:
    """Core scenario execution works with torch hard-blocked (SPEC §12)."""

    script = r"""
import json, sys
from pathlib import Path

class TorchBlocker:
    FORBIDDEN = ("torch", "torchvision", "functorch")
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.FORBIDDEN:
            raise ImportError(f"torch is blocked for this test (tried {name!r})")
        return None

sys.meta_path.insert(0, TorchBlocker())
import json
from inferref.scenario import validate_scenario, run_scenario
from inferref.tensor import codec

scenario = Path(sys.argv[1])
adapter = Path(sys.argv[2])
runs = Path(sys.argv[3])
validation = validate_scenario(scenario)
assert validation["schema_valid"] and validation["runnable"]
report = run_scenario(scenario, adapter, runs, state_mode="engine", compare_state=True)
assert report["status"] == "pass", report["status"]
assert (runs / report["steps"][0]["run"]["testcase"]).is_dir()
print("scenario-without-torch: ok")
"""
    adapter = _adapter(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(SCENARIO_FIXTURE),
            str(adapter),
            str(tmp_path / "torch-free-runs"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "scenario-without-torch: ok" in result.stdout
