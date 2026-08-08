from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from inferref.agent.protocol import ENGINE_ADAPTER_FORMAT, ENGINE_ADAPTER_VERSION
from inferref.cli.main import EXIT_FAIL, EXIT_OK, main
from inferref.suite import load_suite, render_suite_report, run_suite, validate_suite
from inferref.tensor import codec

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "scenarios" / "kv-chain"
COPY_ENGINE = REPO_ROOT / "tests" / "fixtures" / "adapters" / "copy_engine.py"


ENGINE = r"""
import json, sys
from pathlib import Path
import numpy as np
from inferref.tensor import codec
case, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((case / 'testcase.json').read_text())
x = codec.read(case / manifest['inputs'][0]['payload']).as_comparable()
codec.write_array(out / 'out.irtensor', np.asarray(x + 1, dtype=np.float32))
"""


def _case(root: Path, *, contract: str | None = None) -> Path:
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    inp = codec.write_array(root / "inputs" / "x.irtensor", x)
    out = codec.write_array(root / "reference" / "out.irtensor", x + 1)
    im = codec.read(inp).to_metadata()
    om = codec.read(out).to_metadata()
    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.2",
        "name": "add-one",
        "reproducible": True,
        "inputs": [{"name": "x", "value_id": 1, "payload": "inputs/x.irtensor", **im}],
        "outputs": [
            {"name": "out", "value_id": 2, "payload": "reference/out.irtensor", **om}
        ],
        "nodes": [],
        "values": [{"id": 1, **im}, {"id": 2, **om}],
        "requirements": {
            "dtypes": ["float32"],
            "max_rank": 2,
            "features": [],
            **({"contracts": [contract]} if contract else {}),
        },
    }
    if contract:
        manifest["contracts"] = [contract]
    (root / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _fixture(
    tmp_path: Path,
    *,
    dtypes: list[str] | None = None,
    case_contract: str | None = None,
    adapter_contracts: list[str] | None = None,
):
    _case(tmp_path / "cases" / "add", contract=case_contract)
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "format": "inferref-suite",
                "format_version": "0.1",
                "name": "tiny",
                "cases": [{"id": "add", "testcase": "cases/add", "tags": ["smoke"]}],
            }
        ),
        encoding="utf-8",
    )
    script = tmp_path / "engine.py"
    script.write_text(ENGINE, encoding="utf-8")
    adapter = tmp_path / "adapter.json"
    capabilities = {
        "device_types": ["cpu"],
        "dtypes": dtypes or ["float32"],
        "max_rank": 8,
        "features": [],
    }
    if adapter_contracts is not None:
        capabilities["contracts"] = adapter_contracts
    adapter.write_text(
        json.dumps(
            {
                "format": ENGINE_ADAPTER_FORMAT,
                "format_version": ENGINE_ADAPTER_VERSION,
                "name": "python",
                "target_device": "cpu",
                "capabilities": capabilities,
                "command": ["{python}", str(script), "{testcase}", "{output}"],
            }
        ),
        encoding="utf-8",
    )
    return suite, adapter


def test_suite_validate_and_run(tmp_path: Path) -> None:
    suite, adapter = _fixture(tmp_path)
    assert validate_suite(suite)["cases"] == 1
    report = run_suite(suite, adapter, tmp_path / "runs")
    assert report["status"] == "pass"
    assert report["counts"] == {"total": 1, "pass": 1, "unsupported": 0, "failed": 0}


def test_suite_rejects_escape_and_preflights_unsupported(tmp_path: Path) -> None:
    suite, adapter = _fixture(tmp_path, dtypes=["float16"])
    report = run_suite(suite, adapter, tmp_path / "runs")
    assert report["status"] == "fail"
    assert report["cases"][0]["status"] == "unsupported"
    assert report["cases"][0]["run"]["execution"] is None

    data = json.loads(suite.read_text())
    data["cases"][0]["testcase"] = "../outside"
    suite.write_text(json.dumps(data))
    try:
        load_suite(suite)
    except ValueError as exc:
        assert "escape" in str(exc).lower() or "outside" in str(exc).lower()
    else:
        raise AssertionError("suite path escape was accepted")


def test_allow_unsupported_preserves_semantic_status(tmp_path: Path) -> None:
    suite, adapter = _fixture(tmp_path, dtypes=["float16"])
    report = run_suite(suite, adapter, tmp_path / "inventory", allow_unsupported=True)
    assert report["status"] == "unsupported"
    assert report["accepted"] is True
    assert report["exit_code_policy_satisfied"] is True
    assert report["counts"]["pass"] == 0


def test_operation_contract_preflight_and_legacy_unchecked(tmp_path: Path) -> None:
    contract = "softmax/last-dim/v1"
    suite, unchecked = _fixture(tmp_path, case_contract=contract)
    unchecked_report = run_suite(suite, unchecked, tmp_path / "unchecked")
    assert unchecked_report["status"] == "pass"
    assert unchecked_report["cases"][0]["run"]["capability_status"] == "unchecked"

    adapter_data = json.loads(unchecked.read_text(encoding="utf-8"))
    adapter_data["capabilities"]["contracts"] = ["rmsnorm/last-dim/v1"]
    incompatible = tmp_path / "incompatible.adapter.json"
    incompatible.write_text(json.dumps(adapter_data), encoding="utf-8")
    unsupported = run_suite(suite, incompatible, tmp_path / "unsupported")
    run = unsupported["cases"][0]["run"]
    assert run["status"] == "unsupported"
    assert run["execution"] is None
    assert run["unsupported"] == [{"kind": "contract", "required": contract}]


def test_allow_unsupported_matrix_is_partial_when_some_cells_run(
    tmp_path: Path,
) -> None:
    suite, supported = _fixture(
        tmp_path,
        case_contract="softmax/last-dim/v1",
        adapter_contracts=["softmax/last-dim/v1"],
    )
    unsupported_data = json.loads(supported.read_text(encoding="utf-8"))
    unsupported_data["name"] = "unsupported"
    unsupported_data["capabilities"]["contracts"] = ["rmsnorm/last-dim/v1"]
    unsupported = tmp_path / "unsupported.adapter.json"
    unsupported.write_text(json.dumps(unsupported_data), encoding="utf-8")
    report = run_suite(
        suite,
        [supported, unsupported],
        tmp_path / "partial",
        allow_unsupported=True,
    )
    assert report["status"] == "partial"
    assert report["accepted"] is True
    assert report["counts"]["pass"] == 1
    assert report["counts"]["unsupported"] == 1


def test_suite_isolates_cell_configuration_errors(tmp_path: Path) -> None:
    suite, valid = _fixture(tmp_path)
    broken_data = json.loads(valid.read_text(encoding="utf-8"))
    broken_data["name"] = "broken"
    broken_data["cwd"] = "missing-directory"
    broken = tmp_path / "broken.adapter.json"
    broken.write_text(json.dumps(broken_data), encoding="utf-8")

    report = run_suite(suite, [broken, valid], tmp_path / "isolated")
    results = report["cases"][0]["results"]
    assert [item["status"] for item in results] == ["infrastructure_error", "pass"]
    assert results[0]["run"]["error"]["type"] == "AgentProtocolError"
    assert report["status"] == "fail"

    with pytest.raises(Exception, match="working directory"):
        run_suite(suite, broken, tmp_path / "fail-fast", fail_fast=True)


def test_suite_cli(tmp_path: Path, capsys) -> None:
    suite, adapter = _fixture(tmp_path)
    assert main(["suite", "validate", str(suite), "--json"]) == EXIT_OK
    capsys.readouterr()
    assert (
        main(
            [
                "suite",
                "run",
                str(suite),
                "--adapter",
                str(adapter),
                "--runs-dir",
                str(tmp_path / "cli-runs"),
                "--json",
            ]
        )
        == EXIT_OK
    )
    assert json.loads(capsys.readouterr().out)["status"] == "pass"


def test_suite_matrix_and_static_report(tmp_path: Path, capsys) -> None:
    suite, adapter = _fixture(tmp_path)
    second = tmp_path / "adapter-second.json"
    second.write_text(adapter.read_text(encoding="utf-8"), encoding="utf-8")

    report = run_suite(suite, [adapter, second], tmp_path / "matrix-runs")
    assert report["format_version"] == "0.2"
    assert report["counts"] == {"total": 2, "pass": 2, "unsupported": 0, "failed": 0}
    adapter_ids = [item["id"] for item in report["adapters"]]
    assert adapter_ids == ["python", "python-2"]
    assert len(report["cases"][0]["results"]) == 2

    output = tmp_path / "report" / "index.html"
    rendered = render_suite_report(report, output)
    assert rendered["status"] == "pass"
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    assert adapter_ids[1] in output.read_text(encoding="utf-8")

    run_path = tmp_path / "matrix-runs" / "inferref-suite-run.json"
    assert (
        main(
            [
                "suite",
                "report",
                str(run_path),
                "--output",
                str(tmp_path / "cli-report.html"),
                "--json",
            ]
        )
        == EXIT_OK
    )
    assert json.loads(capsys.readouterr().out)["format"] == "inferref-suite-report"


@pytest.mark.parametrize(
    "case_id",
    [
        "../escape",
        "/absolute",
        "C:\\absolute",
        "a/b",
        "a\\b",
        ".",
        "..",
        "case.",
        "case ",
        "CON",
        "nul.txt",
        "LPT1",
    ],
)
def test_suite_rejects_nonportable_case_ids(tmp_path: Path, case_id: str) -> None:
    suite, _ = _fixture(tmp_path)
    data = json.loads(suite.read_text(encoding="utf-8"))
    data["cases"][0]["id"] = case_id
    suite.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_suite(suite)


def test_suite_rejects_casefold_collisions(tmp_path: Path) -> None:
    suite, _ = _fixture(tmp_path)
    data = json.loads(suite.read_text(encoding="utf-8"))
    data["cases"] = [
        {"id": "Case", "testcase": "cases/add"},
        {"id": "case", "testcase": "cases/add"},
    ]
    suite.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="collides"):
        load_suite(suite)


def test_suite_output_uses_hash_backed_contained_artifact_key(tmp_path: Path) -> None:
    suite, adapter = _fixture(tmp_path)
    runs = tmp_path / "runs"
    report = run_suite(suite, adapter, runs)
    output = Path(report["cases"][0]["run"]["output"]).resolve()
    assert output.is_relative_to(runs.resolve())
    assert "add-" in output.parent.name


def test_suite_validate_reports_schema_and_runnable(tmp_path: Path) -> None:
    suite, _ = _fixture(tmp_path)
    report = validate_suite(suite)
    assert report["schema_valid"] is True
    assert report["runnable"] is True
    assert report["status"] == "pass"
    assert report["non_runnable_cases"] == []
    assert report["cases"] == 1


def test_suite_validate_flags_non_reproducible_cases(tmp_path: Path) -> None:
    suite, _ = _fixture(tmp_path)
    case = tmp_path / "cases" / "add" / "testcase.json"
    data = json.loads(case.read_text(encoding="utf-8"))
    data["reproducible"] = False
    case.write_text(json.dumps(data), encoding="utf-8")

    strict = validate_suite(suite)
    assert strict["schema_valid"] is True
    assert strict["runnable"] is False
    assert strict["status"] == "fail"
    assert strict["non_runnable_cases"] == ["add"]

    inventory = validate_suite(suite, allow_nonreproducible=True)
    assert inventory["runnable"] is False
    assert inventory["allow_nonreproducible"] is True
    assert inventory["non_runnable_cases"] == ["add"]


def test_suite_validate_structural_failure_is_schema_invalid(tmp_path: Path) -> None:
    suite, _ = _fixture(tmp_path)
    data = json.loads(suite.read_text(encoding="utf-8"))
    data["cases"][0]["testcase"] = "../outside"
    suite.write_text(json.dumps(data), encoding="utf-8")

    report = validate_suite(suite)
    assert report["schema_valid"] is False
    assert report["runnable"] is False
    assert report["status"] == "fail"
    assert report["error"]


def test_suite_validate_cli_exit_code_follows_runnable_policy(
    tmp_path: Path, capsys
) -> None:
    suite, _ = _fixture(tmp_path)
    case = tmp_path / "cases" / "add" / "testcase.json"
    data = json.loads(case.read_text(encoding="utf-8"))
    data["reproducible"] = False
    case.write_text(json.dumps(data), encoding="utf-8")

    assert main(["suite", "validate", str(suite), "--json"]) == EXIT_FAIL
    capsys.readouterr()
    assert (
        main(["suite", "validate", str(suite), "--allow-nonreproducible", "--json"])
        == EXIT_OK
    )


def _scenario_fixture(tmp_path: Path, *, kind: str = "scenario") -> tuple[Path, Path]:
    suite_dir = tmp_path / "suite-scenario"
    (suite_dir / "scenarios").mkdir(parents=True)
    shutil.copytree(SCENARIO_FIXTURE, suite_dir / "scenarios" / "kv-chain")
    suite = suite_dir / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "format": "inferref-suite",
                "format_version": "0.2",
                "name": "kv-corpus",
                "cases": [
                    {
                        "id": "kv-chain",
                        "kind": kind,
                        "testcase": "scenarios/kv-chain",
                        "tags": ["kv", "stateful"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter.json"
    adapter.write_text(
        json.dumps(
            {
                "format": ENGINE_ADAPTER_FORMAT,
                "format_version": ENGINE_ADAPTER_VERSION,
                "name": "kv-copy",
                "target_device": "cpu",
                "capabilities": {
                    "device_types": ["cpu"],
                    "dtypes": ["float32"],
                    "max_rank": 8,
                    "features": ["multiple_outputs"],
                },
                "command": ["{python}", str(COPY_ENGINE), "{testcase}", "{output}"],
            }
        ),
        encoding="utf-8",
    )
    return suite, adapter


def test_suite_0_2_writer_emits_kind(tmp_path: Path) -> None:
    from inferref.suite.schema import Suite, SuiteCase

    suite = Suite(
        name="tiny",
        source=(tmp_path / "suite.json").resolve(),
        cases=(
            SuiteCase(
                id="kv-chain",
                testcase=(tmp_path / "scenarios" / "kv-chain").resolve(),
                kind="scenario",
            ),
        ),
    )
    data = suite.to_dict()
    assert data["format_version"] == "0.2"
    assert data["cases"][0]["kind"] == "scenario"


def test_suite_0_1_reads_without_kind(tmp_path: Path) -> None:
    suite, _ = _fixture(tmp_path)
    data = json.loads(suite.read_text(encoding="utf-8"))
    assert data["format_version"] == "0.1"
    loaded = load_suite(suite)
    assert loaded.cases[0].kind == "testcase"


def test_suite_scenario_cell_runs_and_report_renders_steps(tmp_path: Path) -> None:
    suite, adapter = _scenario_fixture(tmp_path)
    validation = validate_suite(suite)
    assert validation["schema_valid"] is True
    assert validation["runnable"] is True
    assert validation["non_runnable_cases"] == []

    report = run_suite(suite, adapter, tmp_path / "runs")
    assert report["status"] == "pass"
    assert report["format_version"] == "0.2"
    assert report["counts"] == {"total": 1, "pass": 1, "unsupported": 0, "failed": 0}
    cell = report["cases"][0]["results"][0]
    assert cell["status"] == "pass"
    assert cell["run"]["format"] == "inferref-scenario-run"
    assert [step["id"] for step in cell["run"]["steps"]] == [
        "prefill",
        "decode-0",
        "decode-1",
    ]

    rendered = render_suite_report(report, tmp_path / "report.html")
    summary = rendered["matrix"][0]["engines"]["kv-copy"]
    assert summary["kind"] == "scenario"
    assert summary["step_counts"] == {"pass": 3}
    assert summary["first_failed_step"] is None
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "steps: 3 pass" in html


def test_suite_report_renders_failed_scenario_steps(tmp_path: Path) -> None:
    suite, adapter = _scenario_fixture(tmp_path)
    data = json.loads(adapter.read_text(encoding="utf-8"))
    data["environment"] = {"INFERREF_ENGINE_STATE_CORRUPTION": "value"}
    adapter.write_text(json.dumps(data), encoding="utf-8")
    report = run_suite(
        suite,
        adapter,
        tmp_path / "runs",
    )
    cell = report["cases"][0]["results"][0]
    assert cell["status"] == "fail"
    rendered = render_suite_report(report, tmp_path / "report.html")
    summary = rendered["matrix"][0]["engines"]["kv-copy"]
    assert summary["first_failed_step"] == "prefill"
    assert summary["step_counts"]["mismatch"] == 3
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "first failed step: prefill" in html


def test_suite_rejects_invalid_kind(tmp_path: Path) -> None:
    suite, _ = _scenario_fixture(tmp_path, kind="benchmark")
    with pytest.raises(ValueError, match="kind"):
        load_suite(suite)


def test_suite_flags_non_reproducible_scenario_steps(tmp_path: Path) -> None:
    suite, _ = _scenario_fixture(tmp_path)
    testcase = tmp_path / "suite-scenario" / "scenarios" / "kv-chain"
    case = testcase / "cases" / "decode-0" / "testcase.json"
    data = json.loads(case.read_text(encoding="utf-8"))
    data["reproducible"] = False
    case.write_text(json.dumps(data), encoding="utf-8")

    validation = validate_suite(suite)
    assert validation["schema_valid"] is True
    assert validation["runnable"] is False
    assert validation["non_runnable_cases"] == ["kv-chain"]


def test_suite_scenario_path_escape_is_schema_invalid(tmp_path: Path) -> None:
    suite, _ = _scenario_fixture(tmp_path)
    data = json.loads(suite.read_text(encoding="utf-8"))
    data["cases"][0]["testcase"] = "../outside"
    suite.write_text(json.dumps(data), encoding="utf-8")
    report = validate_suite(suite)
    assert report["schema_valid"] is False
    assert report["status"] == "fail"


def test_report_requires_html_extension(tmp_path: Path) -> None:
    suite, adapter = _fixture(tmp_path)
    report = run_suite(suite, adapter, tmp_path / "runs")
    run_path = tmp_path / "suite-run.json"
    run_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.html"):
        render_suite_report(run_path, tmp_path / "report.json")

    rendered = render_suite_report(run_path, tmp_path / "report.html")
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "report.json").is_file()
    assert rendered["html"] == str((tmp_path / "report.html").resolve())
    assert rendered["json"] == str((tmp_path / "report.json").resolve())
