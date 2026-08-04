from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from inferref.agent.protocol import ENGINE_ADAPTER_FORMAT, ENGINE_ADAPTER_VERSION
from inferref.cli.main import EXIT_FAIL, EXIT_OK, main
from inferref.suite import load_suite, render_suite_report, run_suite, validate_suite
from inferref.tensor import codec


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


def _case(root: Path) -> Path:
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
        "outputs": [{"name": "out", "value_id": 2, "payload": "reference/out.irtensor", **om}],
        "nodes": [],
        "values": [{"id": 1, **im}, {"id": 2, **om}],
        "requirements": {"dtypes": ["float32"], "max_rank": 2, "features": []},
    }
    (root / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _fixture(tmp_path: Path, *, dtypes: list[str] | None = None):
    case = _case(tmp_path / "cases" / "add")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({
        "format": "inferref-suite", "format_version": "0.1", "name": "tiny",
        "cases": [{"id": "add", "testcase": "cases/add", "tags": ["smoke"]}],
    }), encoding="utf-8")
    script = tmp_path / "engine.py"
    script.write_text(ENGINE, encoding="utf-8")
    adapter = tmp_path / "adapter.json"
    adapter.write_text(json.dumps({
        "format": ENGINE_ADAPTER_FORMAT, "format_version": ENGINE_ADAPTER_VERSION,
        "name": "python", "target_device": "cpu",
        "capabilities": {"device_types": ["cpu"], "dtypes": dtypes or ["float32"], "max_rank": 8, "features": []},
        "command": ["{python}", str(script), "{testcase}", "{output}"],
    }), encoding="utf-8")
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


def test_suite_cli(tmp_path: Path, capsys) -> None:
    suite, adapter = _fixture(tmp_path)
    assert main(["suite", "validate", str(suite), "--json"]) == EXIT_OK
    capsys.readouterr()
    assert main(["suite", "run", str(suite), "--adapter", str(adapter), "--runs-dir", str(tmp_path / "cli-runs"), "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["status"] == "pass"


def test_suite_matrix_and_static_report(tmp_path: Path, capsys) -> None:
    suite, adapter = _fixture(tmp_path)
    second = tmp_path / "adapter-second.json"
    second.write_text(adapter.read_text(encoding="utf-8"), encoding="utf-8")

    report = run_suite(suite, [adapter, second], tmp_path / "matrix-runs")
    assert report["format_version"] == "0.2"
    assert report["counts"] == {"total": 2, "pass": 2, "unsupported": 0, "failed": 0}
    assert [item["id"] for item in report["adapters"]] == ["python", "python-2"]
    assert len(report["cases"][0]["results"]) == 2

    output = tmp_path / "report" / "index.html"
    rendered = render_suite_report(report, output)
    assert rendered["status"] == "pass"
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    assert "python-2" in output.read_text(encoding="utf-8")

    run_path = tmp_path / "matrix-runs" / "inferref-suite-run.json"
    assert main(["suite", "report", str(run_path), "--output", str(tmp_path / "cli-report.html"), "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out)["format"] == "inferref-suite-report"
