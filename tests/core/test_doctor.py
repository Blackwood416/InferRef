from __future__ import annotations

import json

from inferref.cli.main import EXIT_OK, main
from inferref.doctor import DOCTOR_FORMAT, DOCTOR_FORMAT_VERSION, run_doctor


def test_doctor_envelope_is_stable() -> None:
    report = run_doctor()
    assert report["format"] == DOCTOR_FORMAT
    assert report["format_version"] == DOCTOR_FORMAT_VERSION
    assert report["status"] in {"pass", "warn"}
    assert {item["id"] for item in report["checks"]} >= {
        "runtime.python",
        "runtime.inferref",
        "runtime.numpy",
    }


def test_doctor_cli_json(capsys) -> None:
    assert main(["doctor", "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["format"] == DOCTOR_FORMAT


def test_doctor_cpu_is_required_and_smoked_when_torch_is_present() -> None:
    report = run_doctor("cpu")
    torch_check = next(
        item for item in report["checks"] if item["id"] == "frontend.torch"
    )
    if torch_check["status"] == "pass":
        smoke = next(
            item for item in report["checks"] if item["id"] == "device.cpu.smoke"
        )
        assert smoke["status"] == "pass"
        assert report["status"] != "fail"
    else:
        assert report["status"] == "fail"
