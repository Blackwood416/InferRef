from __future__ import annotations

import json
import builtins

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


def test_explicit_cpu_fails_when_torch_is_missing(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("blocked for doctor contract test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    report = run_doctor("cpu")
    assert report["status"] == "fail"
    check = next(item for item in report["checks"] if item["id"] == "frontend.torch")
    assert check["status"] == "fail"
