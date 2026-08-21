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
        "comparator.registry",
    }
    assert "comparators" in report


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


def test_invalid_device_returns_structured_failure(capsys) -> None:
    assert main(["doctor", "--device", "banana", "--json"]) != EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "fail"
    check = next(item for item in report["checks"] if item["id"] == "device.request")
    assert "banana" in check["message"]


def test_doctor_only_loads_plugins_when_explicitly_requested(monkeypatch) -> None:
    calls = []
    comp_calls = []

    def descriptors(*, load=False):
        calls.append(load)
        return []

    def comp_statuses(*, load=False):
        comp_calls.append(load)
        return []

    monkeypatch.setattr("inferref.semantic.registry.plugin_descriptors", descriptors)
    monkeypatch.setattr("inferref.comparators.comparator_plugin_statuses", comp_statuses)
    run_doctor()
    run_doctor(verify_plugins=True)
    assert calls == [False, True]
    assert comp_calls == [False, True]


def test_doctor_hardware_details_and_device_names(capsys) -> None:
    report = run_doctor("cpu")
    cpu_check = next(item for item in report["checks"] if item["id"] == "device.cpu.availability")
    if cpu_check["status"] == "pass":
        assert "CPU" in cpu_check["message"]
        assert "device_names" in cpu_check["details"]
        assert "devices" in cpu_check["details"]

    assert main(["doctor"]) == EXIT_OK
    text_out = capsys.readouterr().out
    assert "InferRef doctor:" in text_out
