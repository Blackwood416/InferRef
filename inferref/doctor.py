"""Environment diagnostics for tracing and accelerator capture."""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any

from inferref.ir.version import INFERREF_VERSION

DOCTOR_FORMAT = "inferref-doctor"
DOCTOR_FORMAT_VERSION = "0.1"


@dataclass(frozen=True)
class DoctorCheck:
    id: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            **({"remediation": self.remediation} if self.remediation else {}),
        }


def run_doctor(device: str | None = None) -> dict[str, Any]:
    """Inspect the core install and optionally require a usable device."""
    requested = _normalise_device(device)
    checks = [
        DoctorCheck(
            "runtime.python",
            "pass",
            f"{platform.python_implementation()} {platform.python_version()}",
            {"executable": sys.executable},
        ),
        DoctorCheck("runtime.inferref", "pass", f"InferRef {INFERREF_VERSION}"),
    ]
    try:
        import numpy

        checks.append(DoctorCheck("runtime.numpy", "pass", f"NumPy {numpy.__version__}"))
    except Exception as exc:  # pragma: no cover - core dependency failure
        checks.append(DoctorCheck("runtime.numpy", "fail", str(exc)))

    try:
        import torch
    except Exception as exc:
        status = "fail" if requested not in (None, "cpu") else "warn"
        checks.append(
            DoctorCheck(
                "frontend.torch",
                status,
                f"PyTorch is unavailable: {exc}",
                remediation="Install an InferRef-compatible PyTorch build.",
            )
        )
    else:
        checks.extend(_torch_checks(torch, requested))

    from inferref.semantic.registry import plugin_descriptors

    for plugin in plugin_descriptors(load=True):
        checks.append(
            DoctorCheck(
                f"semantic.plugin.{plugin.name}",
                "pass" if plugin.status == "loaded" else "warn",
                (
                    f"{plugin.distribution or 'unknown distribution'} "
                    f"{plugin.version or ''}: {plugin.status}"
                ).strip(),
                plugin.to_dict(),
                remediation=plugin.error,
            )
        )

    status = "fail" if any(item.status == "fail" for item in checks) else (
        "warn"
        if requested is None and any(item.status == "warn" for item in checks)
        else "pass"
    )
    return {
        "format": DOCTOR_FORMAT,
        "format_version": DOCTOR_FORMAT_VERSION,
        "status": status,
        "requested_device": requested,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "inferref": INFERREF_VERSION,
        },
        "checks": [item.to_dict() for item in checks],
    }


def _normalise_device(device: str | None) -> str | None:
    if device is None or device == "auto":
        return None
    return str(device)


def _torch_checks(torch: Any, requested: str | None) -> list[DoctorCheck]:
    from inferref.frontend.pytorch.accelerator import (
        accelerator_info,
        materialize_to_host,
        synchronize_device,
    )

    checks = [DoctorCheck("frontend.torch", "pass", f"PyTorch {torch.__version__}")]
    private_apis = {
        "TorchDispatchMode": "torch.utils._python_dispatch",
        "StorageWeakRef": "torch.multiprocessing.reductions",
    }
    missing = []
    for symbol, module_name in private_apis.items():
        try:
            module = __import__(module_name, fromlist=[symbol])
            getattr(module, symbol)
        except Exception as exc:
            missing.append(f"{symbol}: {exc}")
    checks.append(
        DoctorCheck(
            "frontend.private_apis",
            "fail" if missing else "pass",
            "; ".join(missing) if missing else "required dispatch APIs are present",
        )
    )

    requested_type = torch.device(requested).type if requested else None
    for device_type in ("cpu", "cuda", "xpu"):
        info = accelerator_info(device_type)
        required = requested_type == device_type
        if not info.available:
            checks.append(
                DoctorCheck(
                    f"device.{device_type}.availability",
                    "fail" if required else "warn",
                    info.reason or f"{device_type} is unavailable",
                    info.to_dict(),
                )
            )
            continue
        checks.append(
            DoctorCheck(
                f"device.{device_type}.availability",
                "pass",
                f"{info.device_count} device(s) available",
                info.to_dict(),
            )
        )
        if required or device_type == "cpu":
            target = requested if required and requested is not None else "cpu"
            try:
                value = torch.arange(8, device=target, dtype=torch.float32)
                result = value.mul(3).add(1)
                synchronize_device(result.device)
                host = materialize_to_host(result)
                expected = torch.arange(8, dtype=torch.float32).mul(3).add(1)
                if not torch.equal(host, expected):
                    raise RuntimeError("host materialization returned incorrect values")
            except Exception as exc:
                checks.append(
                    DoctorCheck(
                        f"device.{device_type}.smoke",
                        "fail",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                checks.append(
                    DoctorCheck(
                        f"device.{device_type}.smoke",
                        "pass",
                        f"allocation, arithmetic, synchronize and host copy passed on {target}",
                    )
                )

    if requested_type not in {"cpu", "cuda", "xpu", None}:
        checks.append(
            DoctorCheck(
                f"device.{requested_type}.availability",
                "fail",
                f"unsupported doctor device type {requested_type!r}",
            )
        )
    return checks


def render_doctor(report: dict[str, Any]) -> str:
    lines = [f"InferRef doctor: {report['status'].upper()}"]
    for check in report["checks"]:
        lines.append(f"  {check['status'].upper():4s}  {check['id']}: {check['message']}")
        if check.get("remediation"):
            lines.append(f"        {check['remediation']}")
    return "\n".join(lines)
