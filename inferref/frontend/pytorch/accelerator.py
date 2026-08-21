"""PyTorch accelerator discovery, synchronization, and host materialization.

The trace format is device-neutral.  This module contains the small amount of
frontend-specific policy needed to make asynchronous accelerator reads
explicit and testable instead of relying on ``Tensor.cpu()`` to happen to
block on a particular backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DeviceDetails:
    id: int
    name: str
    driver_version: str | None = None
    total_memory: int | None = None
    available_memory: int | None = None
    compute_capability: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "name": self.name}
        if self.driver_version is not None:
            out["driver_version"] = self.driver_version
        if self.total_memory is not None:
            out["total_memory"] = self.total_memory
        if self.available_memory is not None:
            out["available_memory"] = self.available_memory
        if self.compute_capability is not None:
            out["compute_capability"] = self.compute_capability
        return out


@dataclass(frozen=True)
class AcceleratorInfo:
    type: str
    available: bool
    device_count: int
    device_names: tuple[str, ...] = ()
    devices: tuple[DeviceDetails, ...] = ()
    runtime_version: str | None = None
    onednn_version: str | None = None
    sycl_version: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type,
            "available": self.available,
            "device_count": self.device_count,
            "device_names": list(self.device_names),
        }
        if self.devices:
            out["devices"] = [d.to_dict() for d in self.devices]
        if self.runtime_version is not None:
            out["runtime_version"] = self.runtime_version
        if self.onednn_version is not None:
            out["onednn_version"] = self.onednn_version
        if self.sycl_version is not None:
            out["sycl_version"] = self.sycl_version
        if self.reason:
            out["reason"] = self.reason
        return out


def accelerator_module(device_type: str) -> Any | None:
    """Return PyTorch's runtime module for an asynchronous device type."""
    if device_type == "cpu":
        return None
    return getattr(torch, device_type, None)


def _get_onednn_version() -> str | None:
    try:
        if hasattr(torch.backends, "mkldnn") and torch.backends.mkldnn.is_available():
            # Best-effort version extraction
            return getattr(torch.backends.mkldnn, "__version__", "enabled")
    except Exception:
        pass
    return None


def _get_sycl_or_oneapi_version() -> str | None:
    try:
        sycl_ver = getattr(torch.version, "sycl", None) or getattr(torch.version, "xpu", None)
        if sycl_ver:
            return str(sycl_ver)
        import os

        oneapi_root = os.environ.get("ONEAPI_ROOT") or os.environ.get("ONEAPI_DIR")
        if oneapi_root:
            return f"installed at {oneapi_root}"
    except Exception:
        pass
    return None


def accelerator_info(device_type: str) -> AcceleratorInfo:
    onednn = _get_onednn_version()
    sycl_ver = _get_sycl_or_oneapi_version() if device_type == "xpu" else None
    if device_type == "cpu":
        cpu_device = DeviceDetails(id=0, name="CPU")
        return AcceleratorInfo(
            "cpu",
            True,
            1,
            ("CPU",),
            devices=(cpu_device,),
            onednn_version=onednn,
            sycl_version=sycl_ver,
        )
    module = accelerator_module(device_type)
    if module is None or not callable(getattr(module, "is_available", None)):
        return AcceleratorInfo(
            device_type,
            False,
            0,
            reason="backend unavailable",
            onednn_version=onednn,
            sycl_version=sycl_ver,
        )
    try:
        available = bool(module.is_available())
        count = int(module.device_count()) if available else 0
        names: list[str] = []
        devices: list[DeviceDetails] = []
        runtime_ver = getattr(torch.version, device_type, None)

        for index in range(count):
            dev_name = str(module.get_device_name(index))
            names.append(dev_name)
            tot_mem: int | None = None
            avail_mem: int | None = None
            cap_str: str | None = None

            try:
                props = module.get_device_properties(index)
                tot_mem = getattr(props, "total_memory", None)
            except Exception:
                pass

            try:
                if hasattr(module, "mem_get_info"):
                    free, total = module.mem_get_info(index)
                    avail_mem = free
                    if tot_mem is None:
                        tot_mem = total
            except Exception:
                pass

            try:
                if hasattr(module, "get_device_capability"):
                    cap = module.get_device_capability(index)
                    if isinstance(cap, (tuple, list)) and len(cap) >= 2:
                        cap_str = f"{cap[0]}.{cap[1]}"
            except Exception:
                pass

            devices.append(
                DeviceDetails(
                    id=index,
                    name=dev_name,
                    driver_version=str(runtime_ver) if runtime_ver else None,
                    total_memory=tot_mem,
                    available_memory=avail_mem,
                    compute_capability=cap_str,
                )
            )

        return AcceleratorInfo(
            device_type,
            available,
            count,
            tuple(names),
            devices=tuple(devices),
            runtime_version=str(runtime_ver) if runtime_ver else None,
            onednn_version=onednn,
        )
    except Exception as exc:
        return AcceleratorInfo(
            device_type,
            False,
            0,
            reason=f"{type(exc).__name__}: {exc}",
            onednn_version=onednn,
        )


def synchronize_device(device: torch.device | str) -> None:
    """Wait for every operation relevant to ``device`` before host access."""
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return
    module = accelerator_module(resolved.type)
    synchronize = getattr(module, "synchronize", None) if module is not None else None
    if not callable(synchronize):
        raise RuntimeError(
            f"PyTorch backend {resolved.type!r} has no synchronize() contract"
        )
    try:
        synchronize(resolved)
    except TypeError:
        # Some PyTorch backends expose a process-wide synchronize() with no
        # device argument (for example older MPS releases).
        synchronize()


def materialize_to_host(tensor: torch.Tensor) -> torch.Tensor:
    """Return a contiguous CPU snapshot whose bytes are safe to read.

    For an accelerator tensor, contiguous packing is first enqueued on the
    source device, then the source device is explicitly synchronized, and the
    transfer is requested as blocking.  Callers therefore never need an
    out-of-band ``cuda.synchronize``/``xpu.synchronize``.
    """
    flat = tensor.detach().contiguous()
    if flat.device.type == "cpu":
        return flat
    synchronize_device(flat.device)
    return flat.to(device="cpu", non_blocking=False, copy=True).contiguous()


def infer_execution_device(devices: set[str]) -> str:
    """Collapse observed value devices into the manifest's legacy label."""
    if not devices:
        return "unknown"
    if len(devices) == 1:
        return next(iter(devices))
    return "mixed"
