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
class AcceleratorInfo:
    type: str
    available: bool
    device_count: int
    device_names: tuple[str, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "available": self.available,
            "device_count": self.device_count,
            "device_names": list(self.device_names),
            **({"reason": self.reason} if self.reason else {}),
        }


def accelerator_module(device_type: str) -> Any | None:
    """Return PyTorch's runtime module for an asynchronous device type."""
    if device_type == "cpu":
        return None
    return getattr(torch, device_type, None)


def accelerator_info(device_type: str) -> AcceleratorInfo:
    if device_type == "cpu":
        return AcceleratorInfo("cpu", True, 1, ("CPU",))
    module = accelerator_module(device_type)
    if module is None or not callable(getattr(module, "is_available", None)):
        return AcceleratorInfo(device_type, False, 0, reason="backend unavailable")
    try:
        available = bool(module.is_available())
        count = int(module.device_count()) if available else 0
        names = tuple(str(module.get_device_name(index)) for index in range(count))
        return AcceleratorInfo(device_type, available, count, names)
    except Exception as exc:
        return AcceleratorInfo(
            device_type,
            False,
            0,
            reason=f"{type(exc).__name__}: {exc}",
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
