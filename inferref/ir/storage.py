"""Storage identity records (IR §14-§16).

A storage record describes *identity*, not bytes: which tensor values alias the
same backing allocation, and which mutation generations of it were observed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferref.ir._common import Record, drop_none
from inferref.ir.tensor_value import Device


@dataclass
class StorageRecord(Record):
    """One backing storage observed during the trace (IR §16)."""

    id: int = -1
    device: Device = field(default_factory=Device)
    storage_dtype: str | None = None
    observed_versions: tuple[int, ...] = ()
    notes: str | None = None

    _KNOWN = ("id", "device", "storage_dtype", "observed_versions", "notes")

    def _encode(self) -> dict[str, Any]:
        return drop_none(
            {
                "id": self.id,
                "device": self.device.to_dict(),
                "storage_dtype": self.storage_dtype,
                "observed_versions": sorted(self.observed_versions),
                "notes": self.notes,
            }
        )

    @classmethod
    def _decode(cls, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(data["id"]),
            "device": Device.from_dict(data.get("device")),
            "storage_dtype": data.get("storage_dtype"),
            "observed_versions": tuple(data.get("observed_versions", ())),
            "notes": data.get("notes"),
        }
