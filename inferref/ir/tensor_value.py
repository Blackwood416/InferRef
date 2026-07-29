"""Tensor value records (IR §12-§20, §38).

A tensor value is an **immutable snapshot** of one tensor state at one point in
the execution (IR §2.4). An in-place op therefore produces two distinct value
records that share ``runtime_object_id`` and ``storage_id`` but differ in
``storage_version``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferref.ir._common import Record, drop_none
from inferref.ir.dtypes import dtype_itemsize

#: Tensor capture modes (IR §18).
CAPTURE_MODES = ("none", "metadata", "hash", "sample", "full")

#: Value roles (IR §38).
ROLES = (
    "activation",
    "parameter",
    "buffer",
    "input",
    "output",
    "constant",
    "unknown",
)


@dataclass(frozen=True)
class Device:
    """Device placement (IR §12)."""

    type: str = "cpu"
    index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "index": self.index}

    @classmethod
    def from_dict(cls, data: Any) -> "Device":
        if data is None:
            return cls()
        if isinstance(data, str):
            # Tolerate the flat form used by the SPEC manifest examples.
            head, _, idx = data.partition(":")
            return cls(type=head, index=int(idx) if idx else None)
        return cls(type=data.get("type", "cpu"), index=data.get("index"))

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


@dataclass(frozen=True)
class TensorHash:
    """A content hash over an explicit domain (IR §19).

    ``domain`` MUST be explicit; hashes over different domains are never
    comparable.
    """

    algorithm: str
    domain: str
    value: str

    def to_dict(self) -> dict[str, Any]:
        return {"algorithm": self.algorithm, "domain": self.domain, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorHash":
        return cls(
            algorithm=data.get("algorithm", "unknown"),
            domain=data.get("domain", "unknown"),
            value=data.get("value", ""),
        )


@dataclass
class CaptureInfo:
    """How much of a tensor's data was retained (IR §18, §19)."""

    mode: str = "metadata"
    payload: str | None = None
    hashes: tuple[TensorHash, ...] = ()
    sample: list[Any] | None = None
    #: Present when capture policy requested more than ``mode`` retained.
    requested_mode: str | None = None
    #: Stable machine-readable reason for the downgrade.
    degraded_reason: str | None = None
    #: Policy limit involved in the downgrade, when applicable.
    limit: int | None = None
    #: Tensor size evaluated against ``limit``, when applicable.
    logical_numel: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode}
        if self.payload is not None:
            out["payload"] = self.payload
        if self.hashes:
            out["hashes"] = [h.to_dict() for h in self.hashes]
        if self.sample is not None:
            out["sample"] = self.sample
        if self.requested_mode is not None:
            out["requested_mode"] = self.requested_mode
        if self.degraded_reason is not None:
            out["degraded_reason"] = self.degraded_reason
        if self.limit is not None:
            out["limit"] = self.limit
        if self.logical_numel is not None:
            out["logical_numel"] = self.logical_numel
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CaptureInfo":
        if not data:
            return cls(mode="none")
        return cls(
            mode=data.get("mode", "metadata"),
            payload=data.get("payload"),
            hashes=tuple(TensorHash.from_dict(h) for h in data.get("hashes", ())),
            sample=data.get("sample"),
            requested_mode=data.get("requested_mode"),
            degraded_reason=data.get("degraded_reason"),
            limit=data.get("limit"),
            logical_numel=data.get("logical_numel"),
        )


@dataclass
class TensorValueRecord(Record):
    """One immutable tensor state (IR §12)."""

    id: int = -1
    dtype: str = "float32"
    shape: tuple[int, ...] = ()
    stride: tuple[int, ...] = ()
    device: Device = field(default_factory=Device)
    storage_offset_elements: int = 0

    #: Framework tensor object identity — diagnostic only, never an address (IR §13).
    runtime_object_id: int | None = None
    #: Aliased backing storage identity within this trace (IR §14).
    storage_id: int | None = None
    #: Logical mutation generation of ``storage_id`` (IR §15).
    storage_version: int = 0

    contiguous: bool = True
    requires_grad: bool = False

    producer: int | None = None
    consumers: tuple[int, ...] = ()

    #: IR §38 classification.
    role: str = "unknown"
    qualified_name: str | None = None
    #: Every name bound to this tensor's storage, present only when a storage is
    #: shared by more than one parameter/buffer — i.e. tied weights. The
    #: canonical name stays in ``qualified_name``; this is additive so older
    #: readers are unaffected (IR §46).
    qualified_names: tuple[str, ...] = ()

    capture: CaptureInfo = field(default_factory=CaptureInfo)

    kind: str = "tensor"

    _KNOWN = (
        "id",
        "kind",
        "dtype",
        "shape",
        "stride",
        "device",
        "storage_offset_elements",
        "logical_numel",
        "logical_nbytes",
        "runtime_object_id",
        "storage_id",
        "storage_version",
        "contiguous",
        "requires_grad",
        "producer",
        "consumers",
        "role",
        "qualified_name",
        "qualified_names",
        "capture",
    )

    @property
    def logical_numel(self) -> int:
        """Number of logical elements (IR §12)."""
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    @property
    def logical_nbytes(self) -> int:
        """Logical byte size, i.e. ``numel * itemsize`` (IR §12, SPEC §12)."""
        return self.logical_numel * dtype_itemsize(self.dtype)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def _encode(self) -> dict[str, Any]:
        return drop_none(
            {
                "id": self.id,
                "kind": "tensor",
                "dtype": self.dtype,
                "shape": list(self.shape),
                "stride": list(self.stride),
                "device": self.device.to_dict(),
                "storage_offset_elements": self.storage_offset_elements,
                "logical_numel": self.logical_numel,
                "logical_nbytes": self.logical_nbytes,
                "runtime_object_id": self.runtime_object_id,
                "storage_id": self.storage_id,
                "storage_version": self.storage_version,
                "contiguous": self.contiguous,
                "requires_grad": self.requires_grad,
                "producer": self.producer,
                "consumers": list(self.consumers),
                "role": self.role,
                "qualified_name": self.qualified_name,
                "qualified_names": list(self.qualified_names) or None,
                "capture": self.capture.to_dict(),
            }
        )

    @classmethod
    def _decode(cls, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(data["id"]),
            "dtype": data.get("dtype", "float32"),
            "shape": tuple(data.get("shape", ())),
            "stride": tuple(data.get("stride", ())),
            "device": Device.from_dict(data.get("device")),
            "storage_offset_elements": int(data.get("storage_offset_elements", 0)),
            "runtime_object_id": data.get("runtime_object_id"),
            "storage_id": data.get("storage_id"),
            "storage_version": int(data.get("storage_version", 0)),
            "contiguous": bool(data.get("contiguous", True)),
            "requires_grad": bool(data.get("requires_grad", False)),
            "producer": data.get("producer"),
            "consumers": tuple(data.get("consumers", ())),
            "role": data.get("role", "unknown"),
            "qualified_name": data.get("qualified_name"),
            "qualified_names": tuple(data.get("qualified_names") or ()),
            "capture": CaptureInfo.from_dict(data.get("capture")),
            "kind": "tensor",
        }

    def layout_signature(self) -> tuple[Any, ...]:
        """Shape/layout signature used for testcase dedup (SPEC §24)."""
        return (
            self.dtype,
            self.rank,
            self.shape,
            self.stride,
            self.storage_offset_elements,
            self.contiguous,
        )
