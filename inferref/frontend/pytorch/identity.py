"""Tensor, storage and value identity tracking (IR §13-§15, §39).

Three distinct identities are maintained:

``runtime_object_id``
    The framework tensor *object* observed. Diagnostic only (IR §13).

``storage_id``
    The backing allocation. Shared by views (IR §14). Keyed by ``data_ptr``, but
    guarded with a :class:`StorageWeakRef`: CPython/PyTorch reuse freed
    addresses, so a bare ``data_ptr`` table would eventually alias two unrelated
    storages together.

``storage_version``
    The logical mutation generation (IR §15). **InferRef maintains this itself.**
    ``torch.Tensor._version`` cannot be used: the version counter is bumped by
    the autograd layer, which sits *above* ``TorchDispatchMode``, so from inside
    ``__torch_dispatch__`` the counter reads identically before and after the
    operator runs. Versions are instead advanced from the operator schema's
    declared writes (see :mod:`inferref.frontend.pytorch.dispatch`).

Value records are interned on ``(storage_id, storage_version, dtype, shape,
stride, storage_offset)``. Two tensors matching on that tuple are by
construction the same immutable logical value, which is what delivers parameter
deduplication (IR §39): one weight shared by 32 layers yields a single value
record and a single payload.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from inferref.ir.storage import StorageRecord
from inferref.ir.tensor_value import Device, TensorValueRecord


def torch_dtype_name(dtype: torch.dtype) -> str:
    """Map a ``torch.dtype`` to its stable InferRef name (IR §2.6, §11)."""
    name = str(dtype)
    if name.startswith("torch."):
        name = name[len("torch.") :]
    # Normalise PyTorch's aliases onto the stable IR names.
    return {"float": "float32", "double": "float64", "half": "float16", "long": "int64"}.get(
        name, name
    )


def torch_device_of(tensor: torch.Tensor) -> Device:
    device = tensor.device
    return Device(type=device.type, index=device.index)


@dataclass
class _StorageSlot:
    storage_id: int
    weak: StorageWeakRef
    device: Device
    dtype: str


@dataclass
class Identity:
    """Assigns and remembers InferRef ids for tensors, storages and values."""

    _storage_by_ptr: dict[int, _StorageSlot] = field(default_factory=dict)
    _storage_versions: dict[int, int] = field(default_factory=dict)
    _storage_observed: dict[int, set[int]] = field(default_factory=dict)
    _storage_records: dict[int, StorageRecord] = field(default_factory=dict)

    _object_by_pyid: dict[int, int] = field(default_factory=dict)
    _object_keepalive: dict[int, Any] = field(default_factory=dict)

    _value_by_key: dict[tuple, int] = field(default_factory=dict)
    _values: dict[int, TensorValueRecord] = field(default_factory=dict)

    _next_storage_id: int = 1
    _next_object_id: int = 1
    _next_value_id: int = 1

    # -- storage ----------------------------------------------------------

    def storage_id_of(self, tensor: torch.Tensor) -> int | None:
        """Return the storage id backing ``tensor``, allocating one if new."""
        try:
            storage = tensor.untyped_storage()
        except (RuntimeError, NotImplementedError, AttributeError):
            # Sparse / meta / functional tensors may have no accessible storage.
            return None

        ptr = storage.data_ptr()
        slot = self._storage_by_ptr.get(ptr)
        if slot is not None and not slot.weak.expired():
            return slot.storage_id

        # Either unseen, or the previous occupant of this address has been freed
        # and the address recycled — issue a fresh identity either way.
        storage_id = self._next_storage_id
        self._next_storage_id += 1
        device = torch_device_of(tensor)
        dtype = torch_dtype_name(tensor.dtype)
        self._storage_by_ptr[ptr] = _StorageSlot(
            storage_id=storage_id, weak=StorageWeakRef(storage), device=device, dtype=dtype
        )
        self._storage_versions[storage_id] = 0
        self._storage_observed[storage_id] = {0}
        self._storage_records[storage_id] = StorageRecord(
            id=storage_id, device=device, storage_dtype=dtype, observed_versions=(0,)
        )
        return storage_id

    def version_of(self, storage_id: int | None) -> int:
        """Current InferRef mutation generation of ``storage_id``."""
        if storage_id is None:
            return 0
        return self._storage_versions.get(storage_id, 0)

    def bump_version(self, storage_id: int) -> tuple[int, int]:
        """Advance ``storage_id`` to its next generation (IR §15).

        Returns ``(version_before, version_after)``.
        """
        before = self._storage_versions.get(storage_id, 0)
        after = before + 1
        self._storage_versions[storage_id] = after
        self._storage_observed.setdefault(storage_id, {before}).add(after)
        return before, after

    def storage_records(self) -> list[StorageRecord]:
        """Finalised storage records, with observed versions filled in (IR §16)."""
        out: list[StorageRecord] = []
        for storage_id, record in sorted(self._storage_records.items()):
            record.observed_versions = tuple(sorted(self._storage_observed.get(storage_id, {0})))
            out.append(record)
        return out

    # -- runtime object ---------------------------------------------------

    def object_id_of(self, tensor: torch.Tensor) -> int:
        """Return the ``runtime_object_id`` for ``tensor`` (IR §13)."""
        pyid = id(tensor)
        existing = self._object_by_pyid.get(pyid)
        if existing is not None and self._object_keepalive.get(pyid) is not None:
            ref = self._object_keepalive[pyid]
            if ref() is not None:
                return existing
        object_id = self._next_object_id
        self._next_object_id += 1
        self._object_by_pyid[pyid] = object_id
        try:
            self._object_keepalive[pyid] = weakref.ref(tensor)
        except TypeError:  # pragma: no cover - tensor subclasses without weakref
            self._object_keepalive[pyid] = lambda: tensor
        return object_id

    # -- values -----------------------------------------------------------

    def value_key(self, tensor: torch.Tensor, storage_id: int | None) -> tuple:
        return (
            storage_id,
            self.version_of(storage_id),
            torch_dtype_name(tensor.dtype),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
        )

    def intern(self, tensor: torch.Tensor, memo: dict[tuple, int] | None = None) -> tuple[int, bool]:
        """Return ``(value_id, is_new)`` for the current state of ``tensor``.

        Repeat observations of an unchanged tensor collapse onto one record —
        this is where parameter deduplication happens (IR §39).

        Passing ``memo`` switches to *output* interning: a fresh record is
        allocated even when an identical one exists, and the intern table is
        rebound to it. Without this, an alias-only operator such as
        ``aten.detach`` would map its output onto its own input, giving a value
        two producers and a self-loop in the dataflow graph. ``memo`` is scoped
        to one operator so a tensor appearing twice in one result still maps to
        a single value.
        """
        storage_id = self.storage_id_of(tensor)
        key = self.value_key(tensor, storage_id)

        if memo is None:
            existing = self._value_by_key.get(key)
            if existing is not None:
                return existing, False
        else:
            memoized = memo.get(key)
            if memoized is not None:
                return memoized, False

        value_id = self._next_value_id
        self._next_value_id += 1
        record = TensorValueRecord(
            id=value_id,
            dtype=torch_dtype_name(tensor.dtype),
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            device=torch_device_of(tensor),
            storage_offset_elements=tensor.storage_offset(),
            runtime_object_id=self.object_id_of(tensor),
            storage_id=storage_id,
            storage_version=self.version_of(storage_id),
            contiguous=bool(tensor.is_contiguous()),
            requires_grad=bool(tensor.requires_grad),
        )
        # Later consumers of this tensor must resolve to the newest value.
        self._value_by_key[key] = value_id
        if memo is not None:
            memo[key] = value_id
        self._values[value_id] = record
        if storage_id is not None:
            self._storage_observed.setdefault(storage_id, set()).add(record.storage_version)
        return value_id, True

    def value(self, value_id: int) -> TensorValueRecord:
        return self._values[value_id]

    def values(self) -> list[TensorValueRecord]:
        return [self._values[k] for k in sorted(self._values)]
