"""Shared plumbing for IR records.

Trace IR §46 (Forward Compatibility) requires that readers:

* ignore unknown object fields;
* preserve unknown extension fields when rewriting a trace where practical.

:class:`Record` implements both by stashing every field it did not recognise in
``unknown`` and merging it back in :meth:`Record.to_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Record:
    """Base for every persisted IR record.

    Subclasses implement :meth:`_encode` (own fields -> dict) and the classmethod
    :meth:`_decode` (dict -> kwargs). The unknown-field round-trip is handled here.
    """

    #: Fields present in the source JSON that this reader did not recognise.
    unknown: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    #: Framework-specific data, namespaced by producer (IR §45).
    extensions: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    #: Field names consumed by ``_decode``; anything else lands in ``unknown``.
    _KNOWN: tuple[str, ...] = ()

    def _encode(self) -> dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def _decode(cls, data: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-ready dict, re-emitting unknown fields verbatim."""
        out = self._encode()
        if self.extensions:
            out["extensions"] = self.extensions
        # Unknown fields must not shadow fields we understand.
        for key, value in self.unknown.items():
            out.setdefault(key, value)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Deserialise, routing unrecognised fields into ``unknown``."""
        kwargs = cls._decode(data)
        consumed = set(cls._KNOWN) | {"extensions"}
        kwargs["unknown"] = {k: v for k, v in data.items() if k not in consumed}
        kwargs["extensions"] = data.get("extensions") or {}
        return cls(**kwargs)


def drop_none(data: dict[str, Any]) -> dict[str, Any]:
    """Remove keys whose value is ``None`` (optional fields stay absent)."""
    return {k: v for k, v in data.items() if v is not None}
