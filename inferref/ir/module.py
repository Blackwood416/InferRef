"""Module records (IR §28, §29; SPEC §7.2).

Modules provide a navigation hierarchy over the physical trace. They are stored
separately and referenced by id from ``OperatorRecord.module_stack``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inferref.ir._common import Record, drop_none


@dataclass
class ModuleRecord(Record):
    """One module in the reference model's hierarchy (IR §28)."""

    id: int = -1
    #: Fully qualified path, e.g. ``model.layers.17.self_attn.q_proj``.
    path: str = ""
    #: Informational type string, e.g. ``torch.nn.Linear``.
    type: str = ""
    parent_id: int | None = None

    _KNOWN = ("id", "path", "type", "parent_id")

    def _encode(self) -> dict[str, Any]:
        return drop_none(
            {
                "id": self.id,
                "path": self.path,
                "type": self.type,
                "parent_id": self.parent_id,
            }
        )

    @classmethod
    def _decode(cls, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": int(data["id"]),
            "path": data.get("path", ""),
            "type": data.get("type", ""),
            "parent_id": data.get("parent_id"),
        }

    @property
    def leaf_name(self) -> str:
        """Last path component (``q_proj`` for ``model...self_attn.q_proj``)."""
        return self.path.rsplit(".", 1)[-1] if self.path else ""


def path_matches(path: str, pattern: str) -> bool:
    """Whether module ``path`` falls under ``pattern``.

    ``pattern`` matches when it names the module exactly or one of its
    ancestors. Leading components of ``pattern`` that the traced hierarchy does
    not have are tolerated, so ``--scope model.layers.0`` still selects
    ``layers.0`` when the traced root *is* the model.

    Lives in the IR layer rather than the PyTorch frontend because region
    creation and inspection need it in environments without PyTorch.
    """
    if not pattern:
        return True
    parts = pattern.split(".")
    for i in range(len(parts)):
        candidate = ".".join(parts[i:])
        if path == candidate or path.startswith(candidate + "."):
            return True
    return False
