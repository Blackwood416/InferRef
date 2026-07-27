"""Reference regions (IR §33-§36; SPEC §18-§20).

A region is a named subgraph with explicit external inputs and outputs. It is
the bridge to fused engine kernels: an engine only has to reproduce the region
boundary contract, not PyTorch's operator partitioning (SPEC §20).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferref.ir._common import Record, drop_none
from inferref.ir.operator import Annotation

#: How a region came to exist (IR §35).
CREATION_METHODS = (
    "manual",
    "module",
    "source_function",
    "semantic_pattern",
    "engine_mapping",
    "graph_selection",
)


@dataclass
class RegionRecord(Record):
    """A named subgraph boundary (IR §33)."""

    id: int = -1
    name: str = ""

    #: Operator ids inside the region.
    node_ids: tuple[int, ...] = ()
    #: External input value ids (derived per IR §34).
    inputs: tuple[int, ...] = ()
    #: External output value ids (derived per IR §34).
    outputs: tuple[int, ...] = ()

    semantic: Annotation | None = None
    source_ids: tuple[int, ...] = ()
    creation_method: str = "manual"
    #: Optional engine kernel this region maps to (SPEC §20).
    engine_op: str | None = None

    _KNOWN = (
        "id",
        "name",
        "node_ids",
        "inputs",
        "outputs",
        "semantic",
        "source_ids",
        "creation",
        "engine_op",
    )

    def _encode(self) -> dict[str, Any]:
        return drop_none(
            {
                "id": self.id,
                "name": self.name,
                "node_ids": list(self.node_ids),
                "inputs": list(self.inputs),
                "outputs": list(self.outputs),
                "semantic": self.semantic.to_dict() if self.semantic else None,
                "source_ids": list(self.source_ids),
                "creation": {"method": self.creation_method},
                "engine_op": self.engine_op,
            }
        )

    @classmethod
    def _decode(cls, data: dict[str, Any]) -> dict[str, Any]:
        semantic = data.get("semantic")
        creation = data.get("creation") or {}
        return {
            "id": int(data["id"]),
            "name": data.get("name", ""),
            "node_ids": tuple(data.get("node_ids", ())),
            "inputs": tuple(data.get("inputs", ())),
            "outputs": tuple(data.get("outputs", ())),
            "semantic": Annotation.from_dict(semantic) if semantic else None,
            "source_ids": tuple(data.get("source_ids", ())),
            "creation_method": creation.get("method", "manual"),
            "engine_op": data.get("engine_op"),
        }
