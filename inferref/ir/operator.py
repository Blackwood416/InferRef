"""Operator records (IR §22-§25).

A physical operator record describes one dispatched operator invocation: its
canonical name, its structured arguments, its result, and the effects it had on
storage (mutation and aliasing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferref.ir._common import Record, drop_none
from inferref.ir.values import Value, value_from_dict

#: Alias relationships recognised in v0.1 (IR §25).
ALIAS_RELATIONSHIPS = ("same_object", "shared_storage", "view", "unknown_alias")


@dataclass(frozen=True)
class StorageMutation:
    """A storage version transition caused by an operator (IR §24)."""

    storage_id: int
    version_before: int
    version_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "storage_id": self.storage_id,
            "version_before": self.version_before,
            "version_after": self.version_after,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StorageMutation:
        return cls(
            storage_id=int(data["storage_id"]),
            version_before=int(data["version_before"]),
            version_after=int(data["version_after"]),
        )


@dataclass(frozen=True)
class AliasEffect:
    """An output that aliases an input (IR §25)."""

    output_value_id: int
    input_value_id: int
    relationship: str = "unknown_alias"

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_value_id": self.output_value_id,
            "input_value_id": self.input_value_id,
            "relationship": self.relationship,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AliasEffect:
        return cls(
            output_value_id=int(data["output_value_id"]),
            input_value_id=int(data["input_value_id"]),
            relationship=data.get("relationship", "unknown_alias"),
        )


@dataclass
class Effects:
    """Mutation and alias effects of one operator (IR §24, §25)."""

    mutated_storages: tuple[StorageMutation, ...] = ()
    aliases: tuple[AliasEffect, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.mutated_storages or self.aliases)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.mutated_storages:
            out["mutated_storages"] = [m.to_dict() for m in self.mutated_storages]
        if self.aliases:
            out["aliases"] = [a.to_dict() for a in self.aliases]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Effects:
        if not data:
            return cls()
        return cls(
            mutated_storages=tuple(
                StorageMutation.from_dict(m) for m in data.get("mutated_storages", ())
            ),
            aliases=tuple(AliasEffect.from_dict(a) for a in data.get("aliases", ())),
        )


@dataclass(frozen=True)
class Annotation:
    """Optional derived metadata attached to an operator or region (IR §31).

    Annotations never rewrite physical truth (IR §2.5).
    """

    type: str
    name: str
    confidence: float = 1.0
    detector: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return drop_none(
            {
                "type": self.type,
                "name": self.name,
                "confidence": self.confidence,
                "detector": self.detector,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Annotation:
        return cls(
            type=data.get("type", "semantic"),
            name=data.get("name", ""),
            confidence=float(data.get("confidence", 1.0)),
            detector=data.get("detector"),
        )


@dataclass
class OperatorRecord(Record):
    """One dispatched operator invocation (IR §22)."""

    id: int = -1
    #: Canonical execution ordering; the basis for first-divergence analysis (IR §7).
    execution_index: int = -1

    namespace: str = "aten"
    op: str = ""
    overload: str = "default"

    positional_args: tuple[Value, ...] = ()
    keyword_args: dict[str, Value] = field(default_factory=dict)
    result: Value | None = None

    source_id: int | None = None
    #: Module ids, outermost -> innermost (IR §29).
    module_stack: tuple[int, ...] = ()

    effects: Effects = field(default_factory=Effects)
    annotations: tuple[Annotation, ...] = ()

    kind: str = "operator"

    _KNOWN = (
        "id",
        "execution_index",
        "kind",
        "namespace",
        "op",
        "overload",
        "canonical_name",
        "positional_args",
        "keyword_args",
        "result",
        "source_id",
        "module_stack",
        "effects",
        "annotations",
    )

    @property
    def canonical_name(self) -> str:
        """``namespace.op.overload`` (IR §23)."""
        return f"{self.namespace}.{self.op}.{self.overload}"

    def _encode(self) -> dict[str, Any]:
        out = drop_none(
            {
                "id": self.id,
                "execution_index": self.execution_index,
                "kind": "operator",
                "namespace": self.namespace,
                "op": self.op,
                "overload": self.overload,
                "canonical_name": self.canonical_name,
                "positional_args": [a.to_dict() for a in self.positional_args],
                "keyword_args": {k: v.to_dict() for k, v in self.keyword_args.items()},
                "result": self.result.to_dict() if self.result is not None else None,
                "source_id": self.source_id,
                "module_stack": list(self.module_stack),
            }
        )
        if self.effects:
            out["effects"] = self.effects.to_dict()
        if self.annotations:
            out["annotations"] = [a.to_dict() for a in self.annotations]
        return out

    @classmethod
    def _decode(cls, data: dict[str, Any]) -> dict[str, Any]:
        namespace = data.get("namespace")
        op = data.get("op")
        overload = data.get("overload")
        if namespace is None or op is None or overload is None:
            # Fall back to splitting canonical_name for traces that only wrote it.
            canonical = data.get("canonical_name", "")
            parts = canonical.split(".")
            if len(parts) >= 3:
                namespace = namespace or parts[0]
                op = op or ".".join(parts[1:-1])
                overload = overload or parts[-1]
        return {
            "id": int(data["id"]),
            "execution_index": int(data.get("execution_index", -1)),
            "namespace": namespace or "unknown",
            "op": op or "unknown",
            "overload": overload or "default",
            "positional_args": tuple(
                value_from_dict(a) for a in data.get("positional_args", ())
            ),
            "keyword_args": {
                k: value_from_dict(v) for k, v in (data.get("keyword_args") or {}).items()
            },
            "result": value_from_dict(data["result"]) if data.get("result") else None,
            "source_id": data.get("source_id"),
            "module_stack": tuple(data.get("module_stack", ())),
            "effects": Effects.from_dict(data.get("effects")),
            "annotations": tuple(Annotation.from_dict(a) for a in data.get("annotations", ())),
            "kind": "operator",
        }
