"""The runtime graph container (IR §8, §37).

``graph.json`` holds the operator list, the value records, and the external
trace inputs/outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from inferref.ir.operator import OperatorRecord
from inferref.ir.tensor_value import TensorValueRecord
from inferref.ir.values import Value, value_from_dict, walk_tensor_refs


@dataclass(frozen=True)
class GraphIO:
    """A named external graph input or output (IR §37)."""

    name: str
    value: Value

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphIO":
        return cls(name=data.get("name", ""), value=value_from_dict(data["value"]))


@dataclass
class Graph:
    """Operators + values + external boundary (IR §8, §37)."""

    operators: list[OperatorRecord] = field(default_factory=list)
    values: list[TensorValueRecord] = field(default_factory=list)
    inputs: list[GraphIO] = field(default_factory=list)
    outputs: list[GraphIO] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._op_index: dict[int, OperatorRecord] = {}
        self._value_index: dict[int, TensorValueRecord] = {}
        self.reindex()

    # -- indexing ---------------------------------------------------------

    def reindex(self) -> None:
        """Rebuild the id lookup tables after mutating the record lists."""
        self._op_index = {op.id: op for op in self.operators}
        self._value_index = {v.id: v for v in self.values}

    def op(self, op_id: int) -> OperatorRecord:
        """Return the operator with ``op_id`` (raises :class:`KeyError`)."""
        return self._op_index[op_id]

    def value(self, value_id: int) -> TensorValueRecord:
        """Return the tensor value with ``value_id`` (raises :class:`KeyError`)."""
        return self._value_index[value_id]

    def has_op(self, op_id: int) -> bool:
        return op_id in self._op_index

    def has_value(self, value_id: int) -> bool:
        return value_id in self._value_index

    # -- traversal --------------------------------------------------------

    def ops_in_execution_order(self) -> list[OperatorRecord]:
        """Operators sorted by ``execution_index`` (IR §7)."""
        return sorted(self.operators, key=lambda o: o.execution_index)

    def op_input_value_ids(self, op: OperatorRecord) -> list[int]:
        """Every tensor value id consumed by ``op``, in argument order."""
        seen: dict[int, None] = {}
        for arg in op.positional_args:
            for ref in walk_tensor_refs(arg):
                seen.setdefault(ref.value_id, None)
        for arg in op.keyword_args.values():
            for ref in walk_tensor_refs(arg):
                seen.setdefault(ref.value_id, None)
        return list(seen)

    def op_output_value_ids(self, op: OperatorRecord) -> list[int]:
        """Every tensor value id produced by ``op``, in result order."""
        seen: dict[int, None] = {}
        for ref in walk_tensor_refs(op.result):
            seen.setdefault(ref.value_id, None)
        return list(seen)

    def iter_values(self) -> Iterator[TensorValueRecord]:
        return iter(self.values)

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "operators": [o.to_dict() for o in self.operators],
            "values": [v.to_dict() for v in self.values],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Graph":
        return cls(
            operators=[OperatorRecord.from_dict(o) for o in data.get("operators", ())],
            values=[TensorValueRecord.from_dict(v) for v in data.get("values", ())],
            inputs=[GraphIO.from_dict(i) for i in data.get("inputs", ())],
            outputs=[GraphIO.from_dict(o) for o in data.get("outputs", ())],
        )

    # -- derived ----------------------------------------------------------

    def derived_links(self) -> tuple[dict[int, int], dict[int, list[int]]]:
        """Derive producer and consumer maps from operators and effects.

        A mutating operator produces more than its Python return object.  It
        also produces a new generation of the entire aliased storage.  A later
        read through the base tensor therefore has a real producer even when
        the operator returned only the written view::

            cache[:, :, pos].copy_(key)   # result is the target view
            live = cache[:, :, :end]      # base tensor at the new generation

        Without this effect edge, ``live`` looks like an external graph input
        and mutation regions cannot have a truthful boundary.  Explicit
        operator results win; mutation effects only fill values that otherwise
        have no producer, so a downstream view keeps its own producing op.
        """
        producers: dict[int, int] = {}
        consumers: dict[int, list[int]] = {}
        for op in self.ops_in_execution_order():
            for vid in self.op_input_value_ids(op):
                consumers.setdefault(vid, []).append(op.id)
            for vid in self.op_output_value_ids(op):
                producers.setdefault(vid, op.id)

        values_by_generation: dict[tuple[int, int], list[int]] = {}
        for value in self.values:
            if value.storage_id is None:
                continue
            values_by_generation.setdefault(
                (value.storage_id, value.storage_version), []
            ).append(value.id)

        for op in self.ops_in_execution_order():
            for mutation in op.effects.mutated_storages:
                for vid in values_by_generation.get(
                    (mutation.storage_id, mutation.version_after), ()
                ):
                    producers.setdefault(vid, op.id)
        return producers, consumers

    def produced_value_ids(self, op: OperatorRecord) -> list[int]:
        """Values produced explicitly or through a storage mutation effect."""
        explicit = self.op_output_value_ids(op)
        seen = set(explicit)
        effect = [
            value.id
            for value in self.values
            if value.producer == op.id and value.id not in seen
        ]
        return explicit + effect

    def recompute_links(self) -> None:
        """Recompute ``producer`` / ``consumers`` from the operator list.

        Producer/consumer links are derived data; recomputing them keeps
        validation invariants 4 and 5 (IR §48) satisfied after edits.
        """
        producers, consumers = self.derived_links()
        for value in self.values:
            object.__setattr__(value, "producer", producers.get(value.id))
            object.__setattr__(value, "consumers", tuple(consumers.get(value.id, ())))

    def values_for(self, value_ids: Iterable[int]) -> list[TensorValueRecord]:
        """Look up several values, skipping ids that are not present."""
        return [self._value_index[v] for v in value_ids if v in self._value_index]
