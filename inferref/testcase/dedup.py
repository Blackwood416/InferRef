"""Operator signature deduplication (SPEC §24).

A real model executes the same operator thousands of times over a handful of
distinct shape/layout combinations::

    226 MM executions
    ->
    11 unique MM signatures

Grouping by signature is what turns a model trace into a compact kernel test
suite rather than a redundant pile of near-identical testcases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferref.ir.graph import Graph
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.values import (
    DictValue,
    ListValue,
    NoneValue,
    OpaqueValue,
    ScalarValue,
    StringValue,
    TensorRef,
    TupleValue,
    Value,
)


def _tensor_signature(graph: Graph, value_id: int) -> tuple:
    if not graph.has_value(value_id):
        return ("tensor", "unknown")
    value = graph.value(value_id)
    return (
        "tensor",
        value.dtype,
        len(value.shape),
        tuple(value.shape),
        tuple(value.stride),
        value.storage_offset_elements,
        value.contiguous,
    )


def _argument_signature(graph: Graph, value: Value) -> tuple:
    """Signature of one argument: layout for tensors, exact value for scalars."""
    if isinstance(value, TensorRef):
        return _tensor_signature(graph, value.value_id)
    if isinstance(value, ScalarValue):
        return ("scalar", value.dtype, value.encoding, value.value)
    if isinstance(value, StringValue):
        return ("string", value.value)
    if isinstance(value, NoneValue):
        return ("none",)
    if isinstance(value, (ListValue, TupleValue)):
        return (value.kind, tuple(_argument_signature(graph, i) for i in value.items))
    if isinstance(value, DictValue):
        return (
            "dict",
            tuple(
                (_argument_signature(graph, k), _argument_signature(graph, v))
                for k, v in value.items
            ),
        )
    if isinstance(value, OpaqueValue):
        return ("opaque", value.type, value.repr)
    return ("unknown",)


def operator_signature(graph: Graph, op: OperatorRecord) -> tuple:
    """Full dedup signature for one operator invocation (SPEC §24)."""
    return (
        op.canonical_name,
        tuple(_argument_signature(graph, a) for a in op.positional_args),
        tuple(
            (name, _argument_signature(graph, value))
            for name, value in sorted(op.keyword_args.items())
        ),
    )


@dataclass
class SignatureGroup:
    """One unique operator signature and the invocations that share it."""

    canonical_name: str
    signature: tuple
    op_ids: list[int] = field(default_factory=list)
    #: Representative — the first invocation, used for testcase extraction.
    representative: int = -1
    input_shapes: list[list[int]] = field(default_factory=list)
    input_dtypes: list[str] = field(default_factory=list)
    input_strides: list[list[int]] = field(default_factory=list)
    output_shapes: list[list[int]] = field(default_factory=list)
    module_paths: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.op_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.canonical_name,
            "count": self.count,
            "representative_op_id": self.representative,
            "op_ids": self.op_ids,
            "inputs": [
                {"dtype": d, "shape": s, "stride": st}
                for d, s, st in zip(self.input_dtypes, self.input_shapes, self.input_strides)
            ],
            "output_shapes": self.output_shapes,
            "modules": sorted(set(self.module_paths)),
        }


def dedup_operators(
    package: TracePackage, *, operator: str | None = None
) -> list[SignatureGroup]:
    """Group the trace's operators into unique signatures (SPEC §24).

    ``operator`` optionally restricts the analysis to one canonical name, e.g.
    ``aten.mm.default``.
    """
    graph = package.graph
    groups: dict[tuple, SignatureGroup] = {}

    for op in graph.ops_in_execution_order():
        if operator and op.canonical_name != operator:
            continue
        signature = operator_signature(graph, op)
        group = groups.get(signature)
        if group is None:
            group = SignatureGroup(
                canonical_name=op.canonical_name,
                signature=signature,
                representative=op.id,
            )
            for value_id in graph.op_input_value_ids(op):
                if graph.has_value(value_id):
                    value = graph.value(value_id)
                    group.input_dtypes.append(value.dtype)
                    group.input_shapes.append(list(value.shape))
                    group.input_strides.append(list(value.stride))
            for value_id in graph.op_output_value_ids(op):
                if graph.has_value(value_id):
                    group.output_shapes.append(list(graph.value(value_id).shape))
            groups[signature] = group
        group.op_ids.append(op.id)
        path = package.module_path(op.module_stack)
        if path:
            group.module_paths.append(path)

    # Most-repeated first: those are the signatures worth a kernel test.
    return sorted(groups.values(), key=lambda g: (-g.count, g.canonical_name))


def summarise(groups: list[SignatureGroup]) -> dict[str, Any]:
    """Per-operator ``executions -> unique signatures`` summary (SPEC §24)."""
    by_operator: dict[str, dict[str, int]] = {}
    for group in groups:
        entry = by_operator.setdefault(
            group.canonical_name, {"executions": 0, "signatures": 0}
        )
        entry["executions"] += group.count
        entry["signatures"] += 1
    return {
        "total_executions": sum(g.count for g in groups),
        "total_signatures": len(groups),
        "by_operator": dict(sorted(by_operator.items(), key=lambda kv: -kv[1]["executions"])),
    }
