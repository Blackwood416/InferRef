"""Region boundary derivation (IR §34).

    A region *input* is a value that is consumed by a node inside the region and
    is produced outside the region (or is an external graph input).

    A region *output* is a value that is produced inside the region and is
    consumed outside the region (or selected as a trace output).

This formal definition is what lets regions be extracted automatically and lets
an engine reproduce only the boundary contract rather than PyTorch's operator
partitioning (SPEC §20).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from inferref.ir.graph import Graph
from inferref.ir.values import walk_tensor_refs


def derive_boundary(
    graph: Graph, node_ids: Iterable[int]
) -> tuple[list[int], list[int]]:
    """Return ``(inputs, outputs)`` value ids for the subgraph ``node_ids``.

    Ordering is deterministic: inputs follow first-consumption order and outputs
    follow production order, both in execution order. This keeps extracted
    testcases stable across runs (SPEC §53 "deterministic output").
    """
    inside = {n for n in node_ids if graph.has_op(n)}
    if not inside:
        return [], []

    ops_inside = sorted(
        (graph.op(n) for n in inside), key=lambda o: o.execution_index
    )

    # Values produced within the region.
    produced_inside: set[int] = set()
    for op in ops_inside:
        produced_inside.update(graph.produced_value_ids(op))

    # Inputs: consumed inside, not produced inside.
    inputs: list[int] = []
    seen_inputs: set[int] = set()
    for op in ops_inside:
        for vid in graph.op_input_value_ids(op):
            if vid not in produced_inside and vid not in seen_inputs:
                seen_inputs.add(vid)
                inputs.append(vid)

    # Outputs: produced inside, and consumed outside (or a graph output, or
    # never consumed at all — a dangling result is still the region's product).
    consumed_outside: set[int] = set()
    for op in graph.operators:
        if op.id in inside:
            continue
        consumed_outside.update(graph.op_input_value_ids(op))

    graph_output_ids: set[int] = set()
    for io in graph.outputs:
        for ref in walk_tensor_refs(io.value):
            graph_output_ids.add(ref.value_id)

    consumed_inside: set[int] = set()
    for op in ops_inside:
        consumed_inside.update(graph.op_input_value_ids(op))

    outputs: list[int] = []
    seen_outputs: set[int] = set()
    for op in ops_inside:
        for vid in graph.produced_value_ids(op):
            if vid in seen_outputs:
                continue
            escapes = (
                vid in consumed_outside
                or vid in graph_output_ids
                or vid not in consumed_inside
            )
            if escapes:
                seen_outputs.add(vid)
                outputs.append(vid)

    return inputs, outputs


def nodes_between(graph: Graph, from_op: int, to_op: int) -> list[int]:
    """Return operator ids whose ``execution_index`` lies in ``[from, to]``.

    ``from_op`` and ``to_op`` are operator **ids**; the range is taken over
    execution order (IR §7), which is the ordering a user reading
    ``inferref inspect`` output actually sees.
    """
    if not graph.has_op(from_op):
        raise KeyError(f"no such operator id: {from_op}")
    if not graph.has_op(to_op):
        raise KeyError(f"no such operator id: {to_op}")
    lo = graph.op(from_op).execution_index
    hi = graph.op(to_op).execution_index
    if lo > hi:
        lo, hi = hi, lo
    return [op.id for op in graph.ops_in_execution_order() if lo <= op.execution_index <= hi]


def nodes_for_module(graph: Graph, module_ids: Sequence[int]) -> list[int]:
    """Return operator ids whose ``module_stack`` contains any of ``module_ids``."""
    wanted = set(module_ids)
    return [
        op.id
        for op in graph.ops_in_execution_order()
        if wanted.intersection(op.module_stack)
    ]
