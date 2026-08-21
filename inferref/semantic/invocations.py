"""Splitting a matched operator set into individual invocations.

A module or helper function that runs once per layer produces one *set* of
matched operators spanning the whole model. Turning that into one region per
layer is what makes the result useful: an engineer validates
``RoPE@layers.0.self_attn``, not "every RoPE in the model at once".

Splitting is on gaps in ``execution_index``: consecutive matched operators are
one invocation, and a gap means the model went off and did something else.

Note that operators from an inlined helper are absorbed by the *detector*, not
here — :mod:`inferref.semantic.source_function` matches the whole source stack,
so ``rotate_half``'s operators match via their ``apply_rotary_pos_emb`` caller
and are already in the set. Span filling below is what guarantees the result is
a contiguous slice of execution regardless, which is what gives a region a
clean boundary (IR §34); with the default ``max_gap`` it is an invariant check
that returns the run unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable

from inferref.ir.graph import Graph


def split_invocations(
    graph: Graph, node_ids: Iterable[int], *, max_gap: int = 1
) -> list[tuple[int, ...]]:
    """Split ``node_ids`` into contiguous execution runs.

    ``max_gap`` is the largest gap in ``execution_index`` still treated as one
    invocation. The default of 1 means strictly consecutive. Raising it lets a
    detector tolerate holes — an operator that lost its source mapping, say —
    at the cost of possibly merging two invocations that ran close together.
    """
    known = [n for n in node_ids if graph.has_op(n)]
    if not known:
        return []

    ordered = sorted(known, key=lambda n: graph.op(n).execution_index)
    runs: list[list[int]] = [[ordered[0]]]
    for node_id in ordered[1:]:
        previous = graph.op(runs[-1][-1]).execution_index
        current = graph.op(node_id).execution_index
        if current - previous <= max_gap:
            runs[-1].append(node_id)
        else:
            runs.append([node_id])

    return [tuple(_fill_span(graph, run)) for run in runs]


def _fill_span(graph: Graph, run: list[int]) -> list[int]:
    """Return every operator executed between the first and last of ``run``."""
    first = graph.op(run[0]).execution_index
    last = graph.op(run[-1]).execution_index
    if first == last:
        return list(run)
    return [
        op.id
        for op in graph.ops_in_execution_order()
        if first <= op.execution_index <= last
    ]


def is_contiguous(graph: Graph, node_ids: Iterable[int]) -> bool:
    """Whether ``node_ids`` form an unbroken run in execution order."""
    indices = sorted(graph.op(n).execution_index for n in node_ids if graph.has_op(n))
    if not indices:
        return False
    return indices[-1] - indices[0] + 1 == len(indices)
