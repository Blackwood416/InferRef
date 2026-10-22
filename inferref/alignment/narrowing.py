"""Causal failure narrowing for Alignment Session v0.1 (L5)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from inferref.alignment.boundaries import (
    Boundary,
    BoundaryInventory,
    select_causal_descendants,
)
from inferref.alignment.schema import Evidence
from inferref.ir.graph import Graph
from inferref.ir.package import TracePackage

#: Reason attached to the planner candidate chosen by coarse-to-fine search.
BINARY_SEARCH_REASON = "binary search midpoint of failing serial chain"


def serial_chain_children(
    inventory: BoundaryInventory,
    parent_id: str | Boundary,
    *,
    graph: Graph | None = None,
) -> tuple[Boundary, ...]:
    """Return ordered child regions when they form a disjoint serial chain.

    Only stored/derived *region* children participate: operator children are
    individual nodes of the same chain, not coarse regions to probe.  The
    children are returned in execution order so the midpoint is stable across
    JSON orderings and lexical ID ordering (``region:10`` sorts before
    ``region:2``).
    """

    parent = inventory.resolve(parent_id)
    if parent is None:
        return ()
    children: list[Boundary] = []
    for child_id in inventory.child_ids(parent.id):
        child = inventory.resolve(child_id)
        if child is None or child.kind != "region" or not child.node_ids:
            continue
        children.append(child)
    if not children:
        return ()
    seen: set[int] = set()
    for child in children:
        nodes = set(child.node_ids)
        if seen.intersection(nodes):
            return ()
        seen.update(nodes)
    ordered = tuple(sorted(children, key=lambda child: _chain_order_key(child, graph)))
    previous_end = -1
    for child in ordered:
        if graph is not None:
            positions = [
                graph.op(node_id).execution_index
                for node_id in child.node_ids
                if graph.has_op(node_id)
            ]
        else:
            positions = list(child.node_ids)
        if positions and min(positions) <= previous_end:
            # Interleaved children are a DAG, not a serial chain.
            return ()
        previous_end = max(positions, default=previous_end)
    return ordered


def _chain_order_key(boundary: Boundary, graph: Graph | None) -> tuple[int, str]:
    if graph is not None:
        positions = [
            graph.op(node_id).execution_index
            for node_id in boundary.node_ids
            if graph.has_op(node_id)
        ]
        if positions:
            return (min(positions), boundary.id)
    return (min(boundary.node_ids, default=0), boundary.id)


def _specificity_key(
    inventory: BoundaryInventory,
    boundary_id: str,
    graph: Graph | None,
) -> tuple[int, tuple[int, str], str]:
    boundary = inventory.resolve(boundary_id)
    if boundary is None:
        return (999999, (999999, boundary_id), boundary_id)
    return (len(boundary.node_ids), _chain_order_key(boundary, graph), boundary_id)


def _candidate_order_key(
    inventory: BoundaryInventory,
    boundary_id: str,
    graph: Graph | None,
) -> tuple[int, str]:
    boundary = inventory.resolve(boundary_id)
    if boundary is None:
        return (999999, boundary_id)
    return _chain_order_key(boundary, graph)


def binary_search_probe(
    inventory: BoundaryInventory,
    parent_id: str | Boundary,
    *,
    verified: Sequence[str] = (),
    failing: Sequence[str] = (),
    graph: Graph | None = None,
) -> str | None:
    """Return the next boundary to probe in a failing serial child chain.

    ``verified`` children are passing prefixes and ``failing`` children are
    known failing prefixes.  The probe is the midpoint of the still-unbounded
    interval, rounded up so a two-child chain probes the second child.  When
    the interval collapses onto an already-failing child it is returned so a
    caller can descend into its own children.  ``None`` means there is no
    untested child (an interaction gap or a fully tested chain).
    """

    chain = serial_chain_children(inventory, parent_id, graph=graph)
    if not chain:
        return None
    verified_ids = {inventory.resolve_id(item) for item in verified}
    failing_ids = {inventory.resolve_id(item) for item in failing}
    verified_positions = [
        index for index, child in enumerate(chain) if child.id in verified_ids
    ]
    failing_positions = [
        index for index, child in enumerate(chain) if child.id in failing_ids
    ]
    low = max(verified_positions, default=-1)
    high = min(failing_positions, default=len(chain))
    if high <= low:
        return None
    midpoint = (low + high + 1) // 2
    if midpoint >= high:
        return chain[high].id if high < len(chain) else None
    return chain[midpoint].id


@dataclass
class NarrowingState:
    """State of failure narrowing within the boundary hierarchy."""

    current_failure: str | None = None
    smallest_confirmed_failure: str | None = None
    failing_descendants: tuple[str, ...] = ()
    verified_exclusions: tuple[str, ...] = ()
    uncovered_causal_nodes: tuple[int, ...] = ()
    next_candidates: tuple[str, ...] = ()
    interaction_gap: bool = False
    confidence: str = "high"
    reason: str = ""
    first_failure: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_failure": self.current_failure,
            "smallest_confirmed_failure": self.smallest_confirmed_failure,
            "failing_descendants": list(self.failing_descendants),
            "verified_exclusions": list(self.verified_exclusions),
            "uncovered_causal_nodes": list(self.uncovered_causal_nodes),
            "next_candidates": list(self.next_candidates),
            "interaction_gap": self.interaction_gap,
            "confidence": self.confidence,
            "reason": self.reason,
            "first_failure": self.first_failure,
        }


def compute_narrowing(
    inventory: BoundaryInventory,
    current_evidence: Mapping[str, Sequence[Evidence]],
    *,
    current_failure: str | None = None,
    package_or_graph: TracePackage | Graph | None = None,
    first_failure: Any = None,
) -> NarrowingState:
    """Determine the smallest confirmed failing boundary and next candidates."""
    graph = (
        package_or_graph
        if isinstance(package_or_graph, Graph)
        else getattr(package_or_graph, "graph", None)
    )
    # Group failing and verified boundaries from current evidence
    failing_boundaries: set[str] = set()
    verified_boundaries: set[str] = set()
    latest_ff: dict[str, Any] | None = None

    for boundary_id, ev_list in current_evidence.items():
        canonical_b = inventory.resolve_id(boundary_id)
        for ev in ev_list:
            if ev.result == "fail":
                failing_boundaries.add(canonical_b)
                if latest_ff is None and ev.observations.get("first_failure"):
                    latest_ff = dict(ev.observations["first_failure"])
            elif ev.result == "pass":
                verified_boundaries.add(canonical_b)

    # Boundaries that are failing cannot be considered verified
    verified_boundaries -= failing_boundaries

    if first_failure is not None and isinstance(first_failure, Mapping):
        latest_ff = dict(first_failure)

    if not current_failure and failing_boundaries:
        # Pick the failing boundary with the largest node set (e.g. root/parent)
        sorted_failures = sorted(
            failing_boundaries,
            key=lambda bid: (
                len(inventory.resolve(bid).node_ids) if inventory.resolve(bid) else 0
            ),
            reverse=True,
        )
        current_failure = sorted_failures[0]

    if not current_failure or inventory.resolve(current_failure) is None:
        return NarrowingState(
            current_failure=current_failure,
            smallest_confirmed_failure=None,
            confidence="low",
            reason="no active failure in inventory",
            first_failure=latest_ff,
        )

    canonical_root = inventory.resolve_id(current_failure)
    descendants = set(inventory.descendants(canonical_root))
    failing_descendants = descendants & failing_boundaries

    # 1. Determine smallest confirmed failing boundary
    if failing_descendants:
        # Prefer the smallest node set, then the earliest boundary in execution
        # order, then the stable ID.  Node IDs alone are not a safe ordering
        # after a trace round-trip, so the graph is used when available.
        smallest = min(
            failing_descendants,
            key=lambda bid: _specificity_key(inventory, bid, graph),
        )
    else:
        smallest = canonical_root

    # 2. Causal selection below the smallest confirmed failure
    confidence = "high"
    reason = "causal narrowing from first failure"
    causal_boundary_ids: tuple[str, ...] = ()
    uncovered_nodes: tuple[int, ...] = ()

    if package_or_graph is not None:
        selection = select_causal_descendants(
            package_or_graph,
            inventory,
            smallest,
            latest_ff,
        )
        causal_boundary_ids = selection.boundary_ids
        confidence = selection.confidence
        reason = selection.reason
        smallest_boundary = inventory.resolve(smallest)
        smallest_nodes = set(smallest_boundary.node_ids) if smallest_boundary else set()
        cone_nodes = set(selection.operator_ids)
        uncovered_nodes = tuple(sorted(smallest_nodes & cone_nodes))

    # 3. Children of smallest confirmed failure
    children = set(inventory.children.get(smallest, ()))
    relevant_children = (
        children & set(causal_boundary_ids) if causal_boundary_ids else children
    )
    if not relevant_children:
        relevant_children = children

    # 4. Check for interaction gap: all relevant children pass but parent fails
    interaction_gap = False
    if relevant_children and relevant_children.issubset(verified_boundaries):
        interaction_gap = True
        reason = "all observed relevant children passed; parent failure indicates interaction or unobserved gap"

    # 5. Next candidates are relevant children not yet verified or failing.
    # For a serial chain, probe the midpoint first so plan and narrow report
    # the same next action (plan/run agreement).
    candidate_ids = relevant_children - verified_boundaries - failing_boundaries
    ordered_candidates = sorted(
        candidate_ids,
        key=lambda bid: _candidate_order_key(inventory, bid, graph),
    )
    probe = binary_search_probe(
        inventory,
        smallest,
        verified=tuple(verified_boundaries),
        failing=tuple(failing_boundaries),
        graph=graph,
    )
    if probe is not None and probe in candidate_ids:
        ordered_candidates = [probe] + [bid for bid in ordered_candidates if bid != probe]
    next_candidates = tuple(ordered_candidates)

    verified_exclusions = tuple(sorted(descendants & verified_boundaries))

    return NarrowingState(
        current_failure=canonical_root,
        smallest_confirmed_failure=smallest,
        failing_descendants=tuple(sorted(failing_descendants)),
        verified_exclusions=verified_exclusions,
        uncovered_causal_nodes=uncovered_nodes,
        next_candidates=next_candidates,
        interaction_gap=interaction_gap,
        confidence=confidence,
        reason=reason,
        first_failure=latest_ff,
    )
