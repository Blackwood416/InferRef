"""Causal failure narrowing for Alignment Session v0.1 (L5)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from inferref.alignment.boundaries import (
    BoundaryInventory,
    select_causal_descendants,
)
from inferref.alignment.schema import Evidence
from inferref.ir.graph import Graph
from inferref.ir.package import TracePackage


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
        # Prefer the smallest node set, then stable ID
        smallest = min(
            failing_descendants,
            key=lambda bid: (
                len(inventory.resolve(bid).node_ids) if inventory.resolve(bid) else 999999,
                bid,
            ),
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

    # 5. Next candidates are relevant children not yet verified or failing
    next_candidates = tuple(
        sorted(relevant_children - verified_boundaries - failing_boundaries)
    )

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
