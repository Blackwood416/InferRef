"""Boundary inventory and containment graph for alignment sessions.

The trace IR stores regions, modules, and operators in three different places.
Alignment needs one stable vocabulary for all three without writing derived
boundaries back into the trace.  This module deliberately contains no session
store code: :class:`BoundaryInventory` is a serialisable projection that can
be cached by a caller or rebuilt from a trace at any time.

There are two slightly non-obvious details here:

* A module boundary contains the operators issued by that module *and its
  descendants*.  This is the same interpretation used by
  :func:`inferref.region.boundary.nodes_for_module`.
* Regions are allowed to overlap.  Consequently containment is a DAG rather
  than a tree.  Equal node sets are represented once in the graph and the
  lower-priority IDs are retained as aliases.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from inferref.ir.graph import Graph
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.region.boundary import derive_boundary, nodes_for_module

BOUNDARY_TYPES = ("region", "module", "op")

# Lower value means higher canonicalisation priority.  A stored region must
# remain the public identity when it has the same node set as a derived module
# or operator boundary.
BOUNDARY_TYPE_PRIORITY: Mapping[str, int] = {
    "region": 0,
    "module": 1,
    "op": 2,
}


def stable_boundary_id(kind: str, value: object) -> str:
    """Return the wire ID for a stored or derived boundary.

    IDs are intentionally based on the immutable IR IDs, not on list position.
    This keeps references stable when a trace is loaded by a different reader.
    """

    if kind not in BOUNDARY_TYPES:
        raise ValueError(f"unknown boundary type {kind!r}")
    return f"{kind}:{value}"


def _stable_sort_id(value: object) -> str:
    return str(value)


def _node_key(node_ids: Iterable[int]) -> tuple[int, ...]:
    """Canonical, hashable representation of an operator node set."""

    # Trace operator IDs are integers.  A defensive int conversion is avoided
    # here so an in-memory test graph using an integer-like ID cannot silently
    # change its identity; Graph.has_op still decides whether it is known.
    return tuple(sorted({int(node_id) for node_id in node_ids}))


def _ordered_nodes(graph: Graph, node_ids: Iterable[int]) -> tuple[int, ...]:
    known = {int(node_id) for node_id in node_ids if graph.has_op(int(node_id))}
    return tuple(
        op.id
        for op in sorted(
            (graph.op(node_id) for node_id in known),
            key=lambda op: (op.execution_index, op.id),
        )
    )


@dataclass(frozen=True)
class Boundary:
    """One stored or derived alignment boundary.

    ``canonical_id`` is populated by :func:`inventory_boundaries`.  Before
    canonicalisation it defaults to ``id`` so a directly constructed boundary
    remains useful in tests and in callers that only need boundary metadata.
    ``metadata`` is optional JSON-compatible display data; no TracePackage or
    store object is retained here.
    """

    id: str
    kind: str
    name: str
    node_ids: tuple[int, ...] = ()
    inputs: tuple[int, ...] = ()
    outputs: tuple[int, ...] = ()
    canonical_id: str | None = None
    aliases: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.kind not in BOUNDARY_TYPES:
            raise ValueError(f"unknown boundary type {self.kind!r}")
        if self.canonical_id is None:
            object.__setattr__(self, "canonical_id", self.id)
        object.__setattr__(self, "node_ids", tuple(int(node) for node in self.node_ids))
        object.__setattr__(self, "inputs", tuple(int(value) for value in self.inputs))
        object.__setattr__(self, "outputs", tuple(int(value) for value in self.outputs))
        object.__setattr__(self, "aliases", tuple(sorted(set(self.aliases))))

    @property
    def type(self) -> str:
        """Wire-friendly alias for ``kind``."""

        return self.kind

    @property
    def boundary_type(self) -> str:
        """Descriptive alias retained for callers that avoid ``type``."""

        return self.kind

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def is_alias(self) -> bool:
        return self.canonical_id != self.id

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "node_ids": list(self.node_ids),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "canonical_id": self.canonical_id,
        }
        if self.aliases:
            result["aliases"] = list(self.aliases)
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True)
class CausalSelection:
    """Causal cone projection used by planning and narrowing.

    The IDs are all stable wire IDs and the value/operator sets are retained
    for diagnostics.  ``confidence`` is ``high`` only when a valid trace value
    or producer ID was supplied; containment-only fallback is explicitly
    ``low`` as required by the 0.10 epistemic contract.
    """

    boundary_ids: tuple[str, ...] = ()
    value_id: int | None = None
    producer_op_id: int | None = None
    value_ids: tuple[int, ...] = ()
    operator_ids: tuple[int, ...] = ()
    confidence: str = "low"
    provenance_available: bool = False
    reason: str = "containment fallback"

    def __iter__(self):
        return iter(self.boundary_ids)

    def __len__(self) -> int:
        return len(self.boundary_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundaries": list(self.boundary_ids),
            "value_id": self.value_id,
            "producer_op_id": self.producer_op_id,
            "value_ids": list(self.value_ids),
            "operator_ids": list(self.operator_ids),
            "confidence": self.confidence,
            "provenance_available": self.provenance_available,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BoundaryInventory:
    """Canonical boundaries and their deterministic containment DAG.

    ``boundaries`` contains one entry per distinct node set.  ``raw_boundaries``
    contains every stored/derived identity, including aliases, and is useful
    when a report wants to show the original region or module name.  All IDs in
    ``aliases`` resolve to a canonical ID through :meth:`resolve`.
    """

    boundaries: tuple[Boundary, ...] = ()
    raw_boundaries: tuple[Boundary, ...] = ()
    aliases: Mapping[str, str] = field(default_factory=dict)
    parents: Mapping[str, str | None] = field(default_factory=dict)
    alternate_parents: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    children: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def canonical_boundaries(self) -> tuple[Boundary, ...]:
        return self.boundaries

    @property
    def canonical_ids(self) -> tuple[str, ...]:
        return tuple(boundary.id for boundary in self.boundaries)

    @property
    def all_boundaries(self) -> tuple[Boundary, ...]:
        return self.raw_boundaries or self.boundaries

    @property
    def by_id(self) -> dict[str, Boundary]:
        """Return a fresh lookup including canonical IDs and aliases."""

        canonical = {boundary.id: boundary for boundary in self.boundaries}
        if self.raw_boundaries:
            canonical.update(
                {
                    boundary.id: canonical.get(
                        boundary.canonical_id or boundary.id, boundary
                    )
                    for boundary in self.raw_boundaries
                }
            )
        return canonical

    @property
    def canonical_by_nodes(self) -> dict[tuple[int, ...], str]:
        return {_node_key(boundary.node_ids): boundary.id for boundary in self.boundaries}

    @property
    def nodes(self) -> tuple[Boundary, ...]:
        """Graph-style alias for the canonical boundary collection."""

        return self.boundaries

    @property
    def canonical(self) -> tuple[Boundary, ...]:
        """Short alias used by graph-oriented callers."""

        return self.boundaries

    def boundary_for_nodes(self, node_ids: Iterable[int]) -> Boundary | None:
        """Resolve the canonical boundary for an operator node set."""

        boundary_id = self.canonical_by_nodes.get(_node_key(node_ids))
        return self.resolve(boundary_id) if boundary_id is not None else None

    def resolve_id(self, boundary_id: str | Boundary) -> str:
        value = boundary_id.id if isinstance(boundary_id, Boundary) else str(boundary_id)
        return self.aliases.get(value, value)

    def resolve(self, boundary_id: str | Boundary) -> Boundary | None:
        resolved = self.resolve_id(boundary_id)
        return self.by_id.get(resolved)

    def __contains__(self, item: Any) -> bool:
        return self.resolve(item) is not None

    def parent_ids(self, boundary_id: str | Boundary) -> tuple[str, ...]:
        resolved = self.resolve_id(boundary_id)
        primary = self.parents.get(resolved)
        alternate = self.alternate_parents.get(resolved, ())
        return tuple(item for item in ((primary,) if primary else ()) + tuple(alternate))

    def child_ids(self, boundary_id: str | Boundary) -> tuple[str, ...]:
        return tuple(self.children.get(self.resolve_id(boundary_id), ()))

    def descendants(self, boundary_id: str | Boundary) -> tuple[str, ...]:
        """Return all strict graph descendants in deterministic order."""

        root = self.resolve_id(boundary_id)
        seen: set[str] = set()
        pending = list(self.children.get(root, ()))
        while pending:
            child = pending.pop(0)
            if child in seen:
                continue
            seen.add(child)
            pending.extend(self.children.get(child, ()))
        return tuple(sorted(seen))

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
            "raw_boundaries": [boundary.to_dict() for boundary in self.raw_boundaries],
            "aliases": dict(sorted(self.aliases.items())),
            "parents": {
                boundary_id: parent
                for boundary_id, parent in sorted(self.parents.items())
            },
            "alternate_parents": {
                boundary_id: list(parents)
                for boundary_id, parents in sorted(self.alternate_parents.items())
            },
            "children": {
                boundary_id: list(children)
                for boundary_id, children in sorted(self.children.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BoundaryInventory:
        """Load the store-independent JSON projection produced by ``to_dict``."""

        def _parse_boundary(item: Mapping[str, Any]) -> Boundary:
            return Boundary(
                id=str(item["id"]),
                kind=str(item.get("kind", item.get("type", "op"))),
                name=str(item.get("name", "")),
                node_ids=tuple(item.get("node_ids", ())),
                inputs=tuple(item.get("inputs", ())),
                outputs=tuple(item.get("outputs", ())),
                canonical_id=item.get("canonical_id"),
                aliases=tuple(item.get("aliases", ())),
                metadata=item.get("metadata") or {},
            )

        boundaries = tuple(
            _parse_boundary(item)
            for item in data.get("boundaries", ())
            if isinstance(item, Mapping)
        )
        canonical_ids = {boundary.id for boundary in boundaries}

        aliases = {
            str(key): str(value)
            for key, value in (data.get("aliases") or {}).items()
        }
        for alias_id, target_id in aliases.items():
            if target_id not in canonical_ids:
                raise ValueError(f"dangling alias target: {alias_id} -> {target_id}")

        raw_items = data.get("raw_boundaries")
        if raw_items is not None and isinstance(raw_items, (list, tuple)):
            raw = tuple(
                _parse_boundary(item)
                for item in raw_items
                if isinstance(item, Mapping)
            )
        else:
            raw = tuple(
                replace(boundary, canonical_id=aliases.get(boundary.id, boundary.id))
                for boundary in boundaries
            )

        all_known_ids = canonical_ids | set(aliases.keys()) | {b.id for b in raw}

        parents = {
            str(key): (str(value) if value is not None else None)
            for key, value in (data.get("parents") or {}).items()
        }
        for child_id, parent_id in parents.items():
            if parent_id is not None and parent_id not in canonical_ids:
                raise ValueError(f"dangling parent ID: {child_id} -> {parent_id}")

        for start_id in parents:
            visited = set()
            curr: str | None = start_id
            while curr is not None:
                if curr in visited:
                    raise ValueError(f"cycle detected in boundary parents involving {curr}")
                visited.add(curr)
                curr = parents.get(curr)

        alternate = {
            str(key): tuple(str(value) for value in values)
            for key, values in (data.get("alternate_parents") or {}).items()
        }
        for child_id, alts in alternate.items():
            for alt_id in alts:
                if alt_id not in canonical_ids:
                    raise ValueError(f"dangling alternate parent ID: {child_id} -> {alt_id}")

        children = {
            str(key): tuple(str(value) for value in values)
            for key, values in (data.get("children") or {}).items()
        }
        for parent_id, chs in children.items():
            for ch_id in chs:
                if ch_id not in all_known_ids:
                    raise ValueError(f"dangling child ID: {parent_id} -> {ch_id}")

        return cls(
            boundaries=boundaries,
            raw_boundaries=raw,
            aliases=aliases,
            parents=parents,
            alternate_parents=alternate,
            children=children,
        )


# Names used by early 0.10 callers and by external integrations.  They are
# aliases, not subclasses, so ``isinstance(x, BoundaryInventory)`` remains
# true for every spelling.
BoundaryGraph = BoundaryInventory
ContainmentGraph = BoundaryInventory
BoundaryRecord = Boundary


def _boundary_io(graph: Graph, nodes: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    inputs, outputs = derive_boundary(graph, nodes)
    return tuple(inputs), tuple(outputs)


def _region_io(graph: Graph, region: RegionRecord, nodes: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if region.inputs or region.outputs:
        return tuple(region.inputs), tuple(region.outputs)
    return _boundary_io(graph, nodes)


def _module_name(module: Any, module_id: object) -> str:
    path = getattr(module, "path", "") if module is not None else ""
    return str(path) if path else stable_boundary_id("module", module_id)


def _operator_name(op: Any) -> str:
    name = getattr(op, "canonical_name", "")
    return str(name) if name else stable_boundary_id("op", getattr(op, "id", ""))


def _raw_boundaries(package: TracePackage) -> list[Boundary]:
    graph = package.graph
    result: list[Boundary] = []

    # Keep region input/output declarations intact when present.  They are
    # part of the stored region contract; deriving them is only a fallback for
    # older traces that omitted both fields.
    for region in package.regions:
        nodes = _ordered_nodes(graph, region.node_ids)
        if not nodes:
            continue
        inputs, outputs = _region_io(graph, region, nodes)
        result.append(
            Boundary(
                id=stable_boundary_id("region", region.id),
                kind="region",
                name=region.name or stable_boundary_id("region", region.id),
                node_ids=nodes,
                inputs=inputs,
                outputs=outputs,
                metadata={
                    "source": "stored",
                    "region_id": region.id,
                    "creation_method": region.creation_method,
                    "engine_op": region.engine_op,
                },
            )
        )

    # ``nodes_for_module`` is the canonical existing implementation of module
    # subtree membership.  Sort modules explicitly because module JSON order is
    # not a semantic ordering guarantee.
    modules = sorted(package.modules, key=lambda module: stable_boundary_id("module", module.id))
    for module in modules:
        nodes = _ordered_nodes(graph, nodes_for_module(graph, (module.id,)))
        if not nodes:
            continue
        inputs, outputs = _boundary_io(graph, nodes)
        result.append(
            Boundary(
                id=stable_boundary_id("module", module.id),
                kind="module",
                name=_module_name(module, module.id),
                node_ids=nodes,
                inputs=inputs,
                outputs=outputs,
                metadata={
                    "source": "derived",
                    "module_id": module.id,
                    "module_type": getattr(module, "type", ""),
                },
            )
        )

    for op in graph.ops_in_execution_order():
        nodes = (op.id,)
        inputs, outputs = _boundary_io(graph, nodes)
        result.append(
            Boundary(
                id=stable_boundary_id("op", op.id),
                kind="op",
                name=_operator_name(op),
                node_ids=nodes,
                inputs=inputs,
                outputs=outputs,
                metadata={
                    "source": "derived",
                    "operator_id": op.id,
                    "execution_index": op.execution_index,
                },
            )
        )
    return result


def _boundary_order(boundary: Boundary) -> tuple[int, str]:
    return (BOUNDARY_TYPE_PRIORITY[boundary.kind], boundary.id)


def _canonicalise(raw: Sequence[Boundary]) -> tuple[tuple[Boundary, ...], tuple[Boundary, ...], dict[str, str]]:
    groups: dict[tuple[int, ...], list[Boundary]] = {}
    for boundary in raw:
        groups.setdefault(_node_key(boundary.node_ids), []).append(boundary)

    aliases: dict[str, str] = {}
    canonical: list[Boundary] = []
    all_boundaries: list[Boundary] = []
    for key, members in groups.items():
        del key  # the key is consumed by the grouping; node IDs stay on records.
        members = sorted(members, key=_boundary_order)
        selected = members[0]
        alias_ids = tuple(member.id for member in members[1:])
        canonical_boundary = replace(
            selected,
            canonical_id=selected.id,
            aliases=alias_ids,
        )
        canonical.append(canonical_boundary)
        for member in members:
            aliases[member.id] = selected.id
            all_boundaries.append(
                replace(
                    member,
                    canonical_id=selected.id,
                    aliases=alias_ids,
                )
            )

    canonical.sort(key=_boundary_order)
    all_boundaries.sort(key=_boundary_order)
    aliases = {key: value for key, value in sorted(aliases.items()) if key != value}
    return tuple(canonical), tuple(all_boundaries), aliases


def _containment(
    boundaries: Sequence[Boundary],
) -> tuple[dict[str, str | None], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Build primary/alternate parents and reverse children for a DAG."""

    node_sets = {boundary.id: frozenset(boundary.node_ids) for boundary in boundaries}
    parents: dict[str, str | None] = {}
    alternates: dict[str, tuple[str, ...]] = {}
    children: dict[str, list[str]] = {boundary.id: [] for boundary in boundaries}

    for child in boundaries:
        child_id = child.id
        child_nodes = node_sets[child_id]
        supersets = [
            candidate
            for candidate in boundaries
            if child_id != candidate.id
            and child_nodes < node_sets[candidate.id]
        ]
        minimal: list[Boundary] = []
        for candidate in supersets:
            candidate_nodes = node_sets[candidate.id]
            if any(
                other.id != candidate.id and node_sets[other.id] < candidate_nodes
                for other in supersets
            ):
                continue
            minimal.append(candidate)
        minimal.sort(key=_boundary_order)
        primary = minimal[0].id if minimal else None
        parents[child_id] = primary
        alternates[child_id] = tuple(candidate.id for candidate in minimal[1:])
        for parent in minimal:
            children[parent.id].append(child_id)

    ordered_children = {
        parent: tuple(sorted(values)) for parent, values in sorted(children.items())
    }
    return parents, alternates, ordered_children


def inventory_boundaries(package: TracePackage) -> BoundaryInventory:
    """Inventory stored regions and derived module/operator boundaries.

    The input package is never mutated.  The returned inventory is deterministic
    for a given graph even when the source JSON lists regions or modules in a
    different order.
    """

    if not hasattr(package, "graph"):
        raise TypeError("inventory_boundaries expects a TracePackage-like object")
    raw = _raw_boundaries(package)
    canonical, all_boundaries, aliases = _canonicalise(raw)
    parents, alternates, children = _containment(canonical)
    return BoundaryInventory(
        boundaries=canonical,
        raw_boundaries=all_boundaries,
        aliases=aliases,
        parents=parents,
        alternate_parents=alternates,
        children=children,
    )


# Public spellings used by the CLI/Agent integration layer.
build_boundary_inventory = inventory_boundaries
build_boundary_graph = inventory_boundaries
boundary_inventory = inventory_boundaries


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict):
        return attrs
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
        except Exception:
            return None
        return data if isinstance(data, Mapping) else None
    return None


def _provenance_fields(first_failure: Any) -> tuple[int | None, int | None, bool]:
    """Extract value/producer IDs without treating a display name as proof."""

    if first_failure is None:
        return None, None, False
    mapping = _as_mapping(first_failure)
    if mapping is not None:
        # Comparison reports use ``value_id`` and ``producer_op_id``.  A
        # persisted Evidence record nests the same facts under observations.
        for nested_key in ("first_failure", "observations", "provenance"):
            nested = mapping.get(nested_key)
            if nested is not None:
                value_id, producer_id, available = _provenance_fields(nested)
                if available:
                    return value_id, producer_id, True
        raw_value = mapping.get("value_id", mapping.get("trace_value_id"))
        raw_producer = mapping.get(
            "producer_op_id",
            mapping.get(
                "producer_operator_id",
                mapping.get("operator_id", mapping.get("producer")),
            ),
        )
    else:
        raw_value = getattr(first_failure, "value_id", None)
        raw_producer = getattr(
            first_failure,
            "producer_op_id",
            getattr(
                first_failure,
                "producer_operator_id",
                getattr(first_failure, "operator_id", getattr(first_failure, "producer", None)),
            ),
        )

    value_id: int | None = None
    producer_id: int | None = None
    try:
        if raw_value is not None:
            value_id = int(raw_value)
    except (TypeError, ValueError):
        value_id = None
    try:
        if raw_producer is not None:
            producer_id = int(raw_producer)
    except (TypeError, ValueError):
        producer_id = None
    return value_id, producer_id, value_id is not None or producer_id is not None


def reverse_data_dependency(
    graph: Graph,
    *,
    value_id: int | None = None,
    producer_op_id: int | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(value_ids, operator_ids)`` in the reverse dependency cone.

    The producer links are derived from the graph when possible so stale
    materialised ``producer`` fields cannot hide a mutation edge.  Operators
    are returned in deterministic execution order.
    """

    producers, _consumers = graph.derived_links()
    values_seen: set[int] = set()
    operators_seen: set[int] = set()

    def visit_value(current: int) -> None:
        if current in values_seen:
            return
        values_seen.add(current)
        producer = producers.get(current)
        if producer is None and graph.has_value(current):
            producer = graph.value(current).producer
        if producer is not None:
            visit_operator(int(producer))

    def visit_operator(current: int) -> None:
        if current in operators_seen:
            return
        if not graph.has_op(current):
            return
        operators_seen.add(current)
        for input_id in graph.op_input_value_ids(graph.op(current)):
            visit_value(int(input_id))

    if value_id is not None:
        visit_value(int(value_id))
    if producer_op_id is not None:
        visit_operator(int(producer_op_id))

    ordered_values = tuple(sorted(values_seen))
    ordered_operators = tuple(
        op.id
        for op in sorted(
            graph.operators,
            key=lambda item: (item.execution_index, item.id),
        )
        if op.id in operators_seen
    )
    return ordered_values, ordered_operators


def select_causal_descendants(
    package_or_graph: TracePackage | Graph,
    inventory: BoundaryInventory,
    failing_boundary: str | Boundary,
    first_failure: Any = None,
) -> CausalSelection:
    """Select descendants relevant to a first failing output.

    A valid value/producer provenance causes the reverse data-dependency cone
    to filter descendants.  If provenance is absent, all containment
    descendants remain candidates and confidence is explicitly low.
    """

    graph = getattr(package_or_graph, "graph", package_or_graph)
    if not isinstance(graph, Graph):
        raise TypeError("select_causal_descendants expects a TracePackage or Graph")
    root = inventory.resolve(failing_boundary)
    if root is None:
        return CausalSelection(reason="unknown failing boundary")

    value_id, producer_id, _supplied = _provenance_fields(first_failure)
    if value_id is not None and not graph.has_value(value_id):
        # An invalid first-failure value_id must not be reported as valid
        # provenance merely because a producer field happens to parse.
        valid_value = False
        valid_producer = False
        provenance_available = False
    else:
        valid_value = value_id is not None and graph.has_value(value_id)
        valid_producer = producer_id is not None and graph.has_op(producer_id)
        provenance_available = valid_value or valid_producer
    descendants = inventory.descendants(root.id)

    if provenance_available:
        cone_values, cone_operators = reverse_data_dependency(
            graph,
            value_id=value_id if valid_value else None,
            producer_op_id=producer_id if valid_producer else None,
        )
        cone = set(cone_operators)
        selected = tuple(
            boundary.id
            for boundary in inventory.boundaries
            if boundary.id in descendants
            and (cone.intersection(boundary.node_ids) or (value_id is not None and value_id in boundary.outputs))
        )
        selected = tuple(sorted(selected))
        reason = "reverse data dependency from first failing output"
        confidence = "high"
    else:
        cone_values = ()
        cone_operators = ()
        selected = tuple(sorted(descendants))
        reason = "first-failure provenance unavailable; containment fallback"
        confidence = "low"

    return CausalSelection(
        boundary_ids=selected,
        value_id=value_id,
        producer_op_id=producer_id,
        value_ids=tuple(cone_values),
        operator_ids=tuple(cone_operators),
        confidence=confidence,
        provenance_available=provenance_available,
        reason=reason,
    )


def causal_descendants(
    package_or_graph: TracePackage | Graph,
    inventory: BoundaryInventory,
    failing_boundary: str | Boundary,
    first_failure: Any = None,
) -> tuple[str, ...]:
    """Return only stable IDs for causal descendants.

    ``select_causal_descendants`` is available when callers also need the
    confidence and dependency-cone details.
    """

    return select_causal_descendants(
        package_or_graph, inventory, failing_boundary, first_failure
    ).boundary_ids


def compute_causal_descendants(
    package_or_graph: TracePackage | Graph,
    inventory: BoundaryInventory,
    failing_boundary: str | Boundary,
    first_failure: Any = None,
) -> tuple[str, ...]:
    """Compatibility alias for :func:`causal_descendants`."""

    return causal_descendants(
        package_or_graph, inventory, failing_boundary, first_failure
    )


def compute_in_scope_frontier(
    inventory: BoundaryInventory,
    scope: Sequence[str] = (),
    package: Any = None,
) -> set[str]:
    """Compute the required in-scope canonical boundary frontier for a session."""
    import fnmatch

    if scope:
        in_scope: set[str] = set()
        has_wildcards = lambda s: any(c in s for c in ("*", "?", "[", "]"))
        for pattern in scope:
            if has_wildcards(pattern):
                for b in inventory.canonical_boundaries:
                    if fnmatch.fnmatchcase(b.id, pattern) or fnmatch.fnmatchcase(b.name, pattern):
                        in_scope.add(b.id)
            else:
                b = inventory.resolve(pattern)
                if b is not None:
                    in_scope.add(b.id)
                    in_scope.update(inventory.descendants(b.id))
                else:
                    for cand in inventory.canonical_boundaries:
                        if cand.name == pattern:
                            in_scope.add(cand.id)
                            in_scope.update(inventory.descendants(cand.id))
        frontier = {
            bid for bid in in_scope
            if inventory.parents.get(bid) not in in_scope
        }
        return frontier if frontier else in_scope

    # Default: stored regions in the package
    stored_region_ids = {
        inventory.resolve_id(f"region:{r.id}")
        for r in getattr(package, "regions", ())
        if inventory.resolve(f"region:{r.id}")
    }
    if stored_region_ids:
        return stored_region_ids

    # Top-level roots in DAG
    top_level = {
        b.id for b in inventory.canonical_boundaries
        if inventory.parents.get(b.id) is None
    }
    if top_level:
        return top_level

    return set(inventory.canonical_ids)


__all__ = [
    "BOUNDARY_TYPES",
    "BOUNDARY_TYPE_PRIORITY",
    "Boundary",
    "BoundaryGraph",
    "BoundaryInventory",
    "BoundaryRecord",
    "CausalSelection",
    "ContainmentGraph",
    "boundary_inventory",
    "build_boundary_graph",
    "build_boundary_inventory",
    "causal_descendants",
    "compute_causal_descendants",
    "compute_in_scope_frontier",
    "inventory_boundaries",
    "reverse_data_dependency",
    "select_causal_descendants",
    "stable_boundary_id",
]
