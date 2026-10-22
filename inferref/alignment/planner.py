"""Deterministic, read-only planning for alignment sessions.

The planner composes existing InferRef primitives.  Region scoring is delegated
to :func:`inferref.region.recommend.recommend_regions` (including for derived
module/operator regions), and adapter compatibility is delegated to
``AdapterCapabilities``.  No testcase is extracted and no engine process is
started here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferref.agent.protocol import (
    AdapterCapabilities,
    AgentProtocolError,
    EngineAdapter,
)
from inferref.alignment.boundaries import (
    Boundary,
    BoundaryInventory,
    CausalSelection,
    inventory_boundaries,
    select_causal_descendants,
)
from inferref.alignment.narrowing import (
    BINARY_SEARCH_REASON,
    binary_search_probe,
)
from inferref.ir.operator import Annotation
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.region.analysis import analyze_region
from inferref.region.recommend import RegionRecommendation, recommend_regions

DEFAULT_FIXTURE_KIND = "reference_derived"
UNKNOWN_CAPABILITY_PENALTY = 10
LARGE_BOUNDARY_PENALTY = 10
IMMEDIATE_CHILD_BONUS = 40
CAUSAL_CONE_BONUS = 20
REUSABLE_TESTCASE_BONUS = 20
# A midpoint probe must outrank the recommender's bounded heuristic score and
# the causal/reuse bonuses; otherwise coarse-to-fine order is not deterministic.
BINARY_SEARCH_BONUS = 1000


@dataclass(frozen=True)
class Runnability:
    """Planner's preflight assessment for one boundary.

    ``status`` is deliberately not an execution result.  It describes whether
    a future runner call appears eligible from the currently available source,
    fixture, and adapter declarations.
    """

    status: str = "unknown"
    basis: str = "estimated"
    reasons: tuple[str, ...] = ()
    requirements: Mapping[str, Any] = field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return self.status not in {"unsupported", "blocked"}

    @property
    def reason(self) -> str:
        return self.reasons[0] if self.reasons else self.status

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "basis": self.basis,
            "reasons": list(self.reasons),
        }
        if self.requirements:
            result["requirements"] = _json_value(self.requirements)
        return result


@dataclass(frozen=True)
class PlanCandidate:
    """One ordered planner action."""

    boundary: str
    name: str
    priority_score: int
    plan_confidence: str
    runnability: Runnability | Mapping[str, Any]
    fixture_kind: str = DEFAULT_FIXTURE_KIND
    action: str = "extract_and_run"
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    canonical_boundary: str | None = None
    node_count: int = 0
    base_score: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def boundary_id(self) -> str:
        return self.boundary

    @property
    def score(self) -> int:
        return self.priority_score

    def to_dict(self) -> dict[str, Any]:
        if isinstance(self.runnability, Runnability):
            runnability = self.runnability.to_dict()
        else:
            runnability = _json_value(self.runnability)
        result: dict[str, Any] = {
            "boundary": self.boundary,
            "name": self.name,
            "priority_score": self.priority_score,
            "plan_confidence": self.plan_confidence,
            "runnability": runnability,
            "fixture_kind": self.fixture_kind,
            "action": self.action,
            "reasons": list(self.reasons),
        }
        if self.warnings:
            result["warnings"] = list(self.warnings)
        if self.canonical_boundary and self.canonical_boundary != self.boundary:
            result["canonical_boundary"] = self.canonical_boundary
        # These fields make the result useful to a report while retaining the
        # compact fields shown in the public 0.10 example.
        result["node_count"] = self.node_count
        result["base_score"] = self.base_score
        if self.details:
            result["details"] = _json_value(self.details)
        return result


@dataclass(frozen=True)
class AlignmentPlan:
    """Store-independent plan projection and causal context."""

    candidates: tuple[PlanCandidate, ...] = ()
    inventory: BoundaryInventory | None = None
    current_failure: str | None = None
    causal_descendants: tuple[str, ...] = ()
    causal_confidence: str = "low"
    causal_reason: str = "no active failure"

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> PlanCandidate:
        return self.candidates[index]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "current_failure": self.current_failure,
            "causal_descendants": list(self.causal_descendants),
            "causal_confidence": self.causal_confidence,
            "causal_reason": self.causal_reason,
        }
        if self.inventory is not None:
            result["inventory"] = self.inventory.to_dict()
        return result


def _json_value(value: Any) -> Any:
    """Convert the small set of public planner values into JSON-safe data."""

    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _json_value(value.to_dict())
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    for attr in ("snapshot", "data"):
        nested = getattr(value, attr, None)
        if isinstance(nested, Mapping):
            return nested
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


def _mapping_value(value: Any, *keys: str, default: Any = None) -> Any:
    mapping = _as_mapping(value)
    if mapping is None:
        return default
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _boundary_state(session: Any, boundary: Boundary, inventory: BoundaryInventory) -> str | None:
    """Read aggregate state from common snapshot representations."""

    session_map = _as_mapping(session)
    if session_map is None:
        return None
    entries = session_map.get("boundaries")
    resolved = inventory.resolve_id(boundary.id)
    entry: Any = None
    if isinstance(entries, Mapping):
        entry = entries.get(boundary.id, entries.get(resolved))
        if entry is None:
            for alias, target in inventory.aliases.items():
                if target == resolved and alias in entries:
                    entry = entries[alias]
                    break
    elif isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
        for item in entries:
            item_map = _as_mapping(item)
            if item_map and str(item_map.get("boundary", item_map.get("id", ""))) in {
                boundary.id,
                resolved,
            }:
                entry = item_map
                break
    if isinstance(entry, str):
        return entry
    entry_map = _as_mapping(entry)
    if entry_map is not None:
        for key in ("state", "status", "aggregate_state"):
            value = entry_map.get(key)
            if isinstance(value, str):
                return value

    for key in ("verified_boundaries", "verified"):
        values = session_map.get(key)
        if _contains_boundary(values, boundary.id, resolved):
            return "verified"
    for key in ("unsupported_boundaries", "unsupported"):
        values = session_map.get(key)
        if _contains_boundary(values, boundary.id, resolved):
            return "unsupported"
    return None


def _contains_boundary(values: Any, boundary_id: str, resolved: str) -> bool:
    if isinstance(values, Mapping):
        values = values.keys()
    if isinstance(values, (str, bytes)) or values is None:
        return values in {boundary_id, resolved}
    try:
        return any(str(value) in {boundary_id, resolved} for value in values)
    except TypeError:
        return False


def _failure_payload(session: Any, explicit: Any = None) -> tuple[str | None, Any]:
    """Return ``(boundary_id, first_failure_payload)`` from a snapshot."""

    if explicit is not None:
        if isinstance(explicit, str):
            return explicit, None
        explicit_map = _as_mapping(explicit)
        if explicit_map is not None:
            boundary = explicit_map.get("boundary", explicit_map.get("subject"))
            if isinstance(boundary, Mapping):
                boundary = boundary.get("boundary")
            return (str(boundary) if boundary is not None else None), explicit

    session_map = _as_mapping(session)
    if session_map is None:
        return None, None
    narrowing = session_map.get("narrowing")
    sources = [narrowing, session_map]
    for source in sources:
        source_map = _as_mapping(source)
        if source_map is None:
            continue
        for key in (
            "smallest_confirmed_failure",
            "current_failure",
            "failure",
            "active_failure",
        ):
            value = source_map.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                return value, None
            value_map = _as_mapping(value)
            if value_map is not None:
                boundary = value_map.get("boundary", value_map.get("subject"))
                if isinstance(boundary, Mapping):
                    boundary = boundary.get("boundary")
                return (str(boundary) if boundary is not None else None), value

    return None, None


def _first_failure(session: Any, failure_payload: Any, explicit: Any = None) -> Any:
    if explicit is not None and not isinstance(explicit, str):
        return explicit
    if failure_payload is not None:
        payload_map = _as_mapping(failure_payload)
        if payload_map is not None:
            for key in ("first_failure", "observations", "provenance"):
                if payload_map.get(key) is not None:
                    return payload_map[key]
            return failure_payload
    session_map = _as_mapping(session)
    if session_map is not None:
        narrowing = _as_mapping(session_map.get("narrowing"))
        for source in (narrowing, session_map):
            if source is not None and source.get("first_failure") is not None:
                return source["first_failure"]
    return None


def _adapter_capabilities(adapter: Any) -> tuple[AdapterCapabilities | None, str | None]:
    """Resolve capabilities without starting an adapter process."""

    if adapter is None:
        return None, None
    if isinstance(adapter, AdapterCapabilities):
        return adapter, None
    capabilities = getattr(adapter, "capabilities", None)
    if isinstance(capabilities, AdapterCapabilities):
        return capabilities, None
    if isinstance(adapter, Mapping):
        raw = adapter.get("capabilities")
        if raw is None:
            return None, "adapter did not declare capabilities"
        try:
            return AdapterCapabilities.from_dict(raw), None
        except (AgentProtocolError, TypeError, ValueError) as exc:
            return None, f"invalid adapter capabilities: {exc}"
    if isinstance(adapter, (str, Path)):
        try:
            loaded = EngineAdapter.load(adapter)
        except (OSError, ValueError, AgentProtocolError) as exc:
            return None, f"cannot load adapter for planning: {exc}"
        return loaded.capabilities, None if loaded.capabilities is not None else "adapter did not declare capabilities"
    if capabilities is None:
        return None, "adapter did not declare capabilities"
    return None, "adapter capabilities are not an AdapterCapabilities record"


def boundary_requirements(
    package: TracePackage,
    boundary: Boundary,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive testcase-like requirements for adapter capability assessment."""

    graph = package.graph
    input_values = [
        graph.value(value_id)
        for value_id in boundary.inputs
        if graph.has_value(value_id)
    ]
    from inferref.quantization.artifacts import feature, validate_registry

    value_ids = set(boundary.inputs) | set(boundary.outputs)
    for node_id in boundary.node_ids:
        if graph.has_op(node_id):
            op = graph.op(node_id)
            value_ids.update(graph.op_input_value_ids(op))
            value_ids.update(graph.op_output_value_ids(op))
            for alias in op.effects.aliases:
                value_ids.update((alias.input_value_id, alias.output_value_id))
    if any(v.quantization is not None for v in graph.values):
        closure = validate_registry([v.to_dict() for v in graph.values], value_ids)
        value_ids.update(closure.value_ids)
    values = [graph.value(value_id) for value_id in value_ids if graph.has_value(value_id)]
    dtypes = sorted({str(value.dtype) for value in values if value.dtype})
    max_rank = max((value.rank for value in values), default=0)
    features: set[str] = {feature(v.quantization) for v in values if v.quantization is not None}
    if len(boundary.outputs) > 1:
        features.add("multiple_outputs")
    for value in input_values:
        expected_stride: list[int] = []
        running = 1
        for dim in reversed(value.shape):
            expected_stride.append(running)
            running *= max(int(dim), 1)
        if value.stride and tuple(value.stride) != tuple(reversed(expected_stride)):
            features.add("strided_inputs")
    for node_id in boundary.node_ids:
        if not graph.has_op(node_id):
            continue
        op = graph.op(node_id)
        if op.effects.aliases:
            features.add("alias_effects")
        if op.effects.mutated_storages:
            features.add("mutation_effects")
    result: dict[str, Any] = {
        "dtypes": dtypes,
        "max_rank": max_rank,
        "features": sorted(features),
    }
    if overrides:
        result.update(_json_value(overrides))
        if "dtypes" in result and isinstance(result["dtypes"], (tuple, set)):
            result["dtypes"] = sorted(result["dtypes"])
        if "features" in result and isinstance(result["features"], (tuple, set)):
            result["features"] = sorted(result["features"])
        if any(v.quantization is not None for v in values):
            result["features"] = sorted(set(result.get("features", ())) | features)
            result["dtypes"] = sorted(set(result.get("dtypes", ())) | set(dtypes))
            result["max_rank"] = max(result.get("max_rank", 0), max_rank)
    return result


def _semantic_annotation(package: TracePackage, node_ids: Iterable[int]) -> Annotation | None:
    graph = package.graph
    annotations: list[Annotation] = []
    for node_id in node_ids:
        if not graph.has_op(node_id):
            continue
        annotations.extend(
            annotation
            for annotation in graph.op(node_id).annotations
            if annotation.type == "semantic"
        )
    if not annotations:
        return None
    # Node IDs passed by the inventory are execution ordered.  Keep the first
    # annotation as the stable display label; its confidence is the evidence
    # actually associated with that operator.
    return annotations[0]


def _region_for_boundary(package: TracePackage, boundary: Boundary, index: int) -> RegionRecord:
    semantic: Annotation | None = None
    if boundary.kind == "region":
        raw_id = boundary.metadata.get("region_id") if boundary.metadata else None
        raw_region = package.region(raw_id) if raw_id is not None else None
        if raw_region is not None:
            semantic = raw_region.semantic
    if semantic is None:
        semantic = _semantic_annotation(package, boundary.node_ids)
    return RegionRecord(
        id=index,
        name=boundary.name,
        node_ids=boundary.node_ids,
        inputs=boundary.inputs,
        outputs=boundary.outputs,
        semantic=semantic,
        engine_op=(
            boundary.metadata.get("engine_op") if boundary.metadata else None
        ),
    )


def _recommendations(
    package: TracePackage,
    inventory: BoundaryInventory,
) -> dict[str, RegionRecommendation]:
    """Use the existing region recommender for every canonical boundary."""

    records = [
        _region_for_boundary(package, boundary, index)
        for index, boundary in enumerate(inventory.boundaries)
    ]
    projected = TracePackage(
        manifest=package.manifest,
        graph=package.graph,
        modules=package.modules,
        sources=package.sources,
        regions=records,
        storages=package.storages,
        root=package.root,
    )
    recommendations = recommend_regions(projected)
    result: dict[str, RegionRecommendation] = {}
    for recommendation in recommendations:
        # The projected records use their list index as an ID.  Mapping by ID
        # avoids assigning two malformed-but-legal duplicate names to the
        # same boundary.
        try:
            index = int(recommendation.region_id)
        except (TypeError, ValueError):
            index = None
        if index is None:
            continue
        result[inventory.boundaries[index].id] = recommendation
    # ``name`` is expected to be unique for normal traces, but a malformed
    # trace can repeat it.  Fill any missing entry by computing details directly
    # while preserving the same RegionDetails shape.
    for index, record in enumerate(records):
        boundary_id = inventory.boundaries[index].id
        if boundary_id not in result:
            details = analyze_region(projected, record)
            result[boundary_id] = RegionRecommendation(
                region_id=str(index),
                region_name=record.name,
                score=-15,
                details=details,
                reasons=("-15: boundary could not be recommended",),
                warnings=(),
            )
    return result


def _lookup_fixture_entry(session: Any, boundary: Boundary, inventory: BoundaryInventory) -> Any:
    session_map = _as_mapping(session)
    if session_map is None:
        return None
    resolved = inventory.resolve_id(boundary.id)
    containers = []
    for key in ("testcases", "testcase_cache", "reusable_testcases", "fixtures"):
        value = session_map.get(key)
        if value is not None:
            containers.append(value)
    boundary_entry = _boundary_state_entry(session_map, boundary.id, resolved)
    if boundary_entry is not None:
        containers.append(boundary_entry)
    for container in containers:
        if isinstance(container, Mapping):
            for key in (boundary.id, resolved):
                if key in container:
                    return container[key]
            # Some projections key by a boundary object with a ``subject``
            # field rather than by the ID itself.
            for key, value in container.items():
                if str(key) in {boundary.id, resolved}:
                    return value
                value_map = _as_mapping(value)
                if value_map and str(value_map.get("boundary", value_map.get("subject", ""))) in {
                    boundary.id,
                    resolved,
                }:
                    return value
        elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
            for value in container:
                value_map = _as_mapping(value)
                if value_map and str(value_map.get("boundary", value_map.get("subject", ""))) in {
                    boundary.id,
                    resolved,
                }:
                    return value
    return None


def _boundary_state_entry(session_map: Mapping[str, Any], boundary_id: str, resolved: str) -> Any:
    entries = session_map.get("boundaries")
    if not isinstance(entries, Mapping):
        return None
    return entries.get(boundary_id, entries.get(resolved))


def _reusable_testcase(session: Any, boundary: Boundary, inventory: BoundaryInventory) -> bool:
    entry = _lookup_fixture_entry(session, boundary, inventory)
    if entry is None:
        return False
    if isinstance(entry, (str, Path)):
        return bool(str(entry))
    entry_map = _as_mapping(entry)
    if entry_map is None:
        return False
    for key in ("reusable", "valid", "reproducible"):
        if key in entry_map and entry_map[key] is False:
            return False
    status = entry_map.get("status")
    if isinstance(status, str) and status.lower() in {"invalid", "stale", "missing", "error"}:
        return False
    return bool(
        entry_map.get("path")
        or entry_map.get("tree_sha256")
        or entry_map.get("digest")
        or entry_map.get("reusable")
        or entry_map.get("valid")
        or entry_map.get("reproducible")
    )


def _fixture_kind(session: Any, requested: str | None) -> str:
    if requested:
        return requested
    values = _mapping_value(session, "required_fixture_kinds", default=None)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        for value in values:
            if isinstance(value, str) and value:
                return value
    return DEFAULT_FIXTURE_KIND


def _runnability(
    package: TracePackage,
    boundary: Boundary,
    details: Any,
    adapter: Any,
    requirements: Mapping[str, Any],
) -> Runnability:
    reasons: list[str] = []
    from inferref.quantization.artifacts import needs_quantization, validate_registry

    if needs_quantization(requirements):
        from inferref.quantization.reference import validate_parameters
        from inferref.quantization.trace import trace_tensor

        try:
            graph = package.graph
            roots = set(boundary.inputs) | set(boundary.outputs)
            for node_id in boundary.node_ids:
                roots.update(graph.op_input_value_ids(graph.op(node_id)))
                roots.update(graph.op_output_value_ids(graph.op(node_id)))
            closure = validate_registry([v.to_dict() for v in graph.values], roots)
            parameters = {i: trace_tensor(package, i, dequantize=False) for i in closure.parameter_ids}
            for value_id in closure.quantized_ids:
                # Intermediate q-values need not carry full code payloads for
                # an extracted region. Only parameters are additional boundary
                # dependencies; avoid materializing entire weights to plan.
                validate_parameters(graph.value(value_id).quantization, parameters)
        except (ValueError, OSError) as exc:
            return Runnability(status="blocked", basis="source", reasons=(str(exc),), requirements=requirements)
    if not details.reproducible:
        reasons.append("boundary lacks complete portable payload coverage")
        return Runnability(
            status="blocked",
            basis="source",
            reasons=tuple(reasons),
            requirements=requirements,
        )

    capabilities, capability_error = _adapter_capabilities(adapter)
    if adapter is not None and capabilities is None and needs_quantization(requirements):
        return Runnability(status="unsupported", basis="declared",
                           reasons=("quantized testcases require explicit adapter capabilities",),
                           requirements=requirements)
    if capability_error and adapter is not None:
        # An old v0.1 adapter intentionally has no declaration.  It can still
        # be run, but compatibility is estimated and receives a planner
        # penalty; malformed declarations are surfaced as unknown as well.
        reasons.append(capability_error)
        return Runnability(
            status="eligible",
            basis="estimated",
            reasons=tuple(reasons),
            requirements=requirements,
        )
    if capabilities is None:
        return Runnability(
            status="eligible",
            basis="estimated",
            reasons=tuple(reasons),
            requirements=requirements,
        )

    incompatibilities = list(capabilities.incompatibilities(requirements))
    for contract in requirements.get("contracts", ()):
        # Contract-specific declarations refine the global declaration.  A
        # caller may supply a separate requirement mapping per contract; when
        # it does not, the boundary requirements are the conservative input.
        per_contract = requirements.get("contract_requirements", {})
        if isinstance(per_contract, Mapping):
            contract_requirements = per_contract.get(contract, requirements)
        else:
            contract_requirements = requirements
        if not isinstance(contract_requirements, Mapping):
            contract_requirements = requirements
        incompatibilities.extend(
            capabilities.contract_incompatibilities(contract, contract_requirements)
        )
    if incompatibilities:
        reasons.extend(
            "adapter capability incompatible: "
            + str(issue.get("kind", "requirement"))
            + (f" requires {issue.get('required')}" if "required" in issue else "")
            for issue in incompatibilities
        )
        return Runnability(
            status="unsupported",
            basis="declared",
            reasons=tuple(reasons),
            requirements=requirements,
        )

    assessment = capabilities.assessment(requirements)
    if assessment == "unchecked":
        reasons.append("adapter capability compatibility is unchecked")
        basis = "unchecked"
    else:
        basis = "declared"
    return Runnability(
        status="eligible",
        basis=basis,
        reasons=tuple(reasons),
        requirements=requirements,
    )


def _plan_confidence(runnability: Runnability, causal: str, has_failure: bool) -> str:
    if runnability.status != "eligible":
        return "low"
    if has_failure and causal == "high" and runnability.basis in {"declared", "estimated"}:
        return "high"
    if runnability.basis == "unchecked" or causal == "low":
        return "medium"
    return "medium"


def plan_alignment(
    package: TracePackage,
    *,
    session: Any = None,
    adapter: Any = None,
    limit: int | None = None,
    from_boundary: str | Boundary | None = None,
    first_failure: Any = None,
    fixture_kind: str | None = None,
    requirements: Mapping[str, Any] | Mapping[str, Mapping[str, Any]] | None = None,
    include_unsupported: bool = False,
) -> AlignmentPlan:
    """Build a deterministic read-only alignment plan.

    ``requirements`` may be a single requirement object or a mapping keyed by
    boundary ID.  The latter is useful when a contract-aware caller has already
    validated extracted testcases; planner logic still remains read-only.
    """

    if not hasattr(package, "graph"):
        raise TypeError("plan_alignment expects a TracePackage-like object")
    inventory = inventory_boundaries(package)
    recommendations = _recommendations(package, inventory)
    failure_id, failure_payload = _failure_payload(session, from_boundary)
    if failure_id is not None:
        failure_id = inventory.resolve_id(failure_id)
        if inventory.resolve(failure_id) is None:
            failure_id = None
    provenance = _first_failure(session, failure_payload, first_failure)
    causal = CausalSelection(reason="no active failure")
    if failure_id is not None:
        causal = select_causal_descendants(package, inventory, failure_id, provenance)
    causal_ids = set(causal.boundary_ids)
    immediate_ids = set(inventory.child_ids(failure_id)) if failure_id is not None else set()
    causal_immediate = immediate_ids.intersection(causal_ids)
    selected_fixture_kind = _fixture_kind(session, fixture_kind)
    boundary_states = {
        boundary.id: _boundary_state(session, boundary, inventory)
        for boundary in inventory.boundaries
    }
    binary_probe: str | None = None
    if failure_id is not None:
        parent = inventory.resolve(failure_id)
        parent_state = boundary_states.get(failure_id) if parent is not None else None
        if parent is not None and parent_state in {None, "failing"}:
            probe = binary_search_probe(
                inventory,
                failure_id,
                verified=tuple(
                    boundary_id
                    for boundary_id, state in boundary_states.items()
                    if state == "verified"
                ),
                failing=tuple(
                    boundary_id
                    for boundary_id, state in boundary_states.items()
                    if state == "failing"
                ),
                graph=package.graph,
            )
            if probe is not None:
                binary_probe = inventory.resolve_id(probe)

    candidates: list[PlanCandidate] = []
    for boundary in inventory.boundaries:
        state = boundary_states.get(boundary.id)
        if state in {"verified", "unsupported"}:
            continue

        override: Mapping[str, Any] | None = None
        if requirements and all(isinstance(key, str) for key in requirements):
            # Distinguish one requirements dict from a map keyed by
                # boundary IDs using the known requirement field names.
                req_keys = {"dtypes", "max_rank", "features", "contracts"}
                if req_keys.intersection(requirements.keys()):  # type: ignore[arg-type]
                    override = requirements  # type: ignore[assignment]
                else:
                    candidate_override = requirements.get(boundary.id)  # type: ignore[union-attr]
                    if candidate_override is None:
                        candidate_override = requirements.get(inventory.resolve_id(boundary.id))  # type: ignore[union-attr]
                    if isinstance(candidate_override, Mapping):
                        override = candidate_override
        req = boundary_requirements(package, boundary, override)
        recommendation = recommendations[boundary.id]
        details = recommendation.details
        runnable = _runnability(package, boundary, details, adapter, req)
        if runnable.status == "unsupported" and not include_unsupported:
            continue

        score = recommendation.score
        reasons = list(recommendation.reasons)
        warnings = list(recommendation.warnings)
        if boundary.id in causal_immediate:
            score += IMMEDIATE_CHILD_BONUS
            reasons.append(f"+{IMMEDIATE_CHILD_BONUS}: immediate child of current failing boundary")
        elif boundary.id in causal_ids:
            score += CAUSAL_CONE_BONUS
            reasons.append(f"+{CAUSAL_CONE_BONUS}: on reverse dependency cone of first failing output")

        if binary_probe is not None and boundary.id == binary_probe:
            score += BINARY_SEARCH_BONUS
            reasons.append(BINARY_SEARCH_REASON)

        if _reusable_testcase(session, boundary, inventory):
            score += REUSABLE_TESTCASE_BONUS
            reasons.append(f"+{REUSABLE_TESTCASE_BONUS}: reusable validated testcase")

        if runnable.basis in {"estimated", "unchecked"}:
            score -= UNKNOWN_CAPABILITY_PENALTY
            reasons.append(
                f"-{UNKNOWN_CAPABILITY_PENALTY}: adapter capability compatibility is unknown"
            )
        total_bytes = details.activation_bytes + details.parameter_bytes
        if total_bytes > 200 * 1024 * 1024:
            score -= LARGE_BOUNDARY_PENALTY
            reasons.append(
                f"-{LARGE_BOUNDARY_PENALTY}: large boundary cost ({total_bytes / (1024 * 1024):.1f} MB)"
            )

        # Runnability facts are part of the explanation, but do not duplicate
        # every adapter diagnostic in the score reasons unless it changes the
        # decision.
        reasons.extend(runnable.reasons)
        action = "extract_and_run" if runnable.status == "eligible" else "inspect"
        confidence = _plan_confidence(runnable, causal.confidence, failure_id is not None)
        candidate = PlanCandidate(
            boundary=boundary.id,
            name=boundary.name,
            priority_score=score,
            plan_confidence=confidence,
            runnability=runnable,
            fixture_kind=selected_fixture_kind,
            action=action,
            reasons=tuple(dict.fromkeys(reasons)),
            warnings=tuple(dict.fromkeys(warnings)),
            canonical_boundary=boundary.canonical_id,
            node_count=boundary.node_count,
            base_score=recommendation.score,
            details=details.to_dict(),
        )
        candidates.append(candidate)

    candidates.sort(key=lambda candidate: (-candidate.priority_score, candidate.node_count, candidate.boundary))
    if limit is not None:
        if limit <= 0:
            candidates = []
        else:
            candidates = candidates[:limit]
    return AlignmentPlan(
        candidates=tuple(candidates),
        inventory=inventory,
        current_failure=failure_id,
        causal_descendants=tuple(sorted(causal_ids)),
        causal_confidence=causal.confidence,
        causal_reason=causal.reason,
    )


def plan_boundaries(
    package: TracePackage,
    **kwargs: Any,
) -> list[PlanCandidate]:
    """Return ordered candidates, matching ``recommend_regions``' list API."""

    return list(plan_alignment(package, **kwargs).candidates)


def build_plan(package: TracePackage, **kwargs: Any) -> AlignmentPlan:
    """Compatibility spelling used by early alignment integrations."""

    return plan_alignment(package, **kwargs)


def recommend_boundaries(package: TracePackage, **kwargs: Any) -> list[PlanCandidate]:
    return plan_boundaries(package, **kwargs)


class AlignmentPlanner:
    """Reusable planner facade for callers that prefer an object API."""

    def __init__(self, package: TracePackage, **defaults: Any) -> None:
        self.package = package
        self.defaults = dict(defaults)

    def plan(self, **overrides: Any) -> AlignmentPlan:
        options = dict(self.defaults)
        options.update(overrides)
        return plan_alignment(self.package, **options)

    def candidates(self, **overrides: Any) -> list[PlanCandidate]:
        return list(self.plan(**overrides).candidates)

    def inventory(self) -> BoundaryInventory:
        return inventory_boundaries(self.package)


__all__ = [
    "BINARY_SEARCH_BONUS",
    "BINARY_SEARCH_REASON",
    "CAUSAL_CONE_BONUS",
    "DEFAULT_FIXTURE_KIND",
    "IMMEDIATE_CHILD_BONUS",
    "LARGE_BOUNDARY_PENALTY",
    "REUSABLE_TESTCASE_BONUS",
    "UNKNOWN_CAPABILITY_PENALTY",
    "AlignmentPlan",
    "AlignmentPlanner",
    "PlanCandidate",
    "Runnability",
    "boundary_requirements",
    "build_plan",
    "plan_alignment",
    "plan_boundaries",
    "recommend_boundaries",
]
