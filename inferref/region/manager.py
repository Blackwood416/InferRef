"""Region creation and management (SPEC §19, §37; IR §33-§35)."""

from __future__ import annotations

from typing import Iterable, Sequence

from inferref.ir.module import path_matches
from inferref.ir.operator import Annotation
from inferref.ir.package import TracePackage
from inferref.ir.region import CREATION_METHODS, RegionRecord
from inferref.region.boundary import derive_boundary, nodes_between, nodes_for_module


class RegionError(ValueError):
    """Raised when a region cannot be created as requested."""


def _next_region_id(package: TracePackage) -> int:
    return max((r.id for r in package.regions), default=0) + 1


def _source_ids(package: TracePackage, node_ids: Sequence[int]) -> tuple[int, ...]:
    seen: dict[int, None] = {}
    for node_id in node_ids:
        if package.graph.has_op(node_id):
            source_id = package.graph.op(node_id).source_id
            if source_id is not None:
                seen.setdefault(source_id, None)
    return tuple(seen)


def create_region(
    package: TracePackage,
    name: str,
    node_ids: Iterable[int],
    *,
    method: str = "manual",
    semantic: str | None = None,
    confidence: float = 1.0,
    engine_op: str | None = None,
) -> RegionRecord:
    """Create a region from an explicit node set, deriving its boundary (IR §34)."""
    nodes = sorted(
        {n for n in node_ids if package.graph.has_op(n)},
        key=lambda n: package.graph.op(n).execution_index,
    )
    if not nodes:
        raise RegionError("region would contain no known operators")
    if method not in CREATION_METHODS:
        raise RegionError(
            f"unknown creation method {method!r}; expected one of {list(CREATION_METHODS)}"
        )
    if any(r.name == name for r in package.regions):
        raise RegionError(f"a region named {name!r} already exists")

    inputs, outputs = derive_boundary(package.graph, nodes)
    region = RegionRecord(
        id=_next_region_id(package),
        name=name,
        node_ids=tuple(nodes),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        semantic=(
            Annotation(type="semantic", name=semantic, confidence=confidence)
            if semantic
            else None
        ),
        source_ids=_source_ids(package, nodes),
        creation_method=method,
        engine_op=engine_op,
    )
    package.regions.append(region)
    return region


def create_region_from_ops(
    package: TracePackage, name: str, from_op: int, to_op: int, **kwargs: object
) -> RegionRecord:
    """``inferref region create --from-op N --to-op M`` (SPEC §19)."""
    nodes = nodes_between(package.graph, from_op, to_op)
    kwargs.setdefault("method", "graph_selection")
    return create_region(package, name, nodes, **kwargs)  # type: ignore[arg-type]


def create_region_from_module(
    package: TracePackage, name: str, module_pattern: str, **kwargs: object
) -> RegionRecord:
    """Create a region from a module boundary (IR §35 ``module``)."""
    module_ids = [m.id for m in package.modules if path_matches(m.path, module_pattern)]
    if not module_ids:
        raise RegionError(f"no module matches {module_pattern!r}")
    nodes = nodes_for_module(package.graph, module_ids)
    if not nodes:
        raise RegionError(f"module {module_pattern!r} executed no operators")
    kwargs.setdefault("method", "module")
    return create_region(package, name, nodes, **kwargs)  # type: ignore[arg-type]


def create_region_from_source_function(
    package: TracePackage, name: str, function: str, **kwargs: object
) -> RegionRecord:
    """Create a region from a source function boundary (IR §35 ``source_function``)."""
    nodes: list[int] = []
    for op in package.graph.ops_in_execution_order():
        source = package.source(op.source_id)
        if source is None:
            continue
        if any(frame.function == function for frame in source.stack):
            nodes.append(op.id)
    if not nodes:
        raise RegionError(f"no operators were issued from a function named {function!r}")
    kwargs.setdefault("method", "source_function")
    return create_region(package, name, nodes, **kwargs)  # type: ignore[arg-type]


def delete_region(package: TracePackage, name_or_id: str | int) -> RegionRecord:
    """Remove a region from the package."""
    region = package.region(name_or_id)
    if region is None:
        raise RegionError(f"no region named {name_or_id!r}")
    package.regions.remove(region)
    return region


def refresh_boundaries(package: TracePackage) -> None:
    """Recompute every region's inputs/outputs from its node set (IR §34)."""
    for region in package.regions:
        inputs, outputs = derive_boundary(package.graph, region.node_ids)
        region.inputs = tuple(inputs)
        region.outputs = tuple(outputs)
