"""Trace IR validation (IR §48).

Implements all ten recommended validation invariants:

1.  every referenced operator exists;
2.  every referenced tensor value exists;
3.  execution indices are unique and monotonic;
4.  producer references are consistent;
5.  consumer references are consistent;
6.  tensor rank matches ``shape`` and ``stride`` lengths;
7.  storage versions never decrease in execution order;
8.  region node references exist;
9.  region boundary inputs/outputs are graph-consistent;
10. full-capture payload byte counts match dtype and logical element count.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inferref.ir.dtypes import dtype_itemsize, is_known_dtype
from inferref.ir.package import TracePackage
from inferref.ir.values import walk_tensor_refs


@dataclass(frozen=True)
class ValidationIssue:
    """One validation failure."""

    #: ``1``-``10``, matching the IR §48 invariant list.
    invariant: int
    severity: str  # "error" | "warning"
    message: str
    where: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant": self.invariant,
            "severity": self.severity,
            "message": self.message,
            "where": self.where,
        }

    def __str__(self) -> str:
        loc = f" [{self.where}]" if self.where else ""
        return f"#{self.invariant} {self.severity}: {self.message}{loc}"


def validate_package(
    package: TracePackage, *, check_payloads: bool = True
) -> list[ValidationIssue]:
    """Validate ``package`` and return every issue found.

    ``check_payloads`` controls invariant 10, which needs the trace root on disk.
    """
    issues: list[ValidationIssue] = []
    graph = package.graph
    graph.reindex()

    _check_reference_integrity(package, issues)
    _check_execution_order(package, issues)
    _check_producer_consumer(package, issues)
    _check_tensor_shapes(package, issues)
    _check_storage_versions(package, issues)
    _check_regions(package, issues)
    if check_payloads:
        _check_payloads(package, issues)

    return issues


# -- 1 & 2: reference integrity -------------------------------------------


def _check_reference_integrity(pkg: TracePackage, issues: list[ValidationIssue]) -> None:
    graph = pkg.graph
    module_ids = {m.id for m in pkg.modules}
    source_ids = {s.id for s in pkg.sources}

    for op in graph.operators:
        where = f"op:{op.id} {op.canonical_name}"
        for vid in graph.op_input_value_ids(op) + graph.op_output_value_ids(op):
            if not graph.has_value(vid):
                issues.append(
                    ValidationIssue(2, "error", f"references missing value:{vid}", where)
                )
        if op.source_id is not None and pkg.sources and op.source_id not in source_ids:
            issues.append(
                ValidationIssue(1, "error", f"references missing source:{op.source_id}", where)
            )
        for mid in op.module_stack:
            if pkg.modules and mid not in module_ids:
                issues.append(
                    ValidationIssue(1, "error", f"references missing module:{mid}", where)
                )
        for mutation in op.effects.mutated_storages:
            if mutation.version_after <= mutation.version_before:
                issues.append(
                    ValidationIssue(
                        7,
                        "error",
                        f"storage:{mutation.storage_id} mutation does not advance version "
                        f"({mutation.version_before} -> {mutation.version_after})",
                        where,
                    )
                )
        for alias in op.effects.aliases:
            for vid in (alias.output_value_id, alias.input_value_id):
                if not graph.has_value(vid):
                    issues.append(
                        ValidationIssue(2, "error", f"alias references missing value:{vid}", where)
                    )

    for io in list(graph.inputs) + list(graph.outputs):
        for ref in walk_tensor_refs(io.value):
            if not graph.has_value(ref.value_id):
                issues.append(
                    ValidationIssue(
                        2, "error", f"graph boundary references missing value:{ref.value_id}",
                        f"io:{io.name}",
                    )
                )

    for value in graph.values:
        where = f"value:{value.id}"
        if value.producer is not None and not graph.has_op(value.producer):
            issues.append(
                ValidationIssue(1, "error", f"producer op:{value.producer} does not exist", where)
            )
        for consumer in value.consumers:
            if not graph.has_op(consumer):
                issues.append(
                    ValidationIssue(1, "error", f"consumer op:{consumer} does not exist", where)
                )


# -- 3: execution order ----------------------------------------------------


def _check_execution_order(pkg: TracePackage, issues: list[ValidationIssue]) -> None:
    seen: dict[int, int] = {}
    for op in pkg.graph.operators:
        if op.execution_index in seen:
            issues.append(
                ValidationIssue(
                    3,
                    "error",
                    f"duplicate execution_index {op.execution_index} "
                    f"(also used by op:{seen[op.execution_index]})",
                    f"op:{op.id}",
                )
            )
        seen[op.execution_index] = op.id
        if op.execution_index < 0:
            issues.append(
                ValidationIssue(
                    3, "error", f"negative execution_index {op.execution_index}", f"op:{op.id}"
                )
            )

    indices = sorted(seen)
    if indices and indices[0] != 0:
        issues.append(
            ValidationIssue(
                3, "warning", f"execution indices start at {indices[0]}, not 0", "graph"
            )
        )
    for prev, curr in zip(indices, indices[1:]):
        if curr != prev + 1:
            issues.append(
                ValidationIssue(
                    3, "warning", f"gap in execution indices: {prev} -> {curr}", "graph"
                )
            )


# -- 4 & 5: producer / consumer consistency --------------------------------


def _check_producer_consumer(pkg: TracePackage, issues: list[ValidationIssue]) -> None:
    graph = pkg.graph
    actual_producers: dict[int, int] = {}
    actual_consumers: dict[int, set[int]] = {}

    for op in graph.operators:
        for vid in graph.op_output_value_ids(op):
            if vid in actual_producers and actual_producers[vid] != op.id:
                issues.append(
                    ValidationIssue(
                        4,
                        "error",
                        f"value:{vid} produced by both op:{actual_producers[vid]} and op:{op.id}",
                        f"value:{vid}",
                    )
                )
            actual_producers.setdefault(vid, op.id)
        for vid in graph.op_input_value_ids(op):
            actual_consumers.setdefault(vid, set()).add(op.id)

    # A mutation produces the next storage generation even when the base
    # tensor carrying that generation is not the Python return object.
    effect_producers, _ = graph.derived_links()
    for vid, op_id in effect_producers.items():
        actual_producers.setdefault(vid, op_id)

    for value in graph.values:
        where = f"value:{value.id}"
        expected = actual_producers.get(value.id)
        if value.producer != expected:
            issues.append(
                ValidationIssue(
                    4,
                    "error",
                    f"producer recorded as {value.producer} but graph says {expected}",
                    where,
                )
            )
        expected_consumers = actual_consumers.get(value.id, set())
        if set(value.consumers) != expected_consumers:
            issues.append(
                ValidationIssue(
                    5,
                    "error",
                    f"consumers recorded as {sorted(value.consumers)} "
                    f"but graph says {sorted(expected_consumers)}",
                    where,
                )
            )


# -- 6: tensor shape/stride ------------------------------------------------


def _check_tensor_shapes(pkg: TracePackage, issues: list[ValidationIssue]) -> None:
    for value in pkg.graph.values:
        where = f"value:{value.id}"
        if not is_known_dtype(value.dtype):
            issues.append(ValidationIssue(6, "error", f"unknown dtype {value.dtype!r}", where))
        if len(value.shape) != len(value.stride):
            issues.append(
                ValidationIssue(
                    6,
                    "error",
                    f"rank mismatch: shape has {len(value.shape)} dims, "
                    f"stride has {len(value.stride)}",
                    where,
                )
            )
        if any(d < 0 for d in value.shape):
            issues.append(ValidationIssue(6, "error", f"negative dim in {value.shape}", where))


# -- 7: storage versions never decrease ------------------------------------


def _check_storage_versions(pkg: TracePackage, issues: list[ValidationIssue]) -> None:
    graph = pkg.graph
    high_water: dict[int, int] = {}

    def observe_value(vid: int, op) -> None:
        if not graph.has_value(vid):
            return
        value = graph.value(vid)
        if value.storage_id is None:
            return
        seen = high_water.get(value.storage_id)
        if seen is not None and value.storage_version < seen:
            issues.append(
                ValidationIssue(
                    7,
                    "error",
                    f"storage:{value.storage_id} version went backwards "
                    f"({seen} -> {value.storage_version}) at execution_index "
                    f"{op.execution_index}",
                    f"value:{vid}",
                )
            )
        high_water[value.storage_id] = max(seen or 0, value.storage_version)

    for op in graph.ops_in_execution_order():
        # Inputs are observed before the operator's effects; outputs are
        # observed afterwards. Mutation effects must participate in the high
        # water mark even when an operator returns no tensor for the written
        # storage (a common custom-cache API shape).
        for vid in graph.op_input_value_ids(op):
            observe_value(vid, op)

        for mutation in op.effects.mutated_storages:
            seen = high_water.get(mutation.storage_id)
            if seen is not None and mutation.version_before < seen:
                issues.append(
                    ValidationIssue(
                        7,
                        "error",
                        f"storage:{mutation.storage_id} mutation version_before "
                        f"went backwards ({seen} -> {mutation.version_before}) "
                        f"at execution_index "
                        f"{op.execution_index}",
                        f"op:{op.id}",
                    )
                )
            high_water[mutation.storage_id] = max(
                seen or 0,
                mutation.version_before,
                mutation.version_after,
            )

        for vid in graph.op_output_value_ids(op):
            observe_value(vid, op)


# -- 8 & 9: regions --------------------------------------------------------


def _check_regions(pkg: TracePackage, issues: list[ValidationIssue]) -> None:
    from inferref.region.boundary import derive_boundary

    graph = pkg.graph
    for region in pkg.regions:
        where = f"region:{region.id} {region.name}"
        for node_id in region.node_ids:
            if not graph.has_op(node_id):
                issues.append(
                    ValidationIssue(8, "error", f"references missing op:{node_id}", where)
                )
        if not region.node_ids:
            issues.append(ValidationIssue(8, "warning", "region has no nodes", where))
            continue

        known = [n for n in region.node_ids if graph.has_op(n)]
        expected_inputs, expected_outputs = derive_boundary(graph, known)
        if set(region.inputs) != set(expected_inputs):
            issues.append(
                ValidationIssue(
                    9,
                    "error",
                    f"recorded inputs {sorted(region.inputs)} != derived "
                    f"{sorted(expected_inputs)}",
                    where,
                )
            )
        if set(region.outputs) != set(expected_outputs):
            issues.append(
                ValidationIssue(
                    9,
                    "error",
                    f"recorded outputs {sorted(region.outputs)} != derived "
                    f"{sorted(expected_outputs)}",
                    where,
                )
            )


# -- 10: payload byte counts -----------------------------------------------


def _check_payloads(pkg: TracePackage, issues: list[ValidationIssue]) -> None:
    if pkg.root is None:
        return
    for value in pkg.graph.values:
        capture = value.capture
        if capture.mode != "full" or not capture.payload:
            continue
        where = f"value:{value.id}"
        path = Path(pkg.root) / capture.payload
        if not path.is_file():
            issues.append(
                ValidationIssue(10, "error", f"missing payload {capture.payload}", where)
            )
            continue
        try:
            expected = value.logical_numel * dtype_itemsize(value.dtype)
        except ValueError:
            continue  # dtype already reported by invariant 6

        header = _read_irtensor_header(path)
        if header is None:
            issues.append(
                ValidationIssue(
                    10, "error", f"payload {capture.payload} is not a valid .irtensor", where
                )
            )
            continue
        hsize, payload_nbytes, shape = header

        if payload_nbytes != expected:
            issues.append(
                ValidationIssue(
                    10,
                    "error",
                    f"payload declares {payload_nbytes} bytes, expected {expected} "
                    f"({value.logical_numel} x {value.dtype})",
                    where,
                )
            )
        if shape != tuple(value.shape):
            issues.append(
                ValidationIssue(
                    10,
                    "error",
                    f"payload {capture.payload} has shape {list(shape)} but value "
                    f"declares {list(value.shape)}",
                    where,
                )
            )
        actual_size = path.stat().st_size
        if actual_size < hsize + payload_nbytes:
            issues.append(
                ValidationIssue(
                    10,
                    "error",
                    f"payload truncated: file is {actual_size} bytes, header needs "
                    f"{hsize + payload_nbytes}",
                    where,
                )
            )


#: Fixed part of the .irtensor header; see :mod:`inferref.tensor.codec`.
_IRTENSOR_FIXED = struct.Struct("<4sHHHIHHQQ")


def _read_irtensor_header(path: Path) -> tuple[int, int, tuple[int, ...]] | None:
    """Read ``(header_size, payload_nbytes, shape)`` from an ``.irtensor``.

    Implemented with :mod:`struct` alone so that validation stays usable in a
    stdlib-only environment.
    """
    try:
        with path.open("rb") as handle:
            fixed = handle.read(_IRTENSOR_FIXED.size)
            if len(fixed) < _IRTENSOR_FIXED.size:
                return None
            magic, _version, hsize, _dtype, _flags, rank, _res, _numel, nbytes = (
                _IRTENSOR_FIXED.unpack(fixed)
            )
            if magic != b"IRTN":
                return None
            shape_raw = handle.read(8 * rank)
            if len(shape_raw) < 8 * rank:
                return None
            shape = struct.unpack(f"<{rank}q", shape_raw) if rank else ()
    except OSError:
        return None
    return hsize, nbytes, tuple(shape)
