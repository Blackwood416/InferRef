"""Textual trace inspection (SPEC §34, Appendix A).

Renders the trace as an operator listing with module ownership, source
location, and producer/consumer links — the MVP viewer.
"""

from __future__ import annotations

from typing import Any

from inferref.ir.module import path_matches
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.tensor_value import TensorValueRecord
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


def format_tensor(value: TensorValueRecord, *, verbose: bool = False) -> str:
    """One-line tensor summary: ``t12 [1,128,3584] float16``."""
    shape = ",".join(str(d) for d in value.shape)
    text = f"t{value.id} [{shape}] {value.dtype}"
    if verbose:
        stride = ",".join(str(s) for s in value.stride)
        text += f" stride=[{stride}]"
        if value.storage_offset_elements:
            text += f" offset={value.storage_offset_elements}"
        if not value.contiguous:
            text += " non-contiguous"
        text += f" storage={value.storage_id}v{value.storage_version}"
        if value.qualified_name:
            text += f" {value.role}={value.qualified_name}"
        elif value.role != "activation":
            text += f" role={value.role}"
        if value.capture.mode != "metadata":
            text += f" capture={value.capture.mode}"
    return text


def format_value(package: TracePackage, value: Value, *, verbose: bool = False) -> str:
    """Render an operator argument."""
    if isinstance(value, TensorRef):
        graph = package.graph
        if graph.has_value(value.value_id):
            return format_tensor(graph.value(value.value_id), verbose=verbose)
        return f"t{value.value_id} <missing>"
    if isinstance(value, ScalarValue):
        return str(value.value)
    if isinstance(value, StringValue):
        return repr(value.value)
    if isinstance(value, NoneValue):
        return "None"
    if isinstance(value, (ListValue, TupleValue)):
        inner = ", ".join(format_value(package, i, verbose=verbose) for i in value.items)
        return f"[{inner}]" if isinstance(value, ListValue) else f"({inner})"
    if isinstance(value, DictValue):
        inner = ", ".join(
            f"{format_value(package, k)}: {format_value(package, v, verbose=verbose)}"
            for k, v in value.items
        )
        return "{" + inner + "}"
    if isinstance(value, OpaqueValue):
        return f"<{value.type}: {value.repr}>"
    return "?"


def _matches_filters(
    package: TracePackage,
    op: OperatorRecord,
    *,
    module: str | None,
    operator: str | None,
) -> bool:
    if operator and op.canonical_name != operator:
        return False
    if module:
        path = package.module_path(op.module_stack)
        if not path_matches(path, module):
            return False
    return True


def render_trace(
    package: TracePackage,
    *,
    verbose: bool = False,
    limit: int | None = None,
    module: str | None = None,
    operator: str | None = None,
    show_sources: bool = True,
) -> str:
    """Render the operator listing (SPEC §34, Appendix A)."""
    lines: list[str] = []
    manifest = package.manifest
    lines.append(f"Model:  {manifest.model.name}")
    lines.append(
        f"Trace:  {manifest.format} {manifest.format_version} "
        f"(frontend {manifest.frontend.name} {manifest.frontend.version}, "
        f"{manifest.reference_framework.name} {manifest.reference_framework.version})"
    )
    lines.append(
        f"Capture: tensors={manifest.capture.tensor_policy} "
        f"device={manifest.execution.device}"
    )
    if manifest.capture.scope:
        lines.append(f"Scope:  {manifest.capture.scope}")
    for warning in manifest.determinism.warnings:
        lines.append(f"WARNING: {warning}")
    lines.append("")

    graph = package.graph
    if graph.inputs:
        lines.append("Graph inputs:")
        for io in graph.inputs:
            lines.append(f"  {io.name}: {format_value(package, io.value, verbose=verbose)}")
        lines.append("")

    ops = [
        op
        for op in graph.ops_in_execution_order()
        if _matches_filters(package, op, module=module, operator=operator)
    ]
    total = len(ops)
    truncated = limit is not None and total > limit
    if truncated:
        ops = ops[:limit]

    lines.append(f"Operators ({total}):")
    lines.append("")
    previous_module = object()
    for op in ops:
        module_path = package.module_path(op.module_stack)
        if module_path != previous_module:
            lines.append(f"  [{module_path or '<root>'}]")
            previous_module = module_path

        args = ", ".join(format_value(package, a, verbose=verbose) for a in op.positional_args)
        if op.keyword_args:
            kwargs = ", ".join(
                f"{k}={format_value(package, v, verbose=verbose)}"
                for k, v in op.keyword_args.items()
            )
            args = f"{args}, {kwargs}" if args else kwargs
        lines.append(f"    #{op.execution_index} {op.canonical_name}({args})")

        if op.result is not None:
            lines.append(f"        -> {format_value(package, op.result, verbose=verbose)}")

        for mutation in op.effects.mutated_storages:
            lines.append(
                f"        mutates storage:{mutation.storage_id} "
                f"v{mutation.version_before} -> v{mutation.version_after}"
            )
        if verbose:
            for alias in op.effects.aliases:
                lines.append(
                    f"        alias t{alias.output_value_id} -> "
                    f"t{alias.input_value_id} ({alias.relationship})"
                )

        if show_sources:
            source = package.source(op.source_id)
            if source is not None and source.primary is not None:
                lines.append(f"        at {source.primary}")

        semantic = [a for a in op.annotations if a.type == "semantic"]
        if semantic:
            # Innermost label last (see semantic.run), so show it first.
            labels = " < ".join(
                f"{a.name}" + ("" if a.confidence >= 1.0 else f"({a.confidence:.2f})")
                for a in reversed(semantic)
            )
            lines.append(f"        semantic: {labels}")

        for region in package.regions:
            if op.id in region.node_ids:
                lines.append(f"        region: {region.name}")

    if truncated:
        lines.append("")
        lines.append(f"  ... {total - len(ops)} more operators (raise --limit to see them)")

    if graph.outputs:
        lines.append("")
        lines.append("Graph outputs:")
        for io in graph.outputs:
            lines.append(f"  {io.name}: {format_value(package, io.value, verbose=verbose)}")

    if package.regions:
        lines.append("")
        lines.append("Regions:")
        for region in package.regions:
            lines.append(
                f"  {region.id}: {region.name} "
                f"({len(region.node_ids)} ops, {len(region.inputs)} in, "
                f"{len(region.outputs)} out)"
            )
    return "\n".join(lines)


def trace_to_dict(package: TracePackage, *, limit: int | None = None) -> dict[str, Any]:
    """Machine-readable form of :func:`render_trace` (SPEC §42)."""
    graph = package.graph
    ops = graph.ops_in_execution_order()
    if limit is not None:
        ops = ops[:limit]
    return {
        "model": package.manifest.model.name,
        "format_version": package.manifest.format_version,
        "counts": {
            "operators": len(graph.operators),
            "values": len(graph.values),
            "modules": len(package.modules),
            "sources": len(package.sources),
            "regions": len(package.regions),
            "storages": len(package.storages),
        },
        "inputs": [
            {"name": io.name, "value": io.value.to_dict()} for io in graph.inputs
        ],
        "outputs": [
            {"name": io.name, "value": io.value.to_dict()} for io in graph.outputs
        ],
        "operators": [
            {
                "id": op.id,
                "execution_index": op.execution_index,
                "canonical_name": op.canonical_name,
                "module_path": package.module_path(op.module_stack) or None,
                "source": str(package.source(op.source_id) or "") or None,
                "inputs": graph.op_input_value_ids(op),
                "outputs": graph.op_output_value_ids(op),
            }
            for op in ops
        ],
    }


def render_tensor_detail(package: TracePackage, value_id: int) -> str:
    """Detailed view of one tensor (SPEC §39)."""
    graph = package.graph
    if not graph.has_value(value_id):
        return f"no such value: {value_id}"
    value = graph.value(value_id)
    lines = [f"Tensor t{value.id}", ""]
    lines.append(f"  dtype:           {value.dtype}")
    lines.append(f"  shape:           {list(value.shape)}")
    lines.append(f"  stride:          {list(value.stride)}")
    lines.append(f"  storage_offset:  {value.storage_offset_elements}")
    lines.append(f"  contiguous:      {value.contiguous}")
    lines.append(f"  device:          {value.device}")
    lines.append(f"  storage:         {value.storage_id} (version {value.storage_version})")
    lines.append(f"  runtime object:  {value.runtime_object_id}")
    lines.append(f"  role:            {value.role}")
    if value.qualified_name:
        lines.append(f"  qualified name:  {value.qualified_name}")
    lines.append(f"  logical numel:   {value.logical_numel}")
    lines.append(f"  logical nbytes:  {value.logical_nbytes}")
    lines.append(f"  capture:         {value.capture.mode}")
    if value.capture.payload:
        lines.append(f"  payload:         {value.capture.payload}")
    for tensor_hash in value.capture.hashes:
        lines.append(
            f"  hash:            {tensor_hash.algorithm}[{tensor_hash.domain}] "
            f"{tensor_hash.value[:16]}..."
        )
    lines.append("")
    if value.producer is not None and graph.has_op(value.producer):
        op = graph.op(value.producer)
        lines.append(f"  producer:        #{op.execution_index} {op.canonical_name}")
    else:
        lines.append("  producer:        <graph input>")
    if value.consumers:
        lines.append("  consumers:")
        for consumer in value.consumers:
            if graph.has_op(consumer):
                op = graph.op(consumer)
                lines.append(f"    #{op.execution_index} {op.canonical_name}")
    else:
        lines.append("  consumers:       <none>")
    return "\n".join(lines)
