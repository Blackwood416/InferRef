"""Testcase extraction (SPEC §23; IR §53, §54).

A testcase is a *projection* of the Trace IR::

    operator/region external inputs -> subgraph -> external outputs

The result is a self-contained directory a coding agent (or a C++ test harness)
can consume without reopening the model or rerunning PyTorch::

    repro/419/
    ├── testcase.json
    ├── inputs/<name>.irtensor
    ├── reference/<name>.irtensor
    └── README.md
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import TensorValueRecord
from inferref.ir.values import (
    DictValue,
    ListValue,
    OpaqueValue,
    ScalarValue,
    StringValue,
    TensorRef,
    TupleValue,
    Value,
)
from inferref.ir.version import TESTCASE_FORMAT, TESTCASE_FORMAT_VERSION


class ExtractionError(RuntimeError):
    """Raised when a testcase cannot be produced from the trace."""


@dataclass
class ExtractedTestcase:
    """Summary of what was written."""

    path: Path
    name: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    #: Values lacking a payload; the testcase cannot run without them.
    missing_payloads: list[int] = field(default_factory=list)
    #: Non-portable opaque arguments (IR §41).
    non_portable: list[str] = field(default_factory=list)

    @property
    def reproducible(self) -> bool:
        return not self.missing_payloads and not self.non_portable

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "reproducible": self.reproducible,
            "missing_payloads": self.missing_payloads,
            "non_portable": self.non_portable,
        }


def _value_label(package: TracePackage, value_id: int, fallback: str) -> str:
    """A stable, filesystem-safe name for a value."""
    graph = package.graph
    if graph.has_value(value_id):
        value = graph.value(value_id)
        if value.qualified_name:
            return value.qualified_name.replace(".", "_")
    return fallback


def _resolve_names(
    package: TracePackage,
    value_ids: Sequence[int],
    explicit: Sequence[str] | None,
    prefix: str,
) -> list[str]:
    """Pick a name per boundary value.

    Explicit names win, then the value's qualified name (parameters and
    buffers), then a positional fallback. Explicit naming exists because region
    boundary values are usually anonymous activations, and ``query``/``cos`` is
    far more useful to a kernel author than ``input_2``.
    """
    if explicit:
        if len(explicit) != len(value_ids):
            raise ExtractionError(
                f"got {len(explicit)} {prefix} name(s) but the boundary has "
                f"{len(value_ids)} value(s)"
            )
        return [name.strip() for name in explicit]
    return [
        _value_label(package, value_id, f"{prefix}_{position}")
        for position, value_id in enumerate(value_ids)
    ]


def _copy_payload(
    package: TracePackage,
    value: TensorValueRecord,
    destination: Path,
) -> bool:
    """Copy a value's ``.irtensor`` payload into the testcase. Returns success."""
    if value.capture.mode != "full" or not value.capture.payload:
        return False
    if package.root is None:
        return False
    source = Path(package.root) / value.capture.payload
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return True


def _describe_argument(value: Value) -> Any:
    """Render a non-tensor argument for ``testcase.json``."""
    if isinstance(value, ScalarValue):
        return value.to_dict()
    if isinstance(value, StringValue):
        return {"kind": "string", "value": value.value}
    if isinstance(value, TensorRef):
        return {"kind": "tensor", "value_id": value.value_id}
    if isinstance(value, (ListValue, TupleValue)):
        return {
            "kind": value.kind,
            "items": [_describe_argument(i) for i in value.items],
        }
    if isinstance(value, DictValue):
        return {
            "kind": "dict",
            "items": [
                {"key": _describe_argument(k), "value": _describe_argument(v)}
                for k, v in value.items
            ],
        }
    return value.to_dict()


def _collect_non_portable(value: Value, into: list[str]) -> None:
    if isinstance(value, OpaqueValue) and not value.portable:
        into.append(value.type)
    elif isinstance(value, (ListValue, TupleValue)):
        for item in value.items:
            _collect_non_portable(item, into)
    elif isinstance(value, DictValue):
        for key, val in value.items:
            _collect_non_portable(key, into)
            _collect_non_portable(val, into)


def extract_operator(
    package: TracePackage,
    op_id: int,
    output: str | Path,
    *,
    name: str | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
) -> ExtractedTestcase:
    """Extract a single-operator testcase (SPEC §23; IR §53)."""
    graph = package.graph
    if not graph.has_op(op_id):
        raise ExtractionError(f"no operator with id {op_id} in this trace")
    op = graph.op(op_id)
    return _write_testcase(
        package,
        output=Path(output),
        name=name or op.canonical_name,
        node_ids=[op_id],
        input_ids=graph.op_input_value_ids(op),
        output_ids=graph.op_output_value_ids(op),
        origin={"trace": str(package.root or ""), "operator_id": op_id},
        operators=[op],
        input_names=input_names,
        output_names=output_names,
    )


def extract_region(
    package: TracePackage,
    region: RegionRecord,
    output: str | Path,
    *,
    name: str | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
) -> ExtractedTestcase:
    """Extract a region testcase (SPEC §37; IR §53)."""
    graph = package.graph
    operators = [graph.op(n) for n in region.node_ids if graph.has_op(n)]
    operators.sort(key=lambda o: o.execution_index)
    return _write_testcase(
        package,
        output=Path(output),
        name=name or region.name,
        node_ids=list(region.node_ids),
        input_ids=list(region.inputs),
        output_ids=list(region.outputs),
        origin={"trace": str(package.root or ""), "region_id": region.id},
        operators=operators,
        input_names=input_names,
        output_names=output_names,
    )


def _write_testcase(
    package: TracePackage,
    *,
    output: Path,
    name: str,
    node_ids: Sequence[int],
    input_ids: Sequence[int],
    output_ids: Sequence[int],
    origin: dict[str, Any],
    operators: Sequence[OperatorRecord],
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
) -> ExtractedTestcase:
    graph = package.graph
    output.mkdir(parents=True, exist_ok=True)

    result = ExtractedTestcase(path=output, name=name)
    manifest_inputs: list[dict[str, Any]] = []
    manifest_outputs: list[dict[str, Any]] = []

    input_names = _resolve_names(package, input_ids, input_names, "input")
    output_names = _resolve_names(package, output_ids, output_names, "output")

    for position, value_id in enumerate(input_ids):
        if not graph.has_value(value_id):
            result.missing_payloads.append(value_id)
            continue
        value = graph.value(value_id)
        label = input_names[position]
        relative = f"inputs/{label}.irtensor"
        ok = _copy_payload(package, value, output / relative)
        if not ok:
            result.missing_payloads.append(value_id)
        manifest_inputs.append(
            {
                "name": label,
                "value_id": value_id,
                "payload": relative if ok else None,
                "role": value.role,
                "qualified_name": value.qualified_name,
                **_tensor_metadata(value),
            }
        )
        result.inputs.append(label)

    for position, value_id in enumerate(output_ids):
        if not graph.has_value(value_id):
            result.missing_payloads.append(value_id)
            continue
        value = graph.value(value_id)
        label = output_names[position]
        relative = f"reference/{label}.irtensor"
        ok = _copy_payload(package, value, output / relative)
        if not ok:
            result.missing_payloads.append(value_id)
        manifest_outputs.append(
            {
                "name": label,
                "value_id": value_id,
                "payload": relative if ok else None,
                **_tensor_metadata(value),
                **_provenance(package, value),
            }
        )
        result.outputs.append(label)

    node_records: list[dict[str, Any]] = []
    for op in operators:
        for arg in list(op.positional_args) + list(op.keyword_args.values()):
            _collect_non_portable(arg, result.non_portable)
        source = package.source(op.source_id)
        node_records.append(
            {
                "id": op.id,
                "execution_index": op.execution_index,
                "canonical_name": op.canonical_name,
                "positional_args": [_describe_argument(a) for a in op.positional_args],
                "keyword_args": {
                    k: _describe_argument(v) for k, v in op.keyword_args.items()
                },
                "result": _describe_argument(op.result) if op.result else None,
                "module_path": package.module_path(op.module_stack) or None,
                "source": (
                    {
                        "file": source.primary.file,
                        "line": source.primary.line,
                        "function": source.primary.function,
                    }
                    if source and source.primary
                    else None
                ),
            }
        )

    manifest = {
        "format": TESTCASE_FORMAT,
        "format_version": TESTCASE_FORMAT_VERSION,
        "name": name,
        "origin": origin,
        "reproducible": result.reproducible,
        "inputs": manifest_inputs,
        "outputs": manifest_outputs,
        "nodes": node_records,
    }
    if result.non_portable:
        # IR §41: say so explicitly rather than shipping a testcase that cannot run.
        manifest["non_portable_values"] = sorted(set(result.non_portable))
    if result.missing_payloads:
        manifest["missing_payloads"] = sorted(set(result.missing_payloads))

    (output / "testcase.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(_render_readme(manifest), encoding="utf-8")
    return result


def _tensor_metadata(value: TensorValueRecord) -> dict[str, Any]:
    return {
        "dtype": value.dtype,
        "shape": list(value.shape),
        "stride": list(value.stride),
        "storage_offset": value.storage_offset_elements,
        "contiguous": value.contiguous,
    }


def _provenance(package: TracePackage, value: TensorValueRecord) -> dict[str, Any]:
    """Where a reference output came from, carried into the testcase.

    Without this, comparing a testcase can only say *which tensor* diverged.
    With it, the report names the producing operator, its module and its source
    line — the context an engineer or agent needs in order to act (SPEC §2.6).
    """
    graph = package.graph
    if value.producer is None or not graph.has_op(value.producer):
        return {}
    op = graph.op(value.producer)
    source = package.source(op.source_id)
    # Regions nest (IR §36), so report the most specific one: "RoPE" tells a
    # kernel author what this tensor is, "TransformerBlock" does not.
    containing = [r for r in package.regions if op.id in r.node_ids]
    region = min(containing, key=lambda r: len(r.node_ids)).name if containing else None
    return {
        "producer": {
            "op_id": op.id,
            "execution_index": op.execution_index,
            "canonical_name": op.canonical_name,
            "module_path": package.module_path(op.module_stack) or None,
            "source": str(source) if source else None,
            "region": region,
        }
    }


def _render_readme(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# InferRef testcase: {manifest['name']}")
    lines.append("")
    if not manifest.get("reproducible", True):
        lines.append("> **Warning:** this testcase is not independently reproducible.")
        if manifest.get("non_portable_values"):
            lines.append(
                "> It depends on non-portable values: "
                + ", ".join(manifest["non_portable_values"])
                + "."
            )
        if manifest.get("missing_payloads"):
            lines.append(
                "> Some tensor payloads were not captured. Re-trace with "
                "`--capture-tensors all`."
            )
        lines.append("")

    lines.append("## Reference operators")
    lines.append("")
    lines.append("```text")
    for node in manifest.get("nodes", ()):
        lines.append(f"#{node['execution_index']} {node['canonical_name']}")
        if node.get("module_path"):
            lines.append(f"    module: {node['module_path']}")
        if node.get("source"):
            src = node["source"]
            lines.append(f"    source: {src['file']}:{src['line']} in {src['function']}")
    lines.append("```")
    lines.append("")

    lines.append("## Inputs")
    lines.append("")
    lines.append("| name | dtype | shape | stride | file |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in manifest.get("inputs", ()):
        lines.append(
            f"| `{entry['name']}` | {entry['dtype']} | {entry['shape']} | "
            f"{entry['stride']} | `{entry.get('payload') or '(not captured)'}` |"
        )
    lines.append("")

    lines.append("## Reference outputs")
    lines.append("")
    lines.append("| name | dtype | shape | stride | file |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in manifest.get("outputs", ()):
        lines.append(
            f"| `{entry['name']}` | {entry['dtype']} | {entry['shape']} | "
            f"{entry['stride']} | `{entry.get('payload') or '(not captured)'}` |"
        )
    lines.append("")

    lines.append("## Validating an engine against this testcase")
    lines.append("")
    lines.append("1. Read each input from `inputs/*.irtensor`.")
    lines.append("2. Run your kernel.")
    lines.append("3. Write each output to `<engine-out>/<name>.irtensor`.")
    lines.append("4. Compare:")
    lines.append("")
    lines.append("```bash")
    lines.append("inferref compare . <engine-out>/ --first-failure")
    lines.append("```")
    lines.append("")
    lines.append(
        "Payloads hold the tensor's logical values in canonical contiguous order; "
        "the `stride` column describes the reference tensor's original layout."
    )
    lines.append("")
    return "\n".join(lines)
