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

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

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
from inferref.testcase.requirements import derive_requirements, is_contract_id


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
    #: Structured reasons for every missing payload, suitable for agents and
    #: user-facing diagnostics.
    missing_payload_details: list[dict[str, Any]] = field(default_factory=list)
    #: Non-portable opaque arguments (IR §41).
    non_portable: list[str] = field(default_factory=list)
    #: Externally visible storage writes for which the trace has no post-state
    #: value. Such a testcase cannot validate the side effect.
    unobservable_mutations: list[dict[str, int]] = field(default_factory=list)

    @property
    def reproducible(self) -> bool:
        return (
            not self.missing_payloads
            and not self.non_portable
            and not self.unobservable_mutations
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "name": self.name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "reproducible": self.reproducible,
            "missing_payloads": self.missing_payloads,
            "missing_payload_details": self.missing_payload_details,
            "non_portable": self.non_portable,
            "unobservable_mutations": self.unobservable_mutations,
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
        names = [name.strip() for name in explicit]
        if len(set(names)) != len(names):
            raise ExtractionError(f"{prefix} names must be unique: {names}")
        return names
    names = [
        _value_label(package, value_id, f"{prefix}_{position}")
        for position, value_id in enumerate(value_ids)
    ]
    # Mutating regions can expose several immutable values for one qualified
    # buffer name (pre-state, target view, post-state).  Keep readable names
    # without allowing later payload copies to overwrite earlier ones.
    used: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        ordinal = used.get(name, 0) + 1
        used[name] = ordinal
        unique.append(name if ordinal == 1 else f"{name}_{ordinal}")
    return unique


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
    try:
        source = package.tensor_payload_path(value.capture.payload)
    except ValueError:
        return False
    if not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return True


def _payload_stems(labels: Sequence[str]) -> list[str]:
    """Return unique path-safe stems while preserving labels in the manifest."""

    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    used: set[str] = set()
    stems: list[str] = []
    for position, label in enumerate(labels):
        head = label.split(".", 1)[0].upper()
        safe = (
            bool(label)
            and label not in (".", "..")
            and not label.endswith((".", " "))
            and all(ch.isalnum() or ch in "._-" for ch in label)
            and head not in reserved
        )
        if safe and label not in used:
            stem = label
        else:
            digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:10]
            stem = f"value_{position}_{digest}"
        while stem in used:
            stem = f"{stem}_{position}"
        used.add(stem)
        stems.append(stem)
    return stems


def _missing_payload_detail(
    package: TracePackage,
    value_id: int,
    name: str,
    boundary: str,
    value: TensorValueRecord | None,
) -> dict[str, Any]:
    """Explain why one testcase boundary has no runnable payload."""
    detail: dict[str, Any] = {
        "value_id": value_id,
        "name": name,
        "boundary": boundary,
    }
    if value is None:
        detail["reason"] = "value_not_found"
        return detail

    capture = value.capture
    detail["capture"] = capture.to_dict()
    if capture.degraded_reason:
        detail["reason"] = capture.degraded_reason
    elif capture.mode != "full":
        detail["reason"] = "capture_mode"
    elif not capture.payload:
        detail["reason"] = "payload_not_referenced"
    elif package.root is None:
        detail["reason"] = "trace_root_unavailable"
    else:
        try:
            source = package.tensor_payload_path(capture.payload)
        except ValueError:
            detail["reason"] = "unsafe_payload_path"
            return detail
        if not source.is_file():
            detail["reason"] = "payload_file_missing"
        else:  # Defensive: _copy_payload() may gain another failure condition.
            detail["reason"] = "payload_unavailable"
    return detail


def format_missing_payload(detail: dict[str, Any]) -> str:
    """Render one structured payload failure as a concise actionable message."""
    boundary = detail.get("boundary", "value")
    name = detail.get("name", "unknown")
    value_id = detail.get("value_id", "unknown")
    prefix = f"{boundary} {name} (value {value_id})"
    reason = detail.get("reason", "payload_unavailable")
    capture = detail.get("capture") or {}

    if reason == "max_capture_elements":
        return (
            f"{prefix}: requested {capture.get('requested_mode', 'full')} capture was "
            f"degraded to {capture.get('mode', 'hash')} by max_capture_elements="
            f"{capture.get('limit', 'unknown')} (logical_numel="
            f"{capture.get('logical_numel', 'unknown')})"
        )
    if reason == "capture_error":
        return f"{prefix}: tensor capture failed and was degraded to metadata"
    if reason == "capture_mode":
        return (
            f"{prefix}: capture mode is {capture.get('mode', 'unknown')}; "
            "re-trace with --capture-tensors all"
        )
    if reason == "payload_file_missing":
        return f"{prefix}: referenced payload file is missing from the trace"
    if reason == "unsafe_payload_path":
        return f"{prefix}: payload path escapes the trace package"
    if reason == "value_not_found":
        return f"{prefix}: value record is missing from the trace"
    return f"{prefix}: payload is unavailable ({reason})"


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
    contracts: Sequence[str] | None = None,
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
        output_ids=graph.produced_value_ids(op),
        origin={"trace": str(package.root or ""), "operator_id": op_id},
        operators=[op],
        input_names=input_names,
        output_names=output_names,
        contracts=contracts,
    )


def extract_region(
    package: TracePackage,
    region: RegionRecord,
    output: str | Path,
    *,
    name: str | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    contracts: Sequence[str] | None = None,
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
        contracts=contracts,
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
    contracts: Sequence[str] | None = None,
) -> ExtractedTestcase:
    graph = package.graph
    output.mkdir(parents=True, exist_ok=True)

    result = ExtractedTestcase(path=output, name=name)
    manifest_inputs: list[dict[str, Any]] = []
    manifest_outputs: list[dict[str, Any]] = []

    input_names = _resolve_names(package, input_ids, input_names, "input")
    output_names = _resolve_names(package, output_ids, output_names, "output")
    input_stems = _payload_stems(input_names)
    output_stems = _payload_stems(output_names)

    for position, value_id in enumerate(input_ids):
        label = input_names[position]
        if not graph.has_value(value_id):
            result.missing_payloads.append(value_id)
            result.missing_payload_details.append(
                _missing_payload_detail(package, value_id, label, "input", None)
            )
            continue
        value = graph.value(value_id)
        relative = f"inputs/{input_stems[position]}.irtensor"
        ok = _copy_payload(package, value, output / relative)
        if not ok:
            result.missing_payloads.append(value_id)
            result.missing_payload_details.append(
                _missing_payload_detail(package, value_id, label, "input", value)
            )
        manifest_inputs.append(
            {
                "name": label,
                "value_id": value_id,
                "payload": relative if ok else None,
                "role": value.role,
                "qualified_name": value.qualified_name,
                **({"capture": value.capture.to_dict()} if not ok else {}),
                **_tensor_metadata(value),
            }
        )
        result.inputs.append(label)

    for position, value_id in enumerate(output_ids):
        label = output_names[position]
        if not graph.has_value(value_id):
            result.missing_payloads.append(value_id)
            result.missing_payload_details.append(
                _missing_payload_detail(package, value_id, label, "output", None)
            )
            continue
        value = graph.value(value_id)
        relative = f"reference/{output_stems[position]}.irtensor"
        ok = _copy_payload(package, value, output / relative)
        if not ok:
            result.missing_payloads.append(value_id)
            result.missing_payload_details.append(
                _missing_payload_detail(package, value_id, label, "output", value)
            )
        manifest_outputs.append(
            {
                "name": label,
                "value_id": value_id,
                "payload": relative if ok else None,
                **({"capture": value.capture.to_dict()} if not ok else {}),
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
        node_record = {
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
        if op.effects:
            node_record["effects"] = op.effects.to_dict()
        node_records.append(node_record)

    input_storage_ids = {
        graph.value(value_id).storage_id
        for value_id in input_ids
        if graph.has_value(value_id) and graph.value(value_id).storage_id is not None
    }
    output_states = {
        (graph.value(value_id).storage_id, graph.value(value_id).storage_version)
        for value_id in output_ids
        if graph.has_value(value_id)
    }
    final_mutations: dict[int, Any] = {}
    for op in operators:
        for mutation in op.effects.mutated_storages:
            if mutation.storage_id in input_storage_ids:
                final_mutations[mutation.storage_id] = mutation
    for mutation in final_mutations.values():
        if (mutation.storage_id, mutation.version_after) not in output_states:
            result.unobservable_mutations.append(mutation.to_dict())

    referenced_value_ids: set[int] = set(input_ids) | set(output_ids)
    for op in operators:
        referenced_value_ids.update(graph.op_input_value_ids(op))
        referenced_value_ids.update(graph.op_output_value_ids(op))
        for alias in op.effects.aliases:
            referenced_value_ids.add(alias.input_value_id)
            referenced_value_ids.add(alias.output_value_id)
    selected_op_ids = {op.id for op in operators}
    manifest_values = [
        {
            "id": value.id,
            **_tensor_metadata(value),
            "role": value.role,
            "qualified_name": value.qualified_name,
            # A standalone testcase cannot reference a producer outside its
            # projected node set. Boundary provenance remains on inputs/outputs.
            "producer": (
                value.producer if value.producer in selected_op_ids else None
            ),
        }
        for value in graph.values
        if value.id in referenced_value_ids
    ]

    selected_contracts = list(dict.fromkeys(contracts or ()))
    invalid_contracts = [item for item in selected_contracts if not is_contract_id(item)]
    if invalid_contracts:
        raise ExtractionError(
            "invalid versioned executable contract(s): " + ", ".join(invalid_contracts)
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
        "values": manifest_values,
    }
    if selected_contracts:
        manifest["contracts"] = selected_contracts
    manifest["requirements"] = derive_requirements(manifest)
    if result.non_portable:
        # IR §41: say so explicitly rather than shipping a testcase that cannot run.
        manifest["non_portable_values"] = sorted(set(result.non_portable))
    if result.missing_payloads:
        manifest["missing_payloads"] = sorted(set(result.missing_payloads))
        manifest["missing_payload_details"] = result.missing_payload_details
    if result.unobservable_mutations:
        manifest["unobservable_mutations"] = result.unobservable_mutations

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
        "runtime_object_id": value.runtime_object_id,
        "storage_id": value.storage_id,
        "storage_version": value.storage_version,
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
            details = manifest.get("missing_payload_details") or ()
            if details:
                for detail in details:
                    lines.append(f"> {format_missing_payload(detail)}.")
            else:
                lines.append(
                    "> Some tensor payloads were not captured. Re-trace with "
                    "`--capture-tensors all`."
                )
        if manifest.get("unobservable_mutations"):
            for mutation in manifest["unobservable_mutations"]:
                lines.append(
                    "> Mutation of storage:"
                    f"{mutation['storage_id']} to version {mutation['version_after']} "
                    "has no captured post-state output."
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
