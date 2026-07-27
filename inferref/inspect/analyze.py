"""Model coverage analysis (SPEC §25).

Answers "how much work is it to support this model?" by summarising which
physical operators appear, how much of the trace is covered by regions, and
which tensors lack payloads needed for testcase extraction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from inferref.ir.package import TracePackage
from inferref.testcase.dedup import dedup_operators, summarise


@dataclass
class Analysis:
    """Result of :func:`analyze`."""

    model: str = ""
    operator_counts: dict[str, int] = field(default_factory=dict)
    module_counts: dict[str, int] = field(default_factory=dict)
    total_operators: int = 0
    total_values: int = 0

    #: Fraction of operators that belong to at least one region (SPEC §25).
    region_coverage: float = 0.0
    covered_operators: int = 0
    regions: list[dict[str, Any]] = field(default_factory=list)

    #: Fraction of operators whose source location is known.
    source_coverage: float = 0.0
    #: Fraction of values with a full payload — what testcase extraction needs.
    payload_coverage: float = 0.0

    capture_modes: dict[str, int] = field(default_factory=dict)
    signature_summary: dict[str, Any] = field(default_factory=dict)
    mutating_operators: dict[str, int] = field(default_factory=dict)
    non_portable_operators: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "totals": {
                "operators": self.total_operators,
                "values": self.total_values,
                "unique_operators": len(self.operator_counts),
            },
            "operator_counts": self.operator_counts,
            "module_counts": self.module_counts,
            "coverage": {
                "region": self.region_coverage,
                "source": self.source_coverage,
                "payload": self.payload_coverage,
                "covered_operators": self.covered_operators,
            },
            "regions": self.regions,
            "capture_modes": self.capture_modes,
            "signatures": self.signature_summary,
            "mutating_operators": self.mutating_operators,
            "non_portable_operators": self.non_portable_operators,
            "warnings": self.warnings,
        }


def analyze(package: TracePackage) -> Analysis:
    """Analyse a trace package (SPEC §25)."""
    graph = package.graph
    result = Analysis(model=package.manifest.model.name)
    result.total_operators = len(graph.operators)
    result.total_values = len(graph.values)

    operator_counter: Counter[str] = Counter()
    module_counter: Counter[str] = Counter()
    mutating: Counter[str] = Counter()
    non_portable: set[str] = set()
    sourced = 0

    for op in graph.operators:
        operator_counter[op.canonical_name] += 1
        path = package.module_path(op.module_stack)
        module_counter[path or "<root>"] += 1
        if op.source_id is not None:
            sourced += 1
        if op.effects.mutated_storages:
            mutating[op.canonical_name] += 1
        for arg in list(op.positional_args) + list(op.keyword_args.values()):
            if getattr(arg, "kind", None) == "opaque" and not getattr(arg, "portable", True):
                non_portable.add(op.canonical_name)

    result.operator_counts = dict(operator_counter.most_common())
    result.module_counts = dict(module_counter.most_common())
    result.mutating_operators = dict(mutating.most_common())
    result.non_portable_operators = sorted(non_portable)

    covered: set[int] = set()
    for region in package.regions:
        covered.update(region.node_ids)
        result.regions.append(
            {
                "id": region.id,
                "name": region.name,
                "nodes": len(region.node_ids),
                "inputs": len(region.inputs),
                "outputs": len(region.outputs),
                "creation": region.creation_method,
                "engine_op": region.engine_op,
            }
        )
    result.covered_operators = len(covered)
    if result.total_operators:
        result.region_coverage = len(covered) / result.total_operators
        result.source_coverage = sourced / result.total_operators

    capture_counter: Counter[str] = Counter()
    full = 0
    for value in graph.values:
        capture_counter[value.capture.mode] += 1
        if value.capture.mode == "full":
            full += 1
    result.capture_modes = dict(capture_counter.most_common())
    if result.total_values:
        result.payload_coverage = full / result.total_values

    result.signature_summary = summarise(dedup_operators(package))

    if result.payload_coverage < 1.0:
        result.warnings.append(
            "not every value has a payload; re-trace with --capture-tensors all "
            "to make every operator independently reproducible"
        )
    if result.non_portable_operators:
        result.warnings.append(
            "some operators take non-portable arguments and cannot be extracted "
            "as standalone testcases: " + ", ".join(result.non_portable_operators)
        )
    result.warnings.extend(package.manifest.determinism.warnings)
    return result


def render_analysis(analysis: Analysis, *, top: int = 15) -> str:
    """Render an analysis as text (SPEC §25)."""
    lines: list[str] = []
    lines.append(f"Model: {analysis.model}")
    lines.append("")
    lines.append(f"Operators:        {analysis.total_operators}")
    lines.append(f"Unique operators: {len(analysis.operator_counts)}")
    lines.append(f"Tensor values:    {analysis.total_values}")
    lines.append("")
    lines.append("Coverage:")
    lines.append(f"  Source mapping:   {analysis.source_coverage:6.1%}")
    lines.append(
        f"  Region coverage:  {analysis.region_coverage:6.1%} "
        f"({analysis.covered_operators}/{analysis.total_operators} operators)"
    )
    lines.append(f"  Payload coverage: {analysis.payload_coverage:6.1%}")
    lines.append("")

    signatures = analysis.signature_summary
    lines.append("Operator signatures (SPEC §24):")
    lines.append(
        f"  {signatures.get('total_executions', 0)} executions "
        f"-> {signatures.get('total_signatures', 0)} unique signatures"
    )
    lines.append("")
    lines.append(f"Top operators (by execution count):")
    for name, count in list(analysis.operator_counts.items())[:top]:
        entry = signatures.get("by_operator", {}).get(name, {})
        unique = entry.get("signatures", "?")
        lines.append(f"  {count:6d}  {name:40s} {unique} unique signature(s)")
    remaining = len(analysis.operator_counts) - top
    if remaining > 0:
        lines.append(f"  ... and {remaining} more operator kinds")

    if analysis.mutating_operators:
        lines.append("")
        lines.append("Mutating operators (in-place / cache writes):")
        for name, count in analysis.mutating_operators.items():
            lines.append(f"  {count:6d}  {name}")

    if analysis.regions:
        lines.append("")
        lines.append("Regions:")
        for region in analysis.regions:
            engine = f" -> {region['engine_op']}" if region.get("engine_op") else ""
            lines.append(
                f"  {region['name']:30s} {region['nodes']:4d} ops  "
                f"{region['inputs']} in / {region['outputs']} out{engine}"
            )

    if analysis.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in analysis.warnings:
            lines.append(f"  - {warning}")
    return "\n".join(lines)
