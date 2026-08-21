"""Comparison engine (SPEC §26, §30, §31, §35).

Two comparison modes:

``compare_testcase``
    A testcase directory against an engine output directory. This is the loop a
    coding agent runs (SPEC Appendix C) and requires nothing of the engine
    beyond writing ``.irtensor`` files (SPEC §22).

``compare_traces``
    Two complete traces, aligned by ``execution_index`` (IR §52), with
    first-divergence search (SPEC §30). Reporting stops at the earliest
    causally meaningful divergence because downstream errors are usually
    symptoms (SPEC §68.8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferref.compare.layout import LayoutDiff, diff_layout
from inferref.compare.metrics import Metrics, compute_metrics
from inferref.compare.tolerance import DEFAULT_TOLERANCES, TolerancePolicy
from inferref.ir.package import TracePackage
from inferref.ir.paths import resolve_contained_path
from inferref.tensor import codec
from inferref.tensor.codec import TensorView
from inferref.testcase.validate import require_valid_testcase

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_MISSING = "missing"
STATUS_ERROR = "error"


@dataclass
class TensorComparison:
    """Result of comparing one reference tensor with one engine tensor."""

    name: str
    status: str
    layout: LayoutDiff | None = None
    metrics: Metrics | None = None
    message: str = ""

    #: Provenance, when the comparison came from a trace (SPEC Appendix B).
    value_id: int | None = None
    producer_op_id: int | None = None
    execution_index: int | None = None
    operator: str | None = None
    module_path: str | None = None
    source: str | None = None
    region: str | None = None

    atol: float = 0.0
    rtol: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == STATUS_PASS

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "tolerance": {"atol": self.atol, "rtol": self.rtol},
        }
        if self.message:
            out["message"] = self.message
        if self.value_id is not None:
            out["value_id"] = self.value_id
        if self.producer_op_id is not None:
            out["producer_op_id"] = self.producer_op_id
        if self.execution_index is not None:
            out["execution_index"] = self.execution_index
        if self.operator:
            out["operator"] = self.operator
        if self.module_path:
            out["module_path"] = self.module_path
        if self.source:
            out["source"] = self.source
        if self.region:
            out["region"] = self.region
        if self.layout is not None:
            out["layout"] = self.layout.to_dict()
        if self.metrics is not None:
            out["metrics"] = self.metrics.to_dict()
        return out


@dataclass
class ComparisonReport:
    """Overall comparison result (SPEC §42, Appendix B)."""

    reference: str = ""
    actual: str = ""
    comparisons: list[TensorComparison] = field(default_factory=list)
    #: Set when ``--first-failure`` stopped the run early.
    stopped_early: bool = False
    tolerance: dict[str, Any] = field(default_factory=dict)
    effective_comparison: dict[str, Any] | None = None

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.comparisons if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.comparisons if c.status in (STATUS_FAIL, STATUS_ERROR))

    @property
    def missing_count(self) -> int:
        return sum(1 for c in self.comparisons if c.status == STATUS_MISSING)

    @property
    def first_failure(self) -> TensorComparison | None:
        for comparison in self.comparisons:
            if not comparison.passed:
                return comparison
        return None

    @property
    def status(self) -> str:
        if not self.comparisons:
            return STATUS_ERROR
        return STATUS_PASS if self.first_failure is None else STATUS_FAIL

    def to_dict(self) -> dict[str, Any]:
        first = self.first_failure
        out: dict[str, Any] = {
            "status": self.status,
            "reference": self.reference,
            "actual": self.actual,
            "summary": {
                "compared": len(self.comparisons),
                "passed": self.passed_count,
                "failed": self.failed_count,
                "missing": self.missing_count,
                "stopped_early": self.stopped_early,
            },
            "tolerance": self.tolerance,
            "first_failure": first.to_dict() if first else None,
            "comparisons": [c.to_dict() for c in self.comparisons],
        }
        if self.effective_comparison is not None:
            out["effective_comparison"] = self.effective_comparison
        return out


# -- tensor-level ----------------------------------------------------------


def compare_tensors(
    name: str,
    reference: TensorView,
    actual: TensorView,
    *,
    policy: TolerancePolicy,
    ignore_stride: bool = False,
    strict_layout: bool = False,
) -> TensorComparison:
    """Compare one decoded tensor pair, separating layout from value errors.

    Shape and dtype must always agree. Stride and storage-offset differences are
    *reported* but do not fail by default: payloads hold canonical logical
    values (IR §20), and a fused engine is explicitly not required to reproduce
    PyTorch's layout (SPEC §20). Pass ``strict_layout`` to enforce them.
    """
    layout = diff_layout(reference, actual, ignore_stride=ignore_stride)
    atol, rtol = policy.for_dtype(reference.dtype)

    if not layout.comparable:
        return TensorComparison(
            name=name,
            status=STATUS_FAIL,
            layout=layout,
            message=layout.summary(),
            atol=atol,
            rtol=rtol,
        )

    metrics = compute_metrics(
        reference.as_comparable(), actual.as_comparable(), atol=atol, rtol=rtol
    )
    values_ok = metrics.mismatch_count == 0
    layout_ok = not layout.any_mismatch

    if not values_ok:
        status = STATUS_FAIL
    elif strict_layout and not layout_ok:
        status = STATUS_FAIL
    else:
        status = STATUS_PASS

    message = ""
    if values_ok and not layout_ok:
        # The case SPEC §29 calls out explicitly.
        message = (
            "values are identical when interpreted logically, but layout differs: "
            + layout.summary()
        )
        if not strict_layout:
            message += " (not a failure; pass --strict-layout to enforce)"
    elif not values_ok and not layout_ok:
        message = "value mismatch and layout mismatch: " + layout.summary()

    return TensorComparison(
        name=name,
        status=status,
        layout=layout,
        metrics=metrics,
        message=message,
        atol=atol,
        rtol=rtol,
    )


# -- testcase vs engine output --------------------------------------------


def _engine_candidates(engine_dir: Path, name: str, value_id: Any) -> list[Path]:
    """Filenames an engine might plausibly have written for one output.

    Supports the SPEC §22 ``tensor_<value_id>.irtensor`` convention as well as
    name-based files, so an engine can emit whichever is more natural.
    """
    relatives = [
        f"{name}.irtensor",
        f"outputs/{name}.irtensor",
    ]
    if value_id is not None:
        relatives += [
            f"tensor_{value_id}.irtensor",
            f"v{int(value_id):08d}.irtensor",
            f"outputs/tensor_{value_id}.irtensor",
        ]
    return [
        resolve_contained_path(
            engine_dir, relative, kind=f"engine output {name!r} candidate path"
        )
        for relative in relatives
    ]


def compare_testcase(
    testcase_dir: str | Path,
    engine_dir: str | Path,
    *,
    policy: TolerancePolicy | None = None,
    ignore_stride: bool = False,
    strict_layout: bool = False,
    first_failure: bool = False,
    effective_comparison: Any | None = None,
    comparator: str | None = None,
    comparison_config: dict[str, Any] | None = None,
) -> ComparisonReport:
    """Compare a testcase's reference outputs against an engine output directory."""
    testcase_dir = Path(testcase_dir)
    engine_dir = Path(engine_dir)

    validation = require_valid_testcase(testcase_dir)
    manifest = validation.manifest

    if effective_comparison is None:
        from inferref.comparison.resolution import resolve_comparison_policy

        tc_spec = manifest.get("comparison")
        effective_comparison = resolve_comparison_policy(
            testcase_spec=tc_spec,
            cli_comparator=comparator,
            cli_atol=policy.override_atol if policy and policy.override_atol is not None else None,
            cli_rtol=policy.override_rtol if policy and policy.override_rtol is not None else None,
            cli_strict_layout=strict_layout if strict_layout else None,
            cli_ignore_stride=ignore_stride if ignore_stride else None,
            cli_tolerance=policy.per_dtype if (policy and policy.per_dtype != DEFAULT_TOLERANCES) else None,
            cli_config=comparison_config,
        )

    resolved_policy = effective_comparison.to_tolerance_policy() if hasattr(effective_comparison, "to_tolerance_policy") else (policy or TolerancePolicy())
    resolved_strict_layout = bool(effective_comparison.config.get("strict_layout", strict_layout)) if hasattr(effective_comparison, "config") else strict_layout
    resolved_ignore_stride = bool(effective_comparison.config.get("ignore_stride", ignore_stride)) if hasattr(effective_comparison, "config") else ignore_stride

    report = ComparisonReport(
        reference=str(testcase_dir),
        actual=str(engine_dir),
        tolerance=resolved_policy.to_dict(),
        effective_comparison=effective_comparison.to_dict() if hasattr(effective_comparison, "to_dict") else effective_comparison,
    )

    # An engine may declare its outputs explicitly; otherwise we probe filenames.
    engine_manifest_path = resolve_contained_path(
        engine_dir, "manifest.json", kind="engine output manifest path"
    )
    engine_map: dict[str, str] = {}
    if engine_manifest_path.is_file():
        engine_manifest = json.loads(engine_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(engine_manifest, dict):
            raise ValueError("engine output manifest root must be a JSON object")
        if not isinstance(engine_manifest.get("outputs", []), list):
            raise ValueError("engine output manifest outputs must be an array")
        for entry in engine_manifest.get("outputs", ()):
            if not isinstance(entry, dict):
                raise ValueError("engine output manifest entries must be objects")
            if "name" in entry and "payload" in entry:
                if not isinstance(entry["name"], str) or not isinstance(
                    entry["payload"], str
                ):
                    raise ValueError(
                        "engine output manifest name and payload must be strings"
                    )
                if entry["name"] in engine_map:
                    raise ValueError(
                        f"duplicate engine output name {entry['name']!r}"
                    )
                engine_map[entry["name"]] = entry["payload"]

    for output in manifest.get("outputs", ()):
        name = output.get("name", "output")
        value_id = output.get("value_id")
        reference_payload = output.get("payload")
        if not reference_payload:
            capture = output.get("capture") or {}
            comparison = TensorComparison(
                name=name,
                status=STATUS_MISSING,
                message=(
                    "reference payload is unavailable"
                    f" (capture mode {capture.get('mode', 'unknown')})"
                ),
                value_id=value_id,
            )
            _annotate_from_testcase(comparison, output)
            report.comparisons.append(comparison)
            if first_failure:
                report.stopped_early = True
                break
            continue
        reference_path = resolve_contained_path(
            testcase_dir,
            reference_payload,
            kind=f"testcase output {name!r} payload path",
        )

        actual_path: Path | None = None
        if name in engine_map:
            actual_path = resolve_contained_path(
                engine_dir,
                engine_map[name],
                kind=f"engine output {name!r} payload path",
            )
        else:
            for candidate in _engine_candidates(engine_dir, name, value_id):
                if candidate.is_file():
                    actual_path = candidate
                    break

        comparison = _compare_one_file(
            name,
            reference_path,
            actual_path,
            policy=resolved_policy,
            ignore_stride=resolved_ignore_stride,
            strict_layout=resolved_strict_layout,
            value_id=value_id,
        )
        _annotate_from_testcase(comparison, output)
        report.comparisons.append(comparison)
        if first_failure and not comparison.passed:
            report.stopped_early = True
            break

    return report


def _annotate_from_testcase(comparison: TensorComparison, output: dict[str, Any]) -> None:
    """Carry the producer context recorded at extraction time into the report."""
    producer = output.get("producer")
    if not producer:
        return
    comparison.producer_op_id = producer.get("op_id")
    comparison.execution_index = producer.get("execution_index")
    comparison.operator = producer.get("canonical_name")
    comparison.module_path = producer.get("module_path")
    comparison.source = producer.get("source")
    comparison.region = producer.get("region")


def _compare_one_file(
    name: str,
    reference_path: Path,
    actual_path: Path | None,
    *,
    policy: TolerancePolicy,
    ignore_stride: bool,
    strict_layout: bool = False,
    value_id: Any = None,
) -> TensorComparison:
    if actual_path is None or not actual_path.is_file():
        return TensorComparison(
            name=name,
            status=STATUS_MISSING,
            message=f"engine produced no output for {name!r}",
            value_id=value_id,
        )
    try:
        reference = codec.read(reference_path)
    except Exception as exc:
        return TensorComparison(
            name=name,
            status=STATUS_ERROR,
            message=f"cannot read reference {reference_path}: {exc}",
            value_id=value_id,
        )
    try:
        actual = codec.read(actual_path)
    except Exception as exc:
        return TensorComparison(
            name=name,
            status=STATUS_ERROR,
            message=f"cannot read engine output {actual_path}: {exc}",
            value_id=value_id,
        )

    comparison = compare_tensors(
        name,
        reference,
        actual,
        policy=policy,
        ignore_stride=ignore_stride,
        strict_layout=strict_layout,
    )
    comparison.value_id = value_id
    return comparison


# -- trace vs trace --------------------------------------------------------


def _annotate_from_trace(
    comparison: TensorComparison, package: TracePackage, value_id: int
) -> None:
    """Attach producer/module/source context to a comparison (SPEC Appendix B)."""
    graph = package.graph
    if not graph.has_value(value_id):
        return
    value = graph.value(value_id)
    comparison.value_id = value_id
    comparison.producer_op_id = value.producer
    if value.producer is not None and graph.has_op(value.producer):
        op = graph.op(value.producer)
        comparison.operator = op.canonical_name
        comparison.execution_index = op.execution_index
        comparison.module_path = package.module_path(op.module_stack) or None
        source = package.source(op.source_id)
        comparison.source = str(source) if source else None
        # Most specific containing region: regions nest (IR §36) and the
        # innermost one is what identifies the tensor.
        containing = [r for r in package.regions if op.id in r.node_ids]
        if containing:
            comparison.region = min(containing, key=lambda r: len(r.node_ids)).name


def _comparable_values(package: TracePackage) -> list[tuple[int, str]]:
    """Value ids in execution order that carry a full payload."""
    graph = package.graph
    out: list[tuple[int, str]] = []
    for op in graph.ops_in_execution_order():
        for value_id in graph.op_output_value_ids(op):
            if not graph.has_value(value_id):
                continue
            value = graph.value(value_id)
            if value.capture.mode == "full" and value.capture.payload:
                out.append((value_id, value.capture.payload))
    return out


def compare_traces(
    reference_dir: str | Path,
    actual_dir: str | Path,
    *,
    policy: TolerancePolicy | None = None,
    ignore_stride: bool = False,
    strict_layout: bool = False,
    first_failure: bool = False,
) -> ComparisonReport:
    """Compare two traces, aligned by ``execution_index`` (IR §52; SPEC §31)."""
    reference_dir = Path(reference_dir)
    actual_dir = Path(actual_dir)
    policy = policy or TolerancePolicy()

    ref_pkg = TracePackage.load(reference_dir)
    act_pkg = TracePackage.load(actual_dir)
    report = ComparisonReport(
        reference=str(reference_dir), actual=str(actual_dir), tolerance=policy.to_dict()
    )

    # Align by (execution_index, output position) — the canonical fallback
    # alignment when no explicit mapping is supplied (IR §52).
    act_index = _index_by_execution(act_pkg)

    for op in ref_pkg.graph.ops_in_execution_order():
        outputs = ref_pkg.graph.op_output_value_ids(op)
        for position, value_id in enumerate(outputs):
            if not ref_pkg.graph.has_value(value_id):
                continue
            value = ref_pkg.graph.value(value_id)
            if value.capture.mode != "full" or not value.capture.payload:
                continue

            name = f"#{op.execution_index} {op.canonical_name}[{position}]"
            counterpart = act_index.get((op.execution_index, position))
            if counterpart is None:
                comparison = TensorComparison(
                    name=name,
                    status=STATUS_MISSING,
                    message="no counterpart at this execution index",
                )
            else:
                _actual_value_id, actual_payload = counterpart
                comparison = _compare_one_file(
                    name,
                    ref_pkg.tensor_payload_path(value.capture.payload),
                    act_pkg.tensor_payload_path(actual_payload),
                    policy=policy,
                    ignore_stride=ignore_stride,
                    strict_layout=strict_layout,
                )
            # Provenance always comes from the reference side: it is the trace
            # that carries authoritative module/source context.
            _annotate_from_trace(comparison, ref_pkg, value_id)
            report.comparisons.append(comparison)
            if first_failure and not comparison.passed:
                report.stopped_early = True
                return report

    return report


def _index_by_execution(package: TracePackage) -> dict[tuple[int, int], tuple[int, str]]:
    """Map ``(execution_index, output position)`` to ``(value_id, payload)``."""
    out: dict[tuple[int, int], tuple[int, str]] = {}
    graph = package.graph
    for op in graph.ops_in_execution_order():
        for position, value_id in enumerate(graph.op_output_value_ids(op)):
            if not graph.has_value(value_id):
                continue
            value = graph.value(value_id)
            if value.capture.mode == "full" and value.capture.payload:
                out[(op.execution_index, position)] = (value_id, value.capture.payload)
    return out


def upstream_context(
    report: ComparisonReport, count: int = 3
) -> list[TensorComparison]:
    """The comparisons immediately preceding the first failure (SPEC Appendix B)."""
    first = report.first_failure
    if first is None:
        return []
    index = report.comparisons.index(first)
    start = max(0, index - count)
    return report.comparisons[start:index]
