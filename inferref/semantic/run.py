"""Running detectors and applying their results (SPEC §17; IR §31, §34).

Detectors observe; this module writes. Keeping the split means a detector can
never corrupt physical trace truth (SPEC §68.3) — the worst it can do is
propose a region that is then derived and validated like any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from inferref.ir.operator import Annotation
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.region.manager import RegionError, create_region
from inferref.semantic.base import CONFIDENCE_FLOOR, Detection, SemanticDetector
from inferref.semantic.registry import select

#: Annotation type written onto operators (IR §31).
ANNOTATION_TYPE = "semantic"


@dataclass
class DetectionResult:
    """What :func:`apply_detections` did."""

    detections: list[Detection] = field(default_factory=list)
    regions: list[RegionRecord] = field(default_factory=list)
    annotated_operators: int = 0
    #: Detections that could not become regions, with the reason.
    skipped: list[tuple[Detection, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detections": [d.to_dict() for d in self.detections],
            "regions": [r.to_dict() for r in self.regions],
            "counts": {
                "detections": len(self.detections),
                "regions": len(self.regions),
                "annotated_operators": self.annotated_operators,
                "skipped": len(self.skipped),
            },
            "skipped": [
                {"detection": d.to_dict(), "reason": reason} for d, reason in self.skipped
            ],
        }

    def summary_by_name(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for detection in self.detections:
            counts[detection.name] = counts.get(detection.name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def detect(
    package: TracePackage,
    *,
    detectors: Sequence[SemanticDetector] | None = None,
    detector_names: Sequence[str] | None = None,
    min_confidence: float = CONFIDENCE_FLOOR,
) -> list[Detection]:
    """Run detectors over ``package`` and return their proposals.

    Nothing is written; the package is not modified.
    """
    chosen = list(detectors) if detectors is not None else select(detector_names)

    found: list[Detection] = []
    for detector in chosen:
        found.extend(detector.detect(package))

    found = [d for d in found if d.confidence >= min_confidence]
    found = _deduplicate(found)
    # Outermost first, then by execution order, so nesting reads naturally.
    found.sort(key=lambda d: (-len(d.node_ids), _execution_start(package, d)))
    return found


def _execution_start(package: TracePackage, detection: Detection) -> float:
    """First known execution index, independent of opaque operator ids."""
    indices = [
        package.graph.op(node_id).execution_index
        for node_id in detection.node_ids
        if package.graph.has_op(node_id)
    ]
    return float(min(indices)) if indices else float("inf")


def _deduplicate(detections: Iterable[Detection]) -> list[Detection]:
    """Collapse duplicate labels covering exactly the same operators.

    A construct can be recognised twice — a ``RotaryEmbedding`` module and the
    ``apply_rotary_pos_emb`` it calls, say. For the *same semantic label*, the
    higher-confidence proposal wins. Different labels over one physical node
    set are retained: ``MLP`` and ``SwiGLU`` can be two valid interpretations
    of a thin wrapper, just as regions may otherwise overlap or nest.
    """
    best: dict[tuple[str, tuple[int, ...]], Detection] = {}
    for detection in detections:
        key = (detection.name, tuple(sorted(detection.node_ids)))
        incumbent = best.get(key)
        if incumbent is None or detection.confidence > incumbent.confidence:
            best[key] = detection
    return list(best.values())


def apply_detections(
    package: TracePackage,
    detections: Sequence[Detection],
    *,
    annotate: bool = True,
    create_regions: bool = True,
) -> DetectionResult:
    """Write ``detections`` into ``package`` as annotations and/or regions."""
    result = DetectionResult(detections=list(detections))

    if annotate:
        result.annotated_operators = _annotate(package, detections)

    if create_regions:
        used_names = {r.name for r in package.regions}
        for detection in detections:
            name = _unique_name(detection.region_name, used_names)
            try:
                region = create_region(
                    package,
                    name,
                    detection.node_ids,
                    method=detection.method,
                    semantic=detection.name,
                    confidence=detection.confidence,
                )
            except RegionError as exc:
                result.skipped.append((detection, str(exc)))
                continue
            used_names.add(name)
            result.regions.append(region)

    return result


def _annotate(package: TracePackage, detections: Sequence[Detection]) -> int:
    """Attach semantic annotations to operators, innermost detection last.

    An operator inside ``Attention`` -> ``Linear`` accumulates both labels; the
    order means the most specific one is last, which is what a reader wants to
    see first.
    """
    graph = package.graph
    # Largest first so the innermost (smallest) detection is appended last.
    ordered = sorted(detections, key=lambda d: -len(d.node_ids))
    touched: set[int] = set()

    for detection in ordered:
        annotation = Annotation(
            type=ANNOTATION_TYPE,
            name=detection.name,
            confidence=detection.confidence,
            detector=detection.detector,
        )
        for node_id in detection.node_ids:
            if not graph.has_op(node_id):
                continue
            op = graph.op(node_id)
            if any(
                a.type == ANNOTATION_TYPE
                and a.name == annotation.name
                and a.detector == annotation.detector
                for a in op.annotations
            ):
                continue
            op.annotations = op.annotations + (annotation,)
            touched.add(node_id)
    return len(touched)


def _unique_name(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    index = 1
    while f"{name}#{index}" in taken:
        index += 1
    return f"{name}#{index}"


def clear_semantic_annotations(package: TracePackage) -> int:
    """Remove every semantic annotation, so detection can be re-run cleanly."""
    removed = 0
    for op in package.graph.operators:
        kept = tuple(a for a in op.annotations if a.type != ANNOTATION_TYPE)
        removed += len(op.annotations) - len(kept)
        op.annotations = kept
    return removed
