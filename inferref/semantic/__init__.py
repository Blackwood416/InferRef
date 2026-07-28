"""Semantic analysis (SPEC §17, §56; IR §31-§32).

Semantic labels are optional annotation over an authoritative physical trace.
Detection runs on a loaded trace package and needs no framework, so this
package is pure stdlib and lives on the torch-free side of the dependency
split.
"""

from inferref.semantic.base import (
    CONFIDENCE_DETERMINISTIC,
    CONFIDENCE_FLOOR,
    CONFIDENCE_LIKELY,
    CONFIDENCE_STRONG,
    CONFIDENCE_VERY_STRONG,
    CONFIDENCE_WEAK,
    Detection,
    SemanticDetector,
)
from inferref.semantic.invocations import is_contiguous, split_invocations
from inferref.semantic.module_type import ModuleTypeDetector, semantic_for_type
from inferref.semantic.registry import builtin_detectors, detector_names, select
from inferref.semantic.run import (
    DetectionResult,
    apply_detections,
    clear_semantic_annotations,
    detect,
)
from inferref.semantic.source_function import (
    SourceFunctionDetector,
    semantic_for_function,
)

__all__ = [
    "CONFIDENCE_DETERMINISTIC",
    "CONFIDENCE_FLOOR",
    "CONFIDENCE_LIKELY",
    "CONFIDENCE_STRONG",
    "CONFIDENCE_VERY_STRONG",
    "CONFIDENCE_WEAK",
    "Detection",
    "DetectionResult",
    "ModuleTypeDetector",
    "SemanticDetector",
    "SourceFunctionDetector",
    "apply_detections",
    "builtin_detectors",
    "clear_semantic_annotations",
    "detect",
    "detector_names",
    "is_contiguous",
    "select",
    "semantic_for_function",
    "semantic_for_type",
    "split_invocations",
]
