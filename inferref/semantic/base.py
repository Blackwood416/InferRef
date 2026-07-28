"""Semantic detector contract (SPEC §17, §56; IR §31, §32).

Semantic information is **annotation**, never a replacement for the physical
trace (SPEC §68.3). A detector therefore returns :class:`Detection` records and
never mutates the package; writing annotations and regions is
:mod:`inferref.semantic.run`'s job.

This package is pure stdlib on purpose. Detection operates on a loaded
:class:`~inferref.ir.package.TracePackage`, so it belongs on the torch-free side
of the dependency split and runs anywhere a trace can be read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from inferref.ir.package import TracePackage

#: Confidence bands from IR §32. Used as named constants so a detector states
#: which band it is claiming rather than inventing a number.
CONFIDENCE_DETERMINISTIC = 1.0   # a mapping that cannot be wrong
CONFIDENCE_VERY_STRONG = 0.95    # 0.90-0.99
CONFIDENCE_STRONG = 0.90
CONFIDENCE_LIKELY = 0.75         # 0.70-0.89
CONFIDENCE_WEAK = 0.60           # 0.50-0.69

#: Below this, IR §32 says to normally omit the annotation entirely.
CONFIDENCE_FLOOR = 0.50


@dataclass(frozen=True)
class Detection:
    """One semantic region proposed by a detector.

    ``node_ids`` is a contiguous run in execution order — one *invocation* of
    the construct, not every invocation of it. A model with 32 layers yields 32
    ``RMSNorm`` detections, which is what makes them individually extractable
    as testcases.
    """

    name: str
    node_ids: tuple[int, ...]
    confidence: float
    #: Versioned detector identity, e.g. ``inferref.semantic.module_type.v1``
    #: (IR §31).
    detector: str
    #: Region creation method (IR §35): ``module`` or ``source_function``.
    method: str
    #: Module path the detection is anchored to; used to build a readable name.
    scope: str = ""
    #: Free-form note explaining *why*, surfaced by ``--dry-run``.
    evidence: str = ""

    def __post_init__(self) -> None:
        if not self.node_ids:
            raise ValueError(f"detection {self.name!r} has no nodes")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"detection {self.name!r} has confidence {self.confidence}, "
                "expected 0.0-1.0"
            )

    @property
    def region_name(self) -> str:
        """``RoPE@layers.0.self_attn`` — readable and stable across runs."""
        return f"{self.name}@{self.scope}" if self.scope else self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "region_name": self.region_name,
            "node_ids": list(self.node_ids),
            "node_count": len(self.node_ids),
            "confidence": self.confidence,
            "detector": self.detector,
            "method": self.method,
            "scope": self.scope,
            "evidence": self.evidence,
        }


@runtime_checkable
class SemanticDetector(Protocol):
    """A semantic analysis pass (SPEC §56).

    Implementations MUST be pure with respect to ``package``: physical trace
    truth is authoritative and detectors only observe it.
    """

    #: Stable identifier, also used to select detectors from the CLI.
    name: str

    def detect(self, package: TracePackage) -> list[Detection]:
        """Return every construct this detector recognises in ``package``."""
        ...
