"""Built-in detector registry (SPEC §55).

Keeps detector selection data-driven so ``--detector`` has something to name
and a future plugin mechanism has an obvious place to hook into.
"""

from __future__ import annotations

from inferref.semantic.base import SemanticDetector
from inferref.semantic.module_type import ModuleTypeDetector
from inferref.semantic.source_function import SourceFunctionDetector


def builtin_detectors() -> list[SemanticDetector]:
    """Every detector enabled by default, in the order they run."""
    return [ModuleTypeDetector(), SourceFunctionDetector()]


def detector_names() -> list[str]:
    return [d.name for d in builtin_detectors()]


def select(names: list[str] | tuple[str, ...] | None) -> list[SemanticDetector]:
    """Resolve ``--detector`` values to detector instances."""
    available = {d.name: d for d in builtin_detectors()}
    if not names:
        return list(available.values())
    chosen: list[SemanticDetector] = []
    for name in names:
        try:
            chosen.append(available[name])
        except KeyError:
            raise ValueError(
                f"unknown detector {name!r}; available: {sorted(available)}"
            ) from None
    return chosen
