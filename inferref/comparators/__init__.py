"""Comparator plugin architecture and registry for task-level semantic validation (SPEC §7).

Public API::

    get_comparator(comparator_id) -> ComparatorPlugin | None
    comparator_list() -> list[ComparatorEntry]
    verify_comparators() -> list[ComparatorPluginStatus]
    run_comparator(comparator, reference, actual, ...) -> ComparatorResult
"""

from __future__ import annotations

from inferref.comparators.numeric import NUMERIC_COMPARATOR_ID, NumericComparator
from inferref.comparators.protocol import (
    Artifact,
    ArtifactSet,
    ComparatorPlugin,
    ComparatorResult,
)
from inferref.comparators.registry import (
    BUILTIN_COMPARATORS,
    BUILTIN_PACK_NAME,
    ENTRY_POINT_GROUP,
    ComparatorEntry,
    ComparatorPluginStatus,
    builtin_comparators,
    comparator_list,
    comparator_plugin_statuses,
    get_comparator,
    register_builtin_comparator,
    verify_comparators,
)
from inferref.comparators.runner import run_comparator

__all__ = [
    "Artifact",
    "ArtifactSet",
    "BUILTIN_COMPARATORS",
    "BUILTIN_PACK_NAME",
    "ComparatorEntry",
    "ComparatorPlugin",
    "ComparatorPluginStatus",
    "ComparatorResult",
    "ENTRY_POINT_GROUP",
    "NUMERIC_COMPARATOR_ID",
    "NumericComparator",
    "builtin_comparators",
    "comparator_list",
    "comparator_plugin_statuses",
    "get_comparator",
    "register_builtin_comparator",
    "run_comparator",
    "verify_comparators",
]
