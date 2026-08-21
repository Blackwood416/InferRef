"""Comparator Plugin Protocol and core data structures (SPEC §7).

Defines the contract for task-level semantic equivalence comparators, the
ArtifactSet input abstraction, and the structured ComparatorResult response.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

NUMERIC_COMPARATOR_ID: str = "tensor/numeric/v1"


@dataclass(frozen=True)
class Artifact:
    """A named output artifact pointing to an on-disk payload file."""

    name: str
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": str(self.path)}


ArtifactSet = Mapping[str, Artifact]


@dataclass(frozen=True)
class ComparatorResult:
    """Structured evaluation result returned by a comparator plugin."""

    status: str  # "pass" | "fail" | "error"
    comparator: str
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    first_failure: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "comparator": self.comparator,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "first_failure": self.first_failure,
        }


@runtime_checkable
class ComparatorPlugin(Protocol):
    """Protocol for task-level semantic and tensor comparators."""

    id: str

    def validate_config(self, config: dict[str, Any] | None = None) -> None:
        """Validate comparator configuration statically before execution.

        Raises:
            ValueError: If configuration contains invalid keys, types, or values.
        """
        ...

    def compare(
        self,
        reference: ArtifactSet,
        actual: ArtifactSet,
        config: dict[str, Any] | None = None,
    ) -> ComparatorResult:
        """Compare candidate output artifacts against reference outputs.

        Args:
            reference: Named reference output artifacts.
            actual: Named engine/candidate output artifacts.
            config: Optional comparator-specific configuration dictionary.

        Returns:
            Structured ComparatorResult.
        """
        ...
