"""Fixture comparator implementations."""

from __future__ import annotations

from typing import Any

from examples.comparators.object_detection import ObjectDetectionComparator
from inferref.comparators.protocol import ArtifactSet, ComparatorResult


class DetectionComparator(ObjectDetectionComparator):
    """Fixture alias for ObjectDetectionComparator."""


class BrokenComparator:
    """Fixture comparator that intentionally fails protocol checks or execution."""

    id: str = "broken/comparator/v1"

    def validate_config(self, config: dict[str, Any] | None = None) -> None:
        if config and config.get("raise_validation"):
            raise ValueError("intentional validation failure in BrokenComparator")

    def compare(
        self,
        reference: ArtifactSet,
        actual: ArtifactSet,
        config: dict[str, Any] | None = None,
    ) -> ComparatorResult:
        if config and config.get("raise_runtime"):
            raise RuntimeError("intentional runtime exception in BrokenComparator")
        return ComparatorResult(
            status="pass",
            comparator=self.id,
            metrics={"fixture": True},
        )


class ShadowComparator:
    """Fixture that attempts to shadow built-in tensor/numeric/v1."""

    id: str = "tensor/numeric/v1"

    def validate_config(self, config: dict[str, Any] | None = None) -> None:
        pass

    def compare(
        self,
        reference: ArtifactSet,
        actual: ArtifactSet,
        config: dict[str, Any] | None = None,
    ) -> ComparatorResult:
        return ComparatorResult(status="pass", comparator=self.id)
