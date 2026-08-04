"""Derive portable engine requirements from a standalone testcase."""

from __future__ import annotations

from typing import Any


def derive_requirements(manifest: dict[str, Any]) -> dict[str, Any]:
    tensors = [
        item
        for key in ("inputs", "outputs", "values")
        for item in manifest.get(key, [])
        if isinstance(item, dict) and isinstance(item.get("shape"), list)
    ]
    dtypes = sorted(
        {item["dtype"] for item in tensors if isinstance(item.get("dtype"), str)}
    )
    max_rank = max((len(item["shape"]) for item in tensors), default=0)
    features: set[str] = set()
    if len(manifest.get("outputs", [])) > 1:
        features.add("multiple_outputs")
    if any(
        _is_strided(item)
        for item in manifest.get("inputs", [])
        if isinstance(item, dict)
    ):
        features.add("strided_inputs")
    for node in manifest.get("nodes", []):
        effects = node.get("effects", {}) if isinstance(node, dict) else {}
        if not isinstance(effects, dict):
            continue
        if effects.get("aliases"):
            features.add("alias_effects")
        if effects.get("mutated_storages"):
            features.add("mutation_effects")
    return {"dtypes": dtypes, "max_rank": max_rank, "features": sorted(features)}


def testcase_requirements(manifest: dict[str, Any]) -> dict[str, Any]:
    declared = manifest.get("requirements")
    return dict(declared) if isinstance(declared, dict) else derive_requirements(manifest)


def _is_strided(item: dict[str, Any]) -> bool:
    shape = item.get("shape")
    stride = item.get("stride")
    if (
        not isinstance(shape, list)
        or not isinstance(stride, list)
        or len(shape) != len(stride)
    ):
        return False
    expected = []
    running = 1
    for dim in reversed(shape):
        expected.append(running)
        running *= max(int(dim), 1)
    return stride != list(reversed(expected))
