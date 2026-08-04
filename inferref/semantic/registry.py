"""Built-in detector registry (SPEC §55).

Keeps detector selection data-driven so ``--detector`` has something to name
and a future plugin mechanism has an obvious place to hook into.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any

from inferref.semantic.base import SemanticDetector
from inferref.semantic.cache_update import CacheUpdateDetector
from inferref.semantic.module_type import ModuleTypeDetector
from inferref.semantic.source_function import SourceFunctionDetector


def builtin_detectors() -> list[SemanticDetector]:
    """Every detector enabled by default, in the order they run."""
    return [ModuleTypeDetector(), SourceFunctionDetector(), CacheUpdateDetector()]


ENTRY_POINT_GROUP = "inferref.semantic_detectors"


@dataclass(frozen=True)
class PluginDescriptor:
    name: str
    distribution: str | None
    version: str | None
    value: str
    status: str = "discovered"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "distribution": self.distribution,
            "version": self.version,
            "value": self.value,
            "status": self.status,
            **({"error": self.error} if self.error else {}),
        }


def _plugin_entry_points() -> list[Any]:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        entries = list(discovered.select(group=ENTRY_POINT_GROUP))
    else:  # pragma: no cover - Python 3.10 compatibility
        entries = list(discovered.get(ENTRY_POINT_GROUP, ()))
    return sorted(entries, key=lambda item: (item.name, item.value))


def plugin_descriptors(*, load: bool = False) -> list[PluginDescriptor]:
    builtins = {detector.name for detector in builtin_detectors()}
    counts: dict[str, int] = {}
    entries = _plugin_entry_points()
    for entry in entries:
        counts[entry.name] = counts.get(entry.name, 0) + 1
    out: list[PluginDescriptor] = []
    for entry in entries:
        distribution = getattr(entry, "dist", None)
        name = getattr(distribution, "name", None)
        version = getattr(distribution, "version", None)
        status = "discovered"
        error = None
        if entry.name in builtins:
            status, error = "error", "plugin shadows a built-in detector"
        elif counts[entry.name] > 1:
            status, error = "error", "duplicate detector entry-point name"
        elif load:
            try:
                _load_plugin(entry)
            except Exception as exc:
                status, error = "error", f"{type(exc).__name__}: {exc}"
            else:
                status = "loaded"
        out.append(PluginDescriptor(entry.name, name, version, entry.value, status, error))
    return out


def _load_plugin(entry: Any) -> SemanticDetector:
    factory = entry.load()
    if not callable(factory):
        raise ValueError(f"detector entry point {entry.name!r} is not a factory")
    detector = factory()
    if not isinstance(detector, SemanticDetector):
        raise ValueError(f"detector entry point {entry.name!r} returned an invalid detector")
    if detector.name != entry.name:
        raise ValueError(
            f"detector entry point name {entry.name!r} != detector.name {detector.name!r}"
        )
    return detector


def detector_names() -> list[str]:
    names = [d.name for d in builtin_detectors()]
    names.extend(item.name for item in plugin_descriptors() if item.status != "error")
    return names


def select(names: list[str] | tuple[str, ...] | None) -> list[SemanticDetector]:
    """Resolve ``--detector`` values to detector instances."""
    available = {d.name: d for d in builtin_detectors()}
    if not names:
        return list(available.values())
    entries: dict[str, Any] = {}
    invalid: dict[str, str] = {}
    for descriptor, entry in zip(plugin_descriptors(), _plugin_entry_points()):
        if descriptor.status == "error":
            invalid[descriptor.name] = descriptor.error or "invalid plugin"
        else:
            entries[entry.name] = entry
    chosen: list[SemanticDetector] = []
    for name in names:
        if name in available:
            chosen.append(available[name])
        elif name in invalid:
            raise ValueError(f"invalid detector plugin {name!r}: {invalid[name]}")
        elif name in entries:
            try:
                chosen.append(_load_plugin(entries[name]))
            except Exception as exc:
                raise ValueError(f"failed to load detector plugin {name!r}: {exc}") from exc
        else:
            raise ValueError(
                f"unknown detector {name!r}; available: {sorted(detector_names())}"
            )
    return chosen
