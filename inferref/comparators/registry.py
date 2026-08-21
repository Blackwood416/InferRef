"""Comparator Plugin Registry (SPEC §7).

Handles discovery, verification, and lookup of comparator plugins via the
``inferref.comparators`` entry point group. Entry point discovery does not
import plugins until explicitly requested or verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any

from inferref.comparators.protocol import ComparatorPlugin, NUMERIC_COMPARATOR_ID

ENTRY_POINT_GROUP = "inferref.comparators"
BUILTIN_PACK_NAME = "builtin"


@dataclass(frozen=True)
class ComparatorEntry:
    """One resolvable comparator in deterministic list order."""

    id: str
    source: str
    distribution: str | None
    entry_point: str | None
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "distribution": self.distribution,
            "entry_point": self.entry_point,
            "status": self.status,
            **({"error": self.error} if self.error else {}),
        }


@dataclass(frozen=True)
class ComparatorPluginStatus:
    """Per-plugin verification status."""

    entry_point: str
    distribution: str | None
    version: str | None
    status: str
    comparator_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_point": self.entry_point,
            "distribution": self.distribution,
            "version": self.version,
            "status": self.status,
            **({"comparator_id": self.comparator_id} if self.comparator_id else {}),
            **({"error": self.error} if self.error else {}),
        }


BUILTIN_COMPARATORS: dict[str, ComparatorPlugin] = {}

_LOADED_PLUGINS: dict[str, ComparatorPlugin] = {}


def _ensure_builtins() -> None:
    if NUMERIC_COMPARATOR_ID not in BUILTIN_COMPARATORS:
        from inferref.comparators.numeric import NumericComparator
        BUILTIN_COMPARATORS[NUMERIC_COMPARATOR_ID] = NumericComparator()


def _reset_registry() -> None:
    """Reset loaded plugins cache and builtins to initial state (for testing)."""
    global BUILTIN_COMPARATORS, _LOADED_PLUGINS
    from inferref.comparators.numeric import NumericComparator
    BUILTIN_COMPARATORS = {
        NUMERIC_COMPARATOR_ID: NumericComparator(),
    }
    _LOADED_PLUGINS.clear()


def register_builtin_comparator(comparator: ComparatorPlugin) -> None:
    """Register a built-in comparator."""
    _ensure_builtins()
    BUILTIN_COMPARATORS[comparator.id] = comparator


def builtin_comparators() -> dict[str, ComparatorPlugin]:
    """Return a mapping of all registered built-in comparators."""
    _ensure_builtins()
    return dict(BUILTIN_COMPARATORS)


def _plugin_entry_points() -> list[Any]:
    """Discover entry points without importing modules."""
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        entries = list(discovered.select(group=ENTRY_POINT_GROUP))
    else:  # pragma: no cover - Python 3.10 compatibility
        entries = list(discovered.get(ENTRY_POINT_GROUP, ()))
    return sorted(
        entries,
        key=lambda item: (
            getattr(getattr(item, "dist", None), "name", None) or "",
            item.name,
            item.value,
        ),
    )


def _load_plugin(entry: Any) -> ComparatorPlugin:
    """Load and validate one comparator plugin entry point."""
    target = entry.load()
    if callable(target) and not isinstance(target, ComparatorPlugin):
        plugin = target()
    else:
        plugin = target

    if not hasattr(plugin, "id") or not hasattr(plugin, "validate_config") or not hasattr(plugin, "compare"):
        raise ValueError(
            f"comparator entry point {entry.name!r} returned an object not conforming to ComparatorPlugin protocol"
        )
    if not callable(getattr(plugin, "validate_config", None)):
        raise ValueError(f"comparator {entry.name!r} validate_config must be callable")
    if not callable(getattr(plugin, "compare", None)):
        raise ValueError(f"comparator {entry.name!r} compare must be callable")
    if plugin.id != entry.name:
        raise ValueError(f"comparator entry point name {entry.name!r} != plugin.id {plugin.id!r}")
    return plugin


def comparator_plugin_statuses(*, load: bool = False) -> list[ComparatorPluginStatus]:
    """Inspect all discovered comparator plugins."""
    _ensure_builtins()
    builtins = set(BUILTIN_COMPARATORS)
    counts: dict[str, int] = {}
    entries = _plugin_entry_points()
    for entry in entries:
        counts[entry.name] = counts.get(entry.name, 0) + 1

    out: list[ComparatorPluginStatus] = []
    for entry in entries:
        distribution = getattr(entry, "dist", None)
        dist_name = getattr(distribution, "name", None)
        version = getattr(distribution, "version", None)
        status = "discovered"
        error = None
        comparator_id = None

        if entry.name in builtins:
            status, error = "error", "plugin shadows a built-in comparator"
        elif counts[entry.name] > 1:
            status, error = "error", "duplicate comparator entry-point name"
        elif entry.name in _LOADED_PLUGINS:
            status = "loaded"
            comparator_id = _LOADED_PLUGINS[entry.name].id
        elif load:
            try:
                plugin = _load_plugin(entry)
                status = "loaded"
                comparator_id = plugin.id
                _LOADED_PLUGINS[entry.name] = plugin
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"

        out.append(
            ComparatorPluginStatus(
                entry_point=entry.name,
                distribution=dist_name,
                version=version,
                status=status,
                comparator_id=comparator_id,
                error=error,
            )
        )
    return out


def comparator_list(*, load: bool = False) -> list[ComparatorEntry]:
    """Return all built-in and discovered comparators in deterministic order."""
    _ensure_builtins()
    entries: list[ComparatorEntry] = []
    for comp_id, _comp in sorted(BUILTIN_COMPARATORS.items()):
        entries.append(
            ComparatorEntry(
                id=comp_id,
                source=BUILTIN_PACK_NAME,
                distribution=None,
                entry_point=None,
                status="loaded",
            )
        )

    for plugin in comparator_plugin_statuses(load=load):
        entries.append(
            ComparatorEntry(
                id=plugin.entry_point,
                source="plugin",
                distribution=plugin.distribution,
                entry_point=plugin.entry_point,
                status=plugin.status,
                error=plugin.error,
            )
        )

    return entries


def get_comparator(comparator_id: str) -> ComparatorPlugin | None:
    """Retrieve a comparator instance by its unique identifier."""
    _ensure_builtins()
    if comparator_id in BUILTIN_COMPARATORS:
        return BUILTIN_COMPARATORS[comparator_id]

    if comparator_id in _LOADED_PLUGINS:
        return _LOADED_PLUGINS[comparator_id]

    # Search in discovered entry points
    entries = _plugin_entry_points()
    matching = [e for e in entries if e.name == comparator_id]
    if not matching:
        return None

    if len(matching) > 1:
        raise ValueError(f"duplicate comparator entry-point name: {comparator_id!r}")

    plugin = _load_plugin(matching[0])
    _LOADED_PLUGINS[comparator_id] = plugin
    return plugin


def verify_comparators() -> list[ComparatorPluginStatus]:
    """Eagerly load and verify all registered comparator plugins."""
    return comparator_plugin_statuses(load=True)
