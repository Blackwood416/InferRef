"""Trace package directory I/O (IR §4; SPEC §38).

::

    example.irtrace/
    ├── manifest.json
    ├── graph.json
    ├── modules.json
    ├── sources.json
    ├── regions.json
    ├── storages.json
    ├── tensors/
    │   └── v00000194.irtensor
    └── reports/

This module is pure stdlib: loading a trace never requires PyTorch or numpy.
Reading tensor *payloads* requires numpy, but that lives in
:mod:`inferref.tensor.codec` and is only imported on demand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.module import ModuleRecord
from inferref.ir.region import RegionRecord
from inferref.ir.source import SourceRecord
from inferref.ir.storage import StorageRecord

MANIFEST_FILE = "manifest.json"
GRAPH_FILE = "graph.json"
MODULES_FILE = "modules.json"
SOURCES_FILE = "sources.json"
REGIONS_FILE = "regions.json"
STORAGES_FILE = "storages.json"
TENSORS_DIR = "tensors"
REPORTS_DIR = "reports"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys=False preserves our declared field order; indent keeps traces
    # human-inspectable (SPEC §8) and diffable in review.
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class TracePackage:
    """An on-disk (or in-memory) InferRef trace package."""

    manifest: Manifest = field(default_factory=Manifest)
    graph: Graph = field(default_factory=Graph)
    modules: list[ModuleRecord] = field(default_factory=list)
    sources: list[SourceRecord] = field(default_factory=list)
    regions: list[RegionRecord] = field(default_factory=list)
    storages: list[StorageRecord] = field(default_factory=list)

    #: Directory this package was loaded from, if any.
    root: Path | None = None

    # -- lookup -----------------------------------------------------------

    def module(self, module_id: int) -> ModuleRecord | None:
        for m in self.modules:
            if m.id == module_id:
                return m
        return None

    def source(self, source_id: int | None) -> SourceRecord | None:
        if source_id is None:
            return None
        for s in self.sources:
            if s.id == source_id:
                return s
        return None

    def region(self, name_or_id: str | int) -> RegionRecord | None:
        for r in self.regions:
            if r.id == name_or_id or r.name == name_or_id:
                return r
        if isinstance(name_or_id, str) and name_or_id.isdigit():
            return self.region(int(name_or_id))
        return None

    def module_path(self, op_module_stack: tuple[int, ...]) -> str:
        """Full module path for an operator's module stack (innermost wins)."""
        if not op_module_stack:
            return ""
        innermost = self.module(op_module_stack[-1])
        return innermost.path if innermost else ""

    def tensor_payload_path(self, relative: str) -> Path:
        """Resolve a ``capture.payload`` reference to an absolute path."""
        if self.root is None:
            raise ValueError("trace package has no root directory; cannot resolve payload")
        return self.root / relative

    # -- I/O --------------------------------------------------------------

    def save(self, root: str | Path) -> Path:
        """Write the package to ``root`` (creating it if needed)."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / MANIFEST_FILE, self.manifest.to_dict())
        _write_json(root / GRAPH_FILE, self.graph.to_dict())
        _write_json(root / MODULES_FILE, {"modules": [m.to_dict() for m in self.modules]})
        _write_json(root / SOURCES_FILE, {"sources": [s.to_dict() for s in self.sources]})
        _write_json(root / REGIONS_FILE, {"regions": [r.to_dict() for r in self.regions]})
        _write_json(root / STORAGES_FILE, {"storages": [s.to_dict() for s in self.storages]})
        (root / TENSORS_DIR).mkdir(exist_ok=True)
        self.root = root
        return root

    def save_regions(self) -> None:
        """Rewrite only ``regions.json`` (used by ``inferref region create``)."""
        if self.root is None:
            raise ValueError("trace package has no root directory")
        _write_json(
            self.root / REGIONS_FILE, {"regions": [r.to_dict() for r in self.regions]}
        )

    @classmethod
    def load(cls, root: str | Path) -> "TracePackage":
        """Load a trace package from ``root``.

        Only ``manifest.json`` and ``graph.json`` are required (IR §47); the
        remaining files are optional and default to empty.
        """
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(f"not a trace package directory: {root}")

        manifest_path = root / MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing {MANIFEST_FILE} in {root}")
        manifest = Manifest.from_dict(_read_json(manifest_path))

        graph_path = root / GRAPH_FILE
        if not graph_path.is_file():
            raise FileNotFoundError(f"missing {GRAPH_FILE} in {root}")
        graph = Graph.from_dict(_read_json(graph_path))

        def _optional(name: str, key: str, record_cls: Any) -> list[Any]:
            path = root / name
            if not path.is_file():
                return []
            return [record_cls.from_dict(d) for d in _read_json(path).get(key, ())]

        return cls(
            manifest=manifest,
            graph=graph,
            modules=_optional(MODULES_FILE, "modules", ModuleRecord),
            sources=_optional(SOURCES_FILE, "sources", SourceRecord),
            regions=_optional(REGIONS_FILE, "regions", RegionRecord),
            storages=_optional(STORAGES_FILE, "storages", StorageRecord),
            root=root,
        )


def is_trace_package(path: str | Path) -> bool:
    """Cheap check for whether ``path`` looks like a trace package directory."""
    path = Path(path)
    return path.is_dir() and (path / MANIFEST_FILE).is_file() and (path / GRAPH_FILE).is_file()
