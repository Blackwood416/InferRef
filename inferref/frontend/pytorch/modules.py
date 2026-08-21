"""Module hierarchy tracking (IR §28, §29; SPEC §7.2, §49).

Uses PyTorch's **global** module forward hooks. This matters: it means
``inferref trace some_script.py --scope model.layers.0`` works against a script
InferRef does not control and never had to modify.

Qualified paths are recovered by walking ``named_modules()`` on the outermost
module the moment it is entered, giving ``layers.0.self_attn.q_proj`` style
paths rather than bare class names.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.modules.module as torch_module

from inferref.ir.module import ModuleRecord, path_matches

ROOT_PATH = ""


def module_type_name(module: torch.nn.Module) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass
class ModuleTracker:
    """Maintains the live module stack and the module record table."""

    scope: str | None = None
    exclude: tuple[str, ...] = ()

    _stack: list[int] = field(default_factory=list)
    _path_stack: list[str] = field(default_factory=list)
    _records: dict[str, ModuleRecord] = field(default_factory=dict)
    _path_by_pyid: dict[int, str] = field(default_factory=dict)
    _handles: list[Any] = field(default_factory=list)
    _next_id: int = 1

    #: Called with the module instance the first time a root module is entered.
    on_root: Callable[[torch.nn.Module], None] | None = None

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._handles.append(torch_module.register_module_forward_pre_hook(self._pre_hook))
        self._handles.append(torch_module.register_module_forward_hook(self._post_hook))

    def stop(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._stack.clear()
        self._path_stack.clear()

    # -- hooks ------------------------------------------------------------

    def _pre_hook(self, module: torch.nn.Module, inputs: Any) -> None:
        if not self._stack:
            # Outermost module of this call: (re)derive qualified paths and let
            # the session index parameters/buffers against it.
            self._index_paths(module)
            if self.on_root is not None:
                self.on_root(module)
        path = self._path_by_pyid.get(id(module))
        if path is None:
            # A module invoked outside the indexed root (e.g. a helper module
            # constructed inline). Name it by position under its parent.
            parent = self._path_stack[-1] if self._path_stack else ""
            leaf = type(module).__name__
            path = f"{parent}.{leaf}" if parent else leaf
        record = self._record_for(path, module)
        self._stack.append(record.id)
        self._path_stack.append(path)

    def _post_hook(self, module: torch.nn.Module, inputs: Any, output: Any) -> None:
        if self._stack:
            self._stack.pop()
            self._path_stack.pop()

    # -- paths ------------------------------------------------------------

    def _index_paths(self, root: torch.nn.Module) -> None:
        self._path_by_pyid.clear()
        for name, sub in root.named_modules():
            self._path_by_pyid[id(sub)] = name

    def _record_for(self, path: str, module: torch.nn.Module) -> ModuleRecord:
        existing = self._records.get(path)
        if existing is not None:
            return existing
        parent_path = path.rsplit(".", 1)[0] if "." in path else ROOT_PATH
        parent = self._records.get(parent_path) if parent_path != path else None
        record = ModuleRecord(
            id=self._next_id,
            path=path,
            type=module_type_name(module),
            parent_id=parent.id if parent else None,
        )
        self._next_id += 1
        self._records[path] = record
        return record

    # -- queries ----------------------------------------------------------

    @property
    def stack(self) -> tuple[int, ...]:
        """Current module id stack, outermost -> innermost (IR §29)."""
        return tuple(self._stack)

    @property
    def current_path(self) -> str:
        return self._path_stack[-1] if self._path_stack else ""

    def in_scope(self) -> bool:
        """Whether the current module context should be traced.

        With no ``--scope`` everything is in scope. ``--exclude`` always wins,
        matching the usual expectation that an exclusion is a hard veto.
        """
        for path in self._path_stack:
            for pattern in self.exclude:
                if path_matches(path, pattern):
                    return False
        if self.scope is None:
            return True
        return any(path_matches(path, self.scope) for path in self._path_stack)

    def records(self) -> list[ModuleRecord]:
        return sorted(self._records.values(), key=lambda m: m.id)

    def module_ids_matching(self, pattern: str) -> list[int]:
        """Module ids whose path falls under ``pattern``."""
        return [r.id for r in self.records() if path_matches(r.path, pattern)]
