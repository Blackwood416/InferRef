"""Parameter and buffer classification (IR §38, §39).

Two lookup tables are built from the traced root module:

* ``id(tensor) -> qualified_name`` catches parameters passed directly;
* ``storage data_ptr -> qualified_name`` additionally catches tensors *derived*
  from a parameter. This matters more than it sounds: ``nn.Linear`` dispatches
  as ``aten.t(weight)`` followed by ``aten.addmm``, so the matrix reaching the
  matmul is a transposed view whose object identity differs from the parameter's.

Classifying these lets tensor capture deduplicate weights instead of writing one
payload per invocation (IR §39).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ParameterIndex:
    """Maps runtime tensors back to parameter/buffer names."""

    _by_pyid: dict[int, tuple[str, str]] = field(default_factory=dict)
    _by_ptr: dict[int, tuple[str, str]] = field(default_factory=dict)
    _indexed_roots: set[int] = field(default_factory=set)

    def index(self, root: torch.nn.Module, prefix: str = "") -> None:
        """Index ``root``'s parameters and buffers. Safe to call repeatedly."""
        if id(root) in self._indexed_roots:
            return
        self._indexed_roots.add(id(root))

        for name, param in root.named_parameters():
            self._register(f"{prefix}{name}", param, "parameter")
        for name, buffer in root.named_buffers():
            self._register(f"{prefix}{name}", buffer, "buffer")

    def _register(self, name: str, tensor: torch.Tensor, role: str) -> None:
        if tensor is None:
            return
        self._by_pyid[id(tensor)] = (name, role)
        try:
            ptr = tensor.untyped_storage().data_ptr()
        except (RuntimeError, NotImplementedError, AttributeError):
            return
        # First writer wins: if two parameters somehow share storage, the
        # earlier (usually the canonical) name is the more useful label.
        self._by_ptr.setdefault(ptr, (name, role))

    def classify(self, tensor: torch.Tensor) -> tuple[str, str | None]:
        """Return ``(role, qualified_name)`` for ``tensor`` (IR §38)."""
        hit = self._by_pyid.get(id(tensor))
        if hit is not None:
            return hit[1], hit[0]
        try:
            ptr = tensor.untyped_storage().data_ptr()
        except (RuntimeError, NotImplementedError, AttributeError):
            return "activation", None
        hit = self._by_ptr.get(ptr)
        if hit is not None:
            # A view of a parameter is still parameter data for dedup purposes,
            # but is not itself the named parameter.
            return hit[1], hit[0]
        return "activation", None

    def is_parameter_data(self, tensor: torch.Tensor) -> bool:
        role, _ = self.classify(tensor)
        return role in ("parameter", "buffer")
