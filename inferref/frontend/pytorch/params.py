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
    #: Every name observed over a storage, in registration order. Tied weights
    #: (a shared embedding / lm_head, say) legitimately have more than one.
    _names_by_ptr: dict[int, list[str]] = field(default_factory=dict)
    _indexed_roots: set[int] = field(default_factory=set)

    def index(self, root: torch.nn.Module, prefix: str = "") -> None:
        """Index ``root``'s parameters and buffers. Safe to call repeatedly."""
        if id(root) in self._indexed_roots:
            return
        self._indexed_roots.add(id(root))

        # remove_duplicate=False is required to see tied weights at all: the
        # default deduplicates shared tensors, so a tied lm_head would never
        # report its own name.
        for name, param in root.named_parameters(remove_duplicate=False):
            self._register(f"{prefix}{name}", param, "parameter")
        for name, buffer in root.named_buffers(remove_duplicate=False):
            self._register(f"{prefix}{name}", buffer, "buffer")

    def _register(self, name: str, tensor: torch.Tensor, role: str) -> None:
        if tensor is None:
            return
        # First registration wins for the canonical name, in both tables:
        # `named_parameters` yields in registration order, so the earlier name
        # is the more useful label. Using setdefault for the object table too
        # keeps a tied weight's canonical name the same whether it is reached
        # directly or through a view.
        self._by_pyid.setdefault(id(tensor), (name, role))
        try:
            ptr = tensor.untyped_storage().data_ptr()
        except (RuntimeError, NotImplementedError, AttributeError):
            return
        self._by_ptr.setdefault(ptr, (name, role))
        names = self._names_by_ptr.setdefault(ptr, [])
        if name not in names:
            names.append(name)

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

    def aliases_of(self, tensor: torch.Tensor) -> tuple[str, ...]:
        """Every parameter/buffer name sharing ``tensor``'s storage (IR §38).

        Returns more than one name only for tied weights. Recording all of them
        matters because an engine needs to know that ``lm_head.weight`` and
        ``model.embed_tokens.weight`` are one allocation, not two.
        """
        try:
            ptr = tensor.untyped_storage().data_ptr()
        except (RuntimeError, NotImplementedError, AttributeError):
            return ()
        names = self._names_by_ptr.get(ptr, ())
        return tuple(names) if len(names) > 1 else ()

    def is_parameter_data(self, tensor: torch.Tensor) -> bool:
        role, _ = self.classify(tensor)
        return role in ("parameter", "buffer")
