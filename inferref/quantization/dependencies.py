"""Stdlib-only dependency projection for quantized artifact extraction.

This validates ID relationships; it does not enable a new trace/testcase
version, read payloads, or replace numerical parameter validation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from inferref.quantization.spec import QuantizationError, QuantizationSpec, _integer


@dataclass(frozen=True)
class DependencyClosure:
    """Deterministic projection without assigning extra executable input roles."""

    value_ids: tuple[int, ...]
    parameter_ids: tuple[int, ...]
    quantized_ids: tuple[int, ...]


def quantization_dependencies(
    roots: Iterable[int], *, specs: Mapping[int, QuantizationSpec], available_ids: Iterable[int],
) -> DependencyClosure:
    """Project selected roots and their quantization dependencies, failing closed.

    IDs must be unique in the available-value registry. Repeated roots are
    harmless. Shared parameters appear once, including parameters already in
    the root set. Cycles are diagnosed before rejecting nested quantization;
    iterative traversal avoids Python recursion limits on malformed chains.
    Unselected relationships are not traversed (validate all roots to validate
    an entire registry). This function never mutates its arguments.
    """
    available: set[int] = set()
    for value_id in available_ids:
        _integer(value_id, "available value ID")
        if value_id in available:
            raise QuantizationError(f"duplicate value ID {value_id}")
        available.add(value_id)
    for value_id, spec in specs.items():
        _integer(value_id, "quantized value ID")
        if value_id not in available:
            raise QuantizationError(f"quantized value ID {value_id} is absent from the registry")
        if not isinstance(spec, QuantizationSpec):
            raise QuantizationError(f"value ID {value_id} requires a validated QuantizationSpec")
    selected: set[int] = set()
    for value_id in roots:
        _integer(value_id, "root value ID")
        selected.add(value_id)

    # 1 = active DFS frame; 2 = completely visited. The exit frame remains on
    # the stack until all dependencies are done, including when roots overlap.
    state: dict[int, int] = {}
    parameters: set[int] = set()
    for root in sorted(selected):
        stack = [(root, False)]
        while stack:
            value_id, exiting = stack.pop()
            if exiting:
                state[value_id] = 2
                continue
            if value_id not in available:
                raise QuantizationError(f"missing tensor value ID {value_id}")
            if state.get(value_id) == 1:
                raise QuantizationError(f"quantization dependency cycle at value ID {value_id}")
            if state.get(value_id) == 2:
                continue
            state[value_id] = 1
            stack.append((value_id, True))
            spec = specs.get(value_id)
            if spec is not None:
                parameters.update(spec.dependencies)
                stack.extend((dependency, False) for dependency in reversed(spec.dependencies))
    nested = parameters.intersection(specs)
    if nested:
        raise QuantizationError(f"nested quantized parameters are unsupported: {sorted(nested)}")
    return DependencyClosure(
        value_ids=tuple(sorted(state)),
        parameter_ids=tuple(sorted(parameters)),
        quantized_ids=tuple(sorted(set(state).intersection(specs))),
    )
