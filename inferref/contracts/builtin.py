"""The four built-in executable contracts (Contract Schema v0.1 section 11).

These are migrated unchanged from the old hardcoded table in
``inferref/testcase/contracts.py``: same IDs, role names, validators, issue
messages and derived requirements. They are registered through the same
registry as third-party plugin contracts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from inferref.contracts.schema import ExecutableContract


def _numel(shape: Sequence[Any]) -> int:
    total = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            return 0
        total *= dimension
    return total


def _shape(value: Any) -> list[int] | None:
    if not isinstance(value, dict):
        return None
    shape = value.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
    ):
        return None
    return shape  # type: ignore[return-value]


def _rmsnorm_inputs(inputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    x = _shape(inputs.get("x"))
    weight = _shape(inputs.get("weight"))
    epsilon = _shape(inputs.get("epsilon"))
    issues: list[str] = []
    if x is None:
        return ["x must be a tensor"]
    if not x or x[-1] <= 0 or _numel(x) <= 0:
        issues.append("x must be non-empty with a positive last dimension")
    elif weight is not None and _numel(weight) != x[-1]:
        issues.append("weight numel must equal x.shape[-1]")
    if epsilon is None or _numel(epsilon) != 1:
        issues.append("epsilon must contain exactly one value")
    return issues


def _rope_inputs(inputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    query = _shape(inputs.get("query"))
    key = _shape(inputs.get("key"))
    cos = _shape(inputs.get("cos"))
    sin = _shape(inputs.get("sin"))
    issues: list[str] = []
    if query is None:
        return ["query must be a tensor"]
    if len(query) < 2 or _numel(query) <= 0 or query[-1] <= 0 or query[-1] % 2:
        issues.append(
            "query must be non-empty with rank >= 2 and a positive even last dimension"
        )
    else:
        if key is None:
            issues.append("key must be a tensor")
        elif len(key) < 2 or _numel(key) <= 0 or key[-1] != query[-1]:
            issues.append("key last dimension must equal query")
        if cos is None or sin is None:
            issues.append("cos and sin must be tensors")
        elif len(cos) != 2 or cos != sin or cos[-1] != query[-1] or cos[0] <= 0:
            issues.append("cos/sin must have matching [sequence, rotary_dim] shapes")
        elif key is not None and (query[-2] != cos[0] or key[-2] != cos[0]):
            issues.append("query/key sequence dimensions must equal cos.shape[0]")
    return issues


def _kv_append_inputs(inputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    cache = _shape(inputs.get("cache"))
    update = _shape(inputs.get("update"))
    issues: list[str] = []
    if cache is None or update is None:
        return ["cache and update must be tensors"]
    if (
        len(cache) < 2
        or len(update) != len(cache)
        or _numel(cache) <= 0
        or _numel(update) <= 0
    ):
        issues.append("cache/update must be non-empty with equal rank >= 2")
        return issues
    sequence_axis = len(cache) - 2
    if any(
        cache[axis] != update[axis]
        for axis in range(len(cache))
        if axis != sequence_axis
    ):
        issues.append("non-sequence dimensions must match")
    if cache[-1] <= 0 or cache[-2] <= 0 or update[-2] <= 0:
        issues.append("sequence and width dimensions must be positive")
    return issues


def _kv_indexed_inputs(inputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    issues = list(_kv_append_inputs(inputs))
    index = _shape(inputs.get("index"))
    if index is None:
        issues.append("index must be a tensor")
    elif _numel(index) != 1:
        issues.append("index must contain exactly one value")
    return issues


def _rmsnorm_outputs(outputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    y = _shape(outputs.get("y"))
    if y is None or _numel(y) <= 0:
        return ["y must be a non-empty tensor"]
    return []


def _rmsnorm_relation(
    inputs: Mapping[str, dict[str, Any]], outputs: Mapping[str, dict[str, Any]]
) -> Sequence[str]:
    x = _shape(inputs.get("x"))
    y = _shape(outputs.get("y"))
    if x is None or y is None:
        return []
    issues: list[str] = []
    if y != x:
        issues.append("y.shape must equal x.shape")
    if inputs.get("x", {}).get("dtype") is not None and outputs.get("y", {}).get(
        "dtype"
    ) != inputs["x"].get("dtype"):
        issues.append("y.dtype must equal x.dtype")
    return issues


def _rope_outputs(outputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    issues: list[str] = []
    for name in ("q_embed", "k_embed"):
        shape = _shape(outputs.get(name))
        if shape is None or _numel(shape) <= 0:
            issues.append(f"{name} must be a non-empty tensor")
    return issues


def _rope_relation(
    inputs: Mapping[str, dict[str, Any]], outputs: Mapping[str, dict[str, Any]]
) -> Sequence[str]:
    query = _shape(inputs.get("query"))
    key = _shape(inputs.get("key"))
    q_embed = _shape(outputs.get("q_embed"))
    k_embed = _shape(outputs.get("k_embed"))
    issues: list[str] = []
    if query is not None and q_embed is not None and q_embed != query:
        issues.append("q_embed.shape must equal query.shape")
    if key is not None and k_embed is not None and k_embed != key:
        issues.append("k_embed.shape must equal key.shape")
    for input_name, output_name in (("query", "q_embed"), ("key", "k_embed")):
        input_dtype = inputs.get(input_name, {}).get("dtype")
        output_dtype = outputs.get(output_name, {}).get("dtype")
        if input_dtype is not None and output_dtype != input_dtype:
            issues.append(f"{output_name}.dtype must equal {input_name}.dtype")
    return issues


def _kv_outputs(outputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    cache_out = _shape(outputs.get("cache_out"))
    if cache_out is None or _numel(cache_out) <= 0:
        return ["cache_out must be a non-empty tensor"]
    return []


def _kv_dtype_issues(
    inputs: Mapping[str, dict[str, Any]], outputs: Mapping[str, dict[str, Any]]
) -> Sequence[str]:
    dtypes = [
        value.get("dtype")
        for value in (
            inputs.get("cache"),
            inputs.get("update"),
            outputs.get("cache_out"),
        )
        if isinstance(value, dict) and value.get("dtype") is not None
    ]
    if len(dtypes) == 3 and len(set(dtypes)) != 1:
        return ["cache, update, and cache_out dtypes must match"]
    return []


def _kv_append_relation(
    inputs: Mapping[str, dict[str, Any]], outputs: Mapping[str, dict[str, Any]]
) -> Sequence[str]:
    cache = _shape(inputs.get("cache"))
    update = _shape(inputs.get("update"))
    cache_out = _shape(outputs.get("cache_out"))
    if cache is None or update is None or cache_out is None:
        return []
    if len(cache_out) != len(cache):
        return ["cache_out rank must equal cache rank"]
    issues: list[str] = []
    sequence_axis = len(cache) - 2
    if any(
        cache_out[axis] != cache[axis]
        for axis in range(len(cache))
        if axis != sequence_axis
    ):
        issues.append("cache_out non-sequence dimensions must equal cache")
    if cache_out[-1] != cache[-1]:
        issues.append("cache_out width must equal cache width")
    if cache_out[sequence_axis] != cache[sequence_axis] + update[sequence_axis]:
        issues.append(
            "cache_out sequence length must equal cache + update sequence lengths"
        )
    return [*issues, *_kv_dtype_issues(inputs, outputs)]


def _kv_indexed_relation(
    inputs: Mapping[str, dict[str, Any]], outputs: Mapping[str, dict[str, Any]]
) -> Sequence[str]:
    cache = _shape(inputs.get("cache"))
    cache_out = _shape(outputs.get("cache_out"))
    if cache is None or cache_out is None:
        return []
    if cache_out != cache:
        return ["cache_out.shape must equal cache.shape for an indexed update"]
    return list(_kv_dtype_issues(inputs, outputs))


def _build_builtin_contracts() -> tuple[ExecutableContract, ...]:
    return tuple(
        sorted(
            (
                ExecutableContract(
                    id="rmsnorm/last-dim/v1",
                    inputs=("x", "weight", "epsilon"),
                    outputs=("y",),
                    validate_inputs=_rmsnorm_inputs,
                    validate_outputs=_rmsnorm_outputs,
                    validate_relation=_rmsnorm_relation,
                    description="RMSNorm over the last dimension with weight and scalar epsilon",
                ),
                ExecutableContract(
                    id="rope/rotate-half/v1",
                    inputs=("query", "key", "cos", "sin"),
                    outputs=("q_embed", "k_embed"),
                    validate_inputs=_rope_inputs,
                    validate_outputs=_rope_outputs,
                    validate_relation=_rope_relation,
                    description="rotate-half RoPE with per-token cos/sin over the last dimension",
                ),
                ExecutableContract(
                    id="kv-cache/append/v1",
                    inputs=("cache", "update"),
                    outputs=("cache_out",),
                    validate_inputs=_kv_append_inputs,
                    validate_outputs=_kv_outputs,
                    validate_relation=_kv_append_relation,
                    description="append a new sequence block to the cache sequence axis",
                ),
                ExecutableContract(
                    id="kv-cache/indexed-update/v1",
                    inputs=("cache", "update", "index"),
                    outputs=("cache_out",),
                    validate_inputs=_kv_indexed_inputs,
                    validate_outputs=_kv_outputs,
                    validate_relation=_kv_indexed_relation,
                    description="overwrite one contiguous sequence block of the cache",
                ),
            ),
            key=lambda contract: contract.id,
        )
    )


EXECUTABLE_CONTRACTS: tuple[ExecutableContract, ...] = _build_builtin_contracts()


def builtin_contracts() -> tuple[ExecutableContract, ...]:
    """Return the always-present built-in contracts in sorted ID order."""

    return EXECUTABLE_CONTRACTS
