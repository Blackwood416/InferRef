"""Built-in executable contracts, including explicit quantized Linear profiles.

The core profiles were migrated unchanged from the old hardcoded table in
``inferref/testcase/contracts.py``: same IDs, role names, validators, issue
messages and derived requirements. 0.12 adds explicit bias, bfloat16, SwiGLU
and SDPA profiles next to them. They are registered through the same registry
as third-party plugin contracts.
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


def _y_matches_x_relation(
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


def _int8_linear_inputs(inputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    from inferref.quantization.spec import QuantizationError, QuantizationSpec

    x, weight = inputs.get("x", {}), inputs.get("weight", {})
    issues = []
    if x.get("dtype") != "float32" or not _shape(x) or "quantization" in x:
        issues.append("x must be an ordinary float32 tensor with rank >= 1")
    try:
        spec = QuantizationSpec.from_dict(weight.get("quantization"))
        if len(spec.logical_shape) != 2 or weight.get("dtype") != "int8":
            issues.append("weight must be affine INT8 [out_features, in_features]")
        elif _shape(x) and x["shape"][-1] != spec.logical_shape[1]:
            issues.append("x last dimension must equal weight in_features")
    except QuantizationError as exc:
        issues.append(str(exc))
    return issues


def _linear_out_features(weight: Mapping[str, Any]) -> int | None:
    """Weight out_features from a valid rank-2 descriptor; None when unusable."""

    from inferref.quantization.spec import QuantizationError, QuantizationSpec

    try:
        spec = QuantizationSpec.from_dict(weight.get("quantization"))
    except QuantizationError:
        return None
    return spec.logical_shape[0] if len(spec.logical_shape) == 2 else None


def _int8_bias_linear_inputs(inputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    issues = list(_int8_linear_inputs(inputs))
    bias = inputs.get("bias", {})
    if bias.get("dtype") != "float32" or "quantization" in bias:
        issues.append("bias must be an ordinary float32 tensor")
        return issues
    shape = _shape(bias)
    if shape is None or len(shape) != 1:
        issues.append("bias must have rank 1 with shape [out_features]")
        return issues
    out_features = _linear_out_features(inputs.get("weight", {}))
    if out_features is not None and shape[0] != out_features:
        issues.append("bias length must equal weight out_features")
    return issues


def _linear_output_shape_issues(
    inputs: Mapping[str, dict[str, Any]], outputs: Mapping[str, dict[str, Any]]
) -> list[str]:
    x, weight, y = inputs.get("x", {}), inputs.get("weight", {}), outputs.get("y", {})
    if _shape(x) and _shape(weight) and y.get("shape") != x["shape"][:-1] + weight["shape"][:1]:
        return ["y shape must equal x leading dimensions plus weight out_features"]
    return []


def _int8_linear_relation(inputs, outputs) -> Sequence[str]:
    y = outputs.get("y", {})
    if y.get("dtype") != "float32" or "quantization" in y:
        return ["y must be an ordinary float32 tensor"]
    return _linear_output_shape_issues(inputs, outputs)


def _affine_w4_weight_issues(
    weight: Mapping[str, Any], x_shape: list[int] | None
) -> list[str]:
    """Affine INT4/UINT4 weight checks shared by the float16 and bf16 profiles."""

    from inferref.quantization.spec import QuantizationError, QuantizationSpec

    try:
        spec = QuantizationSpec.from_dict(weight.get("quantization"))
        if spec.scheme == "nf4/v1" or spec.storage_dtype not in {"int4", "uint4"} or len(spec.logical_shape) != 2 or weight.get("dtype") != spec.storage_dtype:
            return ["weight must be affine INT4/UINT4 [out_features, in_features]"]
        if x_shape and x_shape[-1] != spec.logical_shape[1]:
            return ["x last dimension must equal weight in_features"]
    except QuantizationError as exc:
        return [str(exc)]
    return []


def _nf4_weight_issues(
    weight: Mapping[str, Any], x_shape: list[int] | None
) -> list[str]:
    """NF4 weight checks shared by the float16 and bf16 profiles."""

    from inferref.quantization.spec import QuantizationError, QuantizationSpec

    try:
        spec = QuantizationSpec.from_dict(weight.get("quantization"))
        if spec.scheme != "nf4/v1" or len(spec.logical_shape) != 2 or weight.get("dtype") != "uint4":
            return ["weight must be NF4 [out_features, in_features]"]
        if x_shape and x_shape[-1] != spec.logical_shape[1]:
            return ["x last dimension must equal weight in_features"]
    except QuantizationError as exc:
        return [str(exc)]
    return []


def _w4a16_activation_issues(
    inputs: Mapping[str, dict[str, Any]], *, dtype: str, label: str
) -> list[str]:
    x = inputs.get("x", {})
    if x.get("dtype") != dtype or not _shape(x) or "quantization" in x:
        return [f"{label} x must be ordinary {dtype} with rank >= 1"]
    return []


def _w4a16_inputs(inputs) -> Sequence[str]:
    return [
        *_w4a16_activation_issues(inputs, dtype="float16", label="W4A16"),
        *_affine_w4_weight_issues(inputs.get("weight", {}), _shape(inputs.get("x"))),
    ]


def _w4a16_bf16_inputs(inputs) -> Sequence[str]:
    return [
        *_w4a16_activation_issues(inputs, dtype="bfloat16", label="W4A16 bf16"),
        *_affine_w4_weight_issues(inputs.get("weight", {}), _shape(inputs.get("x"))),
    ]


def _w4a16_relation(inputs, outputs) -> Sequence[str]:
    y = outputs.get("y", {})
    if y.get("dtype") != "float16" or "quantization" in y:
        return ["W4A16 y must be ordinary float16"]
    return _linear_output_shape_issues(inputs, outputs)


def _w4a16_bf16_relation(inputs, outputs) -> Sequence[str]:
    y = outputs.get("y", {})
    if y.get("dtype") != "bfloat16" or "quantization" in y:
        return ["W4A16 bf16 y must be ordinary bfloat16"]
    return _linear_output_shape_issues(inputs, outputs)


def _nf4_inputs(inputs) -> Sequence[str]:
    return [
        *_w4a16_activation_issues(inputs, dtype="float16", label="NF4"),
        *_nf4_weight_issues(inputs.get("weight", {}), _shape(inputs.get("x"))),
    ]


def _nf4_bf16_inputs(inputs) -> Sequence[str]:
    return [
        *_w4a16_activation_issues(inputs, dtype="bfloat16", label="NF4 bf16"),
        *_nf4_weight_issues(inputs.get("weight", {}), _shape(inputs.get("x"))),
    ]


def _y_output_issues(outputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    """Shared non-empty shape/dtype check for ``y`` outputs that mirror ``x``.

    Agreement with ``x`` is a boundary relation (``_y_matches_x_relation``)
    because an output-only validator cannot see the inputs.
    """
    y = outputs.get("y")
    shape = _shape(y)
    if shape is None or _numel(shape) <= 0:
        return ["y must be a non-empty tensor"]
    if not isinstance(y, dict) or not isinstance(y.get("dtype"), str) or not y["dtype"]:
        return ["y must declare a non-empty dtype"]
    return []


# This validator owns the y dtype check, so the registry must not repeat it.
_y_output_issues._inferref_dtype_checked = True  # type: ignore[attr-defined]


def _swiglu_weight_shape(
    name: str, record: Mapping[str, Any]
) -> tuple[list[int] | None, list[str]]:
    """Logical shape of one SwiGLU weight plus any descriptor issue.

    A weight may be ordinary or carry an explicit quantization descriptor; the
    descriptor's logical shape is then the weight shape. Its record dtype is a
    storage dtype, not a logical dtype, so it does not join the activation
    dtype agreement.
    """
    if "quantization" not in record:
        return _shape(record), []

    from inferref.quantization.spec import QuantizationError, QuantizationSpec

    try:
        spec = QuantizationSpec.from_dict(record.get("quantization"))
    except QuantizationError as exc:
        return None, [f"{name} has an invalid quantization descriptor: {exc}"]
    return list(spec.logical_shape), []


def _swiglu_inputs(inputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    x = inputs.get("x", {})
    x_shape = _shape(x)
    issues: list[str] = []
    if "quantization" in x:
        issues.append("x must be an ordinary tensor")
    if x_shape is None or not x_shape or _numel(x_shape) <= 0 or x_shape[-1] <= 0:
        issues.append("x must be non-empty with a positive last dimension")

    shapes: dict[str, list[int] | None] = {}
    quantized: set[str] = set()
    for name in ("gate_weight", "up_weight", "down_weight"):
        record = inputs.get(name, {})
        shape, descriptor_issues = _swiglu_weight_shape(name, record)
        shapes[name] = shape
        issues.extend(descriptor_issues)
        if "quantization" in record:
            quantized.add(name)
        if shape is None or len(shape) != 2 or _numel(shape) <= 0:
            issues.append(f"{name} must be a non-empty rank-2 tensor")

    gate_shape = shapes["gate_weight"]
    up_shape = shapes["up_weight"]
    down_shape = shapes["down_weight"]
    gate_rank2 = gate_shape is not None and len(gate_shape) == 2
    if gate_rank2 and up_shape is not None and up_shape != gate_shape:
        issues.append("gate_weight and up_weight shapes must match")
    if gate_rank2 and x_shape and x_shape[-1] != gate_shape[1]:
        issues.append("x last dimension must equal gate_weight in_features")
    if gate_rank2 and down_shape is not None and down_shape != [gate_shape[1], gate_shape[0]]:
        issues.append("down_weight shape must be [hidden_size, intermediate_size]")
    dtypes = {
        record.get("dtype")
        for name, record in (
            ("x", x),
            ("gate_weight", inputs.get("gate_weight", {})),
            ("up_weight", inputs.get("up_weight", {})),
            ("down_weight", inputs.get("down_weight", {})),
        )
        if name not in quantized and record.get("dtype") is not None
    }
    if len(dtypes) > 1:
        issues.append("x, gate_weight, up_weight, and down_weight dtypes must match")
    return issues


def _sdpa_inputs(inputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    query = inputs.get("query", {})
    key = inputs.get("key", {})
    value = inputs.get("value", {})
    q_shape, k_shape, v_shape = _shape(query), _shape(key), _shape(value)
    if q_shape is None or len(q_shape) != 4 or _numel(q_shape) <= 0:
        return ["query must be a non-empty rank-4 [batch, heads, seq, head_dim] tensor"]
    issues: list[str] = [
        f"{name} must be a non-empty rank-4 [batch, kv_heads, seq, head_dim] tensor"
        for name, shape in (("key", k_shape), ("value", v_shape))
        if shape is None or len(shape) != 4 or _numel(shape) <= 0
    ]
    if issues:
        return issues
    if k_shape[0] != q_shape[0] or v_shape[0] != q_shape[0]:
        issues.append("query, key, and value batch dimensions must match")
    if k_shape != v_shape:
        issues.append("key and value shapes must match")
    if k_shape[3] != q_shape[3] or v_shape[3] != q_shape[3]:
        issues.append("query, key, and value head dimensions must match")
    if q_shape[1] % k_shape[1]:
        issues.append(
            "query heads must be a multiple of key/value kv_heads (MHA, GQA, or MQA)"
        )
    dtypes = {
        record.get("dtype")
        for record in (query, key, value)
        if record.get("dtype") is not None
    }
    if len(dtypes) > 1:
        issues.append("query, key, and value dtypes must match")
    return issues


def _sdpa_outputs(outputs: Mapping[str, dict[str, Any]]) -> Sequence[str]:
    shape = _shape(outputs.get("attn_output"))
    if shape is None or len(shape) != 4 or _numel(shape) <= 0:
        return ["attn_output must be a non-empty rank-4 [batch, heads, seq, head_dim] tensor"]
    return []


def _sdpa_relation(
    inputs: Mapping[str, dict[str, Any]], outputs: Mapping[str, dict[str, Any]]
) -> Sequence[str]:
    query = inputs.get("query", {})
    attn_output = outputs.get("attn_output", {})
    issues: list[str] = []
    if _shape(query) and _shape(attn_output) and _shape(attn_output) != _shape(query):
        issues.append("attn_output.shape must equal query.shape")
    if query.get("dtype") is not None and attn_output.get("dtype") != query.get("dtype"):
        issues.append("attn_output.dtype must equal query.dtype")
    return issues


def _build_builtin_contracts() -> tuple[ExecutableContract, ...]:
    return tuple(
        sorted(
            (
                ExecutableContract(
                    id="linear/affine-int8/v1", inputs=("x", "weight"), outputs=("y",),
                    validate_inputs=_int8_linear_inputs, validate_relation=_int8_linear_relation,
                    description="Float32 x @ dequantized INT8 weight.T; no bias or integer accumulation",
                ),
                ExecutableContract(
                    id="linear/affine-int8-bias/v1", inputs=("x", "weight", "bias"), outputs=("y",),
                    validate_inputs=_int8_bias_linear_inputs, validate_relation=_int8_linear_relation,
                    description="Float32 x @ dequantized INT8 weight.T + float32 bias; no integer accumulation",
                ),
                ExecutableContract(
                    id="linear/affine-w4a16/v1", inputs=("x", "weight"), outputs=("y",),
                    validate_inputs=_w4a16_inputs, validate_relation=_w4a16_relation,
                    description="Float16 x, affine INT4/UINT4 weight, float32 dequant/matmul, float16 y; no bias",
                ),
                ExecutableContract(
                    id="linear/affine-w4a16-bf16/v1", inputs=("x", "weight"), outputs=("y",),
                    validate_inputs=_w4a16_bf16_inputs, validate_relation=_w4a16_bf16_relation,
                    description="Bfloat16 x, affine INT4/UINT4 weight, float32 dequant/matmul, one nearest-even bfloat16 y; no bias",
                ),
                ExecutableContract(
                    id="linear/nf4-w4a16/v1", inputs=("x", "weight"), outputs=("y",),
                    validate_inputs=_nf4_inputs, validate_relation=_w4a16_relation,
                    description="Float16 x, explicit NF4 codebook/block scales, float32 dequant/matmul, float16 y",
                ),
                ExecutableContract(
                    id="linear/nf4-w4a16-bf16/v1", inputs=("x", "weight"), outputs=("y",),
                    validate_inputs=_nf4_bf16_inputs, validate_relation=_w4a16_bf16_relation,
                    description="Bfloat16 x, explicit NF4 codebook/block scales, float32 dequant/matmul, one nearest-even bfloat16 y",
                ),
                ExecutableContract(
                    id="rmsnorm/last-dim/v1",
                    inputs=("x", "weight", "epsilon"),
                    outputs=("y",),
                    validate_inputs=_rmsnorm_inputs,
                    validate_outputs=_rmsnorm_outputs,
                    validate_relation=_y_matches_x_relation,
                    description="RMSNorm over the last dimension with weight and scalar epsilon",
                ),
                ExecutableContract(
                    id="swiglu/mlp/v1",
                    inputs=("x", "gate_weight", "up_weight", "down_weight"),
                    outputs=("y",),
                    validate_inputs=_swiglu_inputs,
                    validate_outputs=_y_output_issues,
                    validate_relation=_y_matches_x_relation,
                    description="SwiGLU MLP: down_weight @ (silu(gate_weight @ x) * (up_weight @ x)) over the last dimension",
                ),
                ExecutableContract(
                    id="attention/sdpa-gqa/v1",
                    inputs=("query", "key", "value"),
                    outputs=("attn_output",),
                    validate_inputs=_sdpa_inputs,
                    validate_outputs=_sdpa_outputs,
                    validate_relation=_sdpa_relation,
                    description="Scaled dot-product attention over rank-4 tensors with MHA, GQA, or MQA head grouping",
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
