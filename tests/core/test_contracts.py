"""Executable contract registry: roles, validators, and per-contract needs."""

from __future__ import annotations

import json
import warnings

# The shim intentionally emits DeprecationWarning; this module exists to prove
# the shim still works, so the warning is suppressed here (keeps `-W error`
# and filterwarnings=error runs green). The filter must be installed before
# the import below because the warning fires at import time.
warnings.filterwarnings(
    "ignore",
    message="inferref.testcase.contracts is deprecated",
    category=DeprecationWarning,
)

import pytest

from inferref.cli.main import EXIT_OK, main
from inferref.quantization.spec import QuantizationSpec
from inferref.testcase.contracts import (
    EXECUTABLE_CONTRACTS,
    contract_boundary_issues,
    contract_input_issues,
    contract_requirements,
    get_contract,
)


def _tensor(shape: list[int], dtype: str = "float32") -> dict[str, object]:
    return {"shape": shape, "dtype": dtype, "stride": list(range(len(shape)))}


def _quantized(
    spec: QuantizationSpec, dtype: str | None = None
) -> dict[str, object]:
    record = _tensor(list(spec.logical_shape), dtype or spec.storage_dtype)
    record["quantization"] = spec.to_dict()
    return record


def _affine_int4() -> QuantizationSpec:
    return QuantizationSpec(
        logical_shape=(2, 4),
        scale_value_id=1,
        scheme="affine-int4/v1",
        storage_dtype="int4",
        granularity="per_axis",
        axis=0,
    )


def _nf4() -> QuantizationSpec:
    return QuantizationSpec(
        logical_shape=(2, 4),
        scale_value_id=1,
        codebook_value_id=2,
        scheme="nf4/v1",
        storage_dtype="uint4",
        codebook="bitsandbytes-nf4/v1",
        granularity="per_block",
        group_size=4,
        rounding="none",
        saturation="none",
    )


def test_registry_covers_builtin_engine_contracts() -> None:
    assert {contract.id for contract in EXECUTABLE_CONTRACTS} == {
        "attention/sdpa-gqa/v1",
        "rmsnorm/last-dim/v1",
        "rope/rotate-half/v1",
        "kv-cache/append/v1",
        "kv-cache/indexed-update/v1",
        "linear/affine-int8/v1",
        "linear/affine-int8-bias/v1",
        "linear/affine-w4a16/v1",
        "linear/affine-w4a16-bf16/v1",
        "linear/nf4-w4a16/v1",
        "linear/nf4-w4a16-bf16/v1",
        "swiglu/mlp/v1",
    }
    assert get_contract("rmsnorm/last-dim/v1").inputs == ("x", "weight", "epsilon")
    assert get_contract("rmsnorm/last-dim/v1").outputs == ("y",)
    assert get_contract("rope/rotate-half/v1").inputs == (
        "query",
        "key",
        "cos",
        "sin",
    )
    assert get_contract("rope/rotate-half/v1").outputs == ("q_embed", "k_embed")
    assert get_contract("kv-cache/indexed-update/v1").inputs == (
        "cache",
        "update",
        "index",
    )
    assert get_contract("kv-cache/append/v1").inputs == ("cache", "update")
    assert get_contract("linear/affine-int8-bias/v1").inputs == ("x", "weight", "bias")
    assert get_contract("swiglu/mlp/v1").inputs == (
        "x",
        "gate_weight",
        "up_weight",
        "down_weight",
    )
    assert get_contract("swiglu/mlp/v1").outputs == ("y",)
    assert get_contract("attention/sdpa-gqa/v1").inputs == ("query", "key", "value")
    assert get_contract("attention/sdpa-gqa/v1").outputs == ("attn_output",)
    assert get_contract("softmax/last-dim/v1") is None


def test_contract_list_cli_reports_new_0_12_ids(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["contract", "list", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    ids = {entry["id"] for entry in payload["contracts"]}
    assert {
        "linear/affine-int8-bias/v1",
        "linear/affine-w4a16-bf16/v1",
        "linear/nf4-w4a16-bf16/v1",
        "swiglu/mlp/v1",
        "attention/sdpa-gqa/v1",
    } <= ids
    assert all(entry["source"] == "builtin" for entry in payload["contracts"])


def test_rmsnorm_validator_accepts_and_rejects_shapes() -> None:
    valid = {
        "x": _tensor([2, 3, 64]),
        "weight": _tensor([64]),
        "epsilon": _tensor([1]),
    }
    assert contract_input_issues("rmsnorm/last-dim/v1", valid) == []
    assert contract_input_issues(
        "rmsnorm/last-dim/v1",
        {"x": _tensor([2, 3, 64]), "weight": _tensor([32]), "epsilon": _tensor([1])},
    )
    assert contract_input_issues(
        "rmsnorm/last-dim/v1",
        {"x": _tensor([2, 3, 0]), "weight": _tensor([0]), "epsilon": _tensor([1])},
    )
    assert contract_input_issues(
        "rmsnorm/last-dim/v1",
        {"x": _tensor([2, 3, 64]), "weight": _tensor([64]), "epsilon": _tensor([0])},
    )
    assert contract_input_issues(
        "rmsnorm/last-dim/v1",
        {"x": _tensor([2, 3, 64]), "weight": _tensor([64]), "epsilon": _tensor([2])},
    )


def test_rope_validator_enforces_even_last_dim_and_cos_sin_contract() -> None:
    valid = {
        "query": _tensor([1, 4, 8]),
        "key": _tensor([1, 4, 8]),
        "cos": _tensor([4, 8]),
        "sin": _tensor([4, 8]),
    }
    assert contract_input_issues("rope/rotate-half/v1", valid) == []
    assert contract_input_issues(
        "rope/rotate-half/v1",
        {**valid, "query": _tensor([1, 4, 9])},
    )
    assert contract_input_issues(
        "rope/rotate-half/v1",
        {**valid, "cos": _tensor([4, 16])},
    )
    assert contract_input_issues(
        "rope/rotate-half/v1",
        {**valid, "sin": _tensor([4, 4])},
    )


def test_kv_validators_cover_indexed_and_append_roles() -> None:
    append = {"cache": _tensor([2, 4, 4, 8]), "update": _tensor([2, 4, 1, 8])}
    assert contract_input_issues("kv-cache/append/v1", append) == []
    indexed = {**append, "index": _tensor([1])}
    assert contract_input_issues("kv-cache/indexed-update/v1", indexed) == []
    assert contract_input_issues("kv-cache/indexed-update/v1", append)
    assert contract_input_issues(
        "kv-cache/append/v1",
        {"cache": _tensor([2, 4, 4, 8]), "update": _tensor([2, 4, 1, 16])},
    )
    assert contract_input_issues(
        "kv-cache/append/v1",
        {"cache": _tensor([2, 4, 0, 8]), "update": _tensor([2, 4, 1, 8])},
    )


def test_unknown_contract_has_no_registry_validation() -> None:
    assert contract_input_issues("softmax/last-dim/v1", {}) == ()


def test_boundary_requires_dtype_metadata_for_known_roles() -> None:
    inputs = {
        "x": {"shape": [2, 3, 64]},
        "weight": _tensor([64], dtype="float16"),
        "epsilon": _tensor([1], dtype="float32"),
    }
    outputs = {"y": _tensor([2, 3, 64], dtype="float16")}
    issues = contract_boundary_issues("rmsnorm/last-dim/v1", inputs, outputs)
    assert any("x must declare a non-empty dtype" in issue for issue in issues)

    outputs_missing = {"y": {"shape": [2, 3, 64]}}
    issues = contract_boundary_issues(
        "rmsnorm/last-dim/v1",
        {
            "x": _tensor([2, 3, 64], dtype="float16"),
            "weight": _tensor([64], dtype="float16"),
            "epsilon": _tensor([1], dtype="float32"),
        },
        outputs_missing,
    )
    assert any("y must declare a non-empty dtype" in issue for issue in issues)


def test_contract_requirements_are_role_scoped() -> None:
    manifest = {
        "inputs": [
            {"name": "query", "shape": [1, 4, 8], "dtype": "float16", "stride": []},
            {"name": "key", "shape": [1, 4, 8], "dtype": "float16", "stride": []},
            {"name": "cos", "shape": [4, 8], "dtype": "float32", "stride": []},
            {"name": "sin", "shape": [4, 8], "dtype": "float32", "stride": []},
            {"name": "other", "shape": [1, 999], "dtype": "bfloat16", "stride": []},
        ],
        "outputs": [
            {"name": "q_embed", "shape": [1, 4, 8], "dtype": "float16", "stride": []},
            {"name": "k_embed", "shape": [1, 4, 8], "dtype": "float16", "stride": []},
        ],
        "nodes": [],
    }
    requirements = contract_requirements(manifest, "rope/rotate-half/v1")
    assert requirements["dtypes"] == ["float16", "float32"]
    assert requirements["max_rank"] == 3
    assert "multiple_outputs" in requirements["features"]
    assert "bfloat16" not in requirements["dtypes"]


def test_contract_requirements_fallback_for_unknown_contract() -> None:
    manifest = {
        "inputs": [{"name": "x", "shape": [2, 3], "dtype": "float32", "stride": []}],
        "outputs": [{"name": "y", "shape": [2, 3], "dtype": "float32", "stride": []}],
        "nodes": [],
    }
    assert contract_requirements(manifest, "softmax/last-dim/v1") == {
        "dtypes": ["float32"],
        "max_rank": 2,
        "features": [],
    }


def test_rmsnorm_boundary_validates_output_shape_and_dtype() -> None:
    inputs = {
        "x": _tensor([2, 3, 64], dtype="float16"),
        "weight": _tensor([64], dtype="float16"),
        "epsilon": _tensor([1], dtype="float32"),
    }
    assert (
        contract_boundary_issues(
            "rmsnorm/last-dim/v1",
            inputs,
            {"y": _tensor([2, 3, 64], dtype="float16")},
        )
        == []
    )
    issues = contract_boundary_issues(
        "rmsnorm/last-dim/v1",
        inputs,
        {"y": _tensor([2, 64], dtype="float16")},
    )
    assert any("y.shape must equal x.shape" in issue for issue in issues)
    issues = contract_boundary_issues(
        "rmsnorm/last-dim/v1",
        inputs,
        {"y": _tensor([2, 3, 64], dtype="float32")},
    )
    assert any("y.dtype must equal x.dtype" in issue for issue in issues)


def test_rope_boundary_validates_both_output_branches() -> None:
    inputs = {
        "query": _tensor([1, 4, 8], dtype="float16"),
        "key": _tensor([1, 4, 8], dtype="float16"),
        "cos": _tensor([4, 8], dtype="float32"),
        "sin": _tensor([4, 8], dtype="float32"),
    }
    assert (
        contract_boundary_issues(
            "rope/rotate-half/v1",
            inputs,
            {
                "q_embed": _tensor([1, 4, 8], dtype="float16"),
                "k_embed": _tensor([1, 4, 8], dtype="float16"),
            },
        )
        == []
    )
    issues = contract_boundary_issues(
        "rope/rotate-half/v1",
        inputs,
        {
            "q_embed": _tensor([1, 4, 8], dtype="float16"),
            "k_embed": _tensor([1, 4, 4], dtype="float16"),
        },
    )
    assert any("k_embed.shape must equal key.shape" in issue for issue in issues)


def test_kv_boundary_validates_append_sequence_and_indexed_identity() -> None:
    append_inputs = {"cache": _tensor([2, 4, 4, 8]), "update": _tensor([2, 4, 1, 8])}
    assert (
        contract_boundary_issues(
            "kv-cache/append/v1",
            append_inputs,
            {"cache_out": _tensor([2, 4, 5, 8])},
        )
        == []
    )
    issues = contract_boundary_issues(
        "kv-cache/append/v1",
        append_inputs,
        {"cache_out": _tensor([2, 4, 4, 8])},
    )
    assert any("cache_out sequence length" in issue for issue in issues)

    indexed_inputs = {
        "cache": _tensor([2, 4, 4, 8]),
        "update": _tensor([2, 4, 1, 8]),
        "index": _tensor([1]),
    }
    assert (
        contract_boundary_issues(
            "kv-cache/indexed-update/v1",
            indexed_inputs,
            {"cache_out": _tensor([2, 4, 4, 8])},
        )
        == []
    )
    issues = contract_boundary_issues(
        "kv-cache/indexed-update/v1",
        indexed_inputs,
        {"cache_out": _tensor([2, 4, 5, 8])},
    )
    assert any("cache_out.shape must equal cache.shape" in issue for issue in issues)


def test_kv_boundary_requires_matching_dtypes() -> None:
    append_inputs = {
        "cache": _tensor([2, 4, 4, 8], dtype="float32"),
        "update": _tensor([2, 4, 1, 8], dtype="float16"),
    }
    issues = contract_boundary_issues(
        "kv-cache/append/v1",
        append_inputs,
        {"cache_out": _tensor([2, 4, 5, 8], dtype="float32")},
    )
    assert any("dtypes must match" in issue for issue in issues)

    indexed_inputs = {
        "cache": _tensor([2, 4, 4, 8], dtype="float32"),
        "update": _tensor([2, 4, 1, 8], dtype="float32"),
        "index": _tensor([1]),
    }
    issues = contract_boundary_issues(
        "kv-cache/indexed-update/v1",
        indexed_inputs,
        {"cache_out": _tensor([2, 4, 4, 8], dtype="float16")},
    )
    assert any("dtypes must match" in issue for issue in issues)


def test_affine_int8_bias_validator_requires_float32_bias_of_out_features() -> None:
    spec = QuantizationSpec(logical_shape=(3, 4), scale_value_id=1)
    inputs = {
        "x": _tensor([2, 4]),
        "weight": _quantized(spec, "int8"),
        "bias": _tensor([3]),
    }
    assert contract_input_issues("linear/affine-int8-bias/v1", inputs) == []
    assert (
        contract_boundary_issues(
            "linear/affine-int8-bias/v1", inputs, {"y": _tensor([2, 3])}
        )
        == []
    )

    missing_bias = {name: record for name, record in inputs.items() if name != "bias"}
    assert any(
        "bias must be an ordinary float32 tensor" in issue
        for issue in contract_input_issues("linear/affine-int8-bias/v1", missing_bias)
    )
    for bias in (
        _tensor([3], "float16"),
        {**_tensor([3]), "quantization": spec.to_dict()},
    ):
        assert any(
            "bias must be an ordinary float32 tensor" in issue
            for issue in contract_input_issues(
                "linear/affine-int8-bias/v1", {**inputs, "bias": bias}
            )
        )
    assert any(
        "bias must have rank 1" in issue
        for issue in contract_input_issues(
            "linear/affine-int8-bias/v1", {**inputs, "bias": _tensor([3, 1])}
        )
    )
    assert any(
        "bias length must equal weight out_features" in issue
        for issue in contract_input_issues(
            "linear/affine-int8-bias/v1", {**inputs, "bias": _tensor([2])}
        )
    )
    assert any(
        "y must be an ordinary float32 tensor" in issue
        for issue in contract_boundary_issues(
            "linear/affine-int8-bias/v1", inputs, {"y": _tensor([2, 3], "float16")}
        )
    )
    assert any(
        "y shape must equal" in issue
        for issue in contract_boundary_issues(
            "linear/affine-int8-bias/v1", inputs, {"y": _tensor([3])}
        )
    )


@pytest.mark.parametrize(
    "spec",
    [
        QuantizationSpec(
            logical_shape=(3, 4), scale_value_id=1, granularity="per_axis", axis=0
        ),
        QuantizationSpec(
            logical_shape=(3, 4),
            scale_value_id=1,
            granularity="per_group",
            axis=1,
            group_size=2,
        ),
    ],
    ids=["per-axis", "per-group"],
)
def test_affine_int8_bias_accepts_per_axis_and_per_group_weights(
    spec: QuantizationSpec,
) -> None:
    inputs = {
        "x": _tensor([2, 4]),
        "weight": _quantized(spec, "int8"),
        "bias": _tensor([3]),
    }
    assert contract_input_issues("linear/affine-int8-bias/v1", inputs) == []
    assert (
        contract_boundary_issues(
            "linear/affine-int8-bias/v1", inputs, {"y": _tensor([2, 3])}
        )
        == []
    )


def test_affine_int8_bias_contract_is_the_biasless_contract_plus_bias() -> None:
    spec = QuantizationSpec(logical_shape=(3, 4), scale_value_id=1)
    without_bias = {"x": _tensor([2, 4]), "weight": _quantized(spec, "int8")}
    with_bias = {**without_bias, "bias": _tensor([3])}
    outputs = {"y": _tensor([2, 3])}
    assert contract_input_issues("linear/affine-int8/v1", without_bias) == []
    assert contract_boundary_issues("linear/affine-int8/v1", without_bias, outputs) == []
    assert contract_input_issues("linear/affine-int8-bias/v1", with_bias) == []
    assert contract_boundary_issues("linear/affine-int8-bias/v1", with_bias, outputs) == []
    # The bias profile requires bias; it is not interchangeable with the
    # biasless profile for a testcase that carries no bias tensor.
    assert any(
        "bias" in issue
        for issue in contract_input_issues("linear/affine-int8-bias/v1", without_bias)
    )


@pytest.mark.parametrize(
    ("contract_id", "spec"),
    [
        ("linear/affine-w4a16-bf16/v1", _affine_int4()),
        ("linear/nf4-w4a16-bf16/v1", _nf4()),
    ],
)
def test_bf16_w4a16_validator_accepts_bfloat16_and_rejects_float16(
    contract_id: str, spec: QuantizationSpec
) -> None:
    inputs = {"x": _tensor([1, 4], "bfloat16"), "weight": _quantized(spec)}
    assert contract_input_issues(contract_id, inputs) == []
    outputs = {"y": _tensor([1, 2], "bfloat16")}
    assert contract_boundary_issues(contract_id, inputs, outputs) == []

    float16_io = {"x": _tensor([1, 4], "float16"), "weight": _quantized(spec)}
    assert any(
        "x must be ordinary bfloat16" in issue
        for issue in contract_input_issues(contract_id, float16_io)
    )
    assert any(
        "y must be ordinary bfloat16" in issue
        for issue in contract_boundary_issues(contract_id, inputs, {"y": _tensor([1, 2], "float16")})
    )
    assert any(
        "x last dimension must equal weight in_features" in issue
        for issue in contract_input_issues(
            contract_id, {"x": _tensor([1, 5], "bfloat16"), "weight": _quantized(spec)}
        )
    )
    assert any(
        "y shape must equal" in issue
        for issue in contract_boundary_issues(contract_id, inputs, {"y": _tensor([1, 3], "bfloat16")})
    )


@pytest.mark.parametrize(
    ("contract_id", "wrong_spec", "message"),
    [
        ("linear/affine-w4a16-bf16/v1", _nf4(), "weight must be affine INT4/UINT4"),
        ("linear/nf4-w4a16-bf16/v1", _affine_int4(), "weight must be NF4"),
    ],
)
def test_bf16_w4a16_validator_rejects_the_other_weight_family(
    contract_id: str, wrong_spec: QuantizationSpec, message: str
) -> None:
    issues = contract_input_issues(
        contract_id, {"x": _tensor([1, 4], "bfloat16"), "weight": _quantized(wrong_spec)}
    )
    assert any(message in issue for issue in issues)


def test_existing_float16_contracts_still_reject_bfloat16_activations() -> None:
    for contract_id, spec in (
        ("linear/affine-w4a16/v1", _affine_int4()),
        ("linear/nf4-w4a16/v1", _nf4()),
    ):
        issues = contract_input_issues(
            contract_id, {"x": _tensor([1, 4], "bfloat16"), "weight": _quantized(spec)}
        )
        assert any("float16" in issue for issue in issues)


@pytest.mark.parametrize(
    ("contract_id", "spec"),
    [
        ("linear/affine-w4a16-bf16/v1", _affine_int4()),
        ("linear/nf4-w4a16-bf16/v1", _nf4()),
    ],
)
def test_bf16_w4a16_validator_rejects_a_quantized_activation(
    contract_id: str, spec: QuantizationSpec
) -> None:
    activation = QuantizationSpec(logical_shape=(1, 4), scale_value_id=9)
    inputs = {
        "x": {**_tensor([1, 4], "bfloat16"), "quantization": activation.to_dict()},
        "weight": _quantized(spec),
    }
    assert any(
        "x must be ordinary bfloat16" in issue
        for issue in contract_input_issues(contract_id, inputs)
    )


def _swiglu_boundary() -> dict[str, dict[str, object]]:
    return {
        "x": _tensor([2, 3, 8], "float16"),
        "gate_weight": _tensor([16, 8], "float16"),
        "up_weight": _tensor([16, 8], "float16"),
        "down_weight": _tensor([8, 16], "float16"),
    }


def test_swiglu_validator_accepts_a_consistent_mlp() -> None:
    inputs = _swiglu_boundary()
    assert contract_input_issues("swiglu/mlp/v1", inputs) == []
    assert (
        contract_boundary_issues(
            "swiglu/mlp/v1", inputs, {"y": _tensor([2, 3, 8], "float16")}
        )
        == []
    )


def test_swiglu_validator_rejects_inconsistent_shapes_dtypes_and_outputs() -> None:
    inputs = _swiglu_boundary()
    cases = [
        ({"up_weight": _tensor([32, 8], "float16")}, "gate_weight and up_weight shapes must match"),
        ({"x": _tensor([2, 3, 7], "float16")}, "x last dimension must equal gate_weight in_features"),
        ({"down_weight": _tensor([16, 8], "float16")}, "down_weight shape must be [hidden_size, intermediate_size]"),
        ({"gate_weight": _tensor([16, 8, 1], "float16")}, "gate_weight must be a non-empty rank-2 tensor"),
        (
            {"down_weight": {**_tensor([8, 16], "float16"), "quantization": {}}},
            "down_weight has an invalid quantization descriptor",
        ),
        ({"up_weight": _tensor([16, 8], "float32")}, "dtypes must match"),
    ]
    for overrides, message in cases:
        issues = contract_input_issues("swiglu/mlp/v1", {**inputs, **overrides})
        assert any(message in issue for issue in issues), (overrides, issues)

    for outputs, message in (
        ({"y": _tensor([3, 8], "float16")}, "y.shape must equal x.shape"),
        ({"y": _tensor([2, 3, 8], "float32")}, "y.dtype must equal x.dtype"),
    ):
        issues = contract_boundary_issues("swiglu/mlp/v1", inputs, outputs)
        assert any(message in issue for issue in issues), (outputs, issues)


def test_swiglu_validator_accepts_quantized_weights_by_logical_shape() -> None:
    inputs = _swiglu_boundary()
    # The record dtype of a quantized weight is its storage dtype (int8 here),
    # so only the descriptor's logical shape joins the shape checks and the
    # storage dtype is exempt from the activation dtype agreement.
    quantized_down = QuantizationSpec(logical_shape=(8, 16), scale_value_id=5)
    inputs["down_weight"] = _quantized(quantized_down, "int8")
    assert contract_input_issues("swiglu/mlp/v1", inputs) == []
    assert (
        contract_boundary_issues(
            "swiglu/mlp/v1", inputs, {"y": _tensor([2, 3, 8], "float16")}
        )
        == []
    )

    wrong_down = QuantizationSpec(logical_shape=(16, 8), scale_value_id=5)
    inputs["down_weight"] = _quantized(wrong_down, "int8")
    assert any(
        "down_weight shape must be [hidden_size, intermediate_size]" in issue
        for issue in contract_input_issues("swiglu/mlp/v1", inputs)
    )


def test_swiglu_validator_rejects_an_absent_or_shape_less_y() -> None:
    inputs = _swiglu_boundary()
    for outputs in ({}, {"y": {}}):
        issues = contract_boundary_issues("swiglu/mlp/v1", inputs, outputs)
        assert any("y must be a non-empty tensor" in issue for issue in issues), (
            outputs,
            issues,
        )


def _sdpa_inputs(kv_heads: int) -> dict[str, dict[str, object]]:
    return {
        "query": _tensor([2, 8, 5, 4], "float16"),
        "key": _tensor([2, kv_heads, 7, 4], "float16"),
        "value": _tensor([2, kv_heads, 7, 4], "float16"),
    }


@pytest.mark.parametrize("kv_heads", [8, 2, 1])
def test_sdpa_validator_accepts_mha_gqa_and_mqa(kv_heads: int) -> None:
    inputs = _sdpa_inputs(kv_heads)
    assert contract_input_issues("attention/sdpa-gqa/v1", inputs) == []
    assert (
        contract_boundary_issues(
            "attention/sdpa-gqa/v1", inputs, {"attn_output": _tensor([2, 8, 5, 4], "float16")}
        )
        == []
    )


def test_sdpa_validator_accepts_cross_attention_sequence_lengths() -> None:
    # seq_q != seq_kv is ordinary cross-attention, not a GQA violation.
    inputs = {
        "query": _tensor([2, 8, 3, 4], "float16"),
        "key": _tensor([2, 2, 11, 4], "float16"),
        "value": _tensor([2, 2, 11, 4], "float16"),
    }
    assert contract_input_issues("attention/sdpa-gqa/v1", inputs) == []
    assert (
        contract_boundary_issues(
            "attention/sdpa-gqa/v1",
            inputs,
            {"attn_output": _tensor([2, 8, 3, 4], "float16")},
        )
        == []
    )


def test_sdpa_validator_rejects_kv_heads_above_query_heads() -> None:
    issues = contract_input_issues("attention/sdpa-gqa/v1", _sdpa_inputs(16))
    assert any(
        "query heads must be a multiple of key/value kv_heads" in issue
        for issue in issues
    )


def test_sdpa_validator_rejects_bad_grouping_ranks_and_dtypes() -> None:
    cases = [
        (_sdpa_inputs(3), "query heads must be a multiple of key/value kv_heads"),
        (
            {
                "key": _tensor([2, 2, 7, 8], "float16"),
                "value": _tensor([2, 2, 7, 8], "float16"),
            },
            "head dimensions must match",
        ),
        (
            {
                "key": _tensor([2, 2, 7, 4], "float16"),
                "value": _tensor([2, 2, 6, 4], "float16"),
            },
            "key and value shapes must match",
        ),
        (
            {
                "key": _tensor([3, 2, 7, 4], "float16"),
                "value": _tensor([3, 2, 7, 4], "float16"),
            },
            "batch dimensions must match",
        ),
        ({"query": _tensor([8, 5, 4], "float16")}, "query must be a non-empty rank-4"),
        ({"value": _tensor([2, 2, 7], "float16")}, "value must be a non-empty rank-4"),
        (
            {
                "key": _tensor([2, 2, 7, 4], "float32"),
                "value": _tensor([2, 2, 7, 4], "float32"),
            },
            "dtypes must match",
        ),
    ]
    for overrides, message in cases:
        issues = contract_input_issues(
            "attention/sdpa-gqa/v1", {**_sdpa_inputs(2), **overrides}
        )
        assert any(message in issue for issue in issues), (overrides, issues)

    inputs = _sdpa_inputs(2)
    for outputs, message in (
        (
            {"attn_output": _tensor([2, 8, 5], "float16")},
            "attn_output must be a non-empty rank-4",
        ),
        (
            {"attn_output": _tensor([2, 8, 6, 4], "float16")},
            "attn_output.shape must equal query.shape",
        ),
        (
            {"attn_output": _tensor([2, 8, 5, 4], "float32")},
            "attn_output.dtype must equal query.dtype",
        ),
    ):
        issues = contract_boundary_issues("attention/sdpa-gqa/v1", inputs, outputs)
        assert any(message in issue for issue in issues), (outputs, issues)
