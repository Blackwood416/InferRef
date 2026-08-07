"""Executable contract registry: roles, validators, and per-contract needs."""

from __future__ import annotations

from inferref.testcase.contracts import (
    EXECUTABLE_CONTRACTS,
    contract_boundary_issues,
    contract_input_issues,
    contract_requirements,
    get_contract,
)


def _tensor(shape: list[int], dtype: str = "float32") -> dict[str, object]:
    return {"shape": shape, "dtype": dtype, "stride": list(range(len(shape)))}


def test_registry_covers_the_four_engine_contracts() -> None:
    assert {contract.id for contract in EXECUTABLE_CONTRACTS} == {
        "rmsnorm/last-dim/v1",
        "rope/rotate-half/v1",
        "kv-cache/append/v1",
        "kv-cache/indexed-update/v1",
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
    assert get_contract("softmax/last-dim/v1") is None


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
