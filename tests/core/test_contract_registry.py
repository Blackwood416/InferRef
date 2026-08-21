"""Contract Schema v0.1 registry, CLI, doctor and preflight tests.

Everything here runs without torch: the registry depends on the standard
library only, exactly like the core suite it lives in.
"""

from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from inferref.agent.adapter import execute_adapter
from inferref.agent.protocol import (
    ENGINE_ADAPTER_FORMAT,
    ENGINE_ADAPTER_VERSION,
    EngineAdapter,
)
from inferref.cli.main import EXIT_FAIL, EXIT_OK, main
from inferref.contracts import (
    EXECUTABLE_CONTRACTS,
    REGISTRY,
    ContractSchemaError,
    ExecutableContract,
    contract_boundary_issues,
    contract_input_issues,
    contract_list,
    contract_plugin_statuses,
    contract_requirements,
    get_contract,
    load_contract_file,
    validate_contract_file,
    verify_contracts,
)
from inferref.contracts import registry as contract_registry
from inferref.contracts.relations import (
    RelationEvaluationError,
    RelationSyntaxError,
    evaluate,
    parse,
    relation_roles,
)
from inferref.contracts.schema import build_contract
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.tensor_value import TensorValueRecord
from inferref.ir.values import TensorRef
from inferref.tensor import codec
from inferref.testcase.extract import ExtractionError, extract_operator
from inferref.testcase.requirements import derive_requirements, is_contract_id
from inferref.testcase.validate import validate_testcase

REPO_ROOT = Path(__file__).resolve().parents[2]

SWIGLU: dict[str, object] = {
    "format": "inferref-contract",
    "format_version": "0.1",
    "id": "swiglu/fused/v1",
    "description": "SwiGLU fusion over the last dimension",
    "inputs": {"x": {"kind": "tensor"}, "gate": {"kind": "tensor"}},
    "outputs": {"y": {"kind": "tensor"}},
    "relations": [
        "y.shape == x.shape",
        "y.dtype == x.dtype",
        "gate.shape == x.shape",
    ],
    "features": ["multiple_outputs"],
    "effects": ["pure"],
}

GDN: dict[str, object] = {
    "format": "inferref-contract",
    "format_version": "0.1",
    "id": "gated-deltanet/step/v1",
    "description": "one Gated DeltaNet recurrence step",
    "inputs": {"state": {"kind": "tensor"}, "x": {"kind": "tensor"}},
    "outputs": {"state_out": {"kind": "tensor"}},
    "relations": ["state_out.shape == state.shape", "state_out.dtype == state.dtype"],
}


def _tensor(
    shape: list[int], dtype: str = "float32"
) -> dict[str, object]:
    return {"shape": shape, "dtype": dtype, "stride": []}


def _descriptor(**overrides: object) -> dict[str, object]:
    descriptor = {**SWIGLU}
    descriptor.update(overrides)
    return descriptor


# -- fake entry-point distribution ----------------------------------------


class Distribution:
    def __init__(self, name: str = "inferref-qwen", version: str = "0.1.0") -> None:
        self.name = name
        self.version = version


class Entry:
    def __init__(
        self,
        name: str,
        value: str,
        factory: object,
        dist: Distribution | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self._factory = factory
        self.dist = dist or Distribution()

    def load(self) -> object:
        return self._factory


class Entries(list):
    def select(self, *, group: str) -> Entries:
        if group == contract_registry.ENTRY_POINT_GROUP:
            return Entries(self)
        return Entries()


def _factory(*descriptors: object) -> object:
    def build() -> list[object]:
        return list(descriptors)

    return build


_real_entry_points = contract_registry.metadata.entry_points


def _install_entry_points(
    monkeypatch: pytest.MonkeyPatch, entries: list[Entry]
) -> None:
    def fake_entry_points(group: str | None = None, **kwargs: object) -> object:
        if group == contract_registry.ENTRY_POINT_GROUP:
            return Entries(entries)
        if group is not None:
            try:
                return _real_entry_points(group=group, **kwargs)
            except Exception:
                return Entries()
        return Entries(entries)

    monkeypatch.setattr(
        contract_registry.metadata, "entry_points", fake_entry_points
    )


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    contract_registry._reset_registry()
    yield
    contract_registry._reset_registry()


# -- schema ----------------------------------------------------------------


def test_schema_accepts_swiglu_example_and_ignores_unknown_fields() -> None:
    contract = build_contract(
        _descriptor(author="fixture-pack", notes=["forward compatible"])
    )
    assert contract.id == "swiglu/fused/v1"
    assert contract.inputs == ("x", "gate")
    assert contract.outputs == ("y",)
    assert contract.features == ("multiple_outputs",)
    assert contract.effects == ("pure",)
    assert contract.to_dict()["relations"] == 3
    assert contract.description == "SwiGLU fusion over the last dimension"


def test_schema_requires_format_and_format_version() -> None:
    with pytest.raises(ContractSchemaError, match="format"):
        build_contract(_descriptor(format="other"))
    with pytest.raises(ContractSchemaError, match="format_version"):
        build_contract(_descriptor(format_version="0.2"))
    with pytest.raises(ContractSchemaError, match="object"):
        build_contract("not a descriptor")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "contract_id",
    [
        "SwiGLU/fused/v1",
        "swiglu/fused/v",
        "swiglu/fused/v1x",
        "swiglu.fused/v1",
        "swiglu/_x/v1",
        "a/b/c",
        "a/b/v1/extra",
    ],
)
def test_schema_rejects_invalid_contract_ids(contract_id: str) -> None:
    with pytest.raises(ContractSchemaError, match="invalid contract id"):
        build_contract(_descriptor(id=contract_id))


def test_schema_accepts_version_zero_and_short_ids() -> None:
    for contract_id in ("a/b/v0", "a/v1"):
        contract = build_contract(
            {
                "format": "inferref-contract",
                "format_version": "0.1",
                "id": contract_id,
                "inputs": {"x": {"kind": "tensor"}},
                "outputs": {"y": {"kind": "tensor"}},
            }
        )
        assert contract.id == contract_id
    assert contract.to_dict().get("relations") is None


def test_schema_enforces_role_rules() -> None:
    with pytest.raises(ContractSchemaError, match="role name"):
        build_contract(_descriptor(inputs={"x-1": {"kind": "tensor"}}))
    with pytest.raises(ContractSchemaError, match="must not be empty"):
        build_contract(_descriptor(inputs={}))
    with pytest.raises(ContractSchemaError, match="outputs"):
        build_contract(_descriptor(outputs={}))
    with pytest.raises(ContractSchemaError, match="kind"):
        build_contract(_descriptor(inputs={"x": {"kind": "scalar"}}))


def test_schema_rejects_shared_input_and_output_role_names() -> None:
    descriptor = _descriptor(
        inputs={"x": {"kind": "tensor"}, "gate": {"kind": "tensor"}},
        outputs={"x": {"kind": "tensor"}},
    )
    with pytest.raises(ContractSchemaError, match="cannot disambiguate"):
        build_contract(descriptor)
    # The Python escape hatch is held to the same rule.
    contract = ExecutableContract(
        id="custom/shared/v1",
        inputs=("x",),
        outputs=("x",),
        validate_inputs=lambda inputs: [],
    )
    with pytest.raises(ContractSchemaError, match="cannot disambiguate"):
        contract_registry._coerce_descriptor(contract, "custom")


def test_schema_rejects_bad_relation_syntax_and_roles() -> None:
    with pytest.raises(ContractSchemaError, match="does not parse"):
        build_contract(_descriptor(relations=["y.shape = x.shape"]))
    with pytest.raises(ContractSchemaError, match="undeclared role"):
        build_contract(_descriptor(relations=["y.shape == z.shape"]))
    with pytest.raises(ContractSchemaError, match="relations must be a string array"):
        build_contract(_descriptor(relations=["y.shape == x.shape", 1]))


def test_schema_enforces_feature_and_effect_vocabulary() -> None:
    with pytest.raises(ContractSchemaError, match="unknown features"):
        build_contract(_descriptor(features=["not_a_feature"]))
    with pytest.raises(ContractSchemaError, match="unknown effects"):
        build_contract(_descriptor(effects=["explosive"]))
    with pytest.raises(ContractSchemaError, match="must not coexist"):
        build_contract(_descriptor(effects=["pure", "mutation_effects"]))


def test_schema_effect_mapping_merges_into_features() -> None:
    contract = build_contract(
        _descriptor(features=["strided_inputs"], effects=["alias_effects"])
    )
    assert contract.features == ("alias_effects", "strided_inputs")
    assert contract.effects == ("alias_effects",)


def test_schema_rejects_pure_with_effect_features_across_fields() -> None:
    with pytest.raises(ContractSchemaError, match="must not coexist"):
        build_contract(_descriptor(features=["mutation_effects"], effects=["pure"]))
    with pytest.raises(ContractSchemaError, match="must not coexist"):
        build_contract(_descriptor(features=["alias_effects"], effects=["pure"]))
    # Python-constructed contracts get the same merged-set check.
    contract = ExecutableContract(
        id="custom/pure/v1",
        inputs=("x",),
        outputs=("y",),
        validate_inputs=lambda inputs: [],
        features=("mutation_effects",),
        effects=("pure",),
    )
    with pytest.raises(ContractSchemaError, match="must not coexist"):
        contract_registry._coerce_descriptor(contract, "custom")


def test_contract_id_requirements_helper_matches_section_4() -> None:
    assert is_contract_id("rmsnorm/last-dim/v1")
    assert is_contract_id("swiglu/fused/v0")
    assert is_contract_id("a/v1")
    assert not is_contract_id("qwen2.5/rope/v2")


# -- relations -------------------------------------------------------------


def _relation_roles() -> dict[str, dict[str, object]]:
    return {
        "x": _tensor([2, 3, 5]),
        "y": _tensor([2, 3, 6], dtype="float16"),
        "gate": _tensor([2, 3, 5]),
    }


def test_relation_evaluator_covers_every_operator() -> None:
    roles = _relation_roles()
    assert evaluate("gate.shape == x.shape", roles) == (True, None)
    assert evaluate("y.dtype != x.dtype", roles) == (True, None)
    assert evaluate("y.shape[0] == x.shape[0]", roles) == (True, None)
    assert evaluate("y.shape[-1] != x.shape[-1]", roles) == (True, None)
    assert evaluate("y.rank == 3", roles) == (True, None)
    assert evaluate("y.numel == 36", roles) == (True, None)
    assert evaluate("len(y.shape) == len(x.shape)", roles) == (True, None)
    assert evaluate('not (y.dtype == "float32")', roles) == (True, None)
    assert evaluate("x.rank == 3 and y.rank == 3", roles) == (True, None)
    assert evaluate("x.rank == 2 or y.rank == 3", roles) == (True, None)


def test_relation_failure_message_matches_spec_format() -> None:
    roles = {
        "x": _tensor([2, 3, 6]),
        "y": _tensor([2, 3, 5]),
    }
    holds, message = evaluate("y.shape == x.shape", roles)
    assert holds is False
    assert message == (
        "relation 'y.shape == x.shape' failed (y.shape=[2,3,5], x.shape=[2,3,6])"
    )


def test_relation_short_circuits_and_or() -> None:
    roles = {"x": _tensor([2, 3, 5]), "y": {"shape": [2, 3, 5]}}
    assert evaluate('x.rank == 3 or y.dtype == "float32"', roles) == (True, None)
    holds, message = evaluate('x.rank == 99 and y.dtype == "float32"', roles)
    assert holds is False
    assert "x.rank=3, 99=99" in message
    with pytest.raises(RelationEvaluationError, match="y.dtype"):
        evaluate('x.rank == 99 or y.dtype == "float32"', roles)


def test_relation_syntax_errors_are_reported() -> None:
    for bad in (
        "y.shape > x.shape",
        "y.shape = x.shape",
        "x.shape[",
        "x.shape[0][1]",
        "x[0] == 2",
        'x.dtype[0] == "f"',
        "x.rank[0] == 2",
        "x.numel[0] == 6",
        "y.shape == ",
        'y.dtype == "float32',
        "y.shape == x.shape == z.shape",
    ):
        with pytest.raises(RelationSyntaxError):
            parse(bad)


def test_relation_multiple_indexes_and_bare_role_indexing_are_schema_errors() -> None:
    for expression in (
        "y.shape[0][1] == x.shape[0]",
        "x[0] == 2",
        'x.dtype[0] == "f"',
        "x.rank[0] == 2",
        "x.numel[0] == 6",
    ):
        with pytest.raises(ContractSchemaError, match="does not parse"):
            build_contract(_descriptor(relations=[expression]))


def test_relation_role_collection_is_sorted_and_unique() -> None:
    assert relation_roles("y.shape[0] == x.shape[0] and len(gate.shape) == 2") == (
        "gate",
        "x",
        "y",
    )


def test_generated_validators_split_input_and_mixed_relations() -> None:
    contract = build_contract(_descriptor())
    inputs_bad_gate = {"x": _tensor([2, 3, 64]), "gate": _tensor([2, 3, 63])}
    issues = contract.validate_inputs(inputs_bad_gate)
    assert any("gate.shape == x.shape" in issue for issue in issues)
    issues = contract.validate_inputs(
        {"x": _tensor([2, 3, 64]), "gate": _tensor([2, 3, 64])}
    )
    assert issues == []
    inputs_mixed = {"x": _tensor([2, 3, 63]), "gate": _tensor([2, 3, 63])}
    assert contract.validate_inputs(inputs_mixed) == []
    issues = contract.validate_relation(inputs_mixed, {"y": _tensor([2, 3, 64])})
    assert any("y.shape == x.shape" in issue for issue in issues)


# -- discovery -------------------------------------------------------------


def test_builtins_are_always_registered_in_sorted_order() -> None:
    assert [entry.id for entry in contract_list()] == [
        "kv-cache/append/v1",
        "kv-cache/indexed-update/v1",
        "rmsnorm/last-dim/v1",
        "rope/rotate-half/v1",
    ]
    assert [contract.id for contract in EXECUTABLE_CONTRACTS] == [
        "kv-cache/append/v1",
        "kv-cache/indexed-update/v1",
        "rmsnorm/last-dim/v1",
        "rope/rotate-half/v1",
    ]
    assert get_contract("rope/rotate-half/v1").inputs == (
        "query",
        "key",
        "cos",
        "sin",
    )
    assert get_contract("softmax/last-dim/v1") is None


def test_plugin_discovery_loads_and_lists_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [
            Entry(
                "qwen",
                "inferref_qwen.contracts:build",
                _factory(_descriptor(), GDN),
                Distribution("inferref-qwen"),
            )
        ],
    )
    assert get_contract("swiglu/fused/v1").inputs == ("x", "gate")
    assert get_contract("gated-deltanet/step/v1").outputs == ("state_out",)
    entries = contract_list()
    assert [entry.id for entry in entries][:4] == [
        "kv-cache/append/v1",
        "kv-cache/indexed-update/v1",
        "rmsnorm/last-dim/v1",
        "rope/rotate-half/v1",
    ]
    assert [entry.id for entry in entries][4:] == [
        "gated-deltanet/step/v1",
        "swiglu/fused/v1",
    ]
    assert all(entry.status == "loaded" for entry in entries)
    assert set(REGISTRY) == {
        "kv-cache/append/v1",
        "kv-cache/indexed-update/v1",
        "rmsnorm/last-dim/v1",
        "rope/rotate-half/v1",
        "gated-deltanet/step/v1",
        "swiglu/fused/v1",
    }


def test_plugins_sort_by_distribution_then_entry_then_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [
            Entry(
                "zep",
                "b.pack:build",
                _factory(_descriptor(id="z/z/v1")),
                Distribution("b-pack"),
            ),
            Entry(
                "aep",
                "a.pack:build",
                _factory(_descriptor(id="a/a/v1")),
                Distribution("a-pack"),
            ),
        ],
    )
    ids = [entry.id for entry in contract_list() if entry.source == "plugin"]
    assert ids == ["a/a/v1", "z/z/v1"]


def test_duplicate_entry_point_name_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [
            Entry("shared", "a.pack:build", _factory(_descriptor())),
            Entry(
                "shared", "b.pack:build", _factory(_descriptor()), Distribution("b")
            ),
        ],
    )
    statuses = verify_contracts()
    assert [status.status for status in statuses] == ["error", "error"]
    assert all(
        "duplicate contract entry-point name" in status.error
        for status in statuses
    )
    assert get_contract("swiglu/fused/v1") is None


def test_entry_point_named_builtin_pack_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [Entry("builtin", "evil.pack:build", _factory(_descriptor()))],
    )
    status = verify_contracts()[0]
    assert status.status == "error"
    assert "contract_shadows_builtin" in status.error
    assert get_contract("swiglu/fused/v1") is None


def test_duplicate_contract_ids_are_rejected_from_both_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [
            Entry("a", "a.pack:build", _factory(_descriptor()), Distribution("a")),
            Entry("b", "b.pack:build", _factory(_descriptor()), Distribution("b")),
        ],
    )
    statuses = verify_contracts()
    assert [status.status for status in statuses] == ["error", "error"]
    assert all("contract_duplicate_id" in status.error for status in statuses)
    plugin_entries = [
        entry for entry in contract_list() if entry.source == "plugin"
    ]
    assert [entry.status for entry in plugin_entries] == ["error", "error"]
    assert get_contract("swiglu/fused/v1") is None


def test_contract_shadowing_a_builtin_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [
            Entry(
                "qwen",
                "qwen.pack:build",
                _factory(_descriptor(id="rmsnorm/last-dim/v1")),
            )
        ],
    )
    status = verify_contracts()[0]
    assert status.status == "error"
    assert "contract_shadows_builtin" in status.error
    assert get_contract("rmsnorm/last-dim/v1").description == (
        "RMSNorm over the last dimension with weight and scalar epsilon"
    )


def test_factory_failure_is_reported_without_breaking_other_plugins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> None:
        raise RuntimeError("boom")

    _install_entry_points(
        monkeypatch,
        [
            Entry(
                "broken", "broken.pack:build", broken, Distribution("broken")
            ),
            Entry(
                "good", "good.pack:build", _factory(_descriptor()), Distribution("good")
            ),
        ],
    )
    statuses = verify_contracts()
    by_name = {status.entry_point: status for status in statuses}
    assert by_name["broken"].status == "error"
    assert "contract_entry_point_error: RuntimeError: boom" in by_name["broken"].error
    assert by_name["good"].status == "loaded"
    assert get_contract("swiglu/fused/v1") is not None


def test_invalid_descriptor_and_non_iterable_factory_are_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [
            Entry("bad-desc", "x.pack:build", _factory({"format": "nope"})),
            Entry("non-iter", "y.pack:build", lambda: 42),
            Entry("non-factory", "z.pack:build", "not callable"),
        ],
    )
    statuses = verify_contracts()
    assert all(status.status == "error" for status in statuses)
    messages = " | ".join(status.error or "" for status in statuses)
    assert "contract_schema_invalid" in messages
    assert "did not return an iterable" in messages
    assert "is not a factory" in messages


def test_generator_factory_failure_discards_partial_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build() -> object:
        yield _descriptor(id="ok/one/v1")
        raise RuntimeError("boom mid-iteration")

    _install_entry_points(
        monkeypatch,
        [Entry("partial", "partial.pack:build", build, Distribution("partial"))],
    )
    statuses = verify_contracts()
    assert statuses[0].status == "error"
    assert "boom mid-iteration" in statuses[0].error
    assert statuses[0].contracts == ()
    plugin_entries = [
        entry for entry in contract_list() if entry.source == "plugin"
    ]
    assert plugin_entries == []
    assert get_contract("ok/one/v1") is None


def test_python_contract_escape_hatch_loads_and_merges_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def validate_inputs(inputs):
        return ["x required"] if "x" not in inputs else []

    def validate_outputs(outputs):
        return []

    def validate_relation(inputs, outputs):
        return []

    contract = ExecutableContract(
        id="custom/python/v1",
        inputs=("x",),
        outputs=("y",),
        validate_inputs=validate_inputs,
        validate_outputs=validate_outputs,
        validate_relation=validate_relation,
        description="trusted Python escape hatch",
        features=("multiple_outputs",),
        effects=("pure",),
    )
    _install_entry_points(
        monkeypatch,
        [Entry("custom", "custom.pack:build", _factory(contract))],
    )
    loaded = get_contract("custom/python/v1")
    assert loaded is not None
    assert loaded.features == ("multiple_outputs",)
    assert contract_input_issues("custom/python/v1", {}) == ["x required"]


def test_python_contract_with_missing_validator_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ExecutableContract(
        id="custom/noop/v1",
        inputs=("x",),
        outputs=("y",),
        validate_inputs=None,  # type: ignore[arg-type]
    )
    _install_entry_points(
        monkeypatch,
        [Entry("noop", "noop.pack:build", _factory(contract))],
    )
    status = verify_contracts()[0]
    assert status.status == "error"
    assert "validate_inputs must be callable" in status.error
    assert get_contract("custom/noop/v1") is None


def test_python_contract_with_invalid_effect_combination_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = ExecutableContract(
        id="custom/bad/v1",
        inputs=("x",),
        outputs=("y",),
        validate_inputs=lambda inputs: [],
        effects=("pure", "mutation_effects"),
    )
    _install_entry_points(
        monkeypatch,
        [Entry("bad", "bad.pack:build", _factory(contract))],
    )
    status = verify_contracts()[0]
    assert status.status == "error"
    assert "must not coexist" in status.error


def test_verify_contracts_is_always_fresh_and_list_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}
    descriptor = _descriptor()

    def build() -> list[object]:
        calls["count"] += 1
        return [descriptor]

    _install_entry_points(monkeypatch, [Entry("qwen", "q:build", build)])
    assert get_contract("swiglu/fused/v1") is not None
    assert calls["count"] == 1
    contract_list()
    assert calls["count"] == 1
    verify_contracts()
    verify_contracts()
    assert calls["count"] == 3


def test_plugin_statuses_without_load_are_discovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build() -> None:
        raise AssertionError("factory must not be called for discovery-only status")

    _install_entry_points(
        monkeypatch,
        [
            Entry("qwen", "q:build", build, Distribution("inferref-qwen")),
            Entry("broken", "b:build", build, Distribution("broken")),
        ],
    )
    statuses = contract_plugin_statuses()
    assert [status.status for status in statuses] == ["discovered", "discovered"]
    assert all(status.contracts == () for status in statuses)


def test_plugin_statuses_discovery_reports_metadata_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [
            Entry("builtin", "evil:build", _factory(_descriptor())),
            Entry("shared", "a:build", _factory(_descriptor())),
            Entry("shared", "b:build", _factory(_descriptor()), Distribution("b")),
        ],
    )
    statuses = contract_plugin_statuses()
    by_name = {status.entry_point: status for status in statuses}
    assert by_name["builtin"].status == "error"
    assert "contract_shadows_builtin" in by_name["builtin"].error
    assert by_name["shared"].status == "error"
    assert "duplicate contract entry-point name" in by_name["shared"].error


def test_contract_registry_never_imports_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.split(".")[0] == "torch":
            raise ImportError("torch is blocked for the contract registry test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    _install_entry_points(
        monkeypatch,
        [
            Entry(
                "qwen",
                "q:build",
                _factory(_descriptor()),
                Distribution("inferref-qwen"),
            )
        ],
    )
    statuses = verify_contracts()
    assert [status.status for status in statuses] == ["loaded"]


def test_load_contract_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "swiglu.contract.json"
    path.write_text(json.dumps(_descriptor()), encoding="utf-8")
    contract = load_contract_file(path)
    assert contract.id == "swiglu/fused/v1"
    assert contract.to_dict()["relations"] == 3

    with pytest.raises(ContractSchemaError, match="single object"):
        array_path = tmp_path / "array.contract.json"
        array_path.write_text(json.dumps([_descriptor()]), encoding="utf-8")
        load_contract_file(array_path)


def _install_fixture_distribution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """Install the tests/fixtures/contract_pack distribution into tmp_path.

    The package source is copied next to a hand-written ``*.dist-info`` so
    ``importlib.metadata`` discovers it through the real ``sys.path`` scanning
    path — the same metadata that ``pip install`` would produce.
    """

    source = REPO_ROOT / "tests" / "fixtures" / "contract_pack" / "inferref_qwen"
    target = tmp_path / "site"
    shutil.copytree(source, target / "inferref_qwen")
    dist_info = target / "inferref_qwen-0.1.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: inferref-qwen\nVersion: 0.1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[inferref.contracts]\n"
        "qwen = inferref_qwen.contracts:build\n"
        "qwen-broken = inferref_qwen.contracts:build_broken\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(target))
    return target


def test_real_distribution_discovery_via_importlib_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fixture_distribution(monkeypatch, tmp_path)
    plugin_ids = [
        entry.id for entry in contract_list() if entry.source == "plugin"
    ]
    assert plugin_ids == ["gated-deltanet/step/v1", "swiglu/fused/v1"]
    statuses = {status.entry_point: status for status in verify_contracts()}
    assert statuses["qwen"].status == "loaded"
    assert statuses["qwen"].contracts == ("swiglu/fused/v1", "gated-deltanet/step/v1")
    assert statuses["qwen-broken"].status == "error"
    assert "contract_entry_point_error" in statuses["qwen-broken"].error


def test_contract_list_cli_with_real_distribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fixture_distribution(monkeypatch, tmp_path)
    assert main(["contract", "list", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    by_id = {entry["id"]: entry for entry in payload["contracts"]}
    assert by_id["swiglu/fused/v1"]["status"] == "loaded"
    assert by_id["swiglu/fused/v1"]["distribution"] == "inferref-qwen"
    assert by_id["gated-deltanet/step/v1"]["status"] == "loaded"
    assert any(
        error["entry_point"] == "qwen-broken" for error in payload["errors"]
    )


@pytest.mark.slow
def test_fixture_pack_pip_install_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Acceptance 1: a real ``pip install`` of the fixture distribution."""

    import subprocess
    import sys

    target = tmp_path / "site"
    fixture = REPO_ROOT / "tests" / "fixtures" / "contract_pack"
    # Use pip's standard build isolation instead of --no-build-isolation: the
    # host venv's setuptools is not guaranteed (Python 3.12+ venvs ship none,
    # Python 3.10 venvs ship a version that needs the separate wheel package),
    # while an isolated build env provisions the declared backend for us.
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--target",
                str(target),
                str(fixture),
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - exotic environments
        pytest.skip(f"pip is unavailable: {exc}")
    assert result.returncode == 0, result.stderr[-2000:]

    monkeypatch.syspath_prepend(str(target))
    contract_registry._reset_registry()
    plugin_ids = [
        entry.id for entry in contract_list() if entry.source == "plugin"
    ]
    assert plugin_ids == ["gated-deltanet/step/v1", "swiglu/fused/v1"]
    statuses = {status.entry_point: status for status in verify_contracts()}
    assert statuses["qwen"].status == "loaded"
    assert statuses["qwen-broken"].status == "error"
    assert "contract_entry_point_error" in statuses["qwen-broken"].error


# -- CLI -------------------------------------------------------------------


def test_contract_list_json_reports_loaded_and_broken_plugins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken() -> None:
        raise RuntimeError("broken fixture factory")

    _install_entry_points(
        monkeypatch,
        [
            Entry(
                "qwen",
                "inferref_qwen.contracts:build",
                _factory(_descriptor(), GDN),
                Distribution("inferref-qwen"),
            ),
            Entry(
                "qwen-broken",
                "inferref_qwen_broken.contracts:build",
                broken,
                Distribution("inferref-qwen-broken"),
            ),
        ],
    )
    assert main(["contract", "list", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "inferref-contract-list"
    by_id = {entry["id"]: entry for entry in payload["contracts"]}
    assert by_id["swiglu/fused/v1"]["status"] == "loaded"
    assert by_id["swiglu/fused/v1"]["distribution"] == "inferref-qwen"
    assert by_id["gated-deltanet/step/v1"]["status"] == "loaded"
    assert all(entry["status"] == "loaded" for entry in payload["contracts"])
    broken_errors = [
        error for error in payload["errors"] if error["entry_point"] == "qwen-broken"
    ]
    assert len(broken_errors) == 1
    assert "contract_entry_point_error" in broken_errors[0]["message"]


def test_contract_list_text_render(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["contract", "list"]) == EXIT_OK
    text = capsys.readouterr().out
    assert "Contract" in text
    assert "rope/rotate-half/v1" in text
    assert "builtin" in text


def test_contract_list_runs_each_factory_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: dict[str, int] = {}
    descriptors = {"a": _descriptor(id="a/a/v1"), "b": GDN}

    def make_factory(name: str) -> object:
        def build() -> list[object]:
            calls[name] = calls.get(name, 0) + 1
            return [descriptors[name]]

        return build

    _install_entry_points(
        monkeypatch,
        [
            Entry("a", "a.pack:build", make_factory("a"), Distribution("a-pack")),
            Entry("b", "b.pack:build", make_factory("b"), Distribution("b-pack")),
        ],
    )
    assert main(["contract", "list", "--json"]) == EXIT_OK
    assert calls == {"a": 1, "b": 1}
    capsys.readouterr()


def test_contract_show_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_entry_points(
        monkeypatch,
        [Entry("qwen", "q:build", _factory(_descriptor()))],
    )
    assert main(["contract", "show", "swiglu/fused/v1", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["contract"]["inputs"] == ["x", "gate"]
    assert payload["contract"]["relations"] == 3

    assert main(["contract", "show", "missing/contract/v1", "--json"]) == EXIT_FAIL
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "contract_unknown"


def test_contract_validate_cli_accepts_swiglu_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "swiglu.contract.json"
    path.write_text(json.dumps(_descriptor()), encoding="utf-8")
    assert main(["contract", "validate", str(path), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "inferref-contract-validation"
    assert payload["status"] == "pass"
    assert payload["contract"] == {
        "id": "swiglu/fused/v1",
        "inputs": ["x", "gate"],
        "outputs": ["y"],
        "relations": 3,
    }


@pytest.mark.parametrize(
    "descriptor,code_fragment",
    [
        (_descriptor(format="inferref-trace"), "contract_schema_invalid"),
        (_descriptor(relations=["y.shape = x.shape"]), "contract_relation_syntax"),
        (_descriptor(relations=["y.shape == z.shape"]), "contract_relation_role"),
    ],
)
def test_contract_validate_cli_rejects_bad_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    descriptor: dict[str, object],
    code_fragment: str,
) -> None:
    path = tmp_path / "bad.contract.json"
    path.write_text(json.dumps(descriptor), encoding="utf-8")
    assert main(["contract", "validate", str(path), "--json"]) == EXIT_FAIL
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert any(code_fragment in issue["code"] for issue in payload["issues"])


def test_contract_validate_cli_rejects_duplicate_ids_in_one_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "duplicate.contract.json"
    path.write_text(json.dumps([_descriptor(), _descriptor()]), encoding="utf-8")
    assert main(["contract", "validate", str(path), "--json"]) == EXIT_FAIL
    payload = json.loads(capsys.readouterr().out)
    assert any(issue["code"] == "contract_duplicate_id" for issue in payload["issues"])


def test_validate_contract_file_public_api_matches_cli(
    tmp_path: Path,
) -> None:
    single = tmp_path / "single.contract.json"
    single.write_text(json.dumps(_descriptor()), encoding="utf-8")
    assert validate_contract_file(single) == {
        "status": "pass",
        "contract": {
            "id": "swiglu/fused/v1",
            "inputs": ["x", "gate"],
            "outputs": ["y"],
            "relations": 3,
        },
    }
    duplicate = tmp_path / "duplicate.contract.json"
    duplicate.write_text(json.dumps([_descriptor(), _descriptor()]), encoding="utf-8")
    result = validate_contract_file(duplicate)
    assert result["status"] == "fail"
    assert any(
        issue["code"] == "contract_duplicate_id" for issue in result["issues"]
    )
    malformed = tmp_path / "malformed.contract.json"
    malformed.write_text("{not json", encoding="utf-8")
    result = validate_contract_file(malformed)
    assert result["status"] == "fail"
    assert result["issues"][0]["code"] == "contract_schema_invalid"


# -- doctor ----------------------------------------------------------------


def test_doctor_verify_plugins_reports_contract_section(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_entry_points(
        monkeypatch,
        [
            Entry(
                "qwen",
                "inferref_qwen.contracts:build",
                _factory(_descriptor()),
                Distribution("inferref-qwen", "1.2.3"),
            )
        ],
    )
    assert main(["doctor", "--verify-plugins", "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["contracts"] == [
        {
            "entry_point": "qwen",
            "distribution": "inferref-qwen",
            "version": "1.2.3",
            "status": "loaded",
            "contracts": ["swiglu/fused/v1"],
        }
    ]
    check_ids = {check["id"] for check in report["checks"]}
    assert "contract.registry" in check_ids
    assert "contract.plugin.qwen" in check_ids


def test_doctor_contracts_section_empty_without_verify(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["doctor", "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["contracts"] == []


def test_doctor_lists_discovered_contracts_without_verify(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_entry_points(
        monkeypatch,
        [Entry("qwen", "q:build", _factory(_descriptor()), Distribution("inferref-qwen"))],
    )
    assert main(["doctor", "--json"]) == EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["contracts"] == [
        {
            "entry_point": "qwen",
            "distribution": "inferref-qwen",
            "version": "0.1.0",
            "status": "discovered",
            "contracts": [],
        }
    ]
    assert any(
        check["id"] == "contract.plugin.qwen" and check["status"] == "warn"
        for check in report["checks"]
    )


# -- preflight (C3) --------------------------------------------------------


def test_contract_requirements_merge_declared_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [
            Entry(
                "qwen",
                "q:build",
                _factory(
                    _descriptor(
                        features=["multiple_outputs"], effects=["mutation_effects"]
                    )
                ),
            )
        ],
    )
    manifest = {
        "inputs": [
            {"name": "x", **_tensor([2, 3, 64])},
            {"name": "gate", **_tensor([2, 3, 64], dtype="float16")},
        ],
        "outputs": [{"name": "y", **_tensor([2, 3, 64])}],
        "nodes": [],
    }
    requirements = contract_requirements(manifest, "swiglu/fused/v1")
    assert requirements["dtypes"] == ["float16", "float32"]
    assert requirements["max_rank"] == 3
    assert requirements["features"] == ["multiple_outputs", "mutation_effects"]


def test_contract_requirements_fallback_when_a_role_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [Entry("qwen", "q:build", _factory(_descriptor()))],
    )
    manifest = {
        "inputs": [
            {"name": "x", **_tensor([2, 3, 64])},
            {"name": "extra", **_tensor([1, 999], dtype="bfloat16")},
        ],
        "outputs": [{"name": "y", **_tensor([2, 3, 64])}],
        "nodes": [],
    }
    requirements = contract_requirements(manifest, "swiglu/fused/v1")
    assert "bfloat16" in requirements["dtypes"]
    assert requirements["max_rank"] == 3


def test_unknown_contract_requirements_unchanged() -> None:
    manifest = {
        "inputs": [{"name": "x", **_tensor([2, 3])}],
        "outputs": [{"name": "y", **_tensor([2, 3])}],
        "nodes": [],
    }
    assert contract_requirements(manifest, "softmax/last-dim/v1") == {
        "dtypes": ["float32"],
        "max_rank": 2,
        "features": [],
    }


def _swiglu_case(
    root: Path,
    *,
    y_shape: list[int] | None = None,
    contract: str = "swiglu/fused/v1",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    x = codec.write_array(
        root / "inputs/x.irtensor", np.zeros((2, 3, 64), dtype=np.float32)
    )
    gate = codec.write_array(
        root / "inputs/gate.irtensor", np.zeros((2, 3, 64), dtype=np.float32)
    )
    y = codec.write_array(
        root / "reference/y.irtensor",
        np.zeros(tuple(y_shape or [2, 3, 64]), dtype=np.float32),
    )
    entries: list[dict[str, object]] = []
    values: list[dict[str, object]] = []
    for value_id, (name, payload) in enumerate(
        [("x", x), ("gate", gate), ("y", y)], start=1
    ):
        metadata = codec.read(payload).to_metadata()
        entries.append(
            {
                "name": name,
                "value_id": value_id,
                "payload": str(payload.relative_to(root)).replace("\\", "/"),
                **metadata,
            }
        )
        values.append({"id": value_id, **metadata})
    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.2",
        "name": "swiglu-probe",
        "reproducible": True,
        "contracts": [contract],
        "inputs": entries[:2],
        "outputs": entries[2:],
        "nodes": [],
        "values": values,
    }
    manifest["requirements"] = derive_requirements(manifest)
    (root / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _adapter(path: Path, capabilities: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(
            {
                "format": ENGINE_ADAPTER_FORMAT,
                "format_version": ENGINE_ADAPTER_VERSION,
                "name": "probe",
                "target_device": "cpu",
                "capabilities": capabilities,
                "command": [
                    "{python}",
                    "-c",
                    "raise SystemExit('must not run')",
                    "{testcase}",
                    "{output}",
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_plugin_contract_preflight_returns_unsupported_before_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_entry_points(
        monkeypatch,
        [
            Entry(
                "qwen",
                "q:build",
                _factory(
                    _descriptor(
                        features=["multiple_outputs"], effects=["mutation_effects"]
                    )
                ),
            )
        ],
    )
    case = _swiglu_case(tmp_path / "cases" / "swiglu")
    adapter = _adapter(
        tmp_path / "adapter.json",
        {
            "device_types": ["cpu"],
            "dtypes": ["float32", "float16"],
            "max_rank": 8,
            "features": ["multiple_outputs", "mutation_effects"],
            "contracts": ["swiglu/fused/v1"],
            "contract_capabilities": {
                "swiglu/fused/v1": {
                    "dtypes": ["float32", "float16"],
                    "max_rank": 4,
                    "features": ["multiple_outputs"],
                }
            },
        },
    )
    result = execute_adapter(case, EngineAdapter.load(adapter), tmp_path / "runs")
    assert result["status"] == "unsupported"
    assert result["execution"] is None
    assert result["unsupported"][0]["kind"] == "contract_feature"
    assert result["unsupported"][0]["required"] == "mutation_effects"


def test_plugin_contract_preflight_dtype_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_entry_points(
        monkeypatch,
        [Entry("qwen", "q:build", _factory(_descriptor()))],
    )
    case = _swiglu_case(tmp_path / "cases" / "swiglu")
    adapter = _adapter(
        tmp_path / "adapter.json",
        {
            "device_types": ["cpu"],
            "dtypes": ["float32", "float16"],
            "max_rank": 8,
            "features": ["multiple_outputs"],
            "contracts": ["swiglu/fused/v1"],
            "contract_capabilities": {
                "swiglu/fused/v1": {
                    "dtypes": ["float16"],
                    "max_rank": 4,
                    "features": ["multiple_outputs"],
                }
            },
        },
    )
    result = execute_adapter(case, EngineAdapter.load(adapter), tmp_path / "runs")
    assert result["status"] == "unsupported"
    assert result["unsupported"][0]["kind"] == "contract_dtype"


# -- extraction acceptance ------------------------------------------------


def _swiglu_trace(*, output_shape: tuple[int, ...] = (2, 3, 64)) -> TracePackage:
    values = [
        TensorValueRecord(
            id=1,
            dtype="float32",
            shape=(2, 3, 64),
            stride=(192, 64, 1),
            storage_id=1,
            role="input",
            qualified_name="x",
        ),
        TensorValueRecord(
            id=2,
            dtype="float32",
            shape=(2, 3, 64),
            stride=(192, 64, 1),
            storage_id=2,
            role="input",
            qualified_name="gate",
        ),
        TensorValueRecord(
            id=3,
            dtype="float32",
            shape=output_shape,
            stride=tuple(range(1, len(output_shape) + 1)),
            storage_id=3,
            role="output",
            qualified_name="y",
        ),
    ]
    op = OperatorRecord(
        id=1,
        execution_index=0,
        namespace="aten",
        op="_swiglu",
        overload="default",
        positional_args=(TensorRef(1), TensorRef(2)),
        result=TensorRef(3),
        module_stack=(),
        source_id=None,
    )
    graph = Graph(operators=[op], values=values)
    graph.recompute_links()
    return TracePackage(manifest=Manifest(), graph=graph, modules=[], sources=[])


def test_extract_binds_plugin_contract_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_entry_points(
        monkeypatch,
        [Entry("qwen", "q:build", _factory(_descriptor()))],
    )
    result = extract_operator(
        _swiglu_trace(),
        1,
        tmp_path / "out",
        input_names=["x", "gate"],
        output_names=["y"],
        contracts=["swiglu/fused/v1"],
    )
    assert result.inputs == ["x", "gate"]
    assert result.outputs == ["y"]
    manifest = json.loads((tmp_path / "out" / "testcase.json").read_text("utf-8"))
    assert manifest["contracts"] == ["swiglu/fused/v1"]
    assert [entry["name"] for entry in manifest["inputs"]] == ["x", "gate"]


def test_extract_refuses_plugin_boundary_violation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_entry_points(
        monkeypatch,
        [Entry("qwen", "q:build", _factory(_descriptor()))],
    )
    package = _swiglu_trace(output_shape=(2, 3, 63))
    with pytest.raises(ExtractionError, match="boundary validation failed"):
        extract_operator(
            package,
            1,
            tmp_path / "out",
            input_names=["x", "gate"],
            output_names=["y"],
            contracts=["swiglu/fused/v1"],
        )
    assert not (tmp_path / "out").exists()


def test_boundary_issues_for_plugin_contract_match_schema_relations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_entry_points(
        monkeypatch,
        [Entry("qwen", "q:build", _factory(_descriptor()))],
    )
    inputs = {"x": _tensor([2, 3, 64]), "gate": _tensor([2, 3, 64])}
    assert (
        contract_boundary_issues(
            "swiglu/fused/v1", inputs, {"y": _tensor([2, 3, 64])}
        )
        == []
    )
    issues = contract_boundary_issues(
        "swiglu/fused/v1", inputs, {"y": _tensor([2, 3, 63])}
    )
    assert any(
        "relation 'y.shape == x.shape' failed (y.shape=[2,3,63], x.shape=[2,3,64])"
        in issue
        for issue in issues
    )
    assert contract_boundary_issues("swiglu/fused/v1", {}, {}) != []
    assert contract_boundary_issues("unknown/contract/v1", {}, {}) == ()


def test_standalone_validation_emits_contract_relation_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_entry_points(
        monkeypatch,
        [Entry("qwen", "q:build", _factory(_descriptor()))],
    )
    case = _swiglu_case(tmp_path / "cases" / "swiglu", y_shape=[2, 3, 63])
    result = validate_testcase(case)
    assert result.valid is False
    assert "contract_relation_failed" in {issue.code for issue in result.errors}
    assert all(
        issue.code != "contract_shape_invalid" for issue in result.errors
    )


def test_python_validator_relation_like_message_keeps_contract_shape_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def validate_relation(inputs, outputs):
        return ["relation 'y.shape == x.shape' failed (y.shape=[2], x.shape=[3])"]

    contract = ExecutableContract(
        id="custom/message/v1",
        inputs=("x", "gate"),
        outputs=("y",),
        validate_inputs=lambda inputs: [],
        validate_outputs=lambda outputs: [],
        validate_relation=validate_relation,
    )
    _install_entry_points(
        monkeypatch,
        [Entry("custom", "custom.pack:build", _factory(contract))],
    )
    case = _swiglu_case(
        tmp_path / "cases" / "custom", contract="custom/message/v1"
    )
    result = validate_testcase(case)
    assert result.valid is False
    assert "contract_shape_invalid" in {issue.code for issue in result.errors}
    assert "contract_relation_failed" not in {issue.code for issue in result.errors}
