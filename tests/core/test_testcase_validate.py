"""Standalone testcase validation is the shared trust boundary."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from inferref.tensor import codec
from inferref.testcase.requirements import derive_requirements
from inferref.testcase.validate import TestcaseValidationError as ValidationError
from inferref.testcase.validate import (
    require_valid_testcase,
    validate_testcase,
)


def _make_testcase(root: Path) -> tuple[Path, dict]:
    payload = codec.write_array(
        root / "reference" / "out.irtensor", np.arange(4, dtype=np.float32)
    )
    metadata = codec.read(payload).to_metadata()
    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.1",
        "name": "validator-fixture",
        "reproducible": True,
        "inputs": [],
        "outputs": [
            {
                "name": "out",
                "value_id": 2,
                "payload": "reference/out.irtensor",
                **metadata,
            }
        ],
        "values": [
            {"id": 1, "storage_id": 11, **metadata},
            {"id": 2, "storage_id": 11, **metadata},
        ],
        "nodes": [
            {
                "id": 7,
                "positional_args": [{"kind": "tensor", "value_id": 1}],
                "keyword_args": {},
                "result": {"kind": "tensor", "value_id": 2},
                "effects": {
                    "mutated_storages": [
                        {
                            "storage_id": 11,
                            "version_before": 0,
                            "version_after": 1,
                        }
                    ],
                    "aliases": [{"input_value_id": 1, "output_value_id": 2}],
                },
            }
        ],
    }
    (root / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_valid_testcase_is_computed_reproducible(tmp_path: Path, monkeypatch) -> None:
    root, _ = _make_testcase(tmp_path / "tc")
    monkeypatch.setattr(
        codec,
        "read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("validator must not materialise payloads")
        ),
    )

    result = validate_testcase(root)

    assert result.valid
    assert result.reproducible
    assert result.issues == []


def test_declared_true_does_not_override_missing_payload(tmp_path: Path) -> None:
    root, manifest = _make_testcase(tmp_path / "tc")
    manifest["outputs"][0]["payload"] = "reference/missing.irtensor"
    _write_manifest(root, manifest)

    result = validate_testcase(root)

    assert not result.valid
    assert not result.reproducible
    assert {issue.code for issue in result.errors} == {"payload_file_missing"}
    with pytest.raises(ValidationError):
        require_valid_testcase(root)


def test_hash_only_boundary_is_valid_but_not_reproducible(tmp_path: Path) -> None:
    root, manifest = _make_testcase(tmp_path / "tc")
    manifest["outputs"][0]["payload"] = None
    manifest["outputs"][0]["capture"] = {"mode": "hash"}
    _write_manifest(root, manifest)

    result = require_valid_testcase(root)

    assert result.valid
    assert not result.reproducible
    assert [issue.code for issue in result.issues] == ["payload_missing"]


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda manifest: manifest["outputs"].append(dict(manifest["outputs"][0])),
            "boundary_name_duplicate",
        ),
        (
            lambda manifest: manifest["outputs"][0].update(dtype="float16"),
            "payload_metadata_mismatch",
        ),
        (
            lambda manifest: manifest["nodes"][0]["result"].update(value_id=99),
            "node_value_unknown",
        ),
        (
            lambda manifest: manifest["nodes"][0]["effects"]["aliases"][0].update(
                output_value_id=99
            ),
            "effect_value_unknown",
        ),
        (
            lambda manifest: manifest["nodes"][0]["effects"]["mutated_storages"][
                0
            ].update(storage_id=99),
            "effect_storage_unknown",
        ),
    ],
)
def test_schema_and_reference_corruption_is_rejected(
    tmp_path: Path, mutation, expected_code: str
) -> None:
    root, manifest = _make_testcase(tmp_path / "tc")
    mutation(manifest)
    _write_manifest(root, manifest)

    result = validate_testcase(root)

    assert expected_code in {issue.code for issue in result.errors}


@pytest.mark.parametrize("payload", ["../secret.irtensor", "C:/secret.irtensor"])
def test_payload_path_escape_is_rejected(tmp_path: Path, payload: str) -> None:
    root, manifest = _make_testcase(tmp_path / "tc")
    manifest["outputs"][0]["payload"] = payload
    _write_manifest(root, manifest)

    result = validate_testcase(root)

    assert "payload_path_unsafe" in {issue.code for issue in result.errors}


def test_payload_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root, manifest = _make_testcase(tmp_path / "tc")
    outside = codec.write_array(
        tmp_path / "outside.irtensor", np.arange(4, dtype=np.float32)
    )
    link = root / "reference" / "linked.irtensor"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    manifest["outputs"][0]["payload"] = "reference/linked.irtensor"
    _write_manifest(root, manifest)

    result = validate_testcase(root)

    assert "payload_path_unsafe" in {issue.code for issue in result.errors}


def test_manifest_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root, _ = _make_testcase(tmp_path / "tc")
    outside = tmp_path / "outside.json"
    manifest_path = root / "testcase.json"
    outside.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    try:
        manifest_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    result = validate_testcase(root)

    assert not result.valid
    assert {issue.code for issue in result.errors} == {"manifest_path_unsafe"}


@pytest.mark.parametrize("effect_name", ["aliases", "mutated_storages"])
@pytest.mark.parametrize("malformed", [1, "invalid", {}, None])
def test_non_array_effect_is_structured_error(
    tmp_path: Path, effect_name: str, malformed
) -> None:
    root, manifest = _make_testcase(tmp_path / "tc")
    manifest["nodes"][0]["effects"][effect_name] = malformed
    _write_manifest(root, manifest)

    result = validate_testcase(root)

    assert not result.valid
    assert "effect_schema_invalid" in {issue.code for issue in result.errors}


@pytest.mark.parametrize("effect_name", ["aliases", "mutated_storages"])
def test_nested_malformed_effect_entries_do_not_escape_validator(
    tmp_path: Path, effect_name: str
) -> None:
    root, manifest = _make_testcase(tmp_path / "tc")
    manifest["nodes"][0]["effects"][effect_name] = [
        1,
        "invalid",
        None,
        ["nested"],
        {},
    ]
    _write_manifest(root, manifest)

    result = validate_testcase(root)

    assert not result.valid
    assert "effect_schema_invalid" in {issue.code for issue in result.errors}


def test_arbitrary_json_manifest_never_leaks_validation_exception(
    tmp_path: Path,
) -> None:
    rng = random.Random(0)
    root = tmp_path / "fuzz"
    root.mkdir()
    manifest_path = root / "testcase.json"
    keys = (
        "format",
        "format_version",
        "reproducible",
        "inputs",
        "outputs",
        "values",
        "nodes",
        "id",
        "value_id",
        "producer",
        "consumers",
        "effects",
        "aliases",
        "mutated_storages",
        "storage_id",
        "version_before",
        "version_after",
        "kind",
        "payload",
    )

    def arbitrary(depth: int):
        scalars = [None, True, False, rng.randint(-3, 3), rng.random(), "value"]
        if depth <= 0:
            return rng.choice(scalars)
        choice = rng.randrange(3)
        if choice == 0:
            return rng.choice(scalars)
        if choice == 1:
            return [arbitrary(depth - 1) for _ in range(rng.randrange(5))]
        return {rng.choice(keys): arbitrary(depth - 1) for _ in range(rng.randrange(6))}

    for _ in range(500):
        manifest_path.write_text(json.dumps(arbitrary(4)), encoding="utf-8")
        result = validate_testcase(root)
        assert result.root == root.resolve()


def test_known_executable_contract_validates_input_roles_and_shapes(
    tmp_path: Path,
) -> None:
    root, manifest = _make_testcase(tmp_path / "tc")
    manifest["format_version"] = "0.2"
    manifest["contracts"] = ["rmsnorm/last-dim/v1"]
    manifest["requirements"] = derive_requirements(manifest)
    _write_manifest(root, manifest)

    result = validate_testcase(root)

    assert "contract_input_missing" in {issue.code for issue in result.errors}


def test_invalid_contract_identifier_is_structured_error(tmp_path: Path) -> None:
    root, manifest = _make_testcase(tmp_path / "tc")
    manifest["format_version"] = "0.2"
    manifest["contracts"] = ["rope"]
    manifest["requirements"] = derive_requirements(manifest)
    _write_manifest(root, manifest)

    result = validate_testcase(root)

    assert "contracts_invalid" in {issue.code for issue in result.errors}


def test_contract_rejects_empty_kernel_input_before_execution(tmp_path: Path) -> None:
    root, manifest = _make_testcase(tmp_path / "tc")
    manifest["format_version"] = "0.2"
    next_id = 3
    for name, array in (
        ("x", np.empty((0, 3), dtype=np.float32)),
        ("weight", np.ones(3, dtype=np.float32)),
        ("epsilon", np.asarray(1e-5, dtype=np.float32)),
    ):
        payload = codec.write_array(root / "inputs" / f"{name}.irtensor", array)
        metadata = codec.read(payload).to_metadata()
        manifest["inputs"].append(
            {
                "name": name,
                "value_id": next_id,
                "payload": f"inputs/{name}.irtensor",
                **metadata,
            }
        )
        manifest["values"].append({"id": next_id, **metadata})
        next_id += 1
    manifest["contracts"] = ["rmsnorm/last-dim/v1"]
    manifest["requirements"] = derive_requirements(manifest)
    _write_manifest(root, manifest)

    result = validate_testcase(root)

    assert "contract_shape_invalid" in {issue.code for issue in result.errors}


def test_contract_rejects_non_scalar_epsilon(tmp_path: Path) -> None:
    root, manifest = _make_testcase(tmp_path / "tc")
    manifest["format_version"] = "0.2"
    next_id = 3
    for name, array in (
        ("x", np.ones((2, 3, 8), dtype=np.float32)),
        ("weight", np.ones(8, dtype=np.float32)),
        ("epsilon", np.ones(2, dtype=np.float32)),
    ):
        payload = codec.write_array(root / "inputs" / f"{name}.irtensor", array)
        metadata = codec.read(payload).to_metadata()
        manifest["inputs"].append(
            {
                "name": name,
                "value_id": next_id,
                "payload": f"inputs/{name}.irtensor",
                **metadata,
            }
        )
        manifest["values"].append({"id": next_id, **metadata})
        next_id += 1
    y_payload = codec.write_array(
        root / "reference" / "y.irtensor",
        np.ones((2, 3, 8), dtype=np.float32),
    )
    y_metadata = codec.read(y_payload).to_metadata()
    manifest["outputs"] = [
        {
            "name": "y",
            "value_id": 2,
            "payload": "reference/y.irtensor",
            **y_metadata,
        }
    ]
    manifest["values"][1] = {"id": 2, **y_metadata}
    manifest["contracts"] = ["rmsnorm/last-dim/v1"]
    manifest["requirements"] = derive_requirements(manifest)
    _write_manifest(root, manifest)

    result = validate_testcase(root)

    assert "contract_shape_invalid" in {issue.code for issue in result.errors}
    assert any("exactly one value" in issue.message for issue in result.errors)


def _rope_testcase(root: Path, *, output_names: list[str]) -> tuple[Path, dict]:
    _, manifest = _make_testcase(root)
    manifest["format_version"] = "0.2"
    next_id = 3
    for name, array in (
        ("query", np.ones((1, 4, 8), dtype=np.float32)),
        ("key", np.ones((1, 4, 8), dtype=np.float32)),
        ("cos", np.ones((4, 8), dtype=np.float32)),
        ("sin", np.ones((4, 8), dtype=np.float32)),
    ):
        payload = codec.write_array(root / "inputs" / f"{name}.irtensor", array)
        metadata = codec.read(payload).to_metadata()
        manifest["inputs"].append(
            {
                "name": name,
                "value_id": next_id,
                "payload": f"inputs/{name}.irtensor",
                **metadata,
            }
        )
        manifest["values"].append({"id": next_id, **metadata})
        next_id += 1
    manifest["outputs"] = []
    for name, shape in (
        ("q_embed", (1, 4, 8)),
        ("k_embed", (1, 4, 8)),
    ):
        if name not in output_names:
            continue
        payload = codec.write_array(
            root / "reference" / f"{name}.irtensor",
            np.ones(shape, dtype=np.float32),
        )
        metadata = codec.read(payload).to_metadata()
        manifest["outputs"].append(
            {
                "name": name,
                "value_id": next_id,
                "payload": f"reference/{name}.irtensor",
                **metadata,
            }
        )
        manifest["values"].append({"id": next_id, **metadata})
        next_id += 1
    manifest["contracts"] = ["rope/rotate-half/v1"]
    manifest["requirements"] = derive_requirements(manifest)
    _write_manifest(root, manifest)
    return root, manifest


def test_contract_requires_every_observable_output(tmp_path: Path) -> None:
    root, _ = _rope_testcase(tmp_path / "tc", output_names=["q_embed"])
    result = validate_testcase(root)
    assert result.valid is False
    assert "contract_output_missing" in {issue.code for issue in result.errors}


def test_contract_rejects_unexpected_observable_output(tmp_path: Path) -> None:
    root, manifest = _rope_testcase(
        tmp_path / "tc", output_names=["q_embed", "k_embed"]
    )
    payload = codec.write_array(
        root / "reference" / "extra.irtensor",
        np.ones((1, 4, 8), dtype=np.float32),
    )
    metadata = codec.read(payload).to_metadata()
    manifest["outputs"].append(
        {
            "name": "extra",
            "value_id": 99,
            "payload": "reference/extra.irtensor",
            **metadata,
        }
    )
    manifest["requirements"] = derive_requirements(manifest)
    _write_manifest(root, manifest)

    result = validate_testcase(root)
    assert "contract_unexpected_output" in {issue.code for issue in result.errors}


def test_contract_rejects_wrong_output_shape_relation(tmp_path: Path) -> None:
    root, manifest = _rope_testcase(
        tmp_path / "tc", output_names=["q_embed", "k_embed"]
    )
    payload = codec.write_array(
        root / "reference" / "k_embed.irtensor",
        np.ones((1, 4, 4), dtype=np.float32),
    )
    metadata = codec.read(payload).to_metadata()
    for entry in manifest["outputs"]:
        if entry["name"] == "k_embed":
            entry.update(metadata)
    manifest["requirements"] = derive_requirements(manifest)
    _write_manifest(root, manifest)

    result = validate_testcase(root)
    assert "contract_shape_invalid" in {issue.code for issue in result.errors}


def test_contracts_array_must_contain_exactly_one(tmp_path: Path) -> None:
    root, manifest = _rope_testcase(
        tmp_path / "tc", output_names=["q_embed", "k_embed"]
    )
    manifest["contracts"] = [
        "rope/rotate-half/v1",
        "kv-cache/append/v1",
    ]
    manifest["requirements"] = derive_requirements(manifest)
    _write_manifest(root, manifest)

    result = validate_testcase(root)
    assert "contracts_invalid" in {issue.code for issue in result.errors}


def test_unknown_contract_warns_but_stays_reproducible(tmp_path: Path) -> None:
    # Reuse the add-one fixture shape from the suite tests: a standalone case
    # with a well-formed contract ID that this build does not define.
    from tests.core.test_suite import _case

    case = _case(tmp_path / "case", contract="softmax/last-dim/v1")
    result = validate_testcase(case)

    assert result.valid is True
    assert result.reproducible is True
    assert "contract_not_in_registry" in {issue.code for issue in result.issues}
    assert any(
        issue.severity == "warning" and not issue.blocks_reproduction
        for issue in result.issues
    )
