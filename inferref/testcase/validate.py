"""Standalone testcase validation before comparison or engine execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferref.contracts import (
    contract_boundary_issues,
    get_contract,
)
from inferref.ir.paths import PathBoundaryError, resolve_contained_path
from inferref.ir.version import TESTCASE_FORMAT, TESTCASE_READ_VERSIONS
from inferref.tensor import codec
from inferref.testcase.requirements import derive_requirements, is_contract_id


@dataclass(frozen=True)
class TestcaseValidationIssue:
    severity: str
    code: str
    message: str
    where: str = ""
    blocks_reproduction: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "where": self.where,
            "blocks_reproduction": self.blocks_reproduction,
        }


@dataclass
class TestcaseValidationResult:
    root: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    issues: list[TestcaseValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[TestcaseValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def reproducible(self) -> bool:
        declared = self.manifest.get("reproducible") is not False
        blocked = any(issue.blocks_reproduction for issue in self.issues)
        return bool(self.manifest) and declared and not blocked

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "pass" if self.valid else "fail",
            "valid": self.valid,
            "reproducible": self.reproducible,
            "declared_reproducible": self.manifest.get("reproducible"),
            "errors": len(self.errors),
            "warnings": len(self.issues) - len(self.errors),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def raise_for_errors(self) -> None:
        if self.errors:
            raise TestcaseValidationError(self)


class TestcaseValidationError(ValueError):
    def __init__(self, result: TestcaseValidationResult):
        self.result = result
        details = "; ".join(
            f"{issue.code}: {issue.message}" for issue in result.errors[:3]
        )
        if len(result.errors) > 3:
            details += f"; and {len(result.errors) - 3} more error(s)"
        super().__init__(f"invalid testcase {result.root}: {details}")


def _parse_version_tuple(v: Any) -> tuple[int, ...]:
    if not isinstance(v, str) or not v.strip():
        return (0, 0)
    try:
        return tuple(int(p) for p in v.strip().split("."))
    except ValueError:
        return (0, 0)


def validate_testcase(root: str | Path) -> TestcaseValidationResult:
    """Validate structure, containment, payload metadata, and internal references."""

    resolved_root = Path(root).resolve()
    result = TestcaseValidationResult(root=resolved_root)
    if not resolved_root.is_dir():
        _error(result, "testcase_not_directory", "testcase root is not a directory")
        return result

    try:
        manifest_path = resolve_contained_path(
            resolved_root, "testcase.json", kind="testcase manifest path"
        )
    except PathBoundaryError as exc:
        _error(result, "manifest_path_unsafe", str(exc), "testcase.json")
        return result
    if not manifest_path.is_file():
        _error(result, "manifest_missing", "testcase.json does not exist")
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _error(result, "manifest_invalid", str(exc), "testcase.json")
        return result
    if not isinstance(manifest, dict):
        _error(result, "manifest_invalid", "manifest root must be a JSON object")
        return result
    result.manifest = manifest

    _validate_manifest(result, manifest)

    values = _records(result, manifest, "values")
    nodes = _records(result, manifest, "nodes")
    value_ids = _unique_ids(result, values, "values")
    node_ids = _unique_ids(result, nodes, "nodes")
    storage_ids = {
        entry.get("storage_id")
        for entry in values
        if _is_integer(entry.get("storage_id"))
    }

    inputs = _boundary(result, manifest, "inputs", value_ids)
    outputs = _boundary(result, manifest, "outputs", value_ids)
    _validate_executable_contracts(result, manifest, inputs, outputs)
    if not outputs:
        _issue(
            result,
            "warning",
            "outputs_empty",
            "testcase declares no comparable outputs",
            "outputs",
        )

    for collection_name, entries in (("inputs", inputs), ("outputs", outputs)):
        _validate_payloads(result, resolved_root, collection_name, entries)

    _validate_value_references(result, values, node_ids)
    _validate_node_references(result, nodes, value_ids, storage_ids)
    for index, output in enumerate(outputs):
        producer = output.get("producer")
        if not isinstance(producer, dict) or producer.get("op_id") is None:
            continue
        producer_id = producer.get("op_id")
        if not _is_integer(producer_id) or producer_id not in node_ids:
            _error(
                result,
                "producer_node_unknown",
                f"producer op_id {producer_id!r} is not in nodes",
                f"outputs[{index}].producer.op_id",
            )

    if manifest.get("reproducible") is False:
        _issue(
            result,
            "warning",
            "declared_non_reproducible",
            "manifest explicitly declares this testcase non-reproducible",
            "reproducible",
        )
    for field_name in ("non_portable_values", "unobservable_mutations"):
        if manifest.get(field_name):
            _issue(
                result,
                "warning",
                field_name,
                f"manifest contains {field_name}",
                field_name,
            )
    return result


def _validate_manifest(
    result: TestcaseValidationResult, manifest: dict[str, Any]
) -> None:
    if manifest.get("format") != TESTCASE_FORMAT:
        _error(
            result,
            "format_invalid",
            f"invalid testcase format {manifest.get('format')!r}; expected {TESTCASE_FORMAT!r}",
            "format",
        )
    if manifest.get("format_version") not in TESTCASE_READ_VERSIONS:
        _error(
            result,
            "format_version_unsupported",
            "unsupported testcase format_version "
            f"{manifest.get('format_version')!r}; expected one of {TESTCASE_READ_VERSIONS!r}",
            "format_version",
        )
    if "comparison" in manifest:
        version_tuple = _parse_version_tuple(manifest.get("format_version"))
        if version_tuple < (0, 3):
            _error(
                result,
                "comparison_requires_0_3",
                "comparison requires format_version 0.3 or higher",
                "comparison",
            )
        _validate_comparison(result, manifest)
    elif manifest.get("format_version") == "0.3":
        _issue(
            result,
            "warning",
            "format_version_0_3_without_comparison",
            "testcase declares format_version 0.3 without comparison section",
            "format_version",
            blocks_reproduction=False,
        )
    _validate_requirements(result, manifest)


def require_valid_testcase(root: str | Path) -> TestcaseValidationResult:
    result = validate_testcase(root)
    result.raise_for_errors()
    return result


def _records(
    result: TestcaseValidationResult, manifest: dict[str, Any], name: str
) -> list[dict[str, Any]]:
    raw = manifest.get(name, [])
    if not isinstance(raw, list):
        _error(result, "schema_invalid", f"{name} must be an array", name)
        return []
    records: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            _error(
                result,
                "schema_invalid",
                f"{name}[{index}] must be an object",
                f"{name}[{index}]",
            )
            continue
        records.append(item)
    return records


def _unique_ids(
    result: TestcaseValidationResult,
    records: list[dict[str, Any]],
    name: str,
) -> set[int]:
    observed: set[int] = set()
    for index, record in enumerate(records):
        identifier = record.get("id")
        if not _is_integer(identifier):
            _error(
                result,
                "id_invalid",
                f"{name}[{index}].id must be an integer",
                f"{name}[{index}].id",
            )
        elif identifier in observed:
            _error(
                result,
                "id_duplicate",
                f"duplicate {name} id {identifier}",
                f"{name}[{index}].id",
            )
        else:
            observed.add(identifier)
    return observed


def _boundary(
    result: TestcaseValidationResult,
    manifest: dict[str, Any],
    name: str,
    value_ids: set[int],
) -> list[dict[str, Any]]:
    records = _records(result, manifest, name)
    observed_names: set[str] = set()
    for index, record in enumerate(records):
        label = record.get("name")
        if not isinstance(label, str) or not label:
            _error(
                result,
                "boundary_name_invalid",
                f"{name}[{index}].name must be a non-empty string",
                f"{name}[{index}].name",
            )
        elif label in observed_names:
            _error(
                result,
                "boundary_name_duplicate",
                f"duplicate {name} name {label!r}",
                f"{name}[{index}].name",
            )
        else:
            observed_names.add(label)
        value_id = record.get("value_id")
        if value_id is not None and (
            not _is_integer(value_id) or value_id not in value_ids
        ):
            _error(
                result,
                "boundary_value_unknown",
                f"value_id {value_id!r} is not in values",
                f"{name}[{index}].value_id",
            )
    return records


def _validate_payloads(
    result: TestcaseValidationResult,
    root: Path,
    collection_name: str,
    records: list[dict[str, Any]],
) -> None:
    for index, record in enumerate(records):
        where = f"{collection_name}[{index}]"
        payload = record.get("payload")
        if payload is None:
            _issue(
                result,
                "warning",
                "payload_missing",
                "boundary tensor has no full payload",
                f"{where}.payload",
            )
            continue
        if not isinstance(payload, str) or not payload:
            _error(
                result,
                "payload_path_invalid",
                "payload must be a non-empty relative string",
                f"{where}.payload",
            )
            continue
        try:
            path = resolve_contained_path(
                root, payload, kind=f"{collection_name} payload path"
            )
        except PathBoundaryError as exc:
            _error(result, "payload_path_unsafe", str(exc), f"{where}.payload")
            continue
        if not path.is_file():
            _error(
                result,
                "payload_file_missing",
                f"payload file does not exist: {payload}",
                f"{where}.payload",
            )
            continue
        try:
            header = codec.read_header(path)
        except (OSError, ValueError) as exc:
            _error(result, "payload_invalid", str(exc), f"{where}.payload")
            continue
        expected = {
            "dtype": header.dtype,
            "shape": list(header.shape),
            "stride": list(header.stride),
            "storage_offset": header.storage_offset,
        }
        for field_name, actual in expected.items():
            if field_name not in record:
                _error(
                    result,
                    "payload_metadata_missing",
                    f"manifest is missing {field_name!r} for payload {payload}",
                    f"{where}.{field_name}",
                )
            elif record[field_name] != actual:
                _error(
                    result,
                    "payload_metadata_mismatch",
                    f"manifest {field_name} {record[field_name]!r} "
                    f"!= payload {actual!r}",
                    f"{where}.{field_name}",
                )


def _validate_node_references(
    result: TestcaseValidationResult,
    nodes: list[dict[str, Any]],
    value_ids: set[int],
    storage_ids: set[int],
) -> None:
    for index, node in enumerate(nodes):
        for where, value_id in _walk_value_references(node, f"nodes[{index}]"):
            if not _is_integer(value_id) or value_id not in value_ids:
                _error(
                    result,
                    "node_value_unknown",
                    f"value_id {value_id!r} is not in values",
                    where,
                )
        effects = node.get("effects")
        if effects is None:
            continue
        if not isinstance(effects, dict):
            _error(
                result,
                "effect_schema_invalid",
                "effects must be an object",
                f"nodes[{index}].effects",
            )
            continue
        aliases = _effect_array(result, effects, "aliases", index)
        mutations = _effect_array(result, effects, "mutated_storages", index)
        for effect_index, alias in enumerate(aliases):
            if not isinstance(alias, dict):
                _error(
                    result,
                    "effect_schema_invalid",
                    "alias effect must be an object",
                    f"nodes[{index}].effects.aliases[{effect_index}]",
                )
                continue
            for field_name in ("input_value_id", "output_value_id"):
                value_id = alias.get(field_name)
                if not _is_integer(value_id) or value_id not in value_ids:
                    _error(
                        result,
                        "effect_value_unknown",
                        f"{field_name} {value_id!r} is not in values",
                        (
                            f"nodes[{index}].effects.aliases[{effect_index}]"
                            f".{field_name}"
                        ),
                    )
        for effect_index, mutation in enumerate(mutations):
            if not isinstance(mutation, dict):
                _error(
                    result,
                    "effect_schema_invalid",
                    "mutation effect must be an object",
                    f"nodes[{index}].effects.mutated_storages[{effect_index}]",
                )
                continue
            storage_id = mutation.get("storage_id")
            if not _is_integer(storage_id) or storage_id not in storage_ids:
                _error(
                    result,
                    "effect_storage_unknown",
                    f"storage_id {storage_id!r} is not represented in values",
                    f"nodes[{index}].effects.mutated_storages[{effect_index}]",
                )
            before = mutation.get("version_before")
            after = mutation.get("version_after")
            if not _is_integer(before) or not _is_integer(after):
                _error(
                    result,
                    "effect_schema_invalid",
                    "mutation versions must be integers",
                    f"nodes[{index}].effects.mutated_storages[{effect_index}]",
                )
            elif after != before + 1:
                _error(
                    result,
                    "effect_version_invalid",
                    f"mutation must advance exactly one version ({before} -> {after})",
                    f"nodes[{index}].effects.mutated_storages[{effect_index}]",
                )


def _validate_value_references(
    result: TestcaseValidationResult,
    values: list[dict[str, Any]],
    node_ids: set[int],
) -> None:
    for index, value in enumerate(values):
        producer = value.get("producer")
        if producer is not None and (
            not _is_integer(producer) or producer not in node_ids
        ):
            _error(
                result,
                "value_producer_unknown",
                f"producer {producer!r} is not in nodes",
                f"values[{index}].producer",
            )
        consumers = value.get("consumers")
        if consumers is None:
            continue
        if not isinstance(consumers, list):
            _error(
                result,
                "value_consumers_invalid",
                "consumers must be an array",
                f"values[{index}].consumers",
            )
            continue
        for consumer_index, consumer in enumerate(consumers):
            if not _is_integer(consumer) or consumer not in node_ids:
                _error(
                    result,
                    "value_consumer_unknown",
                    f"consumer {consumer!r} is not in nodes",
                    f"values[{index}].consumers[{consumer_index}]",
                )


def _walk_value_references(value: Any, where: str):
    if isinstance(value, dict):
        if value.get("kind") == "tensor" and "value_id" in value:
            yield f"{where}.value_id", value.get("value_id")
        for key, item in value.items():
            yield from _walk_value_references(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_value_references(item, f"{where}[{index}]")


def _effect_array(
    result: TestcaseValidationResult,
    effects: dict[str, Any],
    name: str,
    node_index: int,
) -> list[Any]:
    raw = effects.get(name, [])
    if isinstance(raw, list):
        return raw
    _error(
        result,
        "effect_schema_invalid",
        f"effects.{name} must be an array",
        f"nodes[{node_index}].effects.{name}",
    )
    return []


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_requirements(
    result: TestcaseValidationResult, manifest: dict[str, Any]
) -> None:
    requirements = manifest.get("requirements")
    if manifest.get("format_version") == "0.1" and requirements is None:
        return
    if not isinstance(requirements, dict):
        _error(
            result,
            "requirements_invalid",
            "requirements must be an object",
            "requirements",
        )
        return
    declared_contracts = manifest.get("contracts")
    if declared_contracts is not None and (
        not isinstance(declared_contracts, list)
        or len(declared_contracts) != 1
        or not all(is_contract_id(item) for item in declared_contracts)
        or len(set(declared_contracts)) != len(declared_contracts)
    ):
        _error(
            result,
            "contracts_invalid",
            "contracts must contain exactly one unique versioned contract ID "
            "when present in v0.2",
            "contracts",
        )
    dtypes = requirements.get("dtypes")
    features = requirements.get("features")
    max_rank = requirements.get("max_rank")
    contracts = requirements.get("contracts")
    if not isinstance(dtypes, list) or not all(
        isinstance(x, str) and x for x in dtypes
    ):
        _error(
            result,
            "requirements_invalid",
            "dtypes must be a string array",
            "requirements.dtypes",
        )
    if not isinstance(features, list) or not all(
        isinstance(x, str) and x for x in features
    ):
        _error(
            result,
            "requirements_invalid",
            "features must be a string array",
            "requirements.features",
        )
    if not isinstance(max_rank, int) or isinstance(max_rank, bool) or max_rank < 0:
        _error(
            result,
            "requirements_invalid",
            "max_rank must be non-negative",
            "requirements.max_rank",
        )
    if contracts is not None and (
        not isinstance(contracts, list)
        or len(contracts) != 1
        or not all(is_contract_id(x) for x in contracts)
        or len(set(contracts)) != len(contracts)
    ):
        _error(
            result,
            "requirements_invalid",
            "requirements.contracts must contain exactly one unique versioned "
            "contract ID when present in v0.2",
            "requirements.contracts",
        )
    if (
        isinstance(dtypes, list)
        and isinstance(features, list)
        and isinstance(max_rank, int)
    ):
        expected = derive_requirements(
            {k: v for k, v in manifest.items() if k != "requirements"}
        )
        if requirements != expected:
            _error(
                result,
                "requirements_mismatch",
                f"declared requirements {requirements!r} != derived {expected!r}",
                "requirements",
            )


def _validate_executable_contracts(
    result: TestcaseValidationResult,
    manifest: dict[str, Any],
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
) -> None:
    contracts = manifest.get("contracts")
    if not isinstance(contracts, list):
        return
    by_name = {
        item.get("name"): item for item in inputs if isinstance(item.get("name"), str)
    }
    output_by_name = {
        item.get("name"): item for item in outputs if isinstance(item.get("name"), str)
    }

    for contract in contracts:
        profile = get_contract(contract)
        if profile is None:
            _issue(
                result,
                "warning",
                "contract_not_in_registry",
                f"{contract} is not defined in this build's executable contract "
                "registry; role/shape validation is skipped",
                "contracts",
                blocks_reproduction=False,
            )
            continue
        unexpected_outputs = sorted(set(output_by_name) - set(profile.outputs))
        if unexpected_outputs:
            _error(
                result,
                "contract_unexpected_output",
                f"{contract} declares exactly {', '.join(profile.outputs)} "
                f"observable output(s); unexpected: {', '.join(unexpected_outputs)}",
                "outputs",
            )
        missing_outputs = [
            name for name in profile.outputs if name not in output_by_name
        ]
        if missing_outputs:
            _error(
                result,
                "contract_output_missing",
                f"{contract} requires observable output(s): "
                + ", ".join(missing_outputs),
                "outputs",
            )
        missing = [name for name in profile.inputs if name not in by_name]
        if missing:
            _error(
                result,
                "contract_input_missing",
                f"{contract} requires input(s): {', '.join(missing)}",
                "inputs",
            )
            continue
        role_inputs = {name: by_name[name] for name in profile.inputs}
        role_outputs = {
            name: output_by_name[name]
            for name in profile.outputs
            if name in output_by_name
        }
        for message in contract_boundary_issues(contract, role_inputs, role_outputs):
            # Schema-derived relation failures carry their stable code on the
            # message object; every other issue (built-in validators, role or
            # dtype checks, Python escape-hatch validators) keeps
            # contract_shape_invalid. No message-text parsing is involved.
            issue_code = getattr(
                message, "code", "contract_shape_invalid"
            )
            _error(
                result,
                issue_code,
                f"{contract}: {message}",
                "inputs",
            )


def _error(
    result: TestcaseValidationResult,
    code: str,
    message: str,
    where: str = "",
) -> None:
    _issue(result, "error", code, message, where)


def _issue(
    result: TestcaseValidationResult,
    severity: str,
    code: str,
    message: str,
    where: str = "",
    *,
    blocks_reproduction: bool = True,
) -> None:
    result.issues.append(
        TestcaseValidationIssue(
            severity=severity,
            code=code,
            message=message,
            where=where,
            blocks_reproduction=blocks_reproduction,
        )
    )


def _validate_comparison(
    result: TestcaseValidationResult, manifest: dict[str, Any]
) -> None:
    from inferref.comparators.registry import get_comparator
    from inferref.comparison.schema import ComparisonSpec, ComparisonSpecValidationError

    comp_data = manifest.get("comparison")
    if not isinstance(comp_data, dict):
        _error(result, "comparison_invalid", "comparison must be an object", "comparison")
        return

    try:
        spec = ComparisonSpec.from_dict(comp_data)
    except (ComparisonSpecValidationError, ValueError) as exc:
        _error(result, "comparison_invalid", str(exc), "comparison")
        return

    plugin = get_comparator(spec.comparator)
    if plugin is None:
        _error(
            result,
            "comparator_unknown",
            f"unknown comparator {spec.comparator!r}",
            "comparison.comparator",
        )
    else:
        try:
            plugin.validate_config(spec.config)
        except (ValueError, TypeError) as exc:
            _error(
                result,
                "invalid_comparison_config",
                f"invalid comparison config: {exc}",
                "comparison.config",
            )

    for role, out_spec in spec.outputs.items():
        if out_spec.comparator is not None:
            role_comp_id = out_spec.comparator
            role_plugin = get_comparator(role_comp_id)
            if role_plugin is None:
                _error(
                    result,
                    "comparator_unknown",
                    f"unknown comparator {role_comp_id!r} for output role {role!r}",
                    f"comparison.outputs.{role}.comparator",
                )
            else:
                try:
                    role_plugin.validate_config(out_spec.config)
                except (ValueError, TypeError) as exc:
                    _error(
                        result,
                        "invalid_comparison_config",
                        f"invalid comparison config for output role {role!r}: {exc}",
                        f"comparison.outputs.{role}.config",
                    )
        elif plugin is not None and out_spec.config:
            try:
                plugin.validate_config(out_spec.config)
            except (ValueError, TypeError) as exc:
                _error(
                    result,
                    "invalid_comparison_config",
                    f"invalid comparison config for output role {role!r}: {exc}",
                    f"comparison.outputs.{role}.config",
                )
