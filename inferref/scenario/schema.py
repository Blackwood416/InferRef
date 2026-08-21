"""Scenario v0.1 manifest loading and structural validation (SPEC §4, §5).

The loader is deliberately torch-free and mirrors the Suite loader: it raises
``ScenarioError`` for every structural problem (or a collection of them) so
callers can split schema validity from runnability.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferref.ir.paths import PathBoundaryError, resolve_contained_path
from inferref.testcase.validate import validate_testcase

SCENARIO_FORMAT = "inferref-scenario"
SCENARIO_FORMAT_VERSION = "0.1"
SCENARIO_MANIFEST = "scenario.json"

#: Scenario input/state/output names become file components in the run
#: directory and reference suffixes, so they follow the portable ID grammar.
_VALUE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class ScenarioError(ValueError):
    """Raised when a scenario manifest is structurally invalid."""

    def __init__(self, issues: list[dict[str, Any]]):
        self.issues = issues
        details = "; ".join(
            f"{issue['code']}: {issue['message']}" for issue in issues[:3]
        )
        if len(issues) > 3:
            details += f"; and {len(issues) - 3} more error(s)"
        super().__init__(f"invalid scenario: {details}")


@dataclass(frozen=True)
class ScenarioInput:
    name: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind}


@dataclass(frozen=True)
class ScenarioState:
    name: str
    kind: str
    init: str | None = None

    def to_dict(self) -> dict[str, str]:
        result: dict[str, str] = {"kind": self.kind}
        if self.init is not None:
            result["init"] = self.init
        return result


@dataclass(frozen=True)
class ScenarioOutput:
    name: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind}


@dataclass(frozen=True)
class ScenarioStep:
    id: str
    #: Original relative path string from the manifest.
    testcase_ref: str
    #: Resolved, validated testcase directory below the scenario root.
    testcase: Path
    input_bindings: dict[str, str] = field(default_factory=dict)
    output_bindings: dict[str, str] = field(default_factory=dict)
    comparison: Any | None = None

    def to_dict(self, root: Path) -> dict[str, Any]:
        record: dict[str, Any] = {
            "id": self.id,
            "testcase": self.testcase.relative_to(root).as_posix(),
        }
        if self.input_bindings or self.output_bindings:
            record["bindings"] = {}
            if self.input_bindings:
                record["bindings"]["inputs"] = dict(self.input_bindings)
            if self.output_bindings:
                record["bindings"]["outputs"] = dict(self.output_bindings)
        if self.comparison is not None:
            record["comparison"] = (
                self.comparison.to_dict()
                if hasattr(self.comparison, "to_dict")
                else dict(self.comparison)
            )
        return record


@dataclass(frozen=True)
class Scenario:
    #: Scenario root directory as the caller supplied it (used in reports).
    id: str
    source: Path
    description: str = ""
    inputs: tuple[ScenarioInput, ...] = ()
    state: tuple[ScenarioState, ...] = ()
    outputs: tuple[ScenarioOutput, ...] = ()
    steps: tuple[ScenarioStep, ...] = ()

    @property
    def root(self) -> Path:
        return self.source.resolve()

    def to_dict(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            "format": SCENARIO_FORMAT,
            "format_version": SCENARIO_FORMAT_VERSION,
            "id": self.id,
            "inputs": {item.name: item.to_dict() for item in self.inputs},
        }
        if self.description:
            manifest["description"] = self.description
        if self.state:
            manifest["state"] = {item.name: item.to_dict() for item in self.state}
        if self.outputs:
            manifest["outputs"] = {item.name: item.to_dict() for item in self.outputs}
        manifest["steps"] = [step.to_dict(self.root) for step in self.steps]
        return manifest


def load_scenario(path: str | Path) -> Scenario:
    """Load and structurally validate one scenario manifest."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise ScenarioError(
            [
                _issue(
                    "scenario_invalid_manifest",
                    "scenario root is not a directory",
                    str(root),
                )
            ]
        )
    try:
        manifest_path = resolve_contained_path(
            root, SCENARIO_MANIFEST, kind="scenario manifest path"
        )
    except PathBoundaryError as exc:
        raise ScenarioError([_issue("scenario_invalid_manifest", str(exc))]) from exc
    if not manifest_path.is_file():
        raise ScenarioError(
            [_issue("scenario_invalid_manifest", "scenario.json does not exist")]
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScenarioError(
            [_issue("scenario_invalid_manifest", str(exc), "scenario.json")]
        ) from exc
    if not isinstance(data, dict):
        raise ScenarioError(
            [_issue("scenario_invalid_manifest", "manifest root must be an object")]
        )
    issues: list[dict[str, Any]] = []
    if data.get("format") != SCENARIO_FORMAT:
        issues.append(
            _issue(
                "scenario_invalid_manifest",
                f"format must be {SCENARIO_FORMAT!r}, got {data.get('format')!r}",
                "format",
            )
        )
    if data.get("format_version") != SCENARIO_FORMAT_VERSION:
        issues.append(
            _issue(
                "scenario_invalid_manifest",
                "format_version must be "
                f"{SCENARIO_FORMAT_VERSION!r}, got {data.get('format_version')!r}",
                "format_version",
            )
        )
    if issues:
        raise ScenarioError(issues)

    scenario_id = _scenario_id(data, issues)
    description = _description(data, issues)
    inputs = _value_map(data, "inputs", ScenarioInput, issues, required=True)
    state = _value_map(data, "state", ScenarioState, issues, required=False)
    outputs = _value_map(data, "outputs", ScenarioOutput, issues, required=False)
    if not inputs:
        issues.append(
            _issue(
                "scenario_invalid_manifest",
                "inputs must be a non-empty object",
                "inputs",
            )
        )
    input_names = {item.name for item in inputs}
    for slot in state:
        if slot.init is None:
            continue
        parsed = parse_reference(slot.init, issues, f"state.{slot.name}.init")
        if parsed is None:
            continue
        prefix, name = parsed
        if prefix != "scenario.inputs" or name not in input_names:
            issues.append(
                _issue(
                    "scenario_invalid_manifest",
                    "state init must reference a declared scenario input",
                    f"state.{slot.name}.init",
                )
            )
    steps = _steps(data, root, inputs, state, outputs, issues)
    if issues:
        raise ScenarioError(issues)
    return Scenario(
        id=scenario_id,
        source=Path(path),
        description=description,
        inputs=inputs,
        state=state,
        outputs=outputs,
        steps=steps,
    )


def parse_reference(
    reference: Any, issues: list[dict[str, Any]], where: str
) -> tuple[str, str] | None:
    """Parse ``scenario.inputs.<name>`` / ``state.<name>`` / ``scenario.outputs.<name>``."""

    split = split_reference(reference)
    if split is None:
        issues.append(
            _issue(
                "scenario_reference_invalid",
                f"invalid binding reference {reference!r}; expected "
                "scenario.inputs.<name>, state.<name>, or scenario.outputs.<name>",
                where,
            )
        )
    return split


def split_reference(reference: Any) -> tuple[str, str] | None:
    """Non-reporting reference parser used after validation succeeds."""

    if not isinstance(reference, str) or not reference:
        return None
    for prefix in ("scenario.inputs.", "scenario.outputs.", "state."):
        if reference.startswith(prefix):
            name = reference[len(prefix) :]
            if _VALUE_NAME.fullmatch(name):
                return prefix[:-1], name
            return None
    return None


def _scenario_id(data: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    from inferref.suite.paths import validate_case_id

    try:
        return validate_case_id(data.get("id"), where="id")
    except ValueError as exc:
        issues.append(_issue("scenario_invalid_id", str(exc), "id"))
        return ""


def _description(data: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    value = data.get("description")
    if value is None:
        return ""
    if not isinstance(value, str) or not value.strip():
        issues.append(
            _issue(
                "scenario_invalid_manifest",
                "description must be a non-empty string when present",
                "description",
            )
        )
        return ""
    return value


def _value_map(
    data: dict[str, Any],
    field_name: str,
    cls: type,
    issues: list[dict[str, Any]],
    *,
    required: bool,
) -> tuple[Any, ...]:
    from inferref.suite.paths import portable_id_key, validate_case_id

    raw = data.get(field_name)
    if raw is None:
        if required:
            issues.append(
                _issue(
                    "scenario_invalid_manifest",
                    f"{field_name} must be an object",
                    field_name,
                )
            )
        return ()
    if not isinstance(raw, dict):
        issues.append(
            _issue(
                "scenario_invalid_manifest",
                f"{field_name} must be an object",
                field_name,
            )
        )
        return ()
    records: list[Any] = []
    portable_names: set[str] = set()
    for name, record in raw.items():
        where = f"{field_name}.{name}"
        try:
            valid_name = validate_case_id(name, where=where)
        except ValueError as exc:
            issues.append(
                _issue(
                    "scenario_invalid_manifest",
                    str(exc),
                    where,
                )
            )
            continue
        portable_key = portable_id_key(valid_name)
        if portable_key in portable_names:
            issues.append(
                _issue(
                    "scenario_invalid_manifest",
                    f"{field_name} name {name!r} collides on a portable filesystem",
                    where,
                )
            )
            continue
        portable_names.add(portable_key)
        if not isinstance(record, dict):
            issues.append(
                _issue(
                    "scenario_invalid_manifest",
                    f"{field_name} value must be an object",
                    where,
                )
            )
            continue
        kind = record.get("kind")
        if kind != "tensor":
            issues.append(
                _issue(
                    "scenario_invalid_manifest",
                    'v0.1 value records must have kind "tensor"',
                    f"{where}.kind",
                )
            )
        init = record.get("init")
        if cls is ScenarioState:
            if init is not None and not isinstance(init, str):
                issues.append(
                    _issue(
                        "scenario_invalid_manifest",
                        "state init must be a reference string",
                        f"{where}.init",
                    )
                )
                init = None
            records.append(ScenarioState(name=name, kind=kind, init=init))
        else:
            records.append(cls(name=name, kind=kind))
    return tuple(records)


def _steps(
    data: dict[str, Any],
    root: Path,
    inputs: tuple[ScenarioInput, ...],
    state: tuple[ScenarioState, ...],
    outputs: tuple[ScenarioOutput, ...],
    issues: list[dict[str, Any]],
) -> tuple[ScenarioStep, ...]:
    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list) or not raw_steps:
        issues.append(
            _issue(
                "scenario_invalid_manifest",
                "steps must be a non-empty array",
                "steps",
            )
        )
        return ()
    input_names = {item.name for item in inputs}
    state_names = {item.name for item in state}
    output_names = {item.name for item in outputs}
    steps: list[ScenarioStep] = []
    ids: set[str] = set()
    portable_ids: set[str] = set()
    initialized = {item.name for item in state if item.init is not None}
    written_outputs: set[str] = set()

    for index, record in enumerate(raw_steps):
        where = f"steps[{index}]"
        if not isinstance(record, dict):
            issues.append(
                _issue("scenario_invalid_manifest", "step must be an object", where)
            )
            continue
        step_id = _step_id(record, issues, where, ids, portable_ids)
        testcase_ref = record.get("testcase")
        testcase_path: Path | None = None
        if not isinstance(testcase_ref, str) or not testcase_ref:
            issues.append(
                _issue(
                    "scenario_invalid_manifest",
                    "step testcase must be a non-empty relative string",
                    f"{where}.testcase",
                )
            )
        else:
            try:
                testcase_path = resolve_contained_path(
                    root,
                    testcase_ref,
                    kind=f"scenario step {step_id!r} testcase",
                )
            except PathBoundaryError as exc:
                issues.append(
                    _issue("scenario_path_escape", str(exc), f"{where}.testcase")
                )
        input_bindings, output_bindings = _bindings(
            record,
            issues,
            where,
            input_names,
            state_names,
            output_names,
        )
        if testcase_path is not None and not testcase_path.is_dir():
            issues.append(
                _issue(
                    "scenario_testcase_invalid",
                    f"step testcase {testcase_ref!r} does not exist or is not a "
                    "directory",
                    f"{where}.testcase",
                )
            )
        written_in_step: set[str] = set()
        tc_inputs: dict[str, dict[str, Any]] | None = None
        tc_outputs: dict[str, dict[str, Any]] | None = None
        if testcase_path is not None and testcase_path.is_dir():
            tc_inputs, tc_outputs = _testcase_roles(testcase_path, step_id, issues)
        if tc_inputs is not None:
            for role, reference in input_bindings.items():
                role_where = f"{where}.bindings.inputs.{role}"
                prefix, _ = split_reference(reference)
                if role not in tc_inputs:
                    issues.append(
                        _issue(
                            "scenario_role_unknown",
                            f"input role {role!r} is not in testcase "
                            f"{testcase_ref!r}",
                            role_where,
                        )
                    )
                elif not _is_tensor_role(tc_inputs[role]):
                    issues.append(
                        _issue(
                            "scenario_role_kind_mismatch",
                            f"input role {role!r} is not a tensor",
                            role_where,
                        )
                    )
                elif prefix == "state":
                    slot = reference[len("state.") :]
                    if slot not in initialized:
                        issues.append(
                            _issue(
                                "scenario_state_uninitialized",
                                f"state slot {slot!r} is read before initialization",
                                role_where,
                            )
                        )
        if tc_outputs is not None:
            for role, reference in output_bindings.items():
                role_where = f"{where}.bindings.outputs.{role}"
                prefix, name = split_reference(reference)
                if role not in tc_outputs:
                    issues.append(
                        _issue(
                            "scenario_role_unknown",
                            f"output role {role!r} is not in testcase "
                            f"{testcase_ref!r}",
                            role_where,
                        )
                    )
                elif not _is_tensor_role(tc_outputs[role]):
                    issues.append(
                        _issue(
                            "scenario_role_kind_mismatch",
                            f"output role {role!r} is not a tensor",
                            role_where,
                        )
                    )
                if prefix == "state" and name in written_in_step:
                    issues.append(
                        _issue(
                            "scenario_state_written_twice",
                            f"state slot {name!r} is written twice in one step",
                            role_where,
                        )
                    )
                if prefix == "state":
                    written_in_step.add(name)
                elif prefix == "scenario.outputs":
                    written_outputs.add(name)

        step_comparison = None
        raw_comparison = record.get("comparison")
        if raw_comparison is not None:
            if not isinstance(raw_comparison, dict):
                issues.append(
                    _issue(
                        "scenario_invalid_manifest",
                        "step comparison must be an object",
                        f"{where}.comparison",
                    )
                )
            else:
                from inferref.comparison.schema import ComparisonSpec, ComparisonSpecValidationError

                try:
                    step_comparison = ComparisonSpec.from_dict(raw_comparison)
                    step_comparison.validate(check_registry=True)
                except (ComparisonSpecValidationError, ValueError) as exc:
                    issues.append(
                        _issue(
                            "scenario_invalid_manifest",
                            f"step comparison is invalid: {exc}",
                            f"{where}.comparison",
                        )
                    )

        if testcase_path is not None:
            steps.append(
                ScenarioStep(
                    id=step_id or f"step-{index}",
                    testcase_ref=(
                        testcase_ref if isinstance(testcase_ref, str) else ""
                    ),
                    testcase=testcase_path,
                    input_bindings=input_bindings,
                    output_bindings=output_bindings,
                    comparison=step_comparison,
                )
            )
        initialized.update(written_in_step)
    for name in sorted(output_names - written_outputs):
        issues.append(
            _issue(
                "scenario_output_unwritten",
                f"declared scenario output {name!r} is never written by a step",
                f"outputs.{name}",
            )
        )
    return tuple(steps)


def _step_id(
    record: dict[str, Any],
    issues: list[dict[str, Any]],
    where: str,
    ids: set[str],
    portable_ids: set[str],
) -> str:
    from inferref.suite.paths import portable_id_key, validate_case_id

    raw = record.get("id")
    try:
        value = validate_case_id(raw, where=f"{where}.id")
    except ValueError as exc:
        issues.append(_issue("scenario_bad_step_id", str(exc), f"{where}.id"))
        return ""
    if value in ids or portable_id_key(value) in portable_ids:
        issues.append(
            _issue(
                "scenario_duplicate_step_id",
                f"duplicate step id {value!r}",
                f"{where}.id",
            )
        )
        return ""
    ids.add(value)
    portable_ids.add(portable_id_key(value))
    return value


def _bindings(
    record: dict[str, Any],
    issues: list[dict[str, Any]],
    where: str,
    input_names: set[str],
    state_names: set[str],
    output_names: set[str],
) -> tuple[dict[str, str], dict[str, str]]:
    raw = record.get("bindings", {})
    if raw is None:
        return {}, {}
    if not isinstance(raw, dict):
        issues.append(
            _issue(
                "scenario_invalid_manifest",
                "bindings must be an object",
                f"{where}.bindings",
            )
        )
        return {}, {}
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    for side, collection, result in (
        ("inputs", raw.get("inputs", {}), inputs),
        ("outputs", raw.get("outputs", {}), outputs),
    ):
        if collection is None:
            continue
        if not isinstance(collection, dict):
            issues.append(
                _issue(
                    "scenario_invalid_manifest",
                    f"bindings.{side} must be an object",
                    f"{where}.bindings.{side}",
                )
            )
            continue
        for role, reference in collection.items():
            role_where = f"{where}.bindings.{side}.{role}"
            if not isinstance(role, str) or not _VALUE_NAME.fullmatch(role):
                issues.append(
                    _issue(
                        "scenario_invalid_manifest",
                        f"bound role {role!r} is not a valid role name",
                        role_where,
                    )
                )
                continue
            parsed = parse_reference(reference, issues, role_where)
            if parsed is None:
                continue
            prefix, name = parsed
            if side == "inputs" and prefix not in ("scenario.inputs", "state"):
                issues.append(
                    _issue(
                        "scenario_reference_invalid",
                        "input bindings must source scenario.inputs or state",
                        role_where,
                    )
                )
                continue
            if side == "outputs" and prefix not in ("state", "scenario.outputs"):
                issues.append(
                    _issue(
                        "scenario_reference_invalid",
                        "output bindings must target state or scenario.outputs",
                        role_where,
                    )
                )
                continue
            if side == "inputs" and prefix == "scenario.inputs" and name not in input_names:
                issues.append(
                    _issue(
                        "scenario_source_undeclared",
                        f"scenario input {name!r} is not declared",
                        role_where,
                    )
                )
                continue
            if side == "inputs" and prefix == "state" and name not in state_names:
                issues.append(
                    _issue(
                        "scenario_source_undeclared",
                        f"state slot {name!r} is not declared",
                        role_where,
                    )
                )
                continue
            if side == "outputs" and prefix == "state" and name not in state_names:
                issues.append(
                    _issue(
                        "scenario_destination_undeclared",
                        f"state slot {name!r} is not declared",
                        role_where,
                    )
                )
                continue
            if (
                side == "outputs"
                and prefix == "scenario.outputs"
                and name not in output_names
            ):
                issues.append(
                    _issue(
                        "scenario_destination_undeclared",
                        f"scenario output {name!r} is not declared",
                        role_where,
                    )
                )
                continue
            result[role] = reference
    return inputs, outputs


def _testcase_roles(
    testcase_path: Path, step_id: str, issues: list[dict[str, Any]]
) -> tuple[
    dict[str, dict[str, Any]] | None, dict[str, dict[str, Any]] | None
]:
    validation = validate_testcase(testcase_path)
    if not validation.valid:
        issues.append(
            _issue(
                "scenario_testcase_invalid",
                f"step {step_id!r} testcase is invalid: "
                + "; ".join(issue.message for issue in validation.errors[:2]),
                f"steps[{step_id}].testcase",
            )
        )
        return None, None
    manifest = validation.manifest
    inputs = {
        item["name"]: item
        for item in manifest.get("inputs", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    outputs = {
        item["name"]: item
        for item in manifest.get("outputs", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    return inputs, outputs


def _is_tensor_role(record: dict[str, Any]) -> bool:
    return record.get("kind") == "tensor" or (
        isinstance(record.get("shape"), list) and isinstance(record.get("dtype"), str)
    )


def _issue(code: str, message: str, where: str = "") -> dict[str, str]:
    result: dict[str, str] = {"code": code, "message": message}
    if where:
        result["where"] = where
    return result


__all__ = [
    "SCENARIO_FORMAT",
    "SCENARIO_FORMAT_VERSION",
    "SCENARIO_MANIFEST",
    "Scenario",
    "ScenarioError",
    "ScenarioInput",
    "ScenarioOutput",
    "ScenarioState",
    "ScenarioStep",
    "load_scenario",
    "parse_reference",
    "split_reference",
]
