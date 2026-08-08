"""``inferref-contract`` v0.1 schema parsing and validation.

Contract Schema v0.1 section 5. A schema-only descriptor is declarative: the
registry builds the default validators (role checks plus the relation
evaluator) from it. Python validators are the escape hatch for what the schema
cannot express (section 6).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from inferref.contracts.relations import (
    RelationEvaluationError,
    RelationSyntaxError,
    evaluate,
    parse,
    relation_roles,
)

CONTRACT_FORMAT = "inferref-contract"
CONTRACT_FORMAT_VERSION = "0.1"

#: Feature vocabulary (section 7.1), shared with adapter capabilities.
FEATURE_VOCABULARY = frozenset(
    {"multiple_outputs", "strided_inputs", "alias_effects", "mutation_effects"}
)

#: Effect vocabulary (section 7.2) mapped onto the feature vocabulary.
EFFECT_VOCABULARY = frozenset({"pure", "alias_effects", "mutation_effects"})
_EFFECT_TO_FEATURE = {
    "pure": None,
    "alias_effects": "alias_effects",
    "mutation_effects": "mutation_effects",
}

_ROLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Section 4: at least two /-separated segments (e.g. a/v1), a final v<digits>
# segment and lowercase [a-z0-9-] segments in between.
_CONTRACT_ID = re.compile(
    r"^[a-z0-9][a-z0-9-]*(?:/[a-z0-9][a-z0-9-]*)*/v[0-9]+$"
)

InputValidator = Callable[[Mapping[str, dict[str, Any]]], Sequence[str]]
OutputValidator = Callable[[Mapping[str, dict[str, Any]]], Sequence[str]]
BoundaryValidator = Callable[
    [Mapping[str, dict[str, Any]], Mapping[str, dict[str, Any]]], Sequence[str]
]


class ContractSchemaError(ValueError):
    """A contract descriptor or file is invalid.

    ``code`` is one of the stable error codes from section 14
    (``contract_schema_invalid``, ``contract_relation_syntax`` or
    ``contract_relation_role``).
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class ExecutableContract:
    """One versioned semantic operation profile an engine can implement.

    This is the runtime contract object shared by the old
    ``inferref.testcase.contracts`` API and the new registry (section 6.1).
    ``features``/``effects`` are additive fields with defaults so existing
    constructions do not change. ``features`` holds the effective merged set
    after registry loading; ``effects`` records the declared effects.
    """

    id: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    validate_inputs: InputValidator
    validate_outputs: OutputValidator | None = None
    validate_relation: BoundaryValidator | None = None
    description: str = ""
    features: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "description": self.description,
            "features": list(self.features),
            "effects": list(self.effects),
        }
        relations = getattr(self, "_relations", ())
        if relations:
            out["relations"] = len(relations)
        return out


def validate_contract_id(contract_id: Any) -> str:
    """Validate the section 4 versioned-path rules and return the ID."""

    if not isinstance(contract_id, str) or _CONTRACT_ID.fullmatch(contract_id) is None:
        raise ContractSchemaError(
            "contract_schema_invalid",
            (
                f"invalid contract id {contract_id!r}: expected at least two "
                "/-separated segments, lowercase [a-z0-9-] segments and a final "
                "v<digits> version segment"
            ),
        )
    return contract_id


def validate_contract_roles(
    inputs: Sequence[str], outputs: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate role names for a Python-constructed contract."""

    input_roles = tuple(inputs)
    output_roles = tuple(outputs)
    for field_name, roles in (("inputs", input_roles), ("outputs", output_roles)):
        if not roles:
            raise ContractSchemaError(
                "contract_schema_invalid", f"{field_name} must not be empty"
            )
        for role in roles:
            if not isinstance(role, str) or _ROLE_NAME.fullmatch(role) is None:
                raise ContractSchemaError(
                    "contract_schema_invalid",
                    f"invalid {field_name} role name {role!r}: expected "
                    "^[A-Za-z_][A-Za-z0-9_]*$",
                )
    shared = set(input_roles) & set(output_roles)
    if shared:
        raise ContractSchemaError(
            "contract_schema_invalid",
            (
                "input and output role names must be unique across both maps in "
                "v0.1; the relation language cannot disambiguate shared name(s): "
                + ", ".join(sorted(shared))
            ),
        )
    return input_roles, output_roles


def _role_map(
    value: Any,
    field_name: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ContractSchemaError(
            "contract_schema_invalid", f"{field_name} must be an object"
        )
    if not value:
        raise ContractSchemaError(
            "contract_schema_invalid", f"{field_name} must not be empty"
        )
    roles: dict[str, dict[str, Any]] = {}
    for raw_name, record in value.items():
        if not isinstance(raw_name, str) or _ROLE_NAME.fullmatch(raw_name) is None:
            raise ContractSchemaError(
                "contract_schema_invalid",
                f"invalid {field_name} role name {raw_name!r}: expected "
                "^[A-Za-z_][A-Za-z0-9_]*$",
            )
        if not isinstance(record, Mapping):
            raise ContractSchemaError(
                "contract_schema_invalid",
                f"{field_name}.{raw_name} must be a role record object",
            )
        kind = record.get("kind")
        if kind != "tensor":
            raise ContractSchemaError(
                "contract_schema_invalid",
                f"{field_name}.{raw_name}.kind must be 'tensor' in v0.1, "
                f"got {kind!r}",
            )
        roles[raw_name] = {"kind": kind}
    return roles


def _vocabulary(
    value: Any,
    field_name: str,
    vocabulary: frozenset[str],
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise ContractSchemaError(
            "contract_schema_invalid", f"{field_name} must be a string array"
        )
    unknown = sorted(set(value) - vocabulary)
    if unknown:
        raise ContractSchemaError(
            "contract_schema_invalid",
            f"unknown {field_name} value(s): {', '.join(unknown)}",
        )
    return tuple(dict.fromkeys(value))


def _effective_features(
    features: Sequence[str], effects: Sequence[str]
) -> tuple[str, ...]:
    merged = set(features)
    for effect in effects:
        feature = _EFFECT_TO_FEATURE[effect]
        if feature is not None:
            merged.add(feature)
    if "pure" in effects and merged & {"alias_effects", "mutation_effects"}:
        raise ContractSchemaError(
            "contract_schema_invalid",
            "'pure' must not coexist with 'alias_effects' or 'mutation_effects' "
            "(checked on the merged effective set, so features and effects "
            "combined are validated too)",
        )
    return tuple(sorted(merged))


def _validate_relation_roles(
    expression: str,
    input_roles: set[str],
    output_roles: set[str],
) -> None:
    declared = input_roles | output_roles
    roles = relation_roles(expression)
    missing = [role for role in roles if role not in declared]
    if missing:
        raise ContractSchemaError(
            "contract_relation_role",
            (
                f"relation {expression!r} references undeclared role(s): "
                + ", ".join(missing)
            ),
        )


def _relation_issue(
    expression: str,
    inputs: Mapping[str, dict[str, Any]],
    outputs: Mapping[str, dict[str, Any]],
) -> str | None:
    roles: dict[str, Any] = {**inputs, **outputs}
    try:
        holds, message = evaluate(expression, roles)
    except RelationEvaluationError as exc:
        return f"relation {expression!r} failed ({exc})"
    if holds:
        return None
    return message


def _generated_inputs(
    input_roles: tuple[str, ...],
    input_relations: tuple[str, ...],
) -> InputValidator:
    def validate_inputs(inputs: Mapping[str, dict[str, Any]]) -> list[str]:
        issues: list[str] = []
        missing_roles = False
        for role in input_roles:
            record = inputs.get(role)
            if record is None:
                issues.append(f"{role} is a required contract input")
                missing_roles = True
                continue
            if not isinstance(record, dict):
                issues.append(f"{role} must be a tensor")
                continue
            if not isinstance(record.get("dtype"), str) or not record["dtype"]:
                issues.append(f"{role} must declare a non-empty dtype")
        if not missing_roles:
            for expression in input_relations:
                message = _relation_issue(expression, inputs, {})
                if message is not None:
                    issues.append(message)
        return issues

    validate_inputs._inferref_dtype_checked = True  # type: ignore[attr-defined]
    return validate_inputs


def _generated_outputs(
    output_roles: tuple[str, ...],
    output_relations: tuple[str, ...],
) -> OutputValidator:
    def validate_outputs(outputs: Mapping[str, dict[str, Any]]) -> list[str]:
        issues: list[str] = []
        missing_roles = False
        for role in output_roles:
            record = outputs.get(role)
            if record is None:
                issues.append(f"{role} is a required contract output")
                missing_roles = True
                continue
            if not isinstance(record, dict):
                issues.append(f"{role} must be a tensor")
                continue
            if not isinstance(record.get("dtype"), str) or not record["dtype"]:
                issues.append(f"{role} must declare a non-empty dtype")
        if not missing_roles:
            for expression in output_relations:
                message = _relation_issue(expression, {}, outputs)
                if message is not None:
                    issues.append(message)
        return issues

    validate_outputs._inferref_dtype_checked = True  # type: ignore[attr-defined]
    return validate_outputs


def _generated_relation(
    mixed_relations: tuple[str, ...],
) -> BoundaryValidator:
    def validate_relation(
        inputs: Mapping[str, dict[str, Any]],
        outputs: Mapping[str, dict[str, Any]],
    ) -> list[str]:
        issues: list[str] = []
        for expression in mixed_relations:
            message = _relation_issue(expression, inputs, outputs)
            if message is not None:
                issues.append(message)
        return issues

    return validate_relation


def build_contract(descriptor: Mapping[str, Any]) -> ExecutableContract:
    """Validate a descriptor and build its executable contract.

    Raises :class:`ContractSchemaError` (code ``contract_schema_invalid``,
    ``contract_relation_syntax`` or ``contract_relation_role``) when the
    descriptor is malformed. Unknown top-level fields are ignored for forward
    compatibility (section 5.1).
    """

    if not isinstance(descriptor, Mapping):
        raise ContractSchemaError(
            "contract_schema_invalid",
            "contract descriptor must be an object",
        )
    if descriptor.get("format") != CONTRACT_FORMAT:
        raise ContractSchemaError(
            "contract_schema_invalid",
            f"format must be {CONTRACT_FORMAT!r}, got {descriptor.get('format')!r}",
        )
    if descriptor.get("format_version") != CONTRACT_FORMAT_VERSION:
        raise ContractSchemaError(
            "contract_schema_invalid",
            (
                f"format_version must be {CONTRACT_FORMAT_VERSION!r}, got "
                f"{descriptor.get('format_version')!r}"
            ),
        )
    contract_id = validate_contract_id(descriptor.get("id"))

    description_text = ""
    if "description" in descriptor:
        description = descriptor["description"]
        if not isinstance(description, str) or not description.strip():
            raise ContractSchemaError(
                "contract_schema_invalid",
                "description must be a non-empty string",
            )
        description_text = description

    input_roles = tuple(_role_map(descriptor.get("inputs"), "inputs"))
    output_roles = tuple(_role_map(descriptor.get("outputs"), "outputs"))
    validate_contract_roles(input_roles, output_roles)

    raw_relations = descriptor.get("relations", [])
    if not isinstance(raw_relations, list) or not all(
        isinstance(item, str) for item in raw_relations
    ):
        raise ContractSchemaError(
            "contract_schema_invalid", "relations must be a string array"
        )
    relations: tuple[str, ...] = tuple(dict.fromkeys(raw_relations))
    for expression in relations:
        try:
            parse(expression)
        except RelationSyntaxError as exc:
            raise ContractSchemaError(
                "contract_relation_syntax",
                f"relation {expression!r} does not parse: {exc}",
            ) from exc
        _validate_relation_roles(expression, set(input_roles), set(output_roles))

    features = _vocabulary(descriptor.get("features"), "features", FEATURE_VOCABULARY)
    effects = _vocabulary(descriptor.get("effects"), "effects", EFFECT_VOCABULARY)
    effective_features = _effective_features(features, effects)

    input_only: list[str] = []
    output_only: list[str] = []
    mixed: list[str] = []
    for expression in relations:
        roles = set(relation_roles(expression))
        reads_input = bool(roles & set(input_roles))
        reads_output = bool(roles & set(output_roles))
        if reads_input and not reads_output:
            input_only.append(expression)
        elif reads_output and not reads_input:
            output_only.append(expression)
        else:
            mixed.append(expression)

    contract = ExecutableContract(
        id=contract_id,
        inputs=input_roles,
        outputs=output_roles,
        validate_inputs=_generated_inputs(input_roles, tuple(input_only)),
        validate_outputs=_generated_outputs(output_roles, tuple(output_only)),
        validate_relation=_generated_relation(tuple(mixed)),
        description=description_text,
        features=effective_features,
        effects=effects,
    )
    object.__setattr__(contract, "_relations", relations)
    # Structured marker so callers can tell schema-generated validators apart
    # from Python escape-hatch validators (used by testcase validation to
    # classify relation failures without string sniffing).
    object.__setattr__(contract, "_inferref_schema_generated", True)
    return contract


def merge_features(contract: ExecutableContract) -> tuple[str, ...]:
    """Return the effective feature set for a Python-constructed contract."""

    return _effective_features(contract.features, contract.effects)
