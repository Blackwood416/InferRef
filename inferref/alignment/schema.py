"""Strict wire records for InferRef Alignment/Evidence/Event v0.1.

The alignment store is an audit log, so permissive ``dict.get`` based parsing
would be dangerous: a typo in a digest or a newly introduced status could make
old evidence look comparable.  These records therefore validate their known
shape and reject unknown top-level fields.  The intentionally data-oriented
aggregate fields (``boundaries``, ``narrowing`` and friends) are validated as
JSON objects and are owned by later alignment workers.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALIGNMENT_FORMAT = "inferref-alignment"
ALIGNMENT_FORMAT_VERSION = "0.1"
ALIGNMENT_READ_VERSIONS = (ALIGNMENT_FORMAT_VERSION,)

EVIDENCE_FORMAT = "inferref-evidence"
EVIDENCE_FORMAT_VERSION = "0.1"
EVIDENCE_READ_VERSIONS = (EVIDENCE_FORMAT_VERSION,)

EVENT_FORMAT = "inferref-alignment-event"
EVENT_FORMAT_VERSION = "0.1"
EVENT_READ_VERSIONS = (EVENT_FORMAT_VERSION,)

# Public aliases used by code which follows the shorter names from the spec.
ALIGNMENT_VERSION = ALIGNMENT_FORMAT_VERSION
EVIDENCE_VERSION = EVIDENCE_FORMAT_VERSION
EVENT_VERSION = EVENT_FORMAT_VERSION

SESSION_STATUSES = ("new", "aligning", "blocked", "verified", "stale", "error")
BOUNDARY_STATES = (
    "untested",
    "verified",
    "failing",
    "blocked",
    "unsupported",
    "stale",
)
FIXTURE_KINDS = (
    "synthetic",
    "reference_derived",
    "real_weight",
    "end_to_end",
    "other",
)
EVIDENCE_TYPES = ("numerical_comparison", "manual_note")
RESERVED_EVIDENCE_TYPES = ("probe_comparison", "artifact_comparison")
EVIDENCE_RESULTS = ("pass", "fail", "error", "unsupported", "recorded")

EVENT_TYPES = (
    "session_initialized",
    "plan_selected",
    "testcase_reused",
    "testcase_extracted",
    "boundary_verified",
    "boundary_failed",
    "boundary_unsupported",
    "execution_error",
    "first_divergence_changed",
    "failure_domain_changed",
    "note_recorded",
    "classification_rejected",
    "classification_overridden",
    "session_verified",
)

NOTE_CATEGORIES = (
    "missing-observation",
    "custom-tool",
    "external-blocker",
    "milestone",
    "classification-reject",
    "classification-override",
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_UUID_ID = re.compile(r"^(?:al|ev)-[0-9a-fA-F]{8,}(?:-[0-9a-fA-F]{4,})*$")


class AlignmentSchemaError(ValueError):
    """Raised when a v0.1 alignment record is malformed."""

    def __init__(self, message: str, *, issues: list[dict[str, Any]] | None = None):
        self.issues = issues or [{"code": "alignment_schema", "message": message}]
        super().__init__(message)


# These aliases make the failure mode discoverable without requiring clients to
# know which of the three records caused it.
AlignmentValidationError = AlignmentSchemaError
EvidenceValidationError = AlignmentSchemaError
EventValidationError = AlignmentSchemaError


def _copy_json(value: Any) -> Any:
    """Copy a JSON-shaped value and reject Python-only objects/NaN."""

    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AlignmentSchemaError(f"value is not valid JSON data: {exc}") from exc


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AlignmentSchemaError(f"{where} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise AlignmentSchemaError(f"{where} keys must be strings")
    return value


def _strict_object(value: Any, allowed: set[str], where: str) -> dict[str, Any]:
    result = _object(value, where)
    unknown = sorted(set(result) - allowed)
    if unknown:
        raise AlignmentSchemaError(
            f"{where} contains unknown field(s): {', '.join(unknown)}"
        )
    return result


def _required(value: Mapping[str, Any], name: str, where: str) -> Any:
    if name not in value:
        raise AlignmentSchemaError(f"{where}.{name} is required")
    return value[name]


def _string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        expected = "a non-empty string" if nonempty else "a string"
        raise AlignmentSchemaError(f"{where} must be {expected}")
    if "\x00" in value:
        raise AlignmentSchemaError(f"{where} must not contain NUL")
    return value


def _string_array(value: Any, where: str, *, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AlignmentSchemaError(f"{where} must be a string array")
    result = tuple(_string(item, f"{where}[{index}]") for index, item in enumerate(value))
    if unique and len(set(result)) != len(result):
        raise AlignmentSchemaError(f"{where} must not contain duplicates")
    return result


def _int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AlignmentSchemaError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise AlignmentSchemaError(f"{where} must be at least {minimum}")
    return value


def _bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise AlignmentSchemaError(f"{where} must be a boolean")
    return value


def _digest(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise AlignmentSchemaError(f"{where} must be a lowercase SHA-256 hex digest")
    return value


def _timestamp(value: Any, where: str) -> str:
    text = _string(value, where)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AlignmentSchemaError(f"{where} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise AlignmentSchemaError(f"{where} must include a timezone")
    return text


def utc_now() -> str:
    """Return the canonical UTC timestamp used for generated records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _id(value: Any, where: str, prefix: str) -> str:
    result = _string(value, where)
    if not result.startswith(prefix) or not _ID.fullmatch(result[len(prefix) :]):
        raise AlignmentSchemaError(
            f"{where} must start with {prefix!r} and contain only portable id characters"
        )
    return result


def _digest_or_none(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _digest(value, where)


def _validate_extensions(value: Any, where: str) -> dict[str, Any]:
    result = _object(value, where)
    # Extensions are the one explicit forward-compatible namespace.  Their
    # values still need to be JSON data so a record remains serializable.
    return _copy_json(result)


def _validate_trace_ref(value: Any, where: str) -> dict[str, Any]:
    raw = _strict_object(value, {"path", "metadata_sha256"}, where)
    return {
        "path": _string(_required(raw, "path", where), f"{where}.path"),
        "metadata_sha256": _digest(
            _required(raw, "metadata_sha256", where), f"{where}.metadata_sha256"
        ),
    }


def _validate_adapter_ref(value: Any, where: str) -> dict[str, Any]:
    raw = _strict_object(value, {"path", "normalized_sha256"}, where)
    return {
        "path": _string(_required(raw, "path", where), f"{where}.path"),
        "normalized_sha256": _digest(
            _required(raw, "normalized_sha256", where),
            f"{where}.normalized_sha256",
        ),
    }


def _validate_source(value: Any, where: str = "source") -> dict[str, Any]:
    raw = _strict_object(value, {"trace"}, where)
    return {"trace": _validate_trace_ref(_required(raw, "trace", where), f"{where}.trace")}


def _validate_target(value: Any, where: str = "target") -> dict[str, Any]:
    raw = _strict_object(value, {"model", "engine", "adapter"}, where)
    return {
        "model": _string(_required(raw, "model", where), f"{where}.model"),
        "engine": _string(_required(raw, "engine", where), f"{where}.engine"),
        "adapter": _validate_adapter_ref(
            _required(raw, "adapter", where), f"{where}.adapter"
        ),
    }


@dataclass
class AlignmentSnapshot:
    """Materialized Alignment Session v0.1 snapshot."""

    session_id: str
    created_at: str
    updated_at: str
    last_event_seq: int
    source: dict[str, Any]
    target: dict[str, Any]
    scope: tuple[str, ...] = ()
    required_fixture_kinds: tuple[str, ...] = ("reference_derived",)
    status: str = "new"
    boundaries: dict[str, Any] = field(default_factory=dict)
    narrowing: dict[str, Any] = field(default_factory=dict)
    classification: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)
    format: str = ALIGNMENT_FORMAT
    format_version: str = ALIGNMENT_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format": self.format,
            "format_version": self.format_version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_event_seq": self.last_event_seq,
            "source": _copy_json(self.source),
            "target": _copy_json(self.target),
            "scope": list(self.scope),
            "required_fixture_kinds": list(self.required_fixture_kinds),
            "status": self.status,
            "boundaries": _copy_json(self.boundaries),
            "narrowing": _copy_json(self.narrowing),
            "classification": _copy_json(self.classification),
            "metrics": _copy_json(self.metrics),
            "extensions": _copy_json(self.extensions),
        }

    def validate(self) -> None:
        _validate_snapshot_dict(self.to_dict() if False else {
            "format": self.format,
            "format_version": self.format_version,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_event_seq": self.last_event_seq,
            "source": self.source,
            "target": self.target,
            "scope": list(self.scope),
            "required_fixture_kinds": list(self.required_fixture_kinds),
            "status": self.status,
            "boundaries": self.boundaries,
            "narrowing": self.narrowing,
            "classification": self.classification,
            "metrics": self.metrics,
            "extensions": self.extensions,
        })

    @classmethod
    def from_dict(cls, data: Any) -> AlignmentSnapshot:
        parsed = _validate_snapshot_dict(data)
        return cls(**parsed)

    @property
    def format_name(self) -> str:
        return self.format


# Names used interchangeably in the implementation plan and integrations.
AlignmentSession = AlignmentSnapshot
SessionSnapshot = AlignmentSnapshot


def _validate_snapshot_dict(data: Any) -> dict[str, Any]:
    allowed = {
        "format",
        "format_version",
        "session_id",
        "created_at",
        "updated_at",
        "last_event_seq",
        "source",
        "target",
        "scope",
        "required_fixture_kinds",
        "status",
        "boundaries",
        "narrowing",
        "classification",
        "metrics",
        "extensions",
    }
    raw = _strict_object(data, allowed, "alignment snapshot")
    fmt = _string(_required(raw, "format", "alignment snapshot"), "format")
    if fmt != ALIGNMENT_FORMAT:
        raise AlignmentSchemaError(f"format must be {ALIGNMENT_FORMAT!r}, got {fmt!r}")
    version = _string(
        _required(raw, "format_version", "alignment snapshot"), "format_version"
    )
    if version not in ALIGNMENT_READ_VERSIONS:
        raise AlignmentSchemaError(f"unsupported alignment format_version {version!r}")
    session_id = _id(_required(raw, "session_id", "alignment snapshot"), "session_id", "al-")
    created_at = _timestamp(_required(raw, "created_at", "alignment snapshot"), "created_at")
    updated_at = _timestamp(_required(raw, "updated_at", "alignment snapshot"), "updated_at")
    last_event_seq = _int(
        _required(raw, "last_event_seq", "alignment snapshot"),
        "last_event_seq",
        minimum=0,
    )
    scope = _string_array(_required(raw, "scope", "alignment snapshot"), "scope")
    fixture_kinds = _string_array(
        _required(raw, "required_fixture_kinds", "alignment snapshot"),
        "required_fixture_kinds",
    )
    invalid_fixture_kinds = sorted(set(fixture_kinds) - set(FIXTURE_KINDS))
    if invalid_fixture_kinds:
        raise AlignmentSchemaError(
            "required_fixture_kinds contains unsupported kind(s): "
            + ", ".join(invalid_fixture_kinds)
        )
    status = _string(_required(raw, "status", "alignment snapshot"), "status")
    if status not in SESSION_STATUSES:
        raise AlignmentSchemaError(f"unknown alignment session status {status!r}")
    aggregate: dict[str, dict[str, Any]] = {}
    for name in ("boundaries", "narrowing", "classification", "metrics"):
        aggregate[name] = _object(
            _required(raw, name, "alignment snapshot"), f"alignment snapshot.{name}"
        )
    extensions = _validate_extensions(
        raw.get("extensions", {}), "alignment snapshot.extensions"
    )
    return {
        "format": fmt,
        "format_version": version,
        "session_id": session_id,
        "created_at": created_at,
        "updated_at": updated_at,
        "last_event_seq": last_event_seq,
        "source": _validate_source(_required(raw, "source", "alignment snapshot")),
        "target": _validate_target(_required(raw, "target", "alignment snapshot")),
        "scope": scope,
        "required_fixture_kinds": fixture_kinds,
        "status": status,
        "boundaries": aggregate["boundaries"],
        "narrowing": aggregate["narrowing"],
        "classification": aggregate["classification"],
        "metrics": aggregate["metrics"],
        "extensions": extensions,
    }


def _validate_subject(value: Any, where: str = "subject") -> dict[str, str]:
    raw = _strict_object(value, {"boundary", "name"}, where)
    boundary = _string(_required(raw, "boundary", where), f"{where}.boundary")
    name = _string(_required(raw, "name", where), f"{where}.name")
    return {"boundary": boundary, "name": name}


def _validate_fixture(value: Any, where: str = "fixture") -> dict[str, Any]:
    raw = _strict_object(value, {"kind", "tags"}, where)
    kind = _string(_required(raw, "kind", where), f"{where}.kind")
    if kind not in FIXTURE_KINDS:
        raise AlignmentSchemaError(f"{where}.kind has unsupported fixture kind {kind!r}")
    tags_value = raw.get("tags", [])
    tags = _string_array(tags_value, f"{where}.tags")
    return {"kind": kind, "tags": list(tags)}


def _validate_first_failure(value: Any, where: str = "first_failure") -> dict[str, Any]:
    raw = _strict_object(
        value,
        {"output", "max_abs_error", "max_rel_error", "mismatch_count", "index", "message"},
        where,
    )
    result: dict[str, Any] = {}
    if "output" in raw:
        result["output"] = _string(raw["output"], f"{where}.output")
    if "max_abs_error" in raw:
        error = raw["max_abs_error"]
        if isinstance(error, bool) or not isinstance(error, (int, float)) or not math.isfinite(error):
            raise AlignmentSchemaError(f"{where}.max_abs_error must be finite numeric")
        result["max_abs_error"] = error
    if "max_rel_error" in raw:
        error = raw["max_rel_error"]
        if isinstance(error, bool) or not isinstance(error, (int, float)) or not math.isfinite(error):
            raise AlignmentSchemaError(f"{where}.max_rel_error must be finite numeric")
        result["max_rel_error"] = error
    if "mismatch_count" in raw:
        result["mismatch_count"] = _int(raw["mismatch_count"], f"{where}.mismatch_count", minimum=0)
    if "index" in raw:
        result["index"] = _int(raw["index"], f"{where}.index", minimum=0)
    if "message" in raw:
        result["message"] = _string(raw["message"], f"{where}.message")
    return result


def _validate_observations(value: Any, where: str = "observations") -> dict[str, Any]:
    raw = _object(value, where)
    # ``inputs_match`` is deliberately not an accepted fact in 0.10.  A
    # reference testcase input is not an engine-observed input.
    if "inputs_match" in raw:
        raise AlignmentSchemaError(
            "observations.inputs_match is not available in Alignment/Evidence v0.1"
        )
    allowed = {
        "provided_inputs_valid",
        "engine_inputs_observed",
        "engine_input_observation",
        "outputs_match",
        "layout_match",
        "first_failure",
        "output_roles",
        "comparison_status",
        "metrics",
        "diagnostics",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise AlignmentSchemaError(
            f"{where} contains unknown field(s): {', '.join(unknown)}"
        )
    result: dict[str, Any] = {}
    for name in (
        "provided_inputs_valid",
        "engine_inputs_observed",
        "outputs_match",
        "layout_match",
    ):
        if name in raw:
            result[name] = _bool(raw[name], f"{where}.{name}")
    observed = result.get("engine_inputs_observed")
    if observed is True and "engine_input_observation" not in raw:
        raise AlignmentSchemaError(
            f"{where}.engine_inputs_observed=true requires typed engine_input_observation"
        )
    if "engine_input_observation" in raw:
        result["engine_input_observation"] = _object(
            raw["engine_input_observation"], f"{where}.engine_input_observation"
        )
    if "first_failure" in raw and raw["first_failure"] is not None:
        result["first_failure"] = _validate_first_failure(raw["first_failure"], f"{where}.first_failure")
    if "output_roles" in raw:
        result["output_roles"] = list(_string_array(raw["output_roles"], f"{where}.output_roles"))
    if "comparison_status" in raw:
        result["comparison_status"] = _string(raw["comparison_status"], f"{where}.comparison_status")
    if "metrics" in raw:
        result["metrics"] = _copy_json(_object(raw["metrics"], f"{where}.metrics"))
    if "diagnostics" in raw:
        if not isinstance(raw["diagnostics"], list):
            raise AlignmentSchemaError(f"{where}.diagnostics must be an array")
        result["diagnostics"] = _copy_json(raw["diagnostics"])
    return result


def _validate_artifact_ref(value: Any, where: str, digest_name: str) -> dict[str, str]:
    raw = _strict_object(value, {"path", digest_name}, where)
    path = _string(_required(raw, "path", where), f"{where}.path")
    # Internal paths are checked against the concrete session root by store.py;
    # schema validation still rejects obvious absolute/traversal syntax.
    if (
        path.startswith(("/", "\\"))
        or ":" in path.split("/", 1)[0]
        or "\\" in path
        or any(part in ("", ".", "..") for part in path.split("/"))
    ):
        raise AlignmentSchemaError(f"{where}.path must be a relative POSIX path")
    return {"path": path, digest_name: _digest(_required(raw, digest_name, where), f"{where}.{digest_name}")}


def _validate_engine_identity(value: Any, where: str = "engine_identity") -> dict[str, Any]:
    raw = _strict_object(
        value,
        {"kind", "revision", "dirty", "path_sha256", "files"},
        where,
    )
    kind = _string(_required(raw, "kind", where), f"{where}.kind")
    revision = _string(_required(raw, "revision", where), f"{where}.revision")
    dirty = _bool(_required(raw, "dirty", where), f"{where}.dirty")
    result: dict[str, Any] = {"kind": kind, "revision": revision, "dirty": dirty}
    if "path_sha256" in raw:
        result["path_sha256"] = _digest(raw["path_sha256"], f"{where}.path_sha256")
    if "files" in raw:
        result["files"] = _copy_json(_object(raw["files"], f"{where}.files"))
    return result


def _validate_provenance(value: Any, where: str = "provenance", *, required: bool) -> dict[str, Any]:
    raw = _object(value, where)
    allowed = {
        "trace_metadata_sha256",
        "testcase",
        "run_record",
        "adapter_normalized_sha256",
        "engine_identity",
        "effective_comparison_sha256",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise AlignmentSchemaError(f"{where} contains unknown field(s): {', '.join(unknown)}")
    if required:
        for name in allowed:
            if name not in raw:
                raise AlignmentSchemaError(f"{where}.{name} is required")
    result: dict[str, Any] = {}
    if "trace_metadata_sha256" in raw:
        result["trace_metadata_sha256"] = _digest(raw["trace_metadata_sha256"], f"{where}.trace_metadata_sha256")
    if "testcase" in raw:
        result["testcase"] = _validate_artifact_ref(raw["testcase"], f"{where}.testcase", "tree_sha256")
    if "run_record" in raw:
        result["run_record"] = _validate_artifact_ref(raw["run_record"], f"{where}.run_record", "sha256")
    if "adapter_normalized_sha256" in raw:
        result["adapter_normalized_sha256"] = _digest(raw["adapter_normalized_sha256"], f"{where}.adapter_normalized_sha256")
    if "engine_identity" in raw:
        result["engine_identity"] = _validate_engine_identity(raw["engine_identity"], f"{where}.engine_identity")
    if "effective_comparison_sha256" in raw:
        result["effective_comparison_sha256"] = _digest(raw["effective_comparison_sha256"], f"{where}.effective_comparison_sha256")
    return result


@dataclass
class Evidence:
    """Immutable Evidence v0.1 record (the store enforces immutability)."""

    id: str
    created_at: str
    type: str
    subject: dict[str, str]
    fixture: dict[str, Any]
    result: str
    observations: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)
    format: str = EVIDENCE_FORMAT
    format_version: str = EVIDENCE_FORMAT_VERSION
    # Manual-note fields are optional and intentionally separate from
    # observations so a note cannot be mistaken for a numerical observation.
    note: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result: dict[str, Any] = {
            "format": self.format,
            "format_version": self.format_version,
            "id": self.id,
            "created_at": self.created_at,
            "type": self.type,
            "subject": _copy_json(self.subject),
            "fixture": _copy_json(self.fixture),
            "result": self.result,
            "observations": _copy_json(self.observations),
            "provenance": _copy_json(self.provenance),
            "extensions": _copy_json(self.extensions),
        }
        if self.note is not None:
            result["note"] = _copy_json(self.note)
        return result

    def validate(self) -> None:
        _validate_evidence_dict({
            "format": self.format,
            "format_version": self.format_version,
            "id": self.id,
            "created_at": self.created_at,
            "type": self.type,
            "subject": self.subject,
            "fixture": self.fixture,
            "result": self.result,
            "observations": self.observations,
            "provenance": self.provenance,
            "extensions": self.extensions,
            **({"note": self.note} if self.note is not None else {}),
        })

    @classmethod
    def from_dict(cls, data: Any) -> Evidence:
        parsed = _validate_evidence_dict(data)
        return cls(**parsed)


def _validate_note(value: Any, where: str = "note") -> dict[str, Any]:
    raw = _strict_object(value, {"category", "reason", "duration_minutes", "milestone", "target"}, where)
    category = _string(_required(raw, "category", where), f"{where}.category")
    if category not in NOTE_CATEGORIES:
        raise AlignmentSchemaError(f"{where}.category is unsupported: {category!r}")
    reason = _string(_required(raw, "reason", where), f"{where}.reason")
    result: dict[str, Any] = {"category": category, "reason": reason}
    if "duration_minutes" in raw:
        duration = raw["duration_minutes"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            raise AlignmentSchemaError(f"{where}.duration_minutes must be finite and non-negative")
        result["duration_minutes"] = duration
    if "milestone" in raw:
        result["milestone"] = _string(raw["milestone"], f"{where}.milestone")
    if "target" in raw:
        result["target"] = _string(raw["target"], f"{where}.target")
    return result


def _validate_evidence_dict(data: Any) -> dict[str, Any]:
    allowed = {
        "format",
        "format_version",
        "id",
        "created_at",
        "type",
        "subject",
        "fixture",
        "result",
        "observations",
        "provenance",
        "extensions",
        "note",
    }
    raw = _strict_object(data, allowed, "evidence")
    fmt = _string(_required(raw, "format", "evidence"), "evidence.format")
    if fmt != EVIDENCE_FORMAT:
        raise AlignmentSchemaError(f"format must be {EVIDENCE_FORMAT!r}, got {fmt!r}")
    version = _string(_required(raw, "format_version", "evidence"), "evidence.format_version")
    if version not in EVIDENCE_READ_VERSIONS:
        raise AlignmentSchemaError(f"unsupported evidence format_version {version!r}")
    evidence_id = _id(_required(raw, "id", "evidence"), "evidence.id", "ev-")
    created_at = _timestamp(_required(raw, "created_at", "evidence"), "evidence.created_at")
    evidence_type = _string(_required(raw, "type", "evidence"), "evidence.type")
    if evidence_type in RESERVED_EVIDENCE_TYPES:
        raise AlignmentSchemaError(f"evidence type {evidence_type!r} is reserved in 0.10")
    if evidence_type not in EVIDENCE_TYPES:
        raise AlignmentSchemaError(f"unsupported evidence type {evidence_type!r}")
    result = _string(_required(raw, "result", "evidence"), "evidence.result")
    if result not in EVIDENCE_RESULTS:
        raise AlignmentSchemaError(f"unsupported evidence result {result!r}")
    observations = _validate_observations(
        _required(raw, "observations", "evidence"), "evidence.observations"
    )
    provenance = _validate_provenance(
        _required(raw, "provenance", "evidence"),
        "evidence.provenance",
        required=evidence_type == "numerical_comparison",
    )
    extensions = _validate_extensions(raw.get("extensions", {}), "evidence.extensions")
    note = _validate_note(raw["note"], "evidence.note") if "note" in raw else None
    if evidence_type == "manual_note" and note is None:
        raise AlignmentSchemaError("manual_note evidence requires a note object")
    if evidence_type == "numerical_comparison" and note is not None:
        raise AlignmentSchemaError("numerical_comparison evidence cannot contain note")
    return {
        "format": fmt,
        "format_version": version,
        "id": evidence_id,
        "created_at": created_at,
        "type": evidence_type,
        "subject": _validate_subject(_required(raw, "subject", "evidence")),
        "fixture": _validate_fixture(_required(raw, "fixture", "evidence")),
        "result": result,
        "observations": observations,
        "provenance": provenance,
        "extensions": extensions,
        "note": note,
    }


@dataclass(frozen=True)
class AlignmentEvent:
    """One append-only Alignment Event v0.1 record."""

    seq: int
    timestamp: str
    event: str
    data: dict[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    format: str = EVENT_FORMAT
    format_version: str = EVENT_FORMAT_VERSION
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format": self.format,
            "format_version": self.format_version,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event": self.event,
            "data": _copy_json(self.data),
            "evidence_ids": list(self.evidence_ids),
            "extensions": _copy_json(self.extensions),
        }

    def validate(self) -> None:
        _validate_event_dict({
            "format": self.format,
            "format_version": self.format_version,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event": self.event,
            "data": self.data,
            "evidence_ids": list(self.evidence_ids),
            "extensions": self.extensions,
        })

    @classmethod
    def from_dict(cls, data: Any) -> AlignmentEvent:
        return cls(**_validate_event_dict(data))


Event = AlignmentEvent


def _validate_event_dict(data: Any) -> dict[str, Any]:
    allowed = {
        "format",
        "format_version",
        "seq",
        "timestamp",
        "event",
        "data",
        "evidence_ids",
        "extensions",
    }
    raw = _strict_object(data, allowed, "alignment event")
    fmt = _string(_required(raw, "format", "alignment event"), "alignment event.format")
    if fmt != EVENT_FORMAT:
        raise AlignmentSchemaError(f"format must be {EVENT_FORMAT!r}, got {fmt!r}")
    version = _string(_required(raw, "format_version", "alignment event"), "alignment event.format_version")
    if version not in EVENT_READ_VERSIONS:
        raise AlignmentSchemaError(f"unsupported event format_version {version!r}")
    seq = _int(_required(raw, "seq", "alignment event"), "alignment event.seq", minimum=1)
    timestamp = _timestamp(_required(raw, "timestamp", "alignment event"), "alignment event.timestamp")
    name = _string(_required(raw, "event", "alignment event"), "alignment event.event")
    if name not in EVENT_TYPES:
        raise AlignmentSchemaError(f"unsupported alignment event {name!r}")
    data_obj = _object(_required(raw, "data", "alignment event"), "alignment event.data")
    evidence_ids = _string_array(raw.get("evidence_ids", []), "alignment event.evidence_ids")
    for index, evidence_id in enumerate(evidence_ids):
        _id(evidence_id, f"alignment event.evidence_ids[{index}]", "ev-")
    extensions = _validate_extensions(raw.get("extensions", {}), "alignment event.extensions")
    return {
        "format": fmt,
        "format_version": version,
        "seq": seq,
        "timestamp": timestamp,
        "event": name,
        "data": _copy_json(data_obj),
        "evidence_ids": evidence_ids,
        "extensions": extensions,
    }


def make_session_id() -> str:
    """Generate a portable session id without importing ``uuid`` at call sites."""

    import uuid

    return f"al-{uuid.uuid4()}"


def make_evidence_id() -> str:
    """Generate a portable evidence id."""

    import uuid

    return f"ev-{uuid.uuid4()}"


def load_alignment(path: str | Path) -> AlignmentSnapshot:
    """Load and validate an ``alignment.json`` file."""

    return AlignmentSnapshot.from_dict(_read_json(path, "alignment snapshot"))


def load_evidence(path: str | Path) -> Evidence:
    """Load and validate one evidence JSON file."""

    return Evidence.from_dict(_read_json(path, "evidence"))


def load_event(data_or_path: Any) -> AlignmentEvent:
    """Load one event from a mapping or JSON file."""

    if isinstance(data_or_path, (str, Path)):
        return AlignmentEvent.from_dict(_read_json(data_or_path, "alignment event"))
    return AlignmentEvent.from_dict(data_or_path)


def _read_json(path: str | Path, kind: str) -> Any:
    candidate = Path(path)
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AlignmentSchemaError(f"cannot read {kind} {candidate}: {exc}") from exc


__all__ = [
    "ALIGNMENT_FORMAT",
    "ALIGNMENT_FORMAT_VERSION",
    "ALIGNMENT_READ_VERSIONS",
    "ALIGNMENT_VERSION",
    "BOUNDARY_STATES",
    "EVENT_FORMAT",
    "EVENT_FORMAT_VERSION",
    "EVENT_READ_VERSIONS",
    "EVENT_TYPES",
    "EVENT_VERSION",
    "EVIDENCE_FORMAT",
    "EVIDENCE_FORMAT_VERSION",
    "EVIDENCE_READ_VERSIONS",
    "EVIDENCE_RESULTS",
    "EVIDENCE_TYPES",
    "EVIDENCE_VERSION",
    "FIXTURE_KINDS",
    "NOTE_CATEGORIES",
    "RESERVED_EVIDENCE_TYPES",
    "SESSION_STATUSES",
    "AlignmentEvent",
    "AlignmentSchemaError",
    "AlignmentSession",
    "AlignmentSnapshot",
    "AlignmentValidationError",
    "Event",
    "EventValidationError",
    "Evidence",
    "EvidenceValidationError",
    "SessionSnapshot",
    "load_alignment",
    "load_event",
    "load_evidence",
    "make_evidence_id",
    "make_session_id",
    "utc_now",
]
