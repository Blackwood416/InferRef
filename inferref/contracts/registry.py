"""Built-in + entry-point contract discovery and lookup.

Contract Schema v0.1 sections 3, 10 and 11. Built-in contracts are always
present; plugin entry points are discovered lazily through
``importlib.metadata`` (``inferref.contracts`` group) and cached process
locally. Malformed or duplicate plugins are reported, never silently loaded,
and never break operations that do not need them.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any

from inferref.contracts.builtin import builtin_contracts
from inferref.contracts.schema import (
    EFFECT_VOCABULARY,
    FEATURE_VOCABULARY,
    ContractSchemaError,
    ExecutableContract,
    build_contract,
    merge_features,
    validate_contract_id,
    validate_contract_roles,
)

ENTRY_POINT_GROUP = "inferref.contracts"

#: The pseudo pack name owned by core. A plugin entry point with this name
#: would shadow the built-in contracts and is rejected (section 3.3).
BUILTIN_PACK_NAME = "builtin"


@dataclass(frozen=True)
class ContractEntry:
    """One resolvable contract in deterministic list order."""

    id: str
    source: str
    distribution: str | None
    entry_point: str | None
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "distribution": self.distribution,
            "entry_point": self.entry_point,
            "status": self.status,
            **({"error": self.error} if self.error else {}),
        }


@dataclass(frozen=True)
class ContractPluginStatus:
    """Per-plugin verification status (section 3.4)."""

    entry_point: str
    distribution: str | None
    version: str | None
    status: str
    contracts: tuple[str, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_point": self.entry_point,
            "distribution": self.distribution,
            "version": self.version,
            "status": self.status,
            "contracts": list(self.contracts),
            **({"error": self.error} if self.error else {}),
        }


REGISTRY: dict[str, ExecutableContract] = {
    contract.id: contract for contract in builtin_contracts()
}

_ENTRIES: list[ContractEntry] | None = None
_STATUSES: list[ContractPluginStatus] | None = None


def _plugin_entry_points() -> list[Any]:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        entries = list(discovered.select(group=ENTRY_POINT_GROUP))
    else:  # pragma: no cover - Python 3.10 compatibility
        entries = list(discovered.get(ENTRY_POINT_GROUP, ()))
    return sorted(
        entries,
        key=lambda item: (
            getattr(getattr(item, "dist", None), "name", None) or "",
            item.name,
            item.value,
        ),
    )


def _coerce_descriptor(descriptor: Any, entry_point: str) -> ExecutableContract:
    """Turn one plugin descriptor into an executable contract."""

    if isinstance(descriptor, ExecutableContract):
        contract = descriptor
        validate_contract_id(contract.id)
        validate_contract_roles(contract.inputs, contract.outputs)
        if not callable(contract.validate_inputs):
            raise ContractSchemaError(
                "contract_schema_invalid",
                f"contract {contract.id!r} validate_inputs must be callable",
            )
        if contract.validate_outputs is not None and not callable(
            contract.validate_outputs
        ):
            raise ContractSchemaError(
                "contract_schema_invalid",
                f"contract {contract.id!r} validate_outputs must be callable",
            )
        if contract.validate_relation is not None and not callable(
            contract.validate_relation
        ):
            raise ContractSchemaError(
                "contract_schema_invalid",
                f"contract {contract.id!r} validate_relation must be callable",
            )
        unknown_features = set(contract.features) - FEATURE_VOCABULARY
        unknown_effects = set(contract.effects) - EFFECT_VOCABULARY
        if unknown_features or unknown_effects:
            raise ContractSchemaError(
                "contract_schema_invalid",
                (
                    f"contract {contract.id!r} declares unknown feature(s) "
                    f"{sorted(unknown_features)} or effect(s) {sorted(unknown_effects)}"
                ),
            )
        effective = merge_features(contract)
        return replace(contract, features=effective)
    if isinstance(descriptor, Mapping):
        return build_contract(descriptor)
    raise ContractSchemaError(
        "contract_schema_invalid",
        (
            f"entry point {entry_point!r} returned a non-contract descriptor: "
            f"{type(descriptor).__name__}"
        ),
    )


def _entry_point_issue(
    entry: Any, name_counts: Counter[str]
) -> tuple[str, str | None]:
    """Metadata-level status/error for one entry point, without loading it."""

    if entry.name == BUILTIN_PACK_NAME:
        return (
            "error",
            (
                "contract_shadows_builtin: entry point name "
                f"{entry.name!r} shadows the built-in contract pack"
            ),
        )
    if name_counts[entry.name] > 1:
        return (
            "error",
            (
                "contract_entry_point_error: duplicate contract entry-point "
                f"name {entry.name!r}"
            ),
        )
    return "loaded", None


def _discovery_only_statuses() -> list[ContractPluginStatus]:
    """Per-plugin statuses without calling any factory (section 3.4)."""

    plugin_entries = _plugin_entry_points()
    name_counts = Counter(entry.name for entry in plugin_entries)
    statuses: list[ContractPluginStatus] = []
    for entry in plugin_entries:
        distribution = getattr(entry, "dist", None)
        dist_name = getattr(distribution, "name", None)
        dist_version = getattr(distribution, "version", None)
        status, error = _entry_point_issue(entry, name_counts)
        if status == "loaded":
            status = "discovered"
        statuses.append(
            ContractPluginStatus(
                entry.name, dist_name, dist_version, status, (), error
            )
        )
    return statuses


def _discover(
    force: bool = False,
) -> tuple[list[ContractEntry], list[ContractPluginStatus]]:
    global _ENTRIES, _STATUSES
    if not force and _ENTRIES is not None and _STATUSES is not None:
        return _ENTRIES, _STATUSES

    builtins = builtin_contracts()
    # Keep the same dict object so re-exported references (the compatibility
    # shim, inferref.contracts) stay live; plugin contracts are added below.
    REGISTRY.clear()
    REGISTRY.update({contract.id: contract for contract in builtins})
    entries = [
        ContractEntry(
            id=contract.id,
            source="builtin",
            distribution=None,
            entry_point=None,
            status="loaded",
        )
        for contract in builtins
    ]

    plugin_entries = _plugin_entry_points()
    name_counts = Counter(entry.name for entry in plugin_entries)
    statuses: list[ContractPluginStatus] = []
    candidates: list[
        tuple[ExecutableContract, str, str | None, str | None]
    ] = []
    for entry in plugin_entries:
        distribution = getattr(entry, "dist", None)
        dist_name = getattr(distribution, "name", None)
        dist_version = getattr(distribution, "version", None)
        status = "loaded"
        error: str | None = None
        loaded: list[ExecutableContract] = []
        status, error = _entry_point_issue(entry, name_counts)
        if status != "error":
            try:
                factory = entry.load()
                if not callable(factory):
                    raise TypeError(
                        f"contract entry point {entry.name!r} is not a factory"
                    )
                descriptors = factory()
                if isinstance(descriptors, (str, bytes)) or not hasattr(
                    descriptors, "__iter__"
                ):
                    raise ValueError(
                        f"contract entry point {entry.name!r} did not return an "
                        "iterable of contract descriptors"
                    )
                for descriptor in descriptors:
                    loaded.append(_coerce_descriptor(descriptor, entry.name))
            except Exception as exc:  # noqa: BLE001 - plugin failures are reported, never fatal
                # A plugin that fails part-way must not register the contracts
                # it produced before the failure (section 2.4, 3.3.5).
                loaded = []
                if isinstance(exc, ContractSchemaError):
                    error_message = f"{exc.code}: {exc.message}"
                else:
                    error_message = f"{type(exc).__name__}: {exc}"
                status, error = (
                    "error",
                    f"contract_entry_point_error: {error_message}",
                )
        for contract in loaded:
            candidates.append((contract, entry.name, dist_name, dist_version))
        statuses.append(
            ContractPluginStatus(
                entry.name,
                dist_name,
                dist_version,
                status,
                tuple(contract.id for contract in loaded),
                error,
            )
        )

    builtin_ids = {contract.id for contract in builtins}
    id_counts = Counter(contract.id for contract, _, _, _ in candidates)
    rejections: dict[str, list[str]] = {}
    accepted: dict[str, list[str]] = {}
    for contract, entry_name, dist_name, _ in candidates:
        message: str | None = None
        if contract.id in builtin_ids:
            message = (
                "contract_shadows_builtin: contract id "
                f"{contract.id!r} shadows a built-in contract"
            )
        elif id_counts[contract.id] > 1:
            message = (
                "contract_duplicate_id: duplicate contract id "
                f"{contract.id!r}"
            )
        if message is not None:
            rejections.setdefault(entry_name, []).append(message)
            entries.append(
                ContractEntry(
                    id=contract.id,
                    source="plugin",
                    distribution=dist_name,
                    entry_point=entry_name,
                    status="error",
                    error=message,
                )
            )
        else:
            accepted.setdefault(entry_name, []).append(contract.id)
            REGISTRY[contract.id] = contract
            entries.append(
                ContractEntry(
                    id=contract.id,
                    source="plugin",
                    distribution=dist_name,
                    entry_point=entry_name,
                    status="loaded",
                )
            )

    resolved_statuses: list[ContractPluginStatus] = []
    for status in statuses:
        if status.status != "loaded":
            resolved_statuses.append(status)
            continue
        entry_rejections = rejections.get(status.entry_point, [])
        entry_accepted = tuple(accepted.get(status.entry_point, ()))
        if entry_rejections and not entry_accepted:
            resolved_statuses.append(
                ContractPluginStatus(
                    status.entry_point,
                    status.distribution,
                    status.version,
                    "error",
                    (),
                    entry_rejections[0],
                )
            )
        else:
            resolved_statuses.append(
                ContractPluginStatus(
                    status.entry_point,
                    status.distribution,
                    status.version,
                    "loaded",
                    entry_accepted,
                    entry_rejections[0] if entry_rejections else None,
                )
            )

    plugin_entries_out = [
        entry
        for entry in entries
        if entry.source == "plugin"
    ]
    plugin_entries_out.sort(
        key=lambda item: (
            item.distribution or "",
            item.entry_point or "",
            item.id,
        )
    )
    entries = [entry for entry in entries if entry.source == "builtin"] + plugin_entries_out
    _ENTRIES = entries
    _STATUSES = resolved_statuses
    return entries, resolved_statuses


def get_contract(contract_id: str) -> ExecutableContract | None:
    """Return the resolved profile for a contract ID, or None if unknown."""

    if contract_id not in REGISTRY:
        _discover()
    return REGISTRY.get(contract_id)


def contract_list() -> list[ContractEntry]:
    """Every contract in deterministic order (built-ins first, then plugins)."""

    entries, _ = _discover()
    return list(entries)


def verify_contracts() -> list[ContractPluginStatus]:
    """Fresh-load every discovered plugin and report per-plugin status."""

    _, statuses = _discover(force=True)
    return list(statuses)


def contract_plugin_statuses(*, load: bool = False) -> list[ContractPluginStatus]:
    """Per-plugin statuses; ``load=True`` performs a fresh full load."""

    if load:
        return verify_contracts()
    return _discovery_only_statuses()


def load_contract_file(path: str | Path) -> ExecutableContract:
    """Load and validate one ``.contract.json`` file without entry points."""

    contract_path = Path(path)
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractSchemaError(
            "contract_schema_invalid",
            f"{contract_path}: invalid JSON: {exc}",
        ) from exc
    if not isinstance(raw, dict):
        raise ContractSchemaError(
            "contract_schema_invalid",
            f"{contract_path}: contract file must contain a single object",
        )
    return build_contract(raw)


def validate_contract_file(path: str | Path) -> dict[str, Any]:
    """Validate one ``.contract.json`` file and return a structured result.

    Accepts a single descriptor object or a JSON array of descriptors.
    Duplicate IDs within one file are rejected with ``contract_duplicate_id``.
    The CLI ``contract validate`` command and this function share one
    implementation so their behavior cannot drift.
    """

    contract_path = Path(path)
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "status": "fail",
            "issues": [
                {
                    "code": "contract_schema_invalid",
                    "message": f"{contract_path}: invalid JSON: {exc}",
                }
            ],
        }
    descriptors = raw if isinstance(raw, list) else [raw]
    contracts: list[ExecutableContract] = []
    issues: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    for index, descriptor in enumerate(descriptors):
        raw_id = descriptor.get("id") if isinstance(descriptor, dict) else None
        try:
            if not isinstance(descriptor, dict):
                raise ContractSchemaError(
                    "contract_schema_invalid", "descriptor must be an object"
                )
            contract = build_contract(descriptor)
        except ContractSchemaError as exc:
            issue: dict[str, Any] = {
                "code": exc.code,
                "message": exc.message,
                "entry": index,
            }
            if isinstance(raw_id, str):
                issue["id"] = raw_id
            issues.append(issue)
            continue
        contracts.append(contract)
        if isinstance(raw_id, str) and raw_id in seen_ids:
            issues.append(
                {
                    "code": "contract_duplicate_id",
                    "message": f"duplicate contract id {raw_id!r}",
                    "id": raw_id,
                    "entry": index,
                }
            )
        seen_ids[raw_id] = index
    if issues:
        return {"status": "fail", "issues": issues}

    def summary(contract: ExecutableContract) -> dict[str, Any]:
        return {
            "id": contract.id,
            "inputs": list(contract.inputs),
            "outputs": list(contract.outputs),
            "relations": contract.to_dict().get("relations", 0),
        }

    if len(contracts) == 1:
        return {"status": "pass", "contract": summary(contracts[0])}
    return {"status": "pass", "contracts": [summary(contract) for contract in contracts]}


def contract_input_issues(
    contract_id: str, inputs: Mapping[str, dict[str, Any]]
) -> Sequence[str]:
    """Shape-level issues for a known contract, or () for an unknown one."""

    contract = get_contract(contract_id)
    if contract is None:
        return ()
    return contract.validate_inputs(inputs)


def contract_boundary_issues(
    contract_id: str,
    inputs: Mapping[str, dict[str, Any]],
    outputs: Mapping[str, dict[str, Any]],
) -> Sequence[str]:
    """Input, output, and input-output relational invariants for a contract."""

    contract = get_contract(contract_id)
    if contract is None:
        return ()
    issues: list[str] = []
    if not getattr(contract.validate_inputs, "_inferref_dtype_checked", False):
        for role in contract.inputs:
            record = inputs.get(role) or outputs.get(role)
            if record is None:
                continue
            if not isinstance(record.get("dtype"), str) or not record["dtype"]:
                issues.append(f"{role} must declare a non-empty dtype")
    if not getattr(contract.validate_outputs, "_inferref_dtype_checked", False):
        for role in contract.outputs:
            record = inputs.get(role) or outputs.get(role)
            if record is None:
                continue
            if not isinstance(record.get("dtype"), str) or not record["dtype"]:
                issues.append(f"{role} must declare a non-empty dtype")
    issues.extend(contract.validate_inputs(inputs))
    if contract.validate_outputs is not None:
        issues.extend(contract.validate_outputs(outputs))
    if contract.validate_relation is not None:
        issues.extend(contract.validate_relation(inputs, outputs))
    return issues


def contract_requirements(
    manifest: dict[str, Any], contract_id: str
) -> dict[str, Any]:
    """Derive dtype/rank requirements from the tensors bound to contract roles.

    Unknown contracts or role names that do not appear in the manifest fall
    back to the testcase-wide derived requirements so preflight remains
    conservative (section 8). Declared contract features are merged into the
    derived feature set.
    """

    # Imported lazily: inferref.testcase imports this package, so a module-level
    # dependency would create a cycle through inferref.testcase.__init__.
    from inferref.testcase.requirements import derive_requirements

    contract = get_contract(contract_id)
    if contract is None:
        return derive_requirements(manifest)
    role_names = set(contract.inputs) | set(contract.outputs)
    by_name: dict[str, dict[str, Any]] = {}
    for key in ("inputs", "outputs"):
        for item in manifest.get(key, []):
            if (
                isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and item["name"] in role_names
            ):
                by_name[item["name"]] = item
    if not all(role in by_name for role in role_names):
        requirements = derive_requirements(manifest)
    else:
        requirements = derive_requirements(
            {
                "inputs": [by_name[name] for name in contract.inputs],
                "outputs": [by_name[name] for name in contract.outputs],
                "nodes": [],
            }
        )
    requirements["features"] = sorted(
        set(requirements.get("features", [])) | set(contract.features)
    )
    return requirements


def _reset_registry() -> None:
    """Clear the process-local cache (test helper)."""

    global _ENTRIES, _STATUSES
    REGISTRY.clear()
    REGISTRY.update({contract.id: contract for contract in builtin_contracts()})
    _ENTRIES = None
    _STATUSES = None
