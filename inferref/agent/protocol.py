"""Stable wire records for coding-agent and engine-adapter integration."""

from __future__ import annotations

import json
import math
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferref.testcase.requirements import is_contract_id

AGENT_PROTOCOL_FORMAT = "inferref-agent-response"
AGENT_PROTOCOL_VERSION = "0.1"
ENGINE_ADAPTER_FORMAT = "inferref-engine-adapter"
ENGINE_ADAPTER_VERSION = "0.2"
ENGINE_ADAPTER_READ_VERSIONS = ("0.1", "0.2")

_ADAPTER_PLACEHOLDERS = frozenset({"testcase", "output", "adapter_dir", "python"})


class AgentProtocolError(ValueError):
    """Raised when an Agent or adapter request violates its wire contract."""


ADAPTER_FEATURES = frozenset(
    {"multiple_outputs", "strided_inputs", "alias_effects", "mutation_effects"}
)


def _string_array(value: Any, where: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(set(value)) != len(value)
    ):
        raise AgentProtocolError(f"{where} must be a non-empty unique string array")
    return tuple(value)


@dataclass(frozen=True)
class ContractCapability:
    """Per-contract refinement of the adapter-wide capability declaration."""

    dtypes: tuple[str, ...] | None = None
    max_rank: int | None = None
    features: tuple[str, ...] | None = None

    @classmethod
    def from_dict(cls, data: Any, *, contract: str) -> ContractCapability:
        if not isinstance(data, dict):
            raise AgentProtocolError(
                f"capabilities.contract_capabilities[{contract}] must be an object"
            )
        dtypes = data.get("dtypes")
        if dtypes is None:
            dtypes_value = None
        elif isinstance(dtypes, list) and not dtypes:
            raise AgentProtocolError(
                f"capabilities.contract_capabilities[{contract}].dtypes "
                "must be a non-empty string array when present"
            )
        else:
            dtypes_value = _string_array(
                dtypes,
                f"capabilities.contract_capabilities[{contract}].dtypes",
            )
        raw_features = data.get("features")
        if raw_features is None:
            features: tuple[str, ...] | None = None
        elif raw_features == []:
            features = ()
        else:
            features = _string_array(
                raw_features,
                f"capabilities.contract_capabilities[{contract}].features",
            )
        if features is not None:
            unknown = set(features) - ADAPTER_FEATURES
            if unknown:
                raise AgentProtocolError(
                    "unknown contract capability feature(s) for "
                    f"{contract}: " + ", ".join(sorted(unknown))
                )
        max_rank = data.get("max_rank")
        if max_rank is not None and (
            not isinstance(max_rank, int) or isinstance(max_rank, bool) or max_rank < 0
        ):
            raise AgentProtocolError(
                f"capabilities.contract_capabilities[{contract}].max_rank "
                "must be a non-negative integer when present"
            )
        return cls(dtypes=dtypes_value, max_rank=max_rank, features=features)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.dtypes is not None:
            result["dtypes"] = list(self.dtypes)
        if self.max_rank is not None:
            result["max_rank"] = self.max_rank
        if self.features is not None:
            result["features"] = list(self.features)
        return result


@dataclass(frozen=True)
class AdapterCapabilities:
    """Declarative limits checked before an engine process is started."""

    device_types: tuple[str, ...]
    dtypes: tuple[str, ...]
    max_rank: int
    features: tuple[str, ...] = ()
    contracts: tuple[str, ...] | None = None
    contract_capabilities: dict[str, ContractCapability] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> AdapterCapabilities:
        if not isinstance(data, dict):
            raise AgentProtocolError("adapter capabilities must be an object")
        devices = _string_array(data.get("device_types"), "capabilities.device_types")
        dtypes = _string_array(data.get("dtypes"), "capabilities.dtypes")
        raw_features = data.get("features", [])
        if raw_features == []:
            features: tuple[str, ...] = ()
        else:
            features = _string_array(raw_features, "capabilities.features")
        unknown = set(features) - ADAPTER_FEATURES
        if unknown:
            raise AgentProtocolError(
                "unknown adapter capability feature(s): " + ", ".join(sorted(unknown))
            )
        max_rank = data.get("max_rank")
        if not isinstance(max_rank, int) or isinstance(max_rank, bool) or max_rank < 0:
            raise AgentProtocolError(
                "capabilities.max_rank must be a non-negative integer"
            )
        raw_contracts = data.get("contracts")
        if raw_contracts is None:
            contracts = None
        elif isinstance(raw_contracts, list) and not raw_contracts:
            # An explicit empty array means "strictly supports zero contracts",
            # distinct from a missing field which means "unchecked".
            contracts = ()
        else:
            contracts = _string_array(raw_contracts, "capabilities.contracts")
        if contracts is not None:
            invalid = [item for item in contracts if not is_contract_id(item)]
            if invalid:
                raise AgentProtocolError(
                    "capabilities.contracts contains invalid versioned contract(s): "
                    + ", ".join(invalid)
                )
        raw_contract_caps = data.get("contract_capabilities")
        if raw_contract_caps is None:
            contract_capabilities: dict[str, ContractCapability] = {}
        else:
            if not isinstance(raw_contract_caps, dict):
                raise AgentProtocolError(
                    "capabilities.contract_capabilities must be an object"
                )
            if contracts is None:
                raise AgentProtocolError(
                    "capabilities.contract_capabilities requires capabilities.contracts"
                )
            contract_capabilities = {}
            for contract, value in raw_contract_caps.items():
                if not is_contract_id(contract):
                    raise AgentProtocolError(
                        "capabilities.contract_capabilities key must be a versioned "
                        f"contract ID: {contract!r}"
                    )
                if contract not in contracts:
                    raise AgentProtocolError(
                        "capabilities.contract_capabilities names a contract that is "
                        f"not declared in capabilities.contracts: {contract!r}"
                    )
                contract_capabilities[contract] = ContractCapability.from_dict(
                    value, contract=contract
                )
            global_dtypes = set(dtypes)
            global_features = set(features)
            for contract, capability in contract_capabilities.items():
                if capability.dtypes is not None:
                    extra_dtypes = set(capability.dtypes) - global_dtypes
                    if extra_dtypes:
                        raise AgentProtocolError(
                            "capabilities.contract_capabilities["
                            f"{contract}].dtypes exceeds global capabilities.dtypes: "
                            + ", ".join(sorted(extra_dtypes))
                        )
                if capability.max_rank is not None and capability.max_rank > max_rank:
                    raise AgentProtocolError(
                        "capabilities.contract_capabilities["
                        f"{contract}].max_rank {capability.max_rank} exceeds global "
                        f"capabilities.max_rank {max_rank}"
                    )
                if capability.features is not None:
                    extra_features = set(capability.features) - global_features
                    if extra_features:
                        raise AgentProtocolError(
                            "capabilities.contract_capabilities["
                            f"{contract}].features exceeds global "
                            "capabilities.features: "
                            + ", ".join(sorted(extra_features))
                        )
        return cls(
            devices, dtypes, max_rank, features, contracts, contract_capabilities
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "device_types": list(self.device_types),
            "dtypes": list(self.dtypes),
            "max_rank": self.max_rank,
            "features": list(self.features),
        }
        if self.contracts is not None:
            result["contracts"] = list(self.contracts)
        if self.contract_capabilities:
            result["contract_capabilities"] = {
                contract: capability.to_dict()
                for contract, capability in self.contract_capabilities.items()
            }
        return result

    def incompatibilities(
        self, requirements: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        for dtype in requirements.get("dtypes", []):
            if dtype not in self.dtypes:
                issues.append({"kind": "dtype", "required": dtype})
        rank = requirements.get("max_rank", 0)
        if isinstance(rank, int) and rank > self.max_rank:
            issues.append(
                {"kind": "max_rank", "required": rank, "supported": self.max_rank}
            )
        for feature in requirements.get("features", []):
            if feature not in self.features:
                issues.append({"kind": "feature", "required": feature})
        required_contracts = requirements.get("contracts", [])
        if required_contracts and self.contracts is not None:
            for contract in required_contracts:
                if contract not in self.contracts:
                    issues.append({"kind": "contract", "required": contract})
        return issues

    def contract_incompatibilities(
        self, contract: str, requirements: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Per-contract refinement checks against role-derived requirements."""

        capability = self.contract_capabilities.get(contract)
        if capability is None:
            return []
        issues: list[dict[str, Any]] = []
        for dtype in requirements.get("dtypes", []):
            if capability.dtypes is not None and dtype not in capability.dtypes:
                issues.append(
                    {"kind": "contract_dtype", "contract": contract, "required": dtype}
                )
        rank = requirements.get("max_rank", 0)
        if (
            isinstance(rank, int)
            and capability.max_rank is not None
            and rank > capability.max_rank
        ):
            issues.append(
                {
                    "kind": "contract_max_rank",
                    "contract": contract,
                    "required": rank,
                    "supported": capability.max_rank,
                }
            )
        for feature in requirements.get("features", []):
            if capability.features is not None and feature not in capability.features:
                issues.append(
                    {
                        "kind": "contract_feature",
                        "contract": contract,
                        "required": feature,
                    }
                )
        return issues

    def assessment(self, requirements: Mapping[str, Any]) -> str:
        if requirements.get("contracts") and self.contracts is None:
            return "unchecked"
        return "supported"


@dataclass(frozen=True)
class AgentResponse:
    """One self-describing result returned to a CLI, MCP host, or Agent API."""

    operation: str
    status: str
    data: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: tuple[Mapping[str, Any], ...] = ()
    next_actions: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": {
                "format": AGENT_PROTOCOL_FORMAT,
                "version": AGENT_PROTOCOL_VERSION,
            },
            "operation": self.operation,
            "status": self.status,
            "data": dict(self.data),
            "diagnostics": [dict(item) for item in self.diagnostics],
            "next_actions": [dict(item) for item in self.next_actions],
        }

    @classmethod
    def error(
        cls,
        operation: str,
        message: str,
        *,
        code: str = "invalid_request",
        data: Mapping[str, Any] | None = None,
    ) -> AgentResponse:
        return cls(
            operation=operation,
            status="error",
            data=data or {},
            diagnostics=({"severity": "error", "code": code, "message": message},),
        )


@dataclass(frozen=True)
class EngineAdapter:
    """Trusted, shell-free command used to execute one engine testcase.

    Adapter files are executable configuration. InferRef never invokes a shell,
    but the first command argument still names a process to run and therefore
    must come from a trusted engine workspace.
    """

    name: str
    command: tuple[str, ...]
    timeout_seconds: float = 300.0
    cwd: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict)
    max_output_chars: int = 65_536
    max_artifact_bytes: int = 1_073_741_824
    max_artifact_files: int = 10_000
    target_device: str | None = None
    capabilities: AdapterCapabilities | None = None
    format_version: str = ENGINE_ADAPTER_VERSION
    source: Path | None = field(default=None, compare=False)

    @classmethod
    def load(cls, path: str | Path) -> EngineAdapter:
        source = Path(path).resolve()
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise AgentProtocolError("engine adapter root must be a JSON object")
        if data.get("format") != ENGINE_ADAPTER_FORMAT:
            raise AgentProtocolError(
                f"not an InferRef engine adapter: format={data.get('format')!r}"
            )
        format_version = data.get("format_version")
        if format_version not in ENGINE_ADAPTER_READ_VERSIONS:
            raise AgentProtocolError(
                f"unsupported engine adapter format_version "
                f"{format_version!r}; expected one of {ENGINE_ADAPTER_READ_VERSIONS!r}"
            )
        raw_command = data.get("command")
        if (
            not isinstance(raw_command, list)
            or not raw_command
            or not all(isinstance(item, str) and item for item in raw_command)
        ):
            raise AgentProtocolError("adapter command must be a non-empty string array")
        name = data.get("name")
        if not isinstance(name, str) or not name.strip():
            raise AgentProtocolError("adapter name must be a non-empty string")
        try:
            timeout = float(data.get("timeout_seconds", 300.0))
        except (TypeError, ValueError) as exc:
            raise AgentProtocolError("adapter timeout_seconds must be numeric") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise AgentProtocolError("adapter timeout_seconds must be positive")
        try:
            max_output = int(data.get("max_output_chars", 65_536))
        except (TypeError, ValueError) as exc:
            raise AgentProtocolError(
                "adapter max_output_chars must be an integer"
            ) from exc
        if max_output < 1_024:
            raise AgentProtocolError("adapter max_output_chars must be at least 1024")
        try:
            max_artifacts = int(data.get("max_artifact_bytes", 1_073_741_824))
        except (TypeError, ValueError) as exc:
            raise AgentProtocolError(
                "adapter max_artifact_bytes must be an integer"
            ) from exc
        if max_artifacts < 1_024:
            raise AgentProtocolError("adapter max_artifact_bytes must be at least 1024")
        try:
            max_artifact_files = int(data.get("max_artifact_files", 10_000))
        except (TypeError, ValueError) as exc:
            raise AgentProtocolError(
                "adapter max_artifact_files must be an integer"
            ) from exc
        if max_artifact_files < 16:
            raise AgentProtocolError("adapter max_artifact_files must be at least 16")
        cwd = data.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise AgentProtocolError("adapter cwd must be a string or null")
        environment = data.get("environment") or {}
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise AgentProtocolError(
                "adapter environment must be a string-to-string object"
            )
        target_device = data.get("target_device")
        raw_capabilities = data.get("capabilities")
        if format_version == ENGINE_ADAPTER_VERSION:
            if not isinstance(target_device, str) or not target_device:
                raise AgentProtocolError(
                    "adapter target_device must be a non-empty string"
                )
            capabilities = AdapterCapabilities.from_dict(raw_capabilities)
            if target_device.split(":", 1)[0] not in capabilities.device_types:
                raise AgentProtocolError(
                    "adapter target_device must be included in capabilities.device_types"
                )
        else:
            target_device = None
            capabilities = None

        adapter = cls(
            name=name.strip(),
            command=tuple(raw_command),
            timeout_seconds=timeout,
            cwd=cwd,
            environment=dict(environment),
            max_output_chars=max_output,
            max_artifact_bytes=max_artifacts,
            max_artifact_files=max_artifact_files,
            target_device=target_device,
            capabilities=capabilities,
            format_version=format_version,
            source=source,
        )
        adapter._validate_placeholders()
        return adapter

    @property
    def directory(self) -> Path:
        return self.source.parent if self.source is not None else Path.cwd()

    def working_directory(self) -> Path:
        if self.cwd is None:
            return self.directory
        configured = Path(self.cwd)
        if configured.is_absolute():
            return configured.resolve()
        return (self.directory / configured).resolve()

    def expand(
        self, *, testcase: Path, output: Path, python: Path
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        replacements = {
            "testcase": str(testcase),
            "output": str(output),
            "adapter_dir": str(self.directory),
            "python": str(python),
        }
        command = tuple(item.format_map(replacements) for item in self.command)
        environment = {
            key: value.format_map(replacements)
            for key, value in self.environment.items()
        }
        return command, environment

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": ENGINE_ADAPTER_FORMAT,
            "format_version": self.format_version,
            "name": self.name,
            "command": list(self.command),
            "timeout_seconds": self.timeout_seconds,
            "cwd": self.cwd,
            # Values may be credentials. The execution record only needs to show
            # which variables were configured, never their contents.
            "environment_keys": sorted(self.environment),
            "max_output_chars": self.max_output_chars,
            "max_artifact_bytes": self.max_artifact_bytes,
            "max_artifact_files": self.max_artifact_files,
            "target_device": self.target_device,
            "capability_status": "declared" if self.capabilities else "unchecked",
            "capabilities": self.capabilities.to_dict() if self.capabilities else None,
            "source": str(self.source) if self.source else None,
        }

    def _validate_placeholders(self) -> None:
        observed: set[str] = set()
        formatter = string.Formatter()
        for value in (*self.command, *self.environment.values()):
            try:
                fields = [field for _, field, _, _ in formatter.parse(value) if field]
            except ValueError as exc:
                raise AgentProtocolError(
                    f"invalid adapter placeholder syntax: {exc}"
                ) from exc
            unknown = set(fields) - _ADAPTER_PLACEHOLDERS
            if unknown:
                raise AgentProtocolError(
                    "unsupported adapter placeholder(s): " + ", ".join(sorted(unknown))
                )
            observed.update(fields)
        missing = {"testcase", "output"} - observed
        if missing:
            raise AgentProtocolError(
                "adapter command/environment must reference placeholder(s): "
                + ", ".join(sorted(missing))
            )
