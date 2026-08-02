"""Stable wire records for coding-agent and engine-adapter integration."""

from __future__ import annotations

import json
import math
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_PROTOCOL_FORMAT = "inferref-agent-response"
AGENT_PROTOCOL_VERSION = "0.1"
ENGINE_ADAPTER_FORMAT = "inferref-engine-adapter"
ENGINE_ADAPTER_VERSION = "0.1"

_ADAPTER_PLACEHOLDERS = frozenset({"testcase", "output", "adapter_dir", "python"})


class AgentProtocolError(ValueError):
    """Raised when an Agent or adapter request violates its wire contract."""


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
        if data.get("format_version") != ENGINE_ADAPTER_VERSION:
            raise AgentProtocolError(
                f"unsupported engine adapter format_version "
                f"{data.get('format_version')!r}; expected {ENGINE_ADAPTER_VERSION!r}"
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

        adapter = cls(
            name=name.strip(),
            command=tuple(raw_command),
            timeout_seconds=timeout,
            cwd=cwd,
            environment=dict(environment),
            max_output_chars=max_output,
            max_artifact_bytes=max_artifacts,
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
            "format_version": ENGINE_ADAPTER_VERSION,
            "name": self.name,
            "command": list(self.command),
            "timeout_seconds": self.timeout_seconds,
            "cwd": self.cwd,
            # Values may be credentials. The execution record only needs to show
            # which variables were configured, never their contents.
            "environment_keys": sorted(self.environment),
            "max_output_chars": self.max_output_chars,
            "max_artifact_bytes": self.max_artifact_bytes,
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
