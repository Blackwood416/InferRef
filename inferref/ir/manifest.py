"""Trace manifest (IR §42-§44, §27; SPEC §9).

The manifest declares the format contract, how the trace was captured, and
enough environment/determinism metadata for a human to judge reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferref.ir._common import Record, drop_none
from inferref.ir.version import (
    FORMAT,
    FORMAT_VERSION,
    INFERREF_VERSION,
    check_format_version,
)


@dataclass(frozen=True)
class NamedVersion:
    """A ``{name, version}`` pair (IR §42)."""

    name: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, default_name: str = "") -> NamedVersion:
        data = data or {}
        return cls(name=data.get("name", default_name), version=data.get("version", "unknown"))


@dataclass(frozen=True)
class ModelInfo:
    """Identity of the traced model (IR §42)."""

    name: str = "unknown"
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "revision": self.revision}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ModelInfo:
        data = data or {}
        return cls(name=data.get("name", "unknown"), revision=data.get("revision"))


@dataclass(frozen=True)
class Execution:
    """Execution mode and device (IR §42; SPEC §47)."""

    mode: str = "inference"
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "device": self.device}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Execution:
        data = data or {}
        return cls(mode=data.get("mode", "inference"), device=data.get("device", "cpu"))


@dataclass(frozen=True)
class Capture:
    """Capture policy in effect for this trace (IR §42; SPEC §14)."""

    tensor_policy: str = "metadata"
    source_mapping: bool = True
    module_mapping: bool = True
    scope: str | None = None
    exclude: tuple[str, ...] = ()
    max_ops: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return drop_none(
            {
                "tensor_policy": self.tensor_policy,
                "source_mapping": self.source_mapping,
                "module_mapping": self.module_mapping,
                "scope": self.scope,
                "exclude": list(self.exclude) or None,
                "max_ops": self.max_ops,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Capture:
        data = data or {}
        return cls(
            tensor_policy=data.get("tensor_policy", "metadata"),
            source_mapping=bool(data.get("source_mapping", True)),
            module_mapping=bool(data.get("module_mapping", True)),
            scope=data.get("scope"),
            exclude=tuple(data.get("exclude", ())),
            max_ops=data.get("max_ops"),
        )


@dataclass(frozen=True)
class SourcePolicy:
    """How source paths are recorded (IR §27; SPEC §58)."""

    path_mode: str = "relative"
    embed_source_text: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_mode": self.path_mode,
            "embed_source_text": self.embed_source_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SourcePolicy:
        data = data or {}
        return cls(
            path_mode=data.get("path_mode", "relative"),
            embed_source_text=bool(data.get("embed_source_text", False)),
        )


@dataclass(frozen=True)
class Environment:
    """Reproducibility metadata; never affects IR parsing (IR §43)."""

    python: str = ""
    os: str = ""
    architecture: str = ""
    device_name: str | None = None
    driver: str | None = None
    packages: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return drop_none(
            {
                "python": self.python,
                "os": self.os,
                "architecture": self.architecture,
                "device_name": self.device_name,
                "driver": self.driver,
                "packages": self.packages or None,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Environment:
        data = data or {}
        return cls(
            python=data.get("python", ""),
            os=data.get("os", ""),
            architecture=data.get("architecture", ""),
            device_name=data.get("device_name"),
            driver=data.get("driver"),
            packages=data.get("packages") or {},
        )


@dataclass(frozen=True)
class Determinism:
    """Determinism state at capture time (IR §44; SPEC §48)."""

    seed: int | None = None
    training: bool = False
    grad_enabled: bool = False
    autocast: bool = False
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "training": self.training,
            "grad_enabled": self.grad_enabled,
            "autocast": self.autocast,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Determinism:
        data = data or {}
        return cls(
            seed=data.get("seed"),
            training=bool(data.get("training", False)),
            grad_enabled=bool(data.get("grad_enabled", False)),
            autocast=bool(data.get("autocast", False)),
            warnings=tuple(data.get("warnings", ())),
        )


@dataclass
class Manifest(Record):
    """Top-level trace manifest (IR §42)."""

    format: str = FORMAT
    format_version: str = FORMAT_VERSION
    inferref_version: str = INFERREF_VERSION

    frontend: NamedVersion = field(default_factory=lambda: NamedVersion("pytorch", "0.1"))
    reference_framework: NamedVersion = field(
        default_factory=lambda: NamedVersion("pytorch", "unknown")
    )

    model: ModelInfo = field(default_factory=ModelInfo)
    execution: Execution = field(default_factory=Execution)
    capture: Capture = field(default_factory=Capture)
    source_policy: SourcePolicy = field(default_factory=SourcePolicy)
    environment: Environment = field(default_factory=Environment)
    determinism: Determinism = field(default_factory=Determinism)

    _KNOWN = (
        "format",
        "format_version",
        "inferref_version",
        "frontend",
        "reference_framework",
        "model",
        "execution",
        "capture",
        "source_policy",
        "environment",
        "determinism",
    )

    def _encode(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "format_version": self.format_version,
            "inferref_version": self.inferref_version,
            "frontend": self.frontend.to_dict(),
            "reference_framework": self.reference_framework.to_dict(),
            "model": self.model.to_dict(),
            "execution": self.execution.to_dict(),
            "capture": self.capture.to_dict(),
            "source_policy": self.source_policy.to_dict(),
            "environment": self.environment.to_dict(),
            "determinism": self.determinism.to_dict(),
        }

    @classmethod
    def _decode(cls, data: dict[str, Any]) -> dict[str, Any]:
        fmt = data.get("format", FORMAT)
        version = data.get("format_version", FORMAT_VERSION)
        check_format_version(fmt, version)
        return {
            "format": fmt,
            "format_version": version,
            "inferref_version": data.get("inferref_version", "unknown"),
            "frontend": NamedVersion.from_dict(data.get("frontend"), "unknown"),
            "reference_framework": NamedVersion.from_dict(
                data.get("reference_framework"), "unknown"
            ),
            "model": ModelInfo.from_dict(data.get("model")),
            "execution": Execution.from_dict(data.get("execution")),
            "capture": Capture.from_dict(data.get("capture")),
            "source_policy": SourcePolicy.from_dict(data.get("source_policy")),
            "environment": Environment.from_dict(data.get("environment")),
            "determinism": Determinism.from_dict(data.get("determinism")),
        }
