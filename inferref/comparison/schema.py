"""Comparison Spec v0.1 data structures and validation (SPEC §6).

Defines the wire format for task-level comparison specifications attached to
testcases, suite cases, scenario steps, and CLI/Agent requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferref.comparators.numeric import NUMERIC_COMPARATOR_ID
from inferref.comparators.registry import get_comparator

COMPARISON_SPEC_FORMAT = "inferref-comparison"
COMPARISON_SPEC_VERSION = "0.1"
COMPARISON_SPEC_READ_VERSIONS = ("0.1",)


class ComparisonSpecValidationError(ValueError):
    """Raised when a Comparison Spec violates its wire contract or schema."""


@dataclass(frozen=True)
class OutputComparisonSpec:
    """Per-output role comparison configuration override."""

    comparator: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.comparator is not None:
            result["comparator"] = self.comparator
        if self.config:
            result["config"] = dict(self.config)
        return result

    @classmethod
    def from_dict(cls, data: Any) -> OutputComparisonSpec:
        if not isinstance(data, dict):
            raise ComparisonSpecValidationError(
                f"output comparison spec must be a dictionary, got {type(data).__name__}"
            )
        comparator = data.get("comparator")
        if comparator is not None and (not isinstance(comparator, str) or not comparator.strip()):
            raise ComparisonSpecValidationError("output comparator must be a non-empty string or null")
        config = data.get("config", {})
        if config is not None and not isinstance(config, dict):
            raise ComparisonSpecValidationError("output config must be a dictionary")
        return cls(
            comparator=comparator.strip() if comparator else None,
            config=dict(config or {}),
        )


def _clean_custom_config(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not raw_cfg:
        return {}
    numeric_defaults = {"per_dtype", "strict_layout", "ignore_stride", "atol", "rtol"}
    return {k: v for k, v in raw_cfg.items() if k not in numeric_defaults}


@dataclass
class ComparisonSpec:
    """Comparison Spec v0.1 model and wire format (SPEC §6.1)."""

    format: str = COMPARISON_SPEC_FORMAT
    format_version: str = COMPARISON_SPEC_VERSION
    comparator: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, OutputComparisonSpec] = field(default_factory=dict)

    def __init__(
        self,
        *,
        format: str = COMPARISON_SPEC_FORMAT,
        format_version: str = COMPARISON_SPEC_VERSION,
        comparator: str | None = None,
        config: dict[str, Any] | None = None,
        outputs: dict[str, OutputComparisonSpec | dict[str, Any]] | None = None,
        tolerances: dict[str, Any] | None = None,
        tensor_layout: str | bool | None = None,
        output_roles: list[str] | None = None,
    ) -> None:
        self.format = format
        self.format_version = format_version
        self.comparator = comparator.strip() if comparator else None
        cfg = dict(config or {})
        if tolerances is not None and "per_dtype" not in cfg:
            cfg["per_dtype"] = tolerances
        if tensor_layout is not None and "strict_layout" not in cfg:
            if isinstance(tensor_layout, bool):
                cfg["strict_layout"] = tensor_layout
            elif isinstance(tensor_layout, str):
                cfg["strict_layout"] = tensor_layout.lower() in ("strict", "true", "1")
        self.config = cfg

        parsed_outputs: dict[str, OutputComparisonSpec] = {}
        if outputs:
            for role, val in outputs.items():
                if isinstance(val, OutputComparisonSpec):
                    parsed_outputs[role] = val
                elif isinstance(val, dict):
                    parsed_outputs[role] = OutputComparisonSpec.from_dict(val)
                else:
                    raise ComparisonSpecValidationError(
                        f"output {role!r} must be OutputComparisonSpec or dict, got {type(val).__name__}"
                    )
        elif output_roles:
            for role in output_roles:
                parsed_outputs[role] = OutputComparisonSpec()
        self.outputs = parsed_outputs

    @property
    def tolerances(self) -> dict[str, Any] | None:
        return self.config.get("per_dtype")

    @property
    def tensor_layout(self) -> str | None:
        if "strict_layout" in self.config:
            return "strict" if self.config["strict_layout"] else "standard"
        return None

    @property
    def output_roles(self) -> list[str]:
        return list(self.outputs.keys())

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "format": self.format,
            "format_version": self.format_version,
            "comparator": self.comparator or NUMERIC_COMPARATOR_ID,
        }
        if self.config:
            result["config"] = dict(self.config)
        if self.outputs:
            result["outputs"] = {k: v.to_dict() for k, v in self.outputs.items()}
        return result

    @classmethod
    def from_dict(cls, data: Any) -> ComparisonSpec:
        if isinstance(data, ComparisonSpec):
            return data
        if not isinstance(data, dict):
            raise ComparisonSpecValidationError(
                f"comparison spec must be a dictionary, got {type(data).__name__}"
            )
        fmt = data.get("format", COMPARISON_SPEC_FORMAT)
        if fmt != COMPARISON_SPEC_FORMAT:
            raise ComparisonSpecValidationError(
                f"invalid comparison format {fmt!r}, expected {COMPARISON_SPEC_FORMAT!r}"
            )
        ver = data.get("format_version", COMPARISON_SPEC_VERSION)
        if ver not in COMPARISON_SPEC_READ_VERSIONS:
            raise ComparisonSpecValidationError(
                f"unsupported comparison format_version {ver!r}, expected one of {COMPARISON_SPEC_READ_VERSIONS!r}"
            )
        comparator = data.get("comparator")
        if comparator is not None and (not isinstance(comparator, str) or not comparator.strip()):
            raise ComparisonSpecValidationError("comparator must be a non-empty string or null")

        raw_config = data.get("config", {})
        if raw_config is not None and not isinstance(raw_config, dict):
            raise ComparisonSpecValidationError("comparison config must be a dictionary")

        raw_outputs = data.get("outputs", {})
        if raw_outputs is not None and not isinstance(raw_outputs, dict):
            raise ComparisonSpecValidationError("comparison outputs must be a dictionary")

        parsed_outputs: dict[str, OutputComparisonSpec] = {}
        if raw_outputs:
            for role, entry in raw_outputs.items():
                if not isinstance(role, str) or not role.strip():
                    raise ComparisonSpecValidationError("output role name must be a non-empty string")
                parsed_outputs[role] = OutputComparisonSpec.from_dict(entry)

        return cls(
            format=fmt,
            format_version=ver,
            comparator=comparator.strip() if comparator else None,
            config=dict(raw_config or {}),
            outputs=parsed_outputs,
        )

    def validate(self, *, check_registry: bool = True) -> None:
        """Statically validate comparison specification before engine launch.

        Raises:
            ComparisonSpecValidationError: If format, version, comparator, or config is invalid.
        """
        if self.format != COMPARISON_SPEC_FORMAT:
            raise ComparisonSpecValidationError(
                f"invalid comparison format {self.format!r}, expected {COMPARISON_SPEC_FORMAT!r}"
            )
        if self.format_version not in COMPARISON_SPEC_READ_VERSIONS:
            raise ComparisonSpecValidationError(
                f"unsupported comparison format_version {self.format_version!r}, "
                f"expected one of {COMPARISON_SPEC_READ_VERSIONS!r}"
            )

        comp_id = self.comparator or NUMERIC_COMPARATOR_ID
        if not comp_id or not isinstance(comp_id, str):
            raise ComparisonSpecValidationError("comparator must be a non-empty string")

        if check_registry:
            plugin = get_comparator(comp_id)
            if plugin is None:
                raise ComparisonSpecValidationError(f"unknown comparator {comp_id!r}")
            try:
                cfg = self.config if comp_id == NUMERIC_COMPARATOR_ID else _clean_custom_config(self.config)
                plugin.validate_config(cfg)
            except Exception as exc:
                raise ComparisonSpecValidationError(
                    f"invalid_comparison_config: {exc}"
                ) from exc

            for role, out_spec in self.outputs.items():
                target_comp_id = out_spec.comparator or comp_id
                out_plugin = get_comparator(target_comp_id)
                if out_plugin is None:
                    raise ComparisonSpecValidationError(
                        f"unknown comparator {target_comp_id!r} for output role {role!r}"
                    )
                try:
                    out_cfg = out_spec.config if target_comp_id == NUMERIC_COMPARATOR_ID else _clean_custom_config(out_spec.config)
                    out_plugin.validate_config(out_cfg)
                except Exception as exc:
                    raise ComparisonSpecValidationError(
                        f"invalid_comparison_config for output role {role!r}: {exc}"
                    ) from exc


def validate_comparison_spec(spec_or_dict: ComparisonSpec | dict[str, Any]) -> ComparisonSpec:
    """Helper to validate and parse a ComparisonSpec or dict representation."""
    spec = ComparisonSpec.from_dict(spec_or_dict)
    spec.validate(check_registry=True)
    return spec
