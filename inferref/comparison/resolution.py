"""Two-Axis tolerance resolution and effective comparison generator (SPEC §6.3).

Resolves multi-layer comparison specifications (CLI, Suite, Testcase, Defaults)
following the two independent axes:
- Axis A: Per-Dtype Table Replacement (Suite > Testcase > CLI --tolerance > Defaults)
- Axis B: Scalar Overrides (CLI --atol/--rtol > Suite > Testcase > None)

And generates `effective_comparison` with fine-grained per-field `sources` tracking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferref.comparators.numeric import NUMERIC_COMPARATOR_ID
from inferref.comparators.registry import get_comparator
from inferref.compare.tolerance import DEFAULT_TOLERANCES, TolerancePolicy
from inferref.comparison.schema import (
    ComparisonSpec,
    ComparisonSpecValidationError,
    OutputComparisonSpec,
)


def _clean_custom_config(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not raw_cfg:
        return {}
    numeric_defaults = {"per_dtype", "strict_layout", "ignore_stride", "atol", "rtol"}
    return {k: v for k, v in raw_cfg.items() if k not in numeric_defaults}


@dataclass
class EffectiveComparison:
    """Resolved comparison policy with fine-grained per-field sources tracking."""

    comparator: str = NUMERIC_COMPARATOR_ID
    config: dict[str, Any] = field(default_factory=dict)
    per_output: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparator": self.comparator,
            "config": self.config,
            "per_output": self.per_output,
            "sources": self.sources,
        }

    def validate(self) -> None:
        """Validate all resolved comparators and configurations before engine launch.

        Raises:
            ComparisonSpecValidationError: If any comparator is unknown or config is invalid.
        """
        plugin = get_comparator(self.comparator)
        if plugin is None:
            raise ComparisonSpecValidationError(f"unknown comparator {self.comparator!r}")
        try:
            cfg = self.config if self.comparator == NUMERIC_COMPARATOR_ID else _clean_custom_config(self.config)
            plugin.validate_config(cfg)
        except Exception as exc:
            raise ComparisonSpecValidationError(
                f"invalid_comparison_config: {exc}"
            ) from exc

        for role, out_data in self.per_output.items():
            comp_id = out_data.get("comparator", self.comparator)
            out_plugin = get_comparator(comp_id)
            if out_plugin is None:
                raise ComparisonSpecValidationError(
                    f"unknown comparator {comp_id!r} for output {role!r}"
                )
            out_cfg = out_data.get("config", {})
            try:
                target_out_cfg = out_cfg if comp_id == NUMERIC_COMPARATOR_ID else _clean_custom_config(out_cfg)
                out_plugin.validate_config(target_out_cfg)
            except Exception as exc:
                raise ComparisonSpecValidationError(
                    f"invalid_comparison_config for output {role!r}: {exc}"
                ) from exc

    def to_tolerance_policy(self) -> TolerancePolicy:
        """Build a TolerancePolicy for the built-in numeric comparator."""
        per_dtype_table = dict(DEFAULT_TOLERANCES)
        if "per_dtype" in self.config:
            for dtype_name, tol_dict in self.config["per_dtype"].items():
                def_atol, def_rtol = DEFAULT_TOLERANCES.get(dtype_name, (0.0, 0.0))
                if isinstance(tol_dict, dict):
                    per_dtype_table[dtype_name] = (
                        float(tol_dict.get("atol", def_atol)),
                        float(tol_dict.get("rtol", def_rtol)),
                    )
                elif isinstance(tol_dict, (tuple, list)) and len(tol_dict) == 2:
                    per_dtype_table[dtype_name] = (float(tol_dict[0]), float(tol_dict[1]))

        override_atol = float(self.config["atol"]) if "atol" in self.config else None
        override_rtol = float(self.config["rtol"]) if "rtol" in self.config else None

        return TolerancePolicy(
            per_dtype=per_dtype_table,
            override_atol=override_atol,
            override_rtol=override_rtol,
        )


def _load_tolerance_file(path_or_str: str | Path) -> dict[str, dict[str, float]]:
    """Load a per-dtype tolerance table JSON file."""
    path = Path(path_or_str).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"tolerance file {path} must contain a JSON object")
    table: dict[str, dict[str, float]] = {}
    for dtype_name, val in data.items():
        if dtype_name.startswith("_"):
            continue
        if not isinstance(val, dict):
            raise ValueError(f"tolerance file entry for {dtype_name!r} must be an object")
        table[dtype_name] = {
            "atol": float(val.get("atol", DEFAULT_TOLERANCES.get(dtype_name, (0.0, 0.0))[0])),
            "rtol": float(val.get("rtol", DEFAULT_TOLERANCES.get(dtype_name, (0.0, 0.0))[1])),
        }
    return table


def _format_per_dtype_table(table: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Normalize a per_dtype table mapping to {dtype: {'atol': ..., 'rtol': ...}}."""
    result: dict[str, dict[str, float]] = {}
    # First populate all defaults
    for dt, (def_atol, def_rtol) in DEFAULT_TOLERANCES.items():
        result[dt] = {"atol": def_atol, "rtol": def_rtol}

    # Then apply table entries
    for dt, val in table.items():
        if isinstance(val, dict):
            def_atol, def_rtol = DEFAULT_TOLERANCES.get(dt, (0.0, 0.0))
            result[dt] = {
                "atol": float(val.get("atol", def_atol)),
                "rtol": float(val.get("rtol", def_rtol)),
            }
        elif isinstance(val, (tuple, list)) and len(val) == 2:
            result[dt] = {"atol": float(val[0]), "rtol": float(val[1])}
    return result


def resolve_comparison_policy(
    *,
    testcase_spec: ComparisonSpec | dict[str, Any] | None = None,
    suite_spec: ComparisonSpec | dict[str, Any] | None = None,
    cli_comparator: str | None = None,
    cli_atol: float | None = None,
    cli_rtol: float | None = None,
    cli_strict_layout: bool | None = None,
    cli_ignore_stride: bool | None = None,
    cli_tolerance: str | Path | dict[str, Any] | None = None,
    cli_config: dict[str, Any] | None = None,
) -> EffectiveComparison:
    """Resolve two-axis tolerance precedence and generate effective_comparison (SPEC §6.3).

    Precedence rules:
    - Axis A (per-dtype default table, table replacement):
        suite case config.per_dtype > testcase config.per_dtype > CLI --tolerance > DEFAULT_TOLERANCES
    - Axis B (scalar overrides, applied after Axis A):
        CLI --atol/--rtol > suite case config.atol/rtol > testcase config.atol/rtol > None
    - Strict Layout / Ignore Stride:
        CLI > suite case config > testcase config > default (False)
    - Comparator:
        CLI > suite case spec > testcase spec > default ("tensor/numeric/v1")

    Returns:
        EffectiveComparison with resolved config and fine-grained per-field sources.
    """
    tc = ComparisonSpec.from_dict(testcase_spec) if isinstance(testcase_spec, dict) else testcase_spec
    sc = ComparisonSpec.from_dict(suite_spec) if isinstance(suite_spec, dict) else suite_spec

    sources: dict[str, str] = {}
    config: dict[str, Any] = {}

    # 1. Resolve Top-Level Comparator (Strict priority: CLI > Suite > Testcase > Default)
    if cli_comparator is not None and cli_comparator.strip():
        comparator = cli_comparator.strip()
        sources["comparator"] = "cli"
    elif sc is not None and sc.comparator is not None and sc.comparator.strip():
        comparator = sc.comparator.strip()
        sources["comparator"] = "suite"
    elif tc is not None and tc.comparator is not None and tc.comparator.strip():
        comparator = tc.comparator.strip()
        sources["comparator"] = "testcase"
    else:
        comparator = NUMERIC_COMPARATOR_ID
        sources["comparator"] = "default"

    # 2. Axis A: Per-Dtype Table Precedence
    per_dtype_table: dict[str, Any] | None = None
    if sc is not None and "per_dtype" in sc.config:
        per_dtype_table = sc.config["per_dtype"]
        sources["config.per_dtype"] = "suite"
    elif tc is not None and "per_dtype" in tc.config:
        per_dtype_table = tc.config["per_dtype"]
        sources["config.per_dtype"] = "testcase"
    elif cli_tolerance is not None:
        if isinstance(cli_tolerance, (str, Path)):
            per_dtype_table = _load_tolerance_file(cli_tolerance)
        elif isinstance(cli_tolerance, dict):
            per_dtype_table = cli_tolerance
        sources["config.per_dtype"] = "cli"
    else:
        per_dtype_table = {k: {"atol": v[0], "rtol": v[1]} for k, v in DEFAULT_TOLERANCES.items()}
        sources["config.per_dtype"] = "default"

    config["per_dtype"] = _format_per_dtype_table(per_dtype_table)

    # 3. Base/Custom Config Keys inheritance (Testcase -> Suite -> CLI)
    # Collect extra keys from testcase config
    if tc is not None:
        for k, v in tc.config.items():
            if k not in ("per_dtype", "atol", "rtol", "strict_layout", "ignore_stride"):
                config[k] = v
                sources[f"config.{k}"] = "testcase"

    # Overlay extra keys from suite config
    if sc is not None:
        for k, v in sc.config.items():
            if k not in ("per_dtype", "atol", "rtol", "strict_layout", "ignore_stride"):
                config[k] = v
                sources[f"config.{k}"] = "suite"

    # Overlay extra keys from cli_config
    if cli_config:
        for k, v in cli_config.items():
            if k not in ("per_dtype", "atol", "rtol", "strict_layout", "ignore_stride"):
                config[k] = v
                sources[f"config.{k}"] = "cli"

    # 4. Strict Layout & Ignore Stride
    if cli_strict_layout is not None:
        config["strict_layout"] = bool(cli_strict_layout)
        sources["config.strict_layout"] = "cli"
    elif sc is not None and "strict_layout" in sc.config:
        config["strict_layout"] = bool(sc.config["strict_layout"])
        sources["config.strict_layout"] = "suite"
    elif tc is not None and "strict_layout" in tc.config:
        config["strict_layout"] = bool(tc.config["strict_layout"])
        sources["config.strict_layout"] = "testcase"
    else:
        config["strict_layout"] = False
        sources["config.strict_layout"] = "default"

    if cli_ignore_stride is not None:
        config["ignore_stride"] = bool(cli_ignore_stride)
        sources["config.ignore_stride"] = "cli"
    elif sc is not None and "ignore_stride" in sc.config:
        config["ignore_stride"] = bool(sc.config["ignore_stride"])
        sources["config.ignore_stride"] = "suite"
    elif tc is not None and "ignore_stride" in tc.config:
        config["ignore_stride"] = bool(tc.config["ignore_stride"])
        sources["config.ignore_stride"] = "testcase"
    else:
        config["ignore_stride"] = False
        sources["config.ignore_stride"] = "default"

    # 5. Axis B: Scalar Overrides (atol / rtol)
    if cli_atol is not None:
        config["atol"] = float(cli_atol)
        sources["config.atol"] = "cli"
    elif sc is not None and "atol" in sc.config:
        config["atol"] = float(sc.config["atol"])
        sources["config.atol"] = "suite"
    elif tc is not None and "atol" in tc.config:
        config["atol"] = float(tc.config["atol"])
        sources["config.atol"] = "testcase"

    if cli_rtol is not None:
        config["rtol"] = float(cli_rtol)
        sources["config.rtol"] = "cli"
    elif sc is not None and "rtol" in sc.config:
        config["rtol"] = float(sc.config["rtol"])
        sources["config.rtol"] = "suite"
    elif tc is not None and "rtol" in tc.config:
        config["rtol"] = float(tc.config["rtol"])
        sources["config.rtol"] = "testcase"

    # 6. Per-Output Resolution
    per_output: dict[str, Any] = {}
    tc_outputs = tc.outputs if tc is not None else {}
    sc_outputs = sc.outputs if sc is not None else {}

    all_roles = sorted(set(tc_outputs.keys()) | set(sc_outputs.keys()))
    for role in all_roles:
        tc_out = tc_outputs.get(role)
        sc_out = sc_outputs.get(role)
        out_sources: dict[str, str] = {}
        out_config: dict[str, Any] = {}

        # Resolve output comparator
        if sc_out is not None and sc_out.comparator:
            out_comp = sc_out.comparator
            out_sources["comparator"] = "suite"
        elif tc_out is not None and tc_out.comparator:
            out_comp = tc_out.comparator
            out_sources["comparator"] = "testcase"
        else:
            out_comp = comparator
            out_sources["comparator"] = sources["comparator"]

        # Merge output configs
        if tc_out is not None and tc_out.config:
            for k, v in tc_out.config.items():
                out_config[k] = v
                out_sources[f"config.{k}"] = "testcase"

        if sc_out is not None and sc_out.config:
            for k, v in sc_out.config.items():
                out_config[k] = v
                out_sources[f"config.{k}"] = "suite"

        per_output[role] = {
            "comparator": out_comp,
            "config": out_config,
            "sources": out_sources,
        }

    return EffectiveComparison(
        comparator=comparator,
        config=config,
        per_output=per_output,
        sources=sources,
    )
