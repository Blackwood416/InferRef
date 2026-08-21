"""Built-in tensor numeric comparator (tensor/numeric/v1).

Wraps the core numeric tolerance and layout comparison logic as a
ComparatorPlugin compliant with SPEC §7.
"""

from __future__ import annotations

from typing import Any

from inferref.comparators.protocol import (
    NUMERIC_COMPARATOR_ID,
    ArtifactSet,
    ComparatorResult,
)
from inferref.compare.tolerance import DEFAULT_TOLERANCES, TolerancePolicy
from inferref.tensor import codec

__all__ = ["NUMERIC_COMPARATOR_ID", "NumericComparator"]

_ALLOWED_CONFIG_KEYS = {"atol", "rtol", "strict_layout", "ignore_stride", "per_dtype"}


class NumericComparator:
    """Default tensor numeric comparison plugin."""

    id: str = NUMERIC_COMPARATOR_ID

    def validate_config(self, config: dict[str, Any] | None = None) -> None:
        """Validate numeric comparison configuration statically.

        Raises:
            ValueError: If configuration keys or types are invalid.
        """
        if config is None:
            return
        if not isinstance(config, dict):
            raise ValueError(f"numeric comparator config must be a dictionary, got {type(config).__name__}")

        unknown = set(config) - _ALLOWED_CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"unknown numeric comparator config key(s): {sorted(unknown)}; "
                f"allowed keys: {sorted(_ALLOWED_CONFIG_KEYS)}"
            )

        if "atol" in config:
            val = config["atol"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or val < 0:
                raise ValueError(f"atol must be a non-negative float, got {val!r}")

        if "rtol" in config:
            val = config["rtol"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or val < 0:
                raise ValueError(f"rtol must be a non-negative float, got {val!r}")

        if "strict_layout" in config:
            val = config["strict_layout"]
            if not isinstance(val, bool):
                raise ValueError(f"strict_layout must be a boolean, got {val!r}")

        if "ignore_stride" in config:
            val = config["ignore_stride"]
            if not isinstance(val, bool):
                raise ValueError(f"ignore_stride must be a boolean, got {val!r}")

        if "per_dtype" in config:
            per_dtype = config["per_dtype"]
            if not isinstance(per_dtype, dict):
                raise ValueError(f"per_dtype must be a dictionary, got {type(per_dtype).__name__}")
            for dtype_name, entry in per_dtype.items():
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"per_dtype entry for {dtype_name!r} must be a dictionary, got {type(entry).__name__}"
                    )
                entry_unknown = set(entry) - {"atol", "rtol"}
                if entry_unknown:
                    raise ValueError(
                        f"per_dtype entry for {dtype_name!r} has unknown key(s): {sorted(entry_unknown)}"
                    )
                for key in ("atol", "rtol"):
                    if key in entry:
                        t_val = entry[key]
                        if isinstance(t_val, bool) or not isinstance(t_val, (int, float)) or t_val < 0:
                            raise ValueError(
                                f"per_dtype {dtype_name!r} {key} must be a non-negative float, got {t_val!r}"
                            )

    def compare(
        self,
        reference: ArtifactSet,
        actual: ArtifactSet,
        config: dict[str, Any] | None = None,
    ) -> ComparatorResult:
        """Execute numeric comparison across all reference output artifacts."""
        self.validate_config(config)
        cfg = config or {}

        per_dtype_table = dict(DEFAULT_TOLERANCES)
        if "per_dtype" in cfg:
            for dtype_name, tol_dict in cfg["per_dtype"].items():
                def_atol, def_rtol = DEFAULT_TOLERANCES.get(dtype_name, (0.0, 0.0))
                per_dtype_table[dtype_name] = (
                    float(tol_dict.get("atol", def_atol)),
                    float(tol_dict.get("rtol", def_rtol)),
                )

        override_atol = float(cfg["atol"]) if "atol" in cfg else None
        override_rtol = float(cfg["rtol"]) if "rtol" in cfg else None

        policy = TolerancePolicy(
            per_dtype=per_dtype_table,
            override_atol=override_atol,
            override_rtol=override_rtol,
        )
        strict_layout = bool(cfg.get("strict_layout", False))
        ignore_stride = bool(cfg.get("ignore_stride", False))

        diagnostics: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}
        first_failure: dict[str, Any] | None = None
        all_passed = True

        max_abs_errors: list[float] = []
        max_rel_errors: list[float] = []
        mismatch_counts: list[int] = []

        for name, ref_artifact in reference.items():
            if name not in actual:
                all_passed = False
                diag = {
                    "output": name,
                    "code": "missing_output_role",
                    "message": f"engine produced no output for role {name!r}",
                }
                diagnostics.append(diag)
                if first_failure is None:
                    first_failure = {"output": name, "message": diag["message"]}
                continue

            act_artifact = actual[name]

            try:
                ref_view = codec.read(ref_artifact.path)
            except Exception as exc:
                all_passed = False
                diag = {
                    "output": name,
                    "code": "reference_read_error",
                    "message": f"cannot read reference artifact {ref_artifact.path}: {exc}",
                }
                diagnostics.append(diag)
                if first_failure is None:
                    first_failure = {"output": name, "message": diag["message"]}
                continue

            try:
                act_view = codec.read(act_artifact.path)
            except Exception as exc:
                all_passed = False
                diag = {
                    "output": name,
                    "code": "candidate_read_error",
                    "message": f"cannot read candidate artifact {act_artifact.path}: {exc}",
                }
                diagnostics.append(diag)
                if first_failure is None:
                    first_failure = {"output": name, "message": diag["message"]}
                continue

            from inferref.compare.compare import compare_tensors

            comp = compare_tensors(
                name,
                ref_view,
                act_view,
                policy=policy,
                ignore_stride=ignore_stride,
                strict_layout=strict_layout,
            )

            if comp.metrics is not None:
                m = comp.metrics.to_dict()
                if "max_abs_error" in m and m["max_abs_error"] is not None:
                    max_abs_errors.append(float(m["max_abs_error"]))
                if "max_rel_error" in m and m["max_rel_error"] is not None:
                    max_rel_errors.append(float(m["max_rel_error"]))
                if "mismatch_count" in m and m["mismatch_count"] is not None:
                    mismatch_counts.append(int(m["mismatch_count"]))

                if len(reference) == 1:
                    metrics = m
                else:
                    metrics[name] = m

            if not comp.passed:
                all_passed = False
                diag = {
                    "output": name,
                    "code": "tensor_mismatch",
                    "status": comp.status,
                    "message": comp.message or f"tensor {name!r} mismatch",
                    "comparison": comp.to_dict(),
                }
                diagnostics.append(diag)
                if first_failure is None:
                    first_failure = {
                        "output": name,
                        "message": comp.message or f"tensor {name!r} mismatch",
                        "status": comp.status,
                    }

        if len(reference) > 1:
            metrics["summary"] = {
                "max_abs_error": max(max_abs_errors) if max_abs_errors else 0.0,
                "max_rel_error": max(max_rel_errors) if max_rel_errors else 0.0,
                "total_mismatch_count": sum(mismatch_counts),
            }

        return ComparatorResult(
            status="pass" if all_passed else "fail",
            comparator=self.id,
            metrics=metrics,
            diagnostics=diagnostics,
            first_failure=first_failure,
        )
