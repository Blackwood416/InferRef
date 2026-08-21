"""Core comparator execution and exception isolation (SPEC §7.5).

Pre-validates output role presence (short-circuiting missing roles) and isolates
plugin exceptions to prevent suite crashes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from inferref.comparators.protocol import (
    ArtifactSet,
    ComparatorPlugin,
    ComparatorResult,
)
from inferref.comparators.registry import get_comparator


def run_comparator(
    comparator: ComparatorPlugin | str,
    reference: ArtifactSet,
    actual: ArtifactSet,
    *,
    config: dict[str, Any] | None = None,
    required_roles: Sequence[str] | None = None,
) -> ComparatorResult:
    """Execute comparison with core role pre-validation and exception isolation.

    Args:
        comparator: ComparatorPlugin instance or registered comparator ID.
        reference: Reference artifact set.
        actual: Candidate/engine artifact set.
        config: Optional comparator-specific config dictionary.
        required_roles: Optional list of required output roles to pre-validate.
            If omitted, defaults to all roles present in ``reference``.

    Returns:
        Structured ComparatorResult (status="pass", "fail", or "error").
    """
    if isinstance(comparator, str):
        plugin = get_comparator(comparator)
        if plugin is None:
            raise ValueError(f"unknown comparator {comparator!r}")
    else:
        plugin = comparator

    expected = list(required_roles) if required_roles is not None else list(reference.keys())
    missing_roles = [role for role in expected if role not in actual]

    if missing_roles:
        diagnostics = [
            {
                "output": role,
                "code": "missing_output_role",
                "message": f"engine produced no output for role {role!r}",
            }
            for role in missing_roles
        ]
        return ComparatorResult(
            status="fail",
            comparator=plugin.id,
            metrics={},
            diagnostics=diagnostics,
            first_failure={
                "output": missing_roles[0],
                "message": f"engine produced no output for role {missing_roles[0]!r}",
            },
        )

    try:
        plugin.validate_config(config)
        return plugin.compare(reference, actual, config)
    except Exception as exc:
        return ComparatorResult(
            status="error",
            comparator=plugin.id,
            metrics={},
            diagnostics=[
                {
                    "code": "comparator_exception",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            ],
            first_failure={
                "message": f"comparator raised {type(exc).__name__}: {exc}",
            },
        )
