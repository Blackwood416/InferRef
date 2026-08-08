"""Deprecated compatibility shim for the executable contract registry.

Contract Schema v0.1 section 10: the hardcoded contract table moved to
``inferref.contracts``. This module keeps importing so existing tests and
third-party callers do not break; new code should import from
``inferref.contracts``.
"""

from __future__ import annotations

import warnings

from inferref.contracts import (
    EXECUTABLE_CONTRACTS,
    REGISTRY,
    ExecutableContract,
    contract_boundary_issues,
    contract_input_issues,
    contract_requirements,
    get_contract,
)

warnings.warn(
    "inferref.testcase.contracts is deprecated; import from inferref.contracts "
    "instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "EXECUTABLE_CONTRACTS",
    "REGISTRY",
    "ExecutableContract",
    "contract_boundary_issues",
    "contract_input_issues",
    "contract_requirements",
    "get_contract",
]
