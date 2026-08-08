"""Executable contract registry (Contract Schema v0.1).

Public API::

    get_contract(contract_id) -> ExecutableContract | None
    contract_list() -> list[ContractEntry]
    load_contract_file(path) -> ExecutableContract
    verify_contracts() -> list[ContractPluginStatus]
    contract_input_issues(contract_id, inputs) -> list[str]
    contract_boundary_issues(contract_id, inputs, outputs) -> list[str]
    contract_requirements(manifest, contract_id) -> dict

The package depends on the standard library only; the torch-free core boundary
is unchanged (section 2, 15).
"""

from __future__ import annotations

from inferref.contracts.builtin import EXECUTABLE_CONTRACTS, builtin_contracts
from inferref.contracts.registry import (
    REGISTRY,
    ContractEntry,
    ContractPluginStatus,
    contract_boundary_issues,
    contract_input_issues,
    contract_list,
    contract_plugin_statuses,
    contract_requirements,
    get_contract,
    load_contract_file,
    validate_contract_file,
    verify_contracts,
)
from inferref.contracts.schema import (
    CONTRACT_FORMAT,
    CONTRACT_FORMAT_VERSION,
    ContractSchemaError,
    ExecutableContract,
)

__all__ = [
    "CONTRACT_FORMAT",
    "CONTRACT_FORMAT_VERSION",
    "EXECUTABLE_CONTRACTS",
    "REGISTRY",
    "ContractEntry",
    "ContractPluginStatus",
    "ContractSchemaError",
    "ExecutableContract",
    "builtin_contracts",
    "contract_boundary_issues",
    "contract_input_issues",
    "contract_list",
    "contract_plugin_statuses",
    "contract_requirements",
    "get_contract",
    "load_contract_file",
    "validate_contract_file",
    "verify_contracts",
]
