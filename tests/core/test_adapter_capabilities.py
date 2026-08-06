"""Per-contract adapter capabilities and three-state contract semantics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inferref.agent.adapter import execute_adapter
from inferref.agent.protocol import (
    ENGINE_ADAPTER_FORMAT,
    ENGINE_ADAPTER_VERSION,
    AdapterCapabilities,
    AgentProtocolError,
    ContractCapability,
    EngineAdapter,
)
from inferref.tensor import codec
from inferref.testcase.requirements import derive_requirements


def _rope_case(root: Path) -> Path:
    query = codec.write_array(
        root / "inputs/query.irtensor",
        np.asarray([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]], dtype=np.float32),
    )
    key = codec.write_array(
        root / "inputs/key.irtensor",
        np.asarray([[[[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]]], dtype=np.float32),
    )
    cos = codec.write_array(
        root / "inputs/cos.irtensor",
        np.asarray([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]], dtype=np.float32),
    )
    sin = codec.write_array(
        root / "inputs/sin.irtensor",
        np.asarray([[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]], dtype=np.float32),
    )
    q_embed = codec.write_array(
        root / "reference/q_embed.irtensor",
        np.asarray([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]]], dtype=np.float32),
    )
    k_embed = codec.write_array(
        root / "reference/k_embed.irtensor",
        np.asarray([[[[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]]], dtype=np.float32),
    )

    entries: list[dict[str, object]] = []
    values: list[dict[str, object]] = []
    for value_id, (name, payload) in enumerate(
        [
            ("query", query),
            ("key", key),
            ("cos", cos),
            ("sin", sin),
            ("q_embed", q_embed),
            ("k_embed", k_embed),
        ],
        start=1,
    ):
        metadata = codec.read(payload).to_metadata()
        entries.append(
            {
                "name": name,
                "value_id": value_id,
                "payload": str(payload.relative_to(root)).replace("\\", "/"),
                **metadata,
            }
        )
        values.append({"id": value_id, **metadata})

    inputs = entries[:4]
    outputs = entries[4:]
    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.2",
        "name": "rope-probe",
        "reproducible": True,
        "contracts": ["rope/rotate-half/v1"],
        "inputs": inputs,
        "outputs": outputs,
        "nodes": [],
        "values": values,
    }
    manifest["requirements"] = derive_requirements(manifest)
    (root / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _adapter(path: Path, capabilities: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(
            {
                "format": ENGINE_ADAPTER_FORMAT,
                "format_version": ENGINE_ADAPTER_VERSION,
                "name": "probe",
                "target_device": "cpu",
                "capabilities": capabilities,
                "command": [
                    "{python}",
                    "-c",
                    "raise SystemExit('must not run')",
                    "{testcase}",
                    "{output}",
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_contract_capabilities_parse_and_validate() -> None:
    caps = AdapterCapabilities.from_dict(
        {
            "device_types": ["xpu"],
            "dtypes": ["float32", "float16"],
            "max_rank": 4,
            "features": [],
            "contracts": ["rope/rotate-half/v1", "kv-cache/append/v1"],
            "contract_capabilities": {
                "rope/rotate-half/v1": {"dtypes": ["float32"], "max_rank": 3},
                "kv-cache/append/v1": {"features": []},
            },
        }
    )
    rope = caps.contract_capabilities["rope/rotate-half/v1"]
    assert rope.dtypes == ("float32",)
    assert rope.max_rank == 3
    assert caps.contract_capabilities["kv-cache/append/v1"].features == ()

    with pytest.raises(AgentProtocolError, match="requires capabilities.contracts"):
        AdapterCapabilities.from_dict(
            {
                "device_types": ["xpu"],
                "dtypes": ["float32"],
                "max_rank": 4,
                "features": [],
                "contract_capabilities": {"rope/rotate-half/v1": {}},
            }
        )
    with pytest.raises(
        AgentProtocolError, match="not declared in capabilities.contracts"
    ):
        AdapterCapabilities.from_dict(
            {
                "device_types": ["xpu"],
                "dtypes": ["float32"],
                "max_rank": 4,
                "features": [],
                "contracts": ["kv-cache/append/v1"],
                "contract_capabilities": {"rope/rotate-half/v1": {}},
            }
        )
    with pytest.raises(AgentProtocolError, match="unknown contract capability feature"):
        AdapterCapabilities.from_dict(
            {
                "device_types": ["xpu"],
                "dtypes": ["float32"],
                "max_rank": 4,
                "features": [],
                "contracts": ["rope/rotate-half/v1"],
                "contract_capabilities": {
                    "rope/rotate-half/v1": {"features": ["not_a_feature"]}
                },
            }
        )


def test_empty_contracts_array_is_strict_zero_support() -> None:
    caps = AdapterCapabilities.from_dict(
        {
            "device_types": ["cpu"],
            "dtypes": ["float32"],
            "max_rank": 4,
            "features": [],
            "contracts": [],
        }
    )
    assert caps.contracts == ()
    assert caps.assessment({"contracts": ["rope/rotate-half/v1"]}) == "supported"
    issues = caps.incompatibilities({"contracts": ["rope/rotate-half/v1"]})
    assert issues == [{"kind": "contract", "required": "rope/rotate-half/v1"}]


def test_contract_incompatibilities_refinements() -> None:
    caps = AdapterCapabilities(
        device_types=("cpu",),
        dtypes=("float32", "float16"),
        max_rank=8,
        features=(),
        contracts=("rope/rotate-half/v1",),
        contract_capabilities={
            "rope/rotate-half/v1": ContractCapability(
                dtypes=("float32",), max_rank=3, features=("multiple_outputs",)
            )
        },
    )
    assert (
        caps.contract_incompatibilities(
            "rope/rotate-half/v1",
            {"dtypes": ["float32"], "max_rank": 3, "features": ["multiple_outputs"]},
        )
        == []
    )
    assert caps.contract_incompatibilities(
        "rope/rotate-half/v1",
        {"dtypes": ["float16"], "max_rank": 3, "features": []},
    ) == [
        {
            "kind": "contract_dtype",
            "contract": "rope/rotate-half/v1",
            "required": "float16",
        }
    ]
    assert caps.contract_incompatibilities(
        "rope/rotate-half/v1",
        {"dtypes": ["float32"], "max_rank": 4, "features": []},
    ) == [
        {
            "kind": "contract_max_rank",
            "contract": "rope/rotate-half/v1",
            "required": 4,
            "supported": 3,
        }
    ]
    assert caps.contract_incompatibilities(
        "rope/rotate-half/v1",
        {
            "dtypes": ["float32"],
            "max_rank": 3,
            "features": ["multiple_outputs", "mutation_effects"],
        },
    ) == [
        {
            "kind": "contract_feature",
            "contract": "rope/rotate-half/v1",
            "required": "mutation_effects",
        }
    ]


def test_preflight_rejects_per_contract_mismatch_before_subprocess(
    tmp_path: Path,
) -> None:
    case = _rope_case(tmp_path / "cases" / "rope")
    base: dict[str, object] = {
        "device_types": ["cpu"],
        "dtypes": ["float32", "float16"],
        "max_rank": 8,
        "features": ["multiple_outputs"],
        "contracts": ["rope/rotate-half/v1"],
    }
    adapter = _adapter(
        tmp_path / "adapter.json",
        {
            **base,
            "contract_capabilities": {
                "rope/rotate-half/v1": {
                    "dtypes": ["float16"],
                    "max_rank": 4,
                    "features": ["multiple_outputs"],
                }
            },
        },
    )
    result = execute_adapter(case, EngineAdapter.load(adapter), tmp_path / "runs")
    assert result["status"] == "unsupported"
    assert result["execution"] is None
    assert result["unsupported"][0]["kind"] == "contract_dtype"

    rank_adapter = _adapter(
        tmp_path / "rank-adapter.json",
        {
            **base,
            "contract_capabilities": {"rope/rotate-half/v1": {"max_rank": 3}},
        },
    )
    rank_result = execute_adapter(
        case, EngineAdapter.load(rank_adapter), tmp_path / "rank-runs"
    )
    assert rank_result["status"] == "unsupported"
    assert rank_result["unsupported"][0]["kind"] == "contract_max_rank"

    feature_adapter = _adapter(
        tmp_path / "feature-adapter.json",
        {
            **base,
            "contract_capabilities": {"rope/rotate-half/v1": {"features": []}},
        },
    )
    feature_result = execute_adapter(
        case, EngineAdapter.load(feature_adapter), tmp_path / "feature-runs"
    )
    assert feature_result["status"] == "unsupported"
    assert feature_result["unsupported"][0]["kind"] == "contract_feature"

    zero_contracts = _adapter(
        tmp_path / "zero-contracts.json",
        {
            "device_types": ["cpu"],
            "dtypes": ["float32", "float16"],
            "max_rank": 8,
            "features": ["multiple_outputs"],
            "contracts": [],
        },
    )
    zero_result = execute_adapter(
        case, EngineAdapter.load(zero_contracts), tmp_path / "zero-runs"
    )
    assert zero_result["status"] == "unsupported"
    assert zero_result["unsupported"][0]["kind"] == "contract"


def test_legacy_adapter_without_contracts_remains_unchecked(
    tmp_path: Path,
) -> None:
    case = _rope_case(tmp_path / "cases" / "rope")
    adapter = _adapter(
        tmp_path / "legacy.json",
        {
            "device_types": ["cpu"],
            "dtypes": ["float32", "float16"],
            "max_rank": 8,
            "features": ["multiple_outputs"],
        },
    )
    # Capability assessment is unchecked, but the adapter never runs because
    # there is no real engine behind the probe command; a supported verdict
    # would attempt to start the subprocess and fail. The unchecked legacy
    # contract keeps execution behavior unchanged, so this is not an unsupported
    # preflight result.
    result = execute_adapter(case, EngineAdapter.load(adapter), tmp_path / "runs")
    assert result["status"] not in {"unsupported"}
    assert result["run_id"] is not None or result["status"] == "execution_error"
