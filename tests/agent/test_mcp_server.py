"""The optional MCP transport exposes the same structured Agent contract."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from mcp import Client

import inferref.agent.mcp_server as mcp_server_module
from inferref.agent.mcp_server import create_server
from inferref.agent.protocol import AgentResponse
from inferref.ir.version import INFERREF_VERSION


def test_mcp_capabilities_roundtrip_in_memory() -> None:
    async def exercise() -> None:
        async with Client(create_server()) as client:
            result = await client.call_tool("inferref_capabilities", {})
            payload = result.structured_content
            assert payload["operation"] == "capabilities"
            assert payload["status"] == "ok"
            assert payload["data"]["inferref_version"] == INFERREF_VERSION

    asyncio.run(exercise())


def test_mcp_rejects_paths_outside_host_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()

    async def exercise() -> None:
        async with Client(
            create_server(read_roots=[allowed], write_roots=[allowed])
        ) as client:
            result = await client.call_tool("inferref_context", {"path": str(outside)})
            payload = result.structured_content
            assert payload["status"] == "error"
            assert payload["diagnostics"][0]["code"] == "path_not_allowed"

            extraction = await client.call_tool(
                "inferref_extract_testcase",
                {
                    "trace": str(allowed / "trace"),
                    "output": str(outside / "testcase"),
                    "op_id": 1,
                },
            )
            extraction_payload = extraction.structured_content
            assert extraction_payload["status"] == "error"
            assert extraction_payload["diagnostics"][0]["code"] == "path_not_allowed"

    asyncio.run(exercise())


def test_mcp_extract_forwards_executable_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    trace = allowed / "trace"
    allowed.mkdir()
    trace.mkdir()
    observed: dict[str, object] = {}

    def fake_extract(trace_path: Path, output_path: Path, **kwargs: object) -> AgentResponse:
        observed.update(kwargs)
        return AgentResponse(
            operation="extract_testcase",
            status="pass",
            data={"trace": str(trace_path), "output": str(output_path)},
        )

    monkeypatch.setattr(mcp_server_module, "extract_testcase", fake_extract)

    async def exercise() -> None:
        async with Client(
            create_server(read_roots=[allowed], write_roots=[allowed])
        ) as client:
            result = await client.call_tool(
                "inferref_extract_testcase",
                {
                    "trace": str(trace),
                    "output": str(allowed / "testcase"),
                    "op_id": 1,
                    "contracts": ["rope/rotate-half/v1"],
                },
            )
            assert result.structured_content["status"] == "pass"

    asyncio.run(exercise())
    assert observed["contracts"] == ["rope/rotate-half/v1"]


def test_mcp_run_scenario_enforces_paths_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()

    scenario = allowed / "scenarios" / "kv-chain"
    scenario.parent.mkdir(parents=True)
    import shutil
    from pathlib import Path

    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "scenarios"
        / "kv-chain"
    )
    shutil.copytree(fixture, scenario)
    adapter = allowed / "adapter.json"
    engine = fixture.parent.parent / "adapters" / "copy_engine.py"
    adapter.write_text(
        json.dumps(
            {
                "format": "inferref-engine-adapter",
                "format_version": "0.2",
                "name": "kv-copy",
                "target_device": "cpu",
                "capabilities": {
                    "device_types": ["cpu"],
                    "dtypes": ["float32"],
                    "max_rank": 8,
                    "features": ["multiple_outputs"],
                },
                "command": ["{python}", str(engine), "{testcase}", "{output}"],
            }
        ),
        encoding="utf-8",
    )

    async def exercise() -> None:
        async with Client(
            create_server(read_roots=[allowed], write_roots=[allowed])
        ) as client:
            blocked = await client.call_tool(
                "inferref_run_scenario",
                {
                    "scenario": str(outside / "scenario"),
                    "adapter": str(adapter),
                    "runs_root": str(allowed / "runs"),
                },
            )
            payload = blocked.structured_content
            assert payload["status"] == "error"
            assert payload["diagnostics"][0]["code"] == "path_not_allowed"

            result = await client.call_tool(
                "inferref_run_scenario",
                {
                    "scenario": str(scenario),
                    "adapter": str(adapter),
                    "runs_root": str(allowed / "runs"),
                    "state_mode": "engine",
                    "compare_state": True,
                },
            )
            payload = result.structured_content
            assert payload["operation"] == "run_scenario"
            assert payload["status"] == "pass"
            assert payload["data"]["format"] == "inferref-scenario-run"
            assert payload["data"]["status"] == "pass"

    asyncio.run(exercise())
