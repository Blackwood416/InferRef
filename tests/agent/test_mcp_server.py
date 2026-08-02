"""The optional MCP transport exposes the same structured Agent contract."""

from __future__ import annotations

import asyncio

import pytest

mcp = pytest.importorskip("mcp")

from mcp import Client

from inferref.agent.mcp_server import create_server


def test_mcp_capabilities_roundtrip_in_memory() -> None:
    async def exercise() -> None:
        async with Client(create_server()) as client:
            result = await client.call_tool("inferref_capabilities", {})
            payload = result.structured_content
            assert payload["operation"] == "capabilities"
            assert payload["status"] == "ok"
            assert payload["data"]["inferref_version"] == "0.3.0"

    asyncio.run(exercise())
