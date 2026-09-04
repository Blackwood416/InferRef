"""Unit and integration tests for InferRef Alignment MCP tools (0.10 L7)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from inferref.agent.mcp_server import create_server
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.module import ModuleRecord
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef
from inferref.tensor.codec import write_array

pytest.importorskip("mcp")
from mcp import Client


def _setup_trace_and_adapter(tmp_path: Path) -> tuple[Path, Path]:
    trace_dir = tmp_path / "trace"
    payloads = trace_dir / "payloads"
    payloads.mkdir(parents=True, exist_ok=True)

    write_array(payloads / "v1.irtensor", np.array([1.0, 2.0], dtype=np.float32))
    write_array(payloads / "v2.irtensor", np.array([2.0, 4.0], dtype=np.float32))

    graph = Graph()
    graph.values = [
        TensorValueRecord(
            id=1,
            shape=(2,),
            stride=(1,),
            dtype="float32",
            capture=CaptureInfo(mode="full", payload="payloads/v1.irtensor"),
        ),
        TensorValueRecord(
            id=2,
            shape=(2,),
            stride=(1,),
            dtype="float32",
            producer=10,
            capture=CaptureInfo(mode="full", payload="payloads/v2.irtensor"),
        ),
    ]
    graph.operators = [
        OperatorRecord(
            id=10,
            execution_index=0,
            op="double",
            result=TensorRef(2),
            positional_args=(TensorRef(1),),
            module_stack=(100,),
        ),
    ]
    graph.reindex()
    pkg = TracePackage(
        manifest=Manifest(),
        graph=graph,
        modules=[ModuleRecord(id=100, path="model")],
        regions=[RegionRecord(id=1, name="op_double", node_ids=(10,), inputs=(1,), outputs=(2,))],
        root=trace_dir,
    )
    pkg.save(trace_dir)

    adapter_path = tmp_path / "adapter.json"
    adapter_data = {
        "format": "inferref-engine-adapter",
        "format_version": "0.2",
        "name": "test-engine",
        "target_device": "cpu",
        "capabilities": {"device_types": ["cpu"], "dtypes": ["float32"], "max_rank": 4},
        "command": [
            "{python}",
            "-c",
            "import sys, shutil; from pathlib import Path; tc, out = Path(sys.argv[1]), Path(sys.argv[2]); shutil.copytree(tc / 'reference', out, dirs_exist_ok=True)",
            "{testcase}",
            "{output}",
        ],
    }
    adapter_path.write_text(json.dumps(adapter_data, indent=2), encoding="utf-8")
    return trace_dir, adapter_path


def test_mcp_server_alignment_tools(tmp_path: Path) -> None:
    trace_dir, adapter_path = _setup_trace_and_adapter(tmp_path)
    session_dir = tmp_path / "mcp_session"

    async def exercise() -> None:
        async with Client(create_server(read_roots=[tmp_path], write_roots=[tmp_path])) as client:
            tools_list = await client.list_tools()
            tool_names = {t.name for t in tools_list.tools}
            expected_mcp_tools = {
                "inferref_alignment_init",
                "inferref_alignment_plan",
                "inferref_alignment_run",
                "inferref_alignment_status",
                "inferref_alignment_summary",
                "inferref_alignment_narrow",
                "inferref_alignment_note",
                "inferref_alignment_report",
            }
            assert expected_mcp_tools.issubset(tool_names)
            assert not any(t.startswith("inferref_align_") for t in tool_names)

            # Test tool execution
            init_res = await client.call_tool(
                "inferref_alignment_init",
                {
                    "trace": str(trace_dir),
                    "adapter": str(adapter_path),
                    "session_dir": str(session_dir),
                    "name": "mcp-test",
                },
            )
            init_payload = init_res.structured_content
            assert init_payload["status"] == "pass"

            # Test run with explicit adapter parameter
            run_res = await client.call_tool(
                "inferref_alignment_run",
                {
                    "session": str(session_dir),
                    "adapter": str(adapter_path),
                    "limit": 1,
                },
            )
            run_payload = run_res.structured_content
            assert run_payload["status"] == "pass"

            # Test external testcase requires explicit fixture_kind
            dummy_tc = tmp_path / "mcp_dummy_tc"
            dummy_tc.mkdir()
            no_fk_res = await client.call_tool(
                "inferref_alignment_run",
                {
                    "session": str(session_dir),
                    "adapter": str(adapter_path),
                    "external_testcase": str(dummy_tc),
                    "boundaries": ["region:1"],
                },
            )
            no_fk_payload = no_fk_res.structured_content
            assert no_fk_payload["status"] == "error"
            assert no_fk_payload["diagnostics"][0]["code"] == "invalid_request"

            status_res = await client.call_tool(
                "inferref_alignment_status",
                {"session": str(session_dir)},
            )
            status_payload = status_res.structured_content
            assert status_payload["status"] == "ok"

            # Test path policy rejection for out-of-boundary session path
            outside_dir = tmp_path.parent / "outside_forbidden_session"
            policy_res = await client.call_tool(
                "inferref_alignment_status",
                {"session": str(outside_dir)},
            )
            policy_payload = policy_res.structured_content
            assert policy_payload["status"] == "error"
            assert policy_payload["diagnostics"][0]["code"] == "path_not_allowed"

            # Test transitive path policy rejection: session in allowed dir, trace outside allowed dir
            transitive_session = tmp_path / "transitive_session"
            from inferref.alignment.store import AlignmentStore
            outside_trace = tmp_path.parent / "outside_trace"
            AlignmentStore.initialize(
                transitive_session,
                trace_path=outside_trace,
                adapter_path=adapter_path,
                trace_metadata_digest="0" * 64,
                adapter_digest="1" * 64,
                session_id="al-transitive",
            )
            trans_res = await client.call_tool(
                "inferref_alignment_status",
                {"session": str(transitive_session)},
            )
            trans_payload = trans_res.structured_content
            assert trans_payload["status"] == "error"
            assert trans_payload["diagnostics"][0]["code"] == "path_not_allowed"

    asyncio.run(exercise())
