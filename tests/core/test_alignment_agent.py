"""Unit and integration tests for InferRef Alignment Agent service functions (0.10 L7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inferref.agent.service import (
    alignment_init,
    alignment_narrow,
    alignment_note,
    alignment_plan,
    alignment_report,
    alignment_run,
    alignment_status,
    alignment_summary,
    capabilities,
)
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.module import ModuleRecord
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef
from inferref.tensor.codec import write_array


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


def test_capabilities_includes_alignment() -> None:
    resp = capabilities()
    assert resp.status == "ok"
    assert "alignment" in resp.data["formats"]
    assert "evidence" in resp.data["formats"]
    op_names = {op["name"] for op in resp.data["operations"]}
    expected_ops = {
        "alignment_init",
        "alignment_plan",
        "alignment_run",
        "alignment_status",
        "alignment_summary",
        "alignment_narrow",
        "alignment_note",
        "alignment_report",
    }
    assert expected_ops.issubset(op_names)
    mcp_tools = set(resp.data["mcp_tools"])
    for op in expected_ops:
        assert f"inferref_{op}" in mcp_tools
    assert not any(t.startswith("inferref_align_") for t in mcp_tools)


def test_agent_service_alignment_workflow(tmp_path: Path) -> None:
    trace_dir, adapter_path = _setup_trace_and_adapter(tmp_path)
    session_dir = tmp_path / "session"

    # 1. alignment_init returns status="pass" per Spec §11.1
    resp = alignment_init(trace_dir, adapter_path, session_dir=session_dir, name="agent-test")
    assert resp.status == "pass"
    assert resp.operation == "alignment_init"
    assert resp.to_dict()["protocol"]["format"] == "inferref-agent-response"
    assert (session_dir / "alignment.json").is_file()

    # 2. alignment_plan returns status="ok"
    resp = alignment_plan(session_dir)
    assert resp.status == "ok"
    assert resp.data["recommended_boundary"] == "region:1"
    assert any(a["operation"] == "alignment_run" for a in resp.next_actions)

    # 3. alignment_run requires explicit adapter and returns status="pass" on success
    resp = alignment_run(session_dir, adapter=adapter_path, limit=1)
    assert resp.status == "pass"
    assert resp.data["results"][0]["status"] == "pass"

    # Test adapter digest mismatch
    fake_adapter = tmp_path / "fake_adapter.json"
    fake_adapter.write_text(
        json.dumps({
            "format": "inferref-engine-adapter",
            "format_version": "0.2",
            "name": "fake",
            "target_device": "cpu",
            "capabilities": {"device_types": ["cpu"], "dtypes": ["float32"], "max_rank": 4},
            "command": ["echo", "{testcase}", "{output}"],
        }),
        encoding="utf-8",
    )
    err_resp = alignment_run(session_dir, adapter=fake_adapter, limit=1)
    assert err_resp.status == "error"
    assert err_resp.diagnostics[0]["code"] == "adapter_digest_mismatch"

    # 4. alignment_status returns status="ok"
    resp = alignment_status(session_dir)
    assert resp.status == "ok"
    assert resp.data["counts"]["verified"] == 1

    # 5. alignment_summary returns compact summary per Spec §11.3
    resp = alignment_summary(session_dir)
    assert resp.status == "ok"
    assert "counts" in resp.data
    assert "current_failure" in resp.data
    assert "classification_candidates" in resp.data
    assert "next_actions" in resp.data
    assert "artifacts" in resp.data

    # 6. alignment_narrow returns status="ok"
    resp = alignment_narrow(session_dir)
    assert resp.status == "ok"
    assert "narrowing" in resp.data

    # 7. alignment_note returns status="pass" per Spec §11.1
    resp = alignment_note(
        session_dir,
        "region:1",
        "milestone",
        "Agent verified boundary",
        duration_minutes=5.0,
        milestone="first-correct-model",
    )
    assert resp.status == "pass"
    assert resp.data["note"]["category"] == "milestone"
    assert resp.data["note"]["duration_minutes"] == 5.0
    assert resp.data["note"]["milestone"] == "first-correct-model"

    # 8. alignment_report returns status="ok"
    resp = alignment_report(session_dir)
    assert resp.status == "ok"
    assert resp.data["format"] == "inferref-alignment-report"


def test_agent_alignment_run_cli_missing_adapter(tmp_path: Path) -> None:
    from inferref.cli.main import main

    with pytest.raises(SystemExit) as exc_info:
        main(["agent", "alignment_run", str(tmp_path)])
    assert exc_info.value.code == 2


def test_agent_alignment_run_external_testcase_missing_fixture_kind(tmp_path: Path) -> None:
    trace_dir, adapter_path = _setup_trace_and_adapter(tmp_path)
    session_dir = tmp_path / "session_ext_fk"
    alignment_init(trace_dir, adapter_path, session_dir=session_dir)

    dummy_tc = tmp_path / "dummy_tc"
    dummy_tc.mkdir()

    # 1. Service call without fixture_kind returns invalid_request error
    resp = alignment_run(
        session_dir,
        adapter=adapter_path,
        external_testcase=dummy_tc,
        boundary_ids=["region:1"],
    )
    assert resp.status == "error"
    assert resp.diagnostics[0]["code"] == "invalid_request"
    assert "explicit fixture_kind" in resp.diagnostics[0]["message"]

    # 2. CLI call without --fixture-kind returns EXIT_USAGE (2)
    from inferref.cli.main import main

    assert (
        main([
            "agent",
            "alignment_run",
            str(session_dir),
            "--adapter",
            str(adapter_path),
            "--testcase",
            str(dummy_tc),
        ])
        == 2
    )
