"""End-to-end MCP evaluation of the one-file repair protocol harness."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from mcp import Client

from examples.agent_eval.rope_sign.prepare import setup_workspace
from inferref.agent.mcp_server import create_server
from inferref.cli.main import EXIT_FAIL, main

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "examples" / "agent_eval" / "rope_sign"


def _copy_workspace(root: Path) -> tuple[Path, dict]:
    benchmark = json.loads(
        (BENCHMARK_ROOT / "benchmark.json").read_text(encoding="utf-8")
    )
    workspace = root / "workspace"
    setup_workspace(workspace)
    observed = {path.name for path in workspace.iterdir()}
    assert observed == set(benchmark["workspace_template"]) | {"testcase"}
    return workspace, benchmark


def _protected_digest(workspace: Path) -> str:
    digest = hashlib.sha256()
    protected = [workspace / "adapter.json", workspace / "TASK.md"]
    protected.extend(sorted((workspace / "testcase").rglob("*")))
    for path in protected:
        if not path.is_file():
            continue
        digest.update(path.relative_to(workspace).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_benchmark_contract_has_machine_readable_iteration_limits() -> None:
    benchmark = json.loads(
        (BENCHMARK_ROOT / "benchmark.json").read_text(encoding="utf-8")
    )
    assert benchmark["format"] == "inferref-agent-evaluation"
    assert benchmark["format_version"] == "0.1"
    assert benchmark["agent"]["editable_paths"] == ["engine.py"]
    assert benchmark["agent"]["max_iterations"] == 4
    assert benchmark["success"] == {
        "operation": "run_engine",
        "status": "pass",
        "protected_paths_unchanged": True,
    }


def test_cli_exposes_the_actionable_baseline_failure(tmp_path: Path, capsys) -> None:
    workspace, benchmark = _copy_workspace(tmp_path)
    exit_code = main(
        [
            "agent",
            "run",
            str(workspace / "testcase"),
            "--adapter",
            str(workspace / "adapter.json"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--json",
        ]
    )
    assert exit_code == EXIT_FAIL
    payload = json.loads(capsys.readouterr().out)
    failure = payload["data"]["comparison"]["first_failure"]
    assert payload["status"] == benchmark["expected_baseline"]["status"]
    assert failure["name"] == benchmark["expected_baseline"]["first_failure"]
    assert failure["metrics"]["first_mismatch"]["index"] == [0, 0, 0, 1]


def test_mcp_diagnostics_support_a_one_file_repair(tmp_path: Path) -> None:
    workspace, benchmark = _copy_workspace(tmp_path)
    before = _protected_digest(workspace)
    engine = workspace / benchmark["agent"]["editable_paths"][0]

    async def exercise() -> None:
        async with Client(
            create_server(read_roots=[workspace], write_roots=[tmp_path])
        ) as client:
            capabilities = await client.call_tool("inferref_capabilities", {})
            available = capabilities.structured_content["data"]["mcp_tools"]
            assert set(benchmark["agent"]["required_tools"]) <= set(available)

            context = await client.call_tool(
                "inferref_context", {"path": str(workspace / "testcase")}
            )
            context_payload = context.structured_content
            assert context_payload["status"] == "ok"
            assert context_payload["next_actions"][0]["operation"] == "run_engine"

            first = await client.call_tool(
                "inferref_run_engine",
                {
                    "testcase": str(workspace / "testcase"),
                    "adapter": str(workspace / "adapter.json"),
                    "runs_root": str(tmp_path / "runs"),
                },
            )
            first_payload = first.structured_content
            assert first_payload["status"] == benchmark["expected_baseline"]["status"]
            assert first_payload["next_actions"][0]["operation"] == "modify_engine"

            failure = first_payload["data"]["comparison"]["first_failure"]
            assert failure["name"] == benchmark["expected_baseline"]["first_failure"]
            assert failure["operator"] == "aten.add.Tensor"
            assert failure["region"] == "RoPE@agent-eval"
            assert "apply_rotary_pos_emb" in failure["source"]
            assert failure["layout"]["reference"]["shape"] == [1, 2, 4, 8]
            assert failure["layout"]["reference"]["dtype"] == "float32"
            assert failure["metrics"]["max_abs_error"] > 0.1
            assert failure["metrics"]["mismatch_count"] > 0
            assert failure["metrics"]["first_mismatch"]["index"]

            source = engine.read_text(encoding="utf-8")
            wrong = "return np.concatenate((second, -first), axis=-1)"
            fixed = "return np.concatenate((-second, first), axis=-1)"
            assert source.count(wrong) == 1
            engine.write_text(source.replace(wrong, fixed), encoding="utf-8")

            second = await client.call_tool(
                "inferref_run_engine",
                {
                    "testcase": str(workspace / "testcase"),
                    "adapter": str(workspace / "adapter.json"),
                    "runs_root": str(tmp_path / "runs"),
                },
            )
            second_payload = second.structured_content
            assert second_payload["status"] == benchmark["success"]["status"]
            assert second_payload["data"]["comparison"]["status"] == "pass"
            assert first_payload["data"]["output"] != second_payload["data"]["output"]

    asyncio.run(exercise())
    assert _protected_digest(workspace) == before
