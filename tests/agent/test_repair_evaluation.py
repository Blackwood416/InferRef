"""Blind Agent evaluation host, oracle-isolation and MCP proxy tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from mcp import Client

from inferref.agent import evaluation_host
from inferref.agent.evaluation import (
    CANDIDATE_URI,
    RUNS_URI,
    VISIBLE_URI,
    EvaluationBenchmark,
    EvaluationSession,
    execute_case,
    load_audit,
    prepare_workspace,
    workspace_hashes,
)
from inferref.agent.evaluation_host import (
    AgentProcessResult,
    assess_candidate,
    evaluate_benchmark,
    parse_agent_usage,
)
from inferref.agent.evaluation_mcp import create_evaluation_server

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "examples" / "agent_eval" / "rope_sign" / "benchmark.json"
WRONG = "return np.concatenate((second, -first), axis=-1)"
FIXED = "return np.concatenate((-second, first), axis=-1)"


def _benchmark() -> EvaluationBenchmark:
    return EvaluationBenchmark.load(BENCHMARK_PATH)


def _process(
    root: Path,
    *,
    exit_code: int = 0,
    timed_out: bool = False,
    transcript_error: str | None = None,
) -> AgentProcessResult:
    stdout = root / "stdout.jsonl"
    stderr = root / "stderr.log"
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    return AgentProcessResult(
        command=("fake-agent",),
        exit_code=exit_code,
        timed_out=timed_out,
        duration_ms=1.0,
        stdout_path=stdout,
        stderr_path=stderr,
        stdout="",
        stderr="",
        usage={},
        transcript_error=transcript_error,
    )


def _exercise_required_tools(session: EvaluationSession, *, repair: bool) -> None:
    response = session.capabilities()
    session.audit("inferref_capabilities", response)
    response = session.context(VISIBLE_URI)
    session.audit("inferref_context", response)
    first = session.run_visible(VISIBLE_URI, CANDIDATE_URI, RUNS_URI)
    session.audit("inferref_run_engine", first)
    if repair:
        engine = session.workspace / "engine.py"
        source = engine.read_text(encoding="utf-8")
        engine.write_text(source.replace(WRONG, FIXED), encoding="utf-8")
        second = session.run_visible(VISIBLE_URI, CANDIDATE_URI, RUNS_URI)
        session.audit("inferref_run_engine", second)


def _fake_repair_driver(
    agent: str,
    model: str,
    benchmark: EvaluationBenchmark,
    workspace: Path,
    session_root: Path,
    audit_path: Path,
) -> AgentProcessResult:
    del agent, model
    session = EvaluationSession(benchmark, workspace, session_root, audit_path)
    _exercise_required_tools(session, repair=True)
    return _process(session_root.parent)


def test_v02_contract_and_workspace_hide_host_assets(tmp_path: Path) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")

    assert benchmark.max_engine_runs == 4
    assert benchmark.drivers["codex"].model == "gpt-5.6-sol"
    assert benchmark.drivers["claude"].model == "opus"
    assert {path.name for path in workspace.iterdir()} == {"engine.py", "TASK.md"}
    assert "reference" not in " ".join(path.as_posix() for path in workspace.rglob("*"))


def test_candidate_input_view_never_materializes_reference_payloads(
    tmp_path: Path,
) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    outcome = execute_case(
        benchmark.visible_case,
        engine=workspace / "engine.py",
        runs_root=tmp_path / "runs",
    )

    input_view = Path(outcome["input_view"])
    manifest = json.loads((input_view / "testcase.json").read_text(encoding="utf-8"))
    assert manifest["oracle"] == "not_materialized"
    assert not (input_view / "reference").exists()
    assert all("payload" not in entry for entry in manifest["outputs"])
    assert outcome["status"] == "mismatch"


def test_candidate_cannot_modify_input_only_staging(tmp_path: Path) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    engine = workspace / "engine.py"
    source = engine.read_text(encoding="utf-8").replace(WRONG, FIXED)
    source = source.replace(
        "results = {",
        '(testcase / "testcase.json").write_text("tampered", encoding="utf-8")\n'
        "    results = {",
    )
    engine.write_text(source, encoding="utf-8")

    outcome = execute_case(
        benchmark.visible_case,
        engine=engine,
        runs_root=tmp_path / "runs",
    )

    assert outcome["status"] == "integrity_failure"
    assert outcome["integrity"]["input_changes"] == ["testcase.json"]


def test_evaluation_mcp_enforces_opaque_uris_and_run_budget(tmp_path: Path) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    session = EvaluationSession(
        benchmark, workspace, tmp_path / "session", tmp_path / "audit.jsonl"
    )

    async def exercise() -> None:
        async with Client(create_evaluation_server(session)) as client:
            capabilities = await client.call_tool("inferref_capabilities", {})
            assert capabilities.structured_content["status"] == "ok"
            assert (
                capabilities.structured_content["data"]["evaluation"][
                    "visible_testcase"
                ]
                == VISIBLE_URI
            )
            out_of_order = await client.call_tool(
                "inferref_run_engine",
                {
                    "testcase": VISIBLE_URI,
                    "adapter": CANDIDATE_URI,
                    "runs_root": RUNS_URI,
                },
            )
            assert (
                out_of_order.structured_content["diagnostics"][0]["code"]
                == "required_tool_sequence"
            )
            denied = await client.call_tool("inferref_context", {"path": "C:/oracle"})
            assert (
                denied.structured_content["diagnostics"][0]["code"]
                == "path_not_allowed"
            )
            context = await client.call_tool("inferref_context", {"path": VISIBLE_URI})
            assert context.structured_content["data"]["oracle"] == "host_memory"
            for _ in range(benchmark.max_engine_runs):
                await client.call_tool(
                    "inferref_run_engine",
                    {
                        "testcase": VISIBLE_URI,
                        "adapter": CANDIDATE_URI,
                        "runs_root": RUNS_URI,
                    },
                )
            exhausted = await client.call_tool(
                "inferref_run_engine",
                {
                    "testcase": VISIBLE_URI,
                    "adapter": CANDIDATE_URI,
                    "runs_root": RUNS_URI,
                },
            )
            assert exhausted.structured_content["status"] == "error"
            assert (
                exhausted.structured_content["diagnostics"][0]["code"]
                == "budget_exhausted"
            )

    asyncio.run(exercise())
    records = load_audit(tmp_path / "audit.jsonl")
    assert records[-1]["diagnostic_codes"] == ["budget_exhausted"]


def test_fake_dual_agent_evaluation_passes_visible_and_holdouts(tmp_path: Path) -> None:
    report = evaluate_benchmark(
        BENCHMARK_PATH,
        agents=("codex", "claude"),
        report_dir=tmp_path / "report",
        driver=_fake_repair_driver,
    )

    assert report["status"] == "pass"
    assert report["acceptance"]["passed"] == 2
    for agent in report["agents"]:
        assert agent["classification"] == "pass"
        assert agent["visible"] == {"runs": 2, "passed": True}
        assert len(agent["holdouts"]) == 3
        assert all(case["status"] == "pass" for case in agent["holdouts"])


def test_visible_only_patch_is_classified_as_overfit(tmp_path: Path) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    engine = workspace / "engine.py"
    source = engine.read_text(encoding="utf-8")
    replacement = (
        "if value.shape[-1] == 8:\n"
        "        return np.concatenate((-second, first), axis=-1)\n"
        "    return np.concatenate((second, -first), axis=-1)"
    )
    engine.write_text(source.replace(WRONG, replacement), encoding="utf-8")
    initial = workspace_hashes(workspace)
    # The modified engine is the permitted baseline for this focused classifier test.
    initial["engine.py"] = "baseline-placeholder"
    session = EvaluationSession(
        benchmark, workspace, tmp_path / "session", tmp_path / "audit.jsonl"
    )
    _exercise_required_tools(session, repair=False)
    result = assess_candidate(
        benchmark,
        agent="fake",
        model="fake",
        workspace=workspace,
        agent_root=tmp_path / "agent",
        process=_process(tmp_path),
        initial_hashes=initial,
        final_hashes=workspace_hashes(workspace),
        audit=load_audit(tmp_path / "audit.jsonl"),
    )

    assert result["visible"]["passed"] is True
    assert result["classification"] == "overfit_failure"
    assert any(case["status"] != "pass" for case in result["holdouts"])


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("missing_tools", "integrity_failure"),
        ("protected_change", "integrity_failure"),
        ("adapter_change", "integrity_failure"),
        ("timeout", "infrastructure_failure"),
        ("malformed_output", "infrastructure_failure"),
        ("malformed_audit", "infrastructure_failure"),
    ],
)
def test_failure_classification_is_structured(
    tmp_path: Path, mode: str, expected: str
) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    initial = workspace_hashes(workspace)
    audit_path = tmp_path / "audit.jsonl"
    if mode == "missing_tools":
        session = EvaluationSession(
            benchmark, workspace, tmp_path / "session", audit_path
        )
        response = session.capabilities()
        session.audit("inferref_capabilities", response)
    elif mode in {"protected_change", "adapter_change"}:
        session = EvaluationSession(
            benchmark, workspace, tmp_path / "session", audit_path
        )
        _exercise_required_tools(session, repair=True)
        if mode == "protected_change":
            (workspace / "TASK.md").write_text("tampered", encoding="utf-8")
    elif mode == "malformed_audit":
        audit_path.write_text("not-json\n", encoding="utf-8")
    process = _process(
        tmp_path,
        timed_out=mode == "timeout",
        transcript_error=(
            "Agent JSON event stream is malformed at line 1"
            if mode == "malformed_output"
            else None
        ),
    )
    result = assess_candidate(
        benchmark,
        agent="fake",
        model="fake",
        workspace=workspace,
        agent_root=tmp_path / "agent",
        process=process,
        initial_hashes=initial,
        final_hashes=workspace_hashes(workspace),
        audit=load_audit(audit_path),
        initial_host_hashes={"host/adapter.py": "before"},
        final_host_hashes={
            "host/adapter.py": "after" if mode == "adapter_change" else "before"
        },
    )

    assert result["classification"] == expected
    if mode == "protected_change":
        assert result["integrity"]["protected_file_hashes"]["TASK.md"]["before"]
        assert result["integrity"]["protected_file_hashes"]["TASK.md"]["after"]
    if mode == "adapter_change":
        assert result["integrity"]["host_protected_changes"] == ["host/adapter.py"]


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 7}},
        {
            "type": "result",
            "total_cost_usd": 0.125,
            "usage": {"input_tokens": 20, "output_tokens": 9},
        },
    ],
)
def test_codex_and_claude_json_usage_fixtures(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    usage = parse_agent_usage(path)

    assert usage["usage.input_tokens"] >= 12
    assert usage["usage.output_tokens"] >= 7


def test_claude_driver_exposes_only_permitted_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _benchmark()
    monkeypatch.setattr(evaluation_host, "_agent_executable", lambda agent: [agent])
    (tmp_path / "session").mkdir()
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")

    command = evaluation_host._agent_command(
        "claude",
        "opus",
        benchmark,
        tmp_path / "workspace",
        tmp_path / "session",
        tmp_path / "audit.jsonl",
        claude_settings=settings,
        claude_model_override="deepseek-v4-flash",
    )

    tools = command[command.index("--tools") + 1].split(",")
    assert tools == [
        "Read",
        "Edit",
        "mcp__inferref__inferref_capabilities",
        "mcp__inferref__inferref_context",
        "mcp__inferref__inferref_run_engine",
    ]
    assert command[command.index("--allowedTools") + 1].split(",") == tools
    assert command[command.index("--settings") + 1] == str(settings.resolve())
    assert command[command.index("--model") + 1] == "deepseek-v4-flash"


def test_claude_resolved_model_is_read_from_init_event(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "model": "deepseek-v4-flash[1m]",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert evaluation_host._agent_model_from_events(events) == "deepseek-v4-flash[1m]"


def test_agent_api_failure_is_summarized_from_result_event(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "type": "result",
                "api_error_status": 429,
                "result": "Request rejected: rate limit",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert evaluation_host._agent_api_failure(events) == (
        "Agent API failed with HTTP 429: Request rejected: rate limit"
    )


def test_malformed_agent_json_event_is_rejected(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text('{"type":"turn.started"}\nnot-json\n', encoding="utf-8")

    assert evaluation_host._validate_event_stream(events) == (
        "Agent JSON event stream is malformed at line 2"
    )
