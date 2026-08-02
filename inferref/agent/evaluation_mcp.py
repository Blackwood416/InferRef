"""Evaluation-only MCP proxy with opaque URIs and host-held oracle data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from inferref.agent.evaluation import EvaluationBenchmark, EvaluationSession
from inferref.ir.version import INFERREF_VERSION


def create_evaluation_server(session: EvaluationSession) -> Any:
    try:
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "MCP support is not installed; run: pip install 'inferref[agent]'"
        ) from exc

    server = MCPServer(
        "inferref",
        title="InferRef Agent Evaluation",
        description="Blind repair evaluation with a host-side oracle",
        instructions=(
            "Call inferref_capabilities, then inferref_context with eval://visible. "
            "Use only the opaque URIs returned by those tools. Repair engine.py and "
            "finish only when inferref_run_engine returns status pass."
        ),
        version=INFERREF_VERSION,
    )

    @server.tool()
    def inferref_capabilities() -> dict[str, Any]:
        """Discover the evaluation contract and opaque testcase URIs."""

        response = session.capabilities()
        session.audit("inferref_capabilities", response)
        return response.to_dict()

    @server.tool()
    def inferref_context(path: str) -> dict[str, Any]:
        """Inspect the visible testcase without exposing reference payloads."""

        response = session.context(path)
        session.audit("inferref_context", response)
        return response.to_dict()

    @server.tool()
    def inferref_run_engine(
        testcase: str,
        adapter: str,
        runs_root: str,
        atol: float | None = None,
        rtol: float | None = None,
        ignore_stride: bool = False,
        strict_layout: bool = False,
        first_failure: bool = True,
    ) -> dict[str, Any]:
        """Run the candidate against the visible host-oracle testcase."""

        del atol, rtol, ignore_stride, strict_layout, first_failure
        response = session.run_visible(testcase, adapter, runs_root)
        session.audit("inferref_run_engine", response)
        return response.to_dict()

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--session-root", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args(argv)
    benchmark = EvaluationBenchmark.load(args.benchmark)
    session = EvaluationSession(
        benchmark=benchmark,
        workspace=Path(args.workspace),
        root=Path(args.session_root),
        audit_path=Path(args.audit),
    )
    create_evaluation_server(session).run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
