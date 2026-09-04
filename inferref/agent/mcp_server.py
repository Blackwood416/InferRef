"""Optional MCP stdio transport for InferRef's framework-neutral Agent API."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inferref.agent.path_policy import MCPPathPolicy
from inferref.agent.protocol import AgentResponse
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
    compare_outputs,
    context,
    extract_testcase,
    run_engine,
    run_scenario,
)
from inferref.ir.version import INFERREF_VERSION


def create_server(
    *,
    read_roots: Sequence[str | Path] | None = None,
    write_roots: Sequence[str | Path] | None = None,
) -> Any:
    """Create an MCP v2 server without making MCP a core dependency."""

    try:
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - exercised by subprocess test
        raise RuntimeError(
            "MCP support is not installed; run: pip install 'inferref[agent]'"
        ) from exc

    policy = MCPPathPolicy.create(read_roots=read_roots, write_roots=write_roots)
    server = MCPServer(
        "inferref",
        title="InferRef",
        description=(
            "Trace inspection, testcase extraction, engine comparison, and "
            "stateful scenario execution"
        ),
        instructions=(
            "Call inferref_capabilities first. Inspect an artifact with "
            "inferref_context before extraction or execution. Engine adapter files "
            "are executable configuration and must be trusted by the host user."
        ),
        version=INFERREF_VERSION,
    )

    @server.tool()
    def inferref_capabilities() -> dict[str, Any]:
        """Discover InferRef formats, operations, and recommended first action."""

        payload = capabilities().to_dict()
        payload["data"]["mcp_path_policy"] = policy.to_dict()
        return payload

    @server.tool()
    def inferref_context(path: str) -> dict[str, Any]:
        """Summarise and validate an InferRef trace or standalone testcase."""

        try:
            allowed = policy.read(path, kind="context artifact")
        except ValueError as exc:
            return AgentResponse.error(
                "context", str(exc), code="path_not_allowed"
            ).to_dict()
        return context(allowed).to_dict()

    @server.tool()
    def inferref_extract_testcase(
        trace: str,
        output: str,
        region: str | None = None,
        op_id: int | None = None,
        name: str | None = None,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        contracts: list[str] | None = None,
    ) -> dict[str, Any]:
        """Extract one operator or region as a portable engine testcase."""

        try:
            allowed_trace = policy.read(trace, kind="trace input")
            allowed_output = policy.write(output, kind="testcase output")
        except ValueError as exc:
            return AgentResponse.error(
                "extract_testcase", str(exc), code="path_not_allowed"
            ).to_dict()
        return extract_testcase(
            allowed_trace,
            allowed_output,
            region=region,
            op_id=op_id,
            name=name,
            input_names=input_names,
            output_names=output_names,
            contracts=contracts,
        ).to_dict()

    @server.tool()
    def inferref_compare_outputs(
        testcase: str,
        engine_output: str,
        comparator: str | None = None,
        comparison_config: dict[str, Any] | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ignore_stride: bool = False,
        strict_layout: bool = False,
        first_failure: bool = True,
    ) -> dict[str, Any]:
        """Compare engine tensors with a testcase and localise the first divergence."""

        try:
            allowed_testcase = policy.read(testcase, kind="testcase input")
            allowed_engine_output = policy.read(
                engine_output, kind="engine output input"
            )
        except ValueError as exc:
            return AgentResponse.error(
                "compare_outputs", str(exc), code="path_not_allowed"
            ).to_dict()
        return compare_outputs(
            allowed_testcase,
            allowed_engine_output,
            comparator=comparator,
            comparison_config=comparison_config,
            atol=atol,
            rtol=rtol,
            ignore_stride=ignore_stride,
            strict_layout=strict_layout,
            first_failure=first_failure,
        ).to_dict()

    @server.tool()
    def inferref_run_engine(
        testcase: str,
        adapter: str,
        runs_root: str,
        comparator: str | None = None,
        comparison_config: dict[str, Any] | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ignore_stride: bool = False,
        strict_layout: bool = False,
        first_failure: bool = True,
    ) -> dict[str, Any]:
        """Run a trusted adapter in a fresh directory and compare its output."""

        try:
            allowed_testcase = policy.read(testcase, kind="testcase input")
            allowed_adapter = policy.read(adapter, kind="engine adapter")
            allowed_runs = policy.write(runs_root, kind="engine runs output")
        except ValueError as exc:
            return AgentResponse.error(
                "run_engine", str(exc), code="path_not_allowed"
            ).to_dict()
        return run_engine(
            allowed_testcase,
            allowed_adapter,
            allowed_runs,
            comparator=comparator,
            comparison_config=comparison_config,
            atol=atol,
            rtol=rtol,
            ignore_stride=ignore_stride,
            strict_layout=strict_layout,
            first_failure=first_failure,
        ).to_dict()

    @server.tool()
    def inferref_run_scenario(
        scenario: str,
        adapter: str,
        runs_root: str,
        state_mode: str = "reference",
        compare_state: bool = False,
        comparator: str | None = None,
        comparison_config: dict[str, Any] | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ignore_stride: bool = False,
        strict_layout: bool = False,
        first_failure: bool = True,
    ) -> dict[str, Any]:
        """Run a stateful scenario chain through a trusted engine adapter."""

        try:
            allowed_scenario = policy.read(scenario, kind="scenario input")
            allowed_adapter = policy.read(adapter, kind="engine adapter")
            allowed_runs = policy.write(runs_root, kind="scenario runs output")
        except ValueError as exc:
            return AgentResponse.error(
                "run_scenario", str(exc), code="path_not_allowed"
            ).to_dict()
        return run_scenario(
            allowed_scenario,
            allowed_adapter,
            allowed_runs,
            state_mode=state_mode,
            compare_state=compare_state,
            comparator=comparator,
            comparison_config=comparison_config,
            atol=atol,
            rtol=rtol,
            ignore_stride=ignore_stride,
            strict_layout=strict_layout,
            first_failure=first_failure,
        ).to_dict()

    @server.tool()
    def inferref_alignment_init(
        trace: str,
        adapter: str,
        session_dir: str | None = None,
        name: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Initialize an InferRef alignment session."""
        try:
            allowed_trace = policy.read(trace, kind="trace")
            allowed_adapter = policy.read(adapter, kind="adapter")
            s_target = session_dir or (str(Path.cwd() / "alignment"))
            allowed_session = policy.write(s_target, kind="alignment session")
        except ValueError as exc:
            return AgentResponse.error("alignment_init", str(exc), code="path_not_allowed").to_dict()

        return alignment_init(
            allowed_trace,
            allowed_adapter,
            session_dir=allowed_session,
            name=name,
            force=force,
            access_policy=policy.read,
        ).to_dict()

    @server.tool()
    def inferref_alignment_plan(
        session: str | None = None,
        trace: str | None = None,
        adapter: str | None = None,
    ) -> dict[str, Any]:
        """Inspect alignment candidates and recommended next boundary."""
        try:
            allowed_session = policy.read(session, kind="alignment session") if session else None
            allowed_trace = policy.read(trace, kind="trace") if trace else None
            allowed_adapter = policy.read(adapter, kind="adapter") if adapter else None
        except ValueError as exc:
            return AgentResponse.error("alignment_plan", str(exc), code="path_not_allowed").to_dict()

        return alignment_plan(
            session_dir=allowed_session,
            trace_path=allowed_trace,
            adapter_path=allowed_adapter,
            access_policy=policy.read,
        ).to_dict()

    @server.tool()
    def inferref_alignment_run(
        session: str,
        adapter: str,
        limit: int = 1,
        boundaries: list[str] | None = None,
        continue_on_fail: bool = False,
        engine_revision: str | None = None,
        external_testcase: str | None = None,
        fixture_kind: str | None = None,
        fixture_tags: list[str] | None = None,
        comparator: str | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ignore_stride: bool = False,
        strict_layout: bool = False,
    ) -> dict[str, Any]:
        """Execute boundary alignment runs against an explicit engine adapter."""
        try:
            allowed_session = policy.write(session, kind="alignment session")
            allowed_adapter = policy.read(adapter, kind="adapter")
            allowed_external_tc = policy.read(external_testcase, kind="external testcase") if external_testcase else None
        except ValueError as exc:
            return AgentResponse.error("alignment_run", str(exc), code="path_not_allowed").to_dict()

        return alignment_run(
            allowed_session,
            allowed_adapter,
            limit=limit,
            boundary_ids=boundaries or (),
            continue_on_fail=continue_on_fail,
            engine_revision=engine_revision,
            external_testcase=allowed_external_tc,
            fixture_kind=fixture_kind,
            fixture_tags=fixture_tags or (),
            comparator=comparator,
            atol=atol,
            rtol=rtol,
            ignore_stride=ignore_stride,
            strict_layout=strict_layout,
            access_policy=policy.read,
        ).to_dict()

    @server.tool()
    def inferref_alignment_status(session: str) -> dict[str, Any]:
        """Inspect alignment session progress, boundary counts, and active failure."""
        try:
            allowed_session = policy.read(session, kind="alignment session")
        except ValueError as exc:
            return AgentResponse.error("alignment_status", str(exc), code="path_not_allowed").to_dict()

        return alignment_status(allowed_session, access_policy=policy.read).to_dict()

    @server.tool()
    def inferref_alignment_summary(session: str) -> dict[str, Any]:
        """Summarize alignment metrics, TFCM, and adoption ratio (compact Spec §11.3)."""
        try:
            allowed_session = policy.read(session, kind="alignment session")
        except ValueError as exc:
            return AgentResponse.error("alignment_summary", str(exc), code="path_not_allowed").to_dict()

        return alignment_summary(allowed_session, access_policy=policy.read).to_dict()

    @server.tool()
    def inferref_alignment_narrow(
        session: str,
        boundary: str | None = None,
    ) -> dict[str, Any]:
        """Compute smallest confirmed failing boundary and failure domain classification."""
        try:
            allowed_session = policy.read(session, kind="alignment session")
        except ValueError as exc:
            return AgentResponse.error("alignment_narrow", str(exc), code="path_not_allowed").to_dict()

        return alignment_narrow(allowed_session, boundary_id=boundary, access_policy=policy.read).to_dict()

    @server.tool()
    def inferref_alignment_note(
        session: str,
        boundary: str,
        category: str,
        text: str,
        duration_minutes: float | None = None,
        milestone: str | None = None,
        target: str | None = None,
        author: str = "agent",
    ) -> dict[str, Any]:
        """Record human/agent manual note evidence."""
        try:
            allowed_session = policy.write(session, kind="alignment session")
            allowed_author = author
        except ValueError as exc:
            return AgentResponse.error("alignment_note", str(exc), code="path_not_allowed").to_dict()

        return alignment_note(
            allowed_session,
            boundary_id=boundary,
            category=category,
            text=text,
            duration_minutes=duration_minutes,
            milestone=milestone,
            target=target,
            author=allowed_author,
            access_policy=policy.read,
        ).to_dict()

    @server.tool()
    def inferref_alignment_report(
        session: str,
        output: str | None = None,
        json_sidecar: str | None = None,
    ) -> dict[str, Any]:
        """Generate markdown and JSON sidecar alignment reports."""
        try:
            allowed_session = policy.write(session, kind="alignment session")
            allowed_output = policy.write(output, kind="report output") if output else None
            allowed_sidecar = policy.write(json_sidecar, kind="sidecar output") if json_sidecar else None
        except ValueError as exc:
            return AgentResponse.error("alignment_report", str(exc), code="path_not_allowed").to_dict()

        return alignment_report(
            allowed_session,
            output=allowed_output,
            json_sidecar=allowed_sidecar,
            access_policy=policy.read,
        ).to_dict()

    return server


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local stdio server used by MCP hosts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--read-root",
        action="append",
        help="allowed artifact/workspace read root (repeatable; default: cwd)",
    )
    parser.add_argument(
        "--write-root",
        action="append",
        help="allowed extraction/run write root (repeatable; default: read roots)",
    )
    args = parser.parse_args(argv)
    try:
        server = create_server(
            read_roots=args.read_root,
            write_roots=args.write_root,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
