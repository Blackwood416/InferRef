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
            atol=atol,
            rtol=rtol,
            ignore_stride=ignore_stride,
            strict_layout=strict_layout,
            first_failure=first_failure,
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
