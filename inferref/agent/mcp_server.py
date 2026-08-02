"""Optional MCP stdio transport for InferRef's framework-neutral Agent API."""

from __future__ import annotations

import sys
from typing import Any

from inferref.agent.service import (
    capabilities,
    compare_outputs,
    context,
    extract_testcase,
    run_engine,
)
from inferref.ir.version import INFERREF_VERSION


def create_server() -> Any:
    """Create an MCP v2 server without making MCP a core dependency."""

    try:
        from mcp.server import MCPServer
    except ImportError as exc:  # pragma: no cover - exercised by subprocess test
        raise RuntimeError(
            "MCP support is not installed; run: pip install 'inferref[agent]'"
        ) from exc

    server = MCPServer(
        "inferref",
        title="InferRef",
        description="Trace inspection, testcase extraction, and engine comparison",
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

        return capabilities().to_dict()

    @server.tool()
    def inferref_context(path: str) -> dict[str, Any]:
        """Summarise and validate an InferRef trace or standalone testcase."""

        return context(path).to_dict()

    @server.tool()
    def inferref_extract_testcase(
        trace: str,
        output: str,
        region: str | None = None,
        op_id: int | None = None,
        name: str | None = None,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Extract one operator or region as a portable engine testcase."""

        return extract_testcase(
            trace,
            output,
            region=region,
            op_id=op_id,
            name=name,
            input_names=input_names,
            output_names=output_names,
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

        return compare_outputs(
            testcase,
            engine_output,
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

        return run_engine(
            testcase,
            adapter,
            runs_root,
            atol=atol,
            rtol=rtol,
            ignore_stride=ignore_stride,
            strict_layout=strict_layout,
            first_failure=first_failure,
        ).to_dict()

    return server


def main() -> int:
    """Run the local stdio server used by MCP hosts."""

    try:
        server = create_server()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
