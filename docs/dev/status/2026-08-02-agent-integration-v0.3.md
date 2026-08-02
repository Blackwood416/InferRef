# Status: Agent integration vertical slice

**Date:** 2026-08-02

**Version:** `0.3.0`

**Baseline:** commit `6f6f5d4` (testcase and effect hardening)

**Verified environment:** Windows, Python 3.13.11, PyTorch 2.13.0 CPU,
Transformers 5.14.1, MCP Python SDK 2.0.0

## Outcome

InferRef can now drive one complete coding-agent iteration through a stable,
framework-neutral API:

```text
discover capabilities
  -> inspect trace/testcase context
  -> extract an operator or semantic region
  -> execute a trusted engine adapter
  -> receive structured execution + comparison diagnostics
  -> rerun in a fresh directory
```

The same operations are available as Python functions, `inferref agent ...
--json`, and optional MCP stdio tools. MCP contains no independent business
logic, so a transport change cannot silently alter testcase or comparison
semantics.

## Implemented contracts

### Agent response v0.1

Every operation returns a self-describing envelope with:

- protocol format/version;
- operation and one of `ok`, `pass`, `fail`, or `error`;
- structured result data;
- diagnostics with stable codes;
- recommended next actions.

Numerical mismatch is `fail`; invocation, timeout, and output-protocol failures
are `error`. This distinction is load-bearing for an Agent deciding whether to
edit a kernel or repair its test command.

### Engine adapter v0.1

An adapter is a small JSON file containing a process argv array, optional working
directory/environment, timeout, and output-log bound. InferRef:

- validates the adapter before creating a run directory;
- expands only declared placeholders;
- invokes the process with `shell=False`;
- assigns every invocation a UTC/UUID output directory;
- bounds stdout and stderr persisted into the run record;
- records configured environment names but redacts their values;
- compares successful engine output immediately;
- persists command, cwd, duration, exit status, logs, and comparison in
  `inferref-run.json`.

Fresh output directories close an important false-positive path: a failed engine
cannot accidentally pass by leaving an older tensor at a reused output path.

### MCP v2 transport

The optional `agent` extra installs the official MCP Python SDK v2 and adds the
`inferref-mcp` stdio entry point. It exposes capability discovery, artifact
context, testcase extraction, comparison, and adapter execution. In-memory MCP
client/server testing verifies structured-content round trips.

MCP remains optional. Importing InferRef or using any Agent operation through
Python/CLI does not require it.

## CLI surface

```text
inferref agent capabilities
inferref agent context
inferref agent extract
inferref agent compare
inferref agent run
```

The example NumPy RoPE engine now has a checked-in adapter at
`examples/engine_sim/rope_numpy.adapter.json`.

## Dependency and CI boundaries

`pyproject.toml` keeps numpy as the only core runtime dependency and defines MCP
as `inferref[agent]`. CI adds a dedicated Agent job which installs Core + MCP,
asserts that PyTorch is absent, and runs both test groups. This is separate from
the PyTorch compatibility matrix for the same reason as the existing Core job:
frontend churn must not break an Agent reading an existing artifact.

The source-distribution target now explicitly excludes local virtual environments,
caches, scratch artifacts, and C++ build output. This was required after package
verification found Hatchling scanning a 1.1 GB auxiliary virtual environment.

## Verification

```text
Core + MCP without PyTorch                  189 passed
targeted Agent + MCP after hardening         11 passed
full local suite                            325 passed, 1 skipped
Mini-Llama Agent adapter loop               extract PASS, run PASS, compare PASS
sdist + wheel                               built; wheel contains Agent/MCP modules
```

The full loop traced 206 operators, detected 30 semantic regions, extracted
`RoPE@layers.0.self_attn`, executed the NumPy adapter, compared both outputs, and
wrote an `inferref-run.json` record in the fresh run directory.

## Security and correctness boundary

Engine adapters are executable configuration. Avoiding a shell prevents command
string interpretation, but it does not make the named process safe. InferRef
0.3.0 therefore requires the host/user to approve adapter files and documents
that trust boundary in the protocol specification and MCP instructions.

The current server is local stdio only. It does not expose a network listener,
authenticate remote users, sandbox child processes, restrict filesystem access,
or maintain a host-side adapter allowlist.

## Remaining gaps

- No adapter discovery/registration mechanism exists; callers pass a trusted
  JSON path explicitly.
- There is no cancellation or streaming progress for long engine builds/runs.
- Child-process resource control is limited to wall-clock timeout. Persisted text
  is bounded, but capture memory, CPU, GPU, and descendant processes are not.
- MCP tools operate on host paths with the host process's permissions. A future
  remote transport must add workspace roots and authentication before use.
- The adapter contract runs already-built engine commands. Build orchestration
  and structured compiler diagnostics are not yet modelled.
- Trace/testcase inspection is compact JSON only; the planned viewer remains
  unimplemented.
- CUDA/vLLM execution remains unverified on this CPU-only machine.
- An end-to-end real Agent autonomously editing an engine kernel has not yet
  been evaluated; current tests validate the callable loop and its failure modes.

## Recommended next 0.3 slice

1. Add a workspace-level adapter registry with explicit trust/enable state.
2. Model build and run as separate bounded phases with structured diagnostics.
3. Add cancellation and progress events for long-running adapters.
4. Build an Agent evaluation fixture containing a deliberately broken engine,
   success criteria, and iteration budget.
5. Add a small trace/testcase viewer over the same service records rather than
   creating another interpretation layer.
