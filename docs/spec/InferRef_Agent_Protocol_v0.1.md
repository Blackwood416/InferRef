# InferRef Agent Protocol v0.1

> **Status:** Draft implemented by InferRef 0.3.0  
> **Scope:** local coding-agent integration, engine invocation, and structured diagnostics

## 1. Design goal

An Agent should be able to move through this loop without parsing terminal prose or
reopening the complete model implementation on every iteration:

```text
discover -> inspect context -> extract testcase -> run engine -> compare -> fix -> rerun
```

The protocol is framework-neutral. Trace reading, testcase extraction, engine
execution, comparison, the CLI JSON API, and MCP transport do not import PyTorch.
MCP is an optional transport over the same Python service; it is not the source of
business logic.

## 2. Response envelope

Every operation returns one object:

```json
{
  "protocol": {
    "format": "inferref-agent-response",
    "version": "0.1"
  },
  "operation": "run_engine",
  "status": "pass",
  "data": {},
  "diagnostics": [],
  "next_actions": []
}
```

`status` has four meanings:

| Status | Meaning |
| --- | --- |
| `ok` | The inspection/discovery request completed. |
| `pass` | A validation, extraction, execution, or comparison succeeded. |
| `fail` | The request ran correctly and found a reproducibility or numerical failure. |
| `error` | The request or adapter could not be executed correctly. |

`fail` is a valid domain result, not a protocol error. An Agent should inspect
`data` and `next_actions`. `error` carries at least one structured diagnostic.

## 3. Operations

### `capabilities`

Returns the InferRef implementation version, supported wire formats, operation
metadata, and MCP tool names. Clients should call it before relying on optional
features.

### `context(path)`

Returns a compact trace or testcase summary, validation diagnostics, semantic
regions, payload/reproducibility state, and recommended next actions. It does not
return tensor payload bytes or the complete graph.

### `extract_testcase(trace, output, region | op_id, ...)`

Projects exactly one operator or region. A reproducible testcase returns `pass`
and recommends `run_engine`; missing boundary payloads return `fail` with stable
reason codes.

### `compare_outputs(testcase, engine_output, ...)`

Compares engine tensors with reference tensors and returns the ordinary InferRef
comparison report inside `data`. A numerical mismatch returns `fail`, not `error`.

### `run_engine(testcase, adapter, runs_root, ...)`

Loads a trusted adapter, creates a fresh output directory, executes it, and then
compares the result. Process/timeout/output-protocol failures return `error`;
numerical disagreement returns `fail`; agreement returns `pass`.

## 4. Engine adapter v0.1

An adapter is JSON executable configuration:

```json
{
  "format": "inferref-engine-adapter",
  "format_version": "0.1",
  "name": "rope-numpy",
  "command": [
    "{python}",
    "rope_numpy.py",
    "{testcase}",
    "--output",
    "{output}"
  ],
  "cwd": ".",
  "environment": {},
  "timeout_seconds": 60,
  "max_output_chars": 65536
}
```

`command` is a non-empty argv array. InferRef passes it directly to the operating
system with `shell=False`; shell syntax and string interpolation are not executed.
`cwd` is resolved relative to the adapter file. The following placeholders are
available in command arguments and environment values:

| Placeholder | Value |
| --- | --- |
| `{testcase}` | Absolute standalone testcase directory. |
| `{output}` | Absolute, fresh per-run engine output directory. |
| `{adapter_dir}` | Absolute directory containing the adapter JSON. |
| `{python}` | Absolute interpreter running InferRef. Useful for simulators only. |

The adapter must reference `{testcase}` and `{output}`. InferRef also supplies
`INFERREF_TESTCASE` and `INFERREF_OUTPUT` to the child environment. Literal braces
inside an argument must be escaped as `{{` and `}}`.

Each run writes `inferref-run.json` beside the engine outputs. The record contains
the expanded argv, working directory, exit code, bounded persisted stdout/stderr, duration,
adapter metadata, comparison, and final run status. Configured environment names
are recorded but their values are redacted because they may contain credentials.
InferRef never reuses a run directory, preventing stale engine output from
producing a false pass. Secrets should be passed through environment variables,
not command arguments, because the expanded argv is intentionally recorded.

### Trust boundary

An adapter names a process to execute and is therefore trusted code even though no
shell is involved. Hosts must only expose adapters approved by the user or engine
workspace. InferRef 0.3.0 does not sandbox child processes or restrict their file
access. The child output is captured in memory before the persisted record is
truncated; adapters that emit unbounded output are outside the current resource
model.

## 5. MCP transport

Install the optional transport with:

```bash
pip install "inferref[agent]"
```

`inferref-mcp` starts a local stdio MCP server and exposes:

- `inferref_capabilities`
- `inferref_context`
- `inferref_extract_testcase`
- `inferref_compare_outputs`
- `inferref_run_engine`

The server uses the official MCP Python SDK v2. Tool return values are the response
envelope above, so MCP and `inferref agent ... --json` have the same semantics.

## 6. Compatibility

Agent protocol, engine adapter, Trace IR, testcase, and tensor payload versions are
independent. A client must inspect each format/version pair rather than infer wire
compatibility from the InferRef package version.

Protocol v0.1 is additive: new fields may appear. Clients should ignore unknown
fields and branch on `protocol.version`, `operation`, and `status`.

## 7. 0.3 acceptance slice

The first 0.3 vertical slice is complete when all of the following hold:

1. Core Agent operations run with PyTorch absent.
2. CLI and MCP expose the same response envelope.
3. A trusted adapter can run a standalone testcase and receive PASS/FAIL diagnostics.
4. Process failure cannot be confused with numerical mismatch.
5. Repeated runs cannot consume stale output from an earlier invocation.
6. CI covers Core plus MCP without installing PyTorch.
