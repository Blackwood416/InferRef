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

`compare_outputs` and `run_engine` share the same standalone testcase validator.
Execution requires a structurally valid and computed-reproducible testcase; a
manifest's `reproducible: true` is not trusted by itself. Validation checks unique
boundary names, contained and existing payloads, header metadata agreement,
node/value/effect references, and declared non-portable or unobservable state.
Tensor validation reads only the `.irtensor` header and file size, not the full
payload.

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
  "max_output_chars": 65536,
  "max_artifact_bytes": 1073741824,
  "max_artifact_files": 10000
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

Each run writes `inferref-run.json` beside the engine outputs. Stdout and stderr
are streamed to separate files with a hard per-stream byte limit. Crossing that
limit, the wall-clock timeout, the monitored artifact-size limit, or the artifact
file/entry limit terminates the process tree and produces `output_limit`,
`timeout`, `artifact_limit`, or `artifact_file_limit`.
Windows uses a kill-on-close Job Object; POSIX uses a dedicated session/process
group. Descendants are cleaned up after normal parent exit as well as failures.
Artifact traversal runs less frequently than process/deadline polling, checks the
deadline while walking, and has an entry budget so a directory-only file storm
cannot turn one scan into an unbounded operation. The record contains the expanded
argv, working directory, exit code, bounded persisted stdout/stderr, artifact
bytes/files/scan entries, duration, adapter metadata, comparison, and final run
status. Configured environment names
are recorded but their values are redacted because they may contain credentials.
InferRef never reuses a run directory, preventing stale engine output from
producing a false pass. Secrets should be passed through environment variables,
not command arguments, because the expanded argv is intentionally recorded.

### Trust boundary

An adapter names a process to execute and is therefore trusted code even though no
shell is involved. Hosts must only expose adapters approved by the user or engine
workspace. The adapter runner is timeout/output-controlled, not a general-purpose
sandbox: it bounds stream persistence, monitors run artifact bytes and entry
counts, and cleans up the
spawned process tree, but it does not impose CPU, memory, GPU, syscall, network,
or filesystem-access limits. Artifact monitoring can detect and terminate growth;
only an OS/container quota can guarantee that a hostile process never transiently
exceeds the configured byte limit.

Root containment is accidental path-escape protection, not a handle-based
security boundary. The current implementation rejects absolute paths, lexical
parents, and resolved paths outside configured roots, but check and use are
separate filesystem operations. A same-user adversary may race a directory
symlink, junction, or reparse point. Strong isolation requires directory-handle
relative access (`openat2`/`O_NOFOLLOW` on Linux and final-handle/reparse checks on
Windows) or an OS sandbox.

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

The host configures repeatable `--read-root` and `--write-root` arguments when
starting the server. Every trace, testcase, adapter, engine-output, extraction,
and run path is resolved against those roots before an operation begins. Payload
paths inside trace, testcase, and engine manifests must be relative and remain
contained after symlink/junction resolution.

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

## 8. Agent evaluation fixture v0.2

Evaluation is a host-side layer over Agent protocol v0.1. It deliberately does
not add production MCP operations. A blind repair benchmark declares:

```json
{
  "format": "inferref-agent-evaluation",
  "format_version": "0.2",
  "id": "rope-half-rotation-sign",
  "task": "TASK.md",
  "workspace_template": ["engine.py", "TASK.md"],
  "candidate": {
    "editable_paths": ["engine.py"],
    "required_tools": [
      "inferref_capabilities",
      "inferref_context",
      "inferref_run_engine"
    ],
    "max_engine_runs": 4,
    "max_wall_seconds": 900
  },
  "cases": {
    "visible": {"id": "visible", "seed": 20260802, "shape": [1, 2, 4, 8]},
    "holdouts": [
      {"id": "dim4", "seed": 20260811, "shape": [1, 2, 3, 4]}
    ]
  },
  "drivers": {
    "codex": {"model": "gpt-5.6-sol"},
    "claude": {"model": "opus"}
  },
  "success": {
    "required_agent_passes": 2,
    "visible_status": "pass",
    "all_holdouts_pass": true,
    "protected_paths_unchanged": true
  }
}
```

The candidate sees opaque `eval://` identifiers through tools with the ordinary
InferRef MCP names and response envelope. Reference arrays are not materialized;
only input payloads are staged for the candidate engine. The MCP proxy records
required tool use and rejects calls beyond `max_engine_runs`. After visible PASS,
the historical MCP result is used only as evidence that the Agent completed the
required interaction. After the Agent exits, the host silently executes the final
candidate against the visible case and every holdout. All acceptance results must
therefore describe one final `engine.py`.

The evaluator snapshots the workspace, allows changes only to `editable_paths`,
and distinguishes infrastructure, Agent, integrity, and overfit failures. The
`success` object is required, parsed, validated, and controls the overall pass
threshold plus final visible, holdout, and protected-path requirements.

Evaluation “blindness” is specifically candidate-input blindness, in-memory oracle
isolation, and holdouts hidden from the Agent workspace. Holdout metadata remains
public, evaluator and Agent share an OS user, and final-state hashes cannot detect
reads or modify-then-restore behavior. This is not an adversarial secrecy or OS
sandbox claim.

The evaluation MCP audit is a fail-closed JSONL evidence stream. Tool-call records
have one continuous global `call_index`; tool/operation/status combinations and
exact `engine_runs` transitions are checked as an explicit evaluation state
machine. A terminal `session_footer` binds the record count and SHA-256 of all
preceding record bytes.
Malformed JSON, invalid record schema, a missing footer, or a count/digest mismatch
is an `infrastructure_failure`; records are never silently skipped.
The host atomically replaces the complete stream, including its footer, after
every tool call. This makes the evidence crash-consistent even when an Agent CLI
terminates the MCP child without allowing process-shutdown handlers to run. A
zero-call probe does not reserve or create the audit path.
The cumulative digest detects corruption, truncation, and torn writes. It is not
a keyed authenticator and does not resist a same-user process able to rewrite the
entire evidence stream.

The evaluator may emit a redacted `inferref-agent-evaluation-attestation` v0.5.
Formal publication is delegated to a fresh isolated Python worker (`python -I`);
ordinary in-process API and custom/test-driver runs can only produce
`attestation_level: development` evidence. The worker requires a clean Git
worktree before and after the run and unchanged repository, evaluator, benchmark,
runtime, Agent executable, model-identity, and external-file evidence.

Each Agent executable chain is resolved once. The same frozen command prefix is
used for version detection and execution, and every component is hashed before
version detection, after version detection, and after execution. Public evidence
contains component roles, basenames, sizes, SHA-256 values, an instance argv
digest, a path-free normalized argv policy with a verifiable digest, version
output, and the aggregate unchanged verdict, but no absolute executable paths.
For Windows Codex the runtime (`node.exe`) and CLI entry (`codex.js`) are
separate components. POSIX shebang resolution honors `env -S` and direct
interpreter arguments (`#!/usr/bin/python3 -s`), keeping interpreter flags as
launch arguments while hashing only executable file components.

Model identity is explicit evidence rather than a single resolved-model string:
`requested_only` means only the evaluator request is known;
`cli_self_reported` means an Agent-specific JSON event reported the model. The
record includes request, report, source and match verdict. Neither level is
provider verification, and a missing Codex model event does not silently become a
confirmed identity. Evidence validity and request satisfaction are separate
verdicts: a self-reported model that does not match the request makes the
candidate an `identity_policy_failure`, fails the benchmark, and is refused by
formal attestation even if the numerical results pass.

The isolated worker records a canonical launch policy (`-I -m`, request
transport, request schema version, SHA-256 of the exact request bytes, and the
worker Python executable hash) with a recomputable `launch_policy_sha256`. The
digest never includes temporary request paths or other instance randomness, and
the parent process independently hashes the request bytes it wrote and rejects a
worker report whose evidence does not match.

When Claude Code runs with `--claude-settings`, the settings file is captured
no-follow before and after the Agent process, using the same regular-file
semantics as workspace hashing (symlink/reparse points are rejected). The
published record contains `present`, `kind`, `size`, `sha256`, `after_sha256`,
`after_kind`, `after_size`, `after_file_id`, and `unchanged`; absolute paths and
contents are never published. The verdict requires the same file identity and
bytes before and after, so replacement or delete-and-recreate with identical
content is still detected. A Claude run without settings records an explicit
`present: false`. Both per-agent runner evidence and benchmark-level
`external_files` are validated, and an unchanged verdict is a formal-attestation
requirement.

The benchmark digest is fixed from the exact bytes parsed at load time rather
than reread during publication. Runtime and installed-distribution evidence is
captured before and after and carries an unchanged verdict. The attestation also
records the complete imported `inferref` source-tree manifest/hash, sealed tool
audit, final patch, final visible/holdout verdicts, raw transcript hashes, and
available usage. `report_json_sha256` hashes exactly `report.json`, not the whole
private report directory. The attestation excludes transcript text, reasoning,
credentials, reference payloads, and absolute local paths.
