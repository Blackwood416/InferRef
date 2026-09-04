---
name: inferref
description: "Validate and debug inference engines with InferRef: alignment orchestrator sessions, trace inspection, testcase extraction, trusted engine-adapter execution, numerical comparison, first-divergence debugging, stateful scenario runs, and suite validation. Use when a coding agent must validate an engine or kernel against InferRef artifacts, align an engine through an alignment session, isolate a failing region from a trace, write or run an engine adapter, interpret an InferRef agent response envelope, or run a prefill/decode scenario."
---

# InferRef Engine Validation

## Standard loop

Alignment-first workflow (SPEC §14.4):

```text
capabilities -> align init / resume -> align plan -> align run -> [narrow / ad-hoc note] -> session verified
```

For legacy standalone extraction and scenario chains:

```text
discover -> context -> extract -> run / scenario -> compare -> fix -> rerun
```

## Choose the operation

| Task | Operation | Notes |
| --- | --- | --- |
| Inventory formats, operations, and MCP tools | capabilities | call once before relying on optional features |
| Initialize or resume an alignment session | alignment_init / alignment_status | required first step for numerical debugging (SPEC §14.4) |
| Inspect alignment candidate frontier | alignment_plan | plans in-scope boundaries in topological order |
| Execute alignment boundary run | alignment_run | executes candidate boundary against explicit adapter |
| Narrow causal divergence subtree | alignment_narrow | narrows to child boundaries on numerical failure |
| Record missing capability or escape-hatch | alignment_note | must record `missing-observation` when using custom tools |
| Summarise or validate one trace/testcase | context | read `data.analysis`, `data.regions`, `data.validation` |
| Project one operator or region into a testcase | extract_testcase | pass exactly one of `--op` / `--region` |
| Compare an existing engine output with a testcase | compare_outputs | use when the output directory already exists |
| Execute a trusted adapter and compare | run_engine | always creates a fresh run directory |
| Execute a stateful chain of testcases | run_scenario | `reference` state by default; `engine` state opt-in |

## Read the references

| File | When to read |
| --- | --- |
| [references/cli.md](references/cli.md) | need the exact CLI subcommand, flag, or MCP tool name |
| [references/adapter.md](references/adapter.md) | create, edit, or debug an adapter JSON |
| [references/protocol.md](references/protocol.md) | interpret an envelope, status, diagnostics, or `next_actions` |
| [references/workflows.md](references/workflows.md) | follow a full workflow with commands |

## Procedure

### 1. Discover capabilities

Run `inferref agent capabilities --json` (or MCP `inferref_capabilities`) and read
the returned formats, operations, and MCP tool names before relying on optional
features.

### 2. Alignment-first session (SPEC §14.4)

Always begin numerical engine debugging by initializing or resuming an alignment session:
- Initialize: `inferref align init <trace> <adapter> --dir <session-dir> --json` (or MCP `inferref_alignment_init`).
- Check status: `inferref align status <session-dir> --json` (or MCP `inferref_alignment_status`).
Numerical alignment debugging must start with alignment session state before using any ad-hoc tooling.

### 3. Plan alignment frontier

Run `inferref align plan <session-dir> --json` (or MCP `inferref_alignment_plan`).
Inspect the candidate boundaries recommended in topological order along the in-scope frontier.

### 4. Run boundary alignment

Run `inferref align run <session-dir> --boundary <id> --json`
(or MCP `inferref_alignment_run`).
If an external testcase is supplied, pass `--testcase <path>` and explicitly specify `--fixture-kind <kind>`.

### 5. Narrow divergence and diagnose

When a boundary run fails:
- Run `inferref align narrow <session-dir> --boundary <id> --json` (or MCP `inferref_alignment_narrow`) to identify the earliest failing causal descendant.
- Inspect diagnostic classification candidates (e.g. implementation, layout, model_artifact, quantization).

### 6. Ad-hoc tooling escape hatch

Whenever custom numerical inspection, ad-hoc scripts, or external debuggers are used outside InferRef, the agent must record the missing capability:
- Run `inferref align note <session-dir> <boundary> missing-observation "<reason>" --duration-minutes <minutes> --json` (or MCP `inferref_alignment_note`).
This records the escape-hatch reason and tracks debugging metrics without compromising session integrity.

### 7. Standalone extraction, runs, and scenarios

For standalone workflows or multi-step chains:
- Extract testcase: `inferref agent extract <trace> --op <id> --output <dir> --json` or `--region <name>`.
- Run engine adapter: `inferref agent run <testcase> --adapter <adapter.json> --runs-dir <runs> --json`.
- Compare output: `inferref agent compare <testcase> <engine-output> --json`.
- Run scenario: `inferref agent run_scenario <scenario-dir> --adapter <adapter.json> --runs-dir <runs> --json` (validate first: `inferref scenario validate <scenario-dir> --json`).

### 8. Interpret the envelope and iterate

- `ok` means discovery/inspection succeeded; `pass` means validation, extraction, or boundary verification passed; `fail` indicates numerical mismatch; `error` indicates process or protocol failure.
- Iterate fixes and rerun candidate boundaries until all in-scope frontier boundaries reach `verified` state, emitting `session_verified`.
