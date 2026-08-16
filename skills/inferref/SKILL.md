---
name: inferref
description: "Validate and debug inference engines with InferRef: trace inspection, testcase extraction, trusted engine-adapter execution, numerical comparison, first-divergence debugging, stateful scenario runs, and suite validation. Use when a coding agent must validate an engine or kernel against InferRef artifacts, isolate a failing region from a trace, write or run an engine adapter, interpret an InferRef agent response envelope, or run a prefill/decode scenario."
---

# InferRef Engine Validation

## Standard loop

```text
discover -> context -> extract -> run / scenario -> compare -> fix -> rerun
```

## Choose the operation

| Task | Operation | Notes |
| --- | --- | --- |
| Inventory formats, operations, and MCP tools | capabilities | call once before relying on optional features |
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

### 1. Discover

Run `inferref agent capabilities --json` (or MCP `inferref_capabilities`) and read
the returned formats, operations, and MCP tool names before relying on optional
features.

### 2. Inspect

Run `inferref agent context <artifact> --json` (or MCP `inferref_context`). For a
trace, read `data.analysis`, `data.regions`, and `data.validation`; follow
`next_actions` when present.

### 3. Extract

Run `inferref agent extract <trace> --op <id> --output <dir> --json` or
`--region <name>`. Name boundary inputs/outputs with `--input-names` and
`--output-names`, and pass `--contract <id>` when the operation has an executable
contract. Prefer a region over a single operator when the failing kernel is
fused or stateful.

### 4. Run or compare

- Run: `inferref agent run <testcase> --adapter <adapter.json> --runs-dir <runs> --json`.
- Compare existing output: `inferref agent compare <testcase> <engine-output> --json`.

Both paths share the same testcase validator. A manifest's `reproducible: true`
is not trusted until validation passes.

### 5. Run a scenario

Validate first: `inferref scenario validate <scenario-dir> --json`. Then run:
`inferref agent run_scenario <scenario-dir> --adapter <adapter.json> --runs-dir <runs> --json`.
Use `--state-mode engine` with `--compare-state` to feed engine state forward and
localise state divergence at the producing step.

### 6. Interpret the envelope

`ok` means discovery/inspection succeeded; `pass` means validation, extraction,
or execution succeeded; `fail` means a numerical mismatch or non-reproducible
artifact; `error` means an adapter, process, or protocol failure. Read `data` and
`next_actions` before deciding the next step. `fail` is a domain result, not a
crash.

### 7. Iterate

Localise the first divergence, fix the engine, and rerun the same command.
Never reuse a run directory; InferRef always creates a fresh one.
