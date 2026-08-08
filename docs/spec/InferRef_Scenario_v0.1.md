# InferRef Scenario v0.1 Specification

> **Project:** InferRef
> **Document:** Scenario Artifact and Execution
> **Scenario Version:** 0.1
> **Status:** Draft / 0.6.0 scope
> **Audience:** InferRef core maintainers, engine adapter authors, suite/report
> tooling authors, worker agents

---

# 1. Purpose

A Scenario is an ordered chain of standalone testcases with explicit state
binding. It is the artifact between a single Testcase and a Suite:

```text
Testcase   one semantic operation, one executable contract

Scenario   an ordered chain of testcases with explicit state binding

Suite      many independent Testcases/Scenarios x many Engines
```

Scenarios make stateful execution first-class:

```text
prefill -> decode #0 -> decode #1 -> decode #2
```

They cover Static/Dynamic/Paged KV cache, sliding-window cache, speculative
fork/rollback, Mamba recurrent state, Gated DeltaNet recurrent state, and
causal conv state.

---

# 2. Scope of v0.1

v0.1 defines:

- the `inferref-scenario` manifest format;
- structural and runnable validation;
- the step executor over the existing engine adapter ABI;
- two replay modes: `reference-state` (default) and `engine-state`;
- the `inferref-scenario-run` v0.1 report;
- CLI `inferref scenario {validate,run}`;
- Suite 0.2 integration with `kind`;
- Agent protocol operation `run_scenario` and MCP tool
  `inferref_run_scenario`.

Not in v0.1:

- automatic scenario derivation from traces (future work);
- branching, loops, or conditional steps;
- scenario-level composition of scenarios;
- scalar/string state slots (tensor state only);
- engine-side session protocol (`{scenario}`/`{state}` placeholders).

---

# 3. Design Decisions

Approved in the 0.6 milestone review:

- D3: `reference-state` is the default replay mode; `engine-state` is opt-in.
- D4: Suite moves to 0.2 with an additive `kind` field; readers accept 0.1.
- D5: the executor compiles each step into an effective testcase and reuses
  `execute_adapter`. No adapter ABI change, no new placeholders.
- D6: the executor inherits the existing adapter trust model and artifact
  controls.

---

# 4. Manifest Format

```json
{
  "format": "inferref-scenario",
  "format_version": "0.1",
  "id": "qwen35-gdn-prefill-decode",
  "description": "GDN prefill followed by two decode steps",
  "inputs": {
    "prefill_kv": {"kind": "tensor"},
    "prefill_tokens": {"kind": "tensor"},
    "decode_tokens": {"kind": "tensor"}
  },
  "state": {
    "kv": {"kind": "tensor", "init": "scenario.inputs.prefill_kv"}
  },
  "outputs": {
    "logits": {"kind": "tensor"}
  },
  "steps": [
    {
      "id": "prefill",
      "testcase": "cases/gdn-prefill",
      "bindings": {
        "inputs": {
          "cache": "scenario.inputs.prefill_kv",
          "update": "scenario.inputs.prefill_tokens"
        },
        "outputs": {
          "cache_out": "state.kv",
          "logits": "scenario.outputs.logits"
        }
      }
    },
    {
      "id": "decode-0",
      "testcase": "cases/gdn-decode",
      "bindings": {
        "inputs": {
          "cache": "state.kv",
          "update": "scenario.inputs.decode_tokens"
        },
        "outputs": {
          "cache_out": "state.kv"
        }
      }
    }
  ]
}
```

## 4.1 Top-level fields

| Field | Required | Type | Rules |
| --- | --- | --- | --- |
| `format` | yes | string | exactly `inferref-scenario` |
| `format_version` | yes | string | exactly `0.1` |
| `id` | yes | string | `^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`, portable case-fold unique |
| `description` | no | string | non-empty |
| `inputs` | yes | object | non-empty scenario input map |
| `state` | no | object | named state slots; may be empty/absent for pure chains |
| `outputs` | no | object | scenario output map; may be empty/absent |
| `steps` | yes | array | non-empty ordered step list |

Every value record has `"kind": "tensor"` in v0.1.

Every `inputs`/`state`/`outputs` name must satisfy the section 4.1 ID rules
(including Windows reserved-name and trailing dot/space checks) and must be
case-fold unique within its own map: v0.1 materializes each value as an
`<name>.irtensor` file, so a case collision would silently merge two slots on
case-insensitive filesystems.

## 4.2 Step record

| Field | Required | Type | Rules |
| --- | --- | --- | --- |
| `id` | yes | string | unique, section 4.1 ID rules |
| `testcase` | yes | string | relative path to a standalone testcase directory, contained below the scenario root |
| `bindings` | no | object | `inputs` and `outputs` role maps |
| `bindings.inputs` | no | object | testcase input role -> source reference |
| `bindings.outputs` | no | object | testcase output role -> destination reference |

## 4.3 Value references

```text
scenario.inputs.<name>     source: scenario input tensor
state.<name>               source/destination: state slot
scenario.outputs.<name>    destination: scenario output tensor
```

Reference rules:

- source references may be `scenario.inputs.<name>` or `state.<name>`;
- destination references may be `state.<name>` or
  `scenario.outputs.<name>`;
- a source role may be bound to `state.<name>` only if that slot is
  initialized before this step (section 5.4);
- a destination role may write a state slot only if the slot exists in
  `state`; a scenario output only if it exists in `outputs`.

## 4.4 Binding semantics

- A testcase input role that is bound is overridden at run time by the bound
  tensor.
- A testcase input role that is NOT bound keeps the testcase's own embedded
  input payload. This lets scenarios reuse testcases that contain fixed
  constants (for example `epsilon`).
- Every testcase output is always compared against the step's reference
  outputs, whether bound or not.
- A bound output also propagates: into a state slot or the scenario output
  area.
- An output role bound to `scenario.outputs.<name>` must exist in the
  testcase's outputs.

---

# 5. Validation

Validation is split into `schema_valid` and `runnable`, mirroring Suite
validation.

## 5.1 Structural rules

1. Manifest parses as JSON with the exact `format`/`format_version`.
2. `id` follows the ID rules and is unique within one Scenario.
3. `inputs` is non-empty; every declared input is a valid value record.
4. `state` and `outputs`, when present, are valid value records.
5. Step IDs are unique.
6. Every `testcase` path is relative, resolves below the scenario root, and
   exists.
7. Every referenced role exists in the referenced testcase with matching
   kind.
8. Every bound input source is declared (`scenario.inputs` or `state`) and
   every bound output destination is declared (`state` or
   `scenario.outputs`).
9. No state slot is written twice within one step.
10. No state slot is read before initialization (section 5.4).
11. Every scenario output is written by at least one step.
12. Reference syntax is valid (section 4.3).

## 5.2 Runnable rules

Every referenced testcase must pass standalone validation and be
reproducible. A non-reproducible step makes the Scenario non-runnable but may
still be schema-valid.

## 5.3 `inferref scenario validate`

```bash
inferref scenario validate scenarios/kv-chain [--allow-nonreproducible] [--json]
```

JSON:

```json
{
  "format": "inferref-scenario-validation",
  "format_version": "0.1",
  "status": "pass",
  "schema_valid": true,
  "runnable": true,
  "non_runnable_steps": [],
  "issues": []
}
```

Exit code: 0 on pass; 1 when schema-invalid, or non-runnable unless
`--allow-nonreproducible` (matching Suite validate policy).

## 5.4 State initialization

A state slot may declare `init`:

```json
"kv": {"kind": "tensor", "init": "scenario.inputs.prefill_kv"}
```

Rules:

- `init` must reference `scenario.inputs.<name>` in v0.1;
- a slot with `init` is considered initialized before step 0;
- a slot without `init` must be written by a step before any later step reads
  it;
- a slot may be written before being read even without `init`.

---

# 6. Execution

## 6.1 Effective testcase compilation

For each step, the executor materializes an effective testcase:

```text
<scenario-run>/steps/<step-id>/testcase/
  testcase.json
  inputs/...
  reference/...
```

Algorithm:

1. Copy the referenced testcase tree into the effective directory.
2. For each bound input role, resolve the source tensor file
   (`scenario.inputs/<name>.irtensor` or the current state slot file), copy it
   to `effective/inputs/<role>.irtensor`, and patch the manifest record for
   that role with the copied file's header metadata (`payload`, `dtype`,
   `shape`, `stride`, `storage_offset`, `numel`, `nbytes`).
3. Run the existing adapter via `execute_adapter(effective, adapter,
   step_runs_dir)` with the adapter's normal placeholders.
4. Compare engine outputs against the effective testcase reference with the
   requested tolerance/layout policy and `--first-failure`.
5. Apply output bindings (section 6.3).

The effective testcase must pass standalone validation after patching. If the
patched manifest is invalid, the step fails with a validation error, not an
engine error.

## 6.2 Replay modes

### `reference-state` (default)

State slots are filled from the step's reference outputs. The engine never
drives state. This validates every kernel at every step independently and is
the mode for corpus and semantic validation.

### `engine-state`

State slots are filled from the engine's outputs. The engine's state feeds the
next step. This validates state continuity end to end.

In `engine-state` mode, after copying the engine output into a state slot, the
executor ALWAYS validates shape and dtype against the step's reference output:

- mismatch -> `state_shape_mismatch` / `state_dtype_mismatch`, step fails and
  the scenario stops (fail-fast) to avoid propagating garbage;
- when `--compare-state` is set, numeric comparison is also performed against
  the reference state; numeric mismatch -> `state_mismatch`, step fails and
  the scenario stops.

Without `--compare-state`, numeric state divergence is allowed to propagate;
the failure then surfaces at a later step or at final outputs.

If the engine produces no tensor for a bound state output in `engine-state`
mode, the step fails with `state_status = "state_missing"` and the scenario
stops immediately; no stale state is fed to later steps. An `unsupported`
step in `engine-state` mode also stops the chain, because no engine output is
available to advance the state slots.

## 6.3 Output binding application

| Destination | Mode | Tensor copied |
| --- | --- | --- |
| `state.<name>` | reference-state | step reference output |
| `state.<name>` | engine-state | engine output, then validated |
| `scenario.outputs.<name>` | either | engine output (must exist; missing output is an engine output error) |

## 6.4 Run directory layout

```text
<runs-root>/scenario-<artifact-key>/
  state/<slot>/<slot>.irtensor            latest slot content
  inputs/<name>.irtensor                  materialized scenario inputs
  outputs/<name>.irtensor                 scenario outputs (engine-produced)
  steps/<step-id>/testcase/               effective testcase
  steps/<step-id>/output/                 engine output dir
  steps/<step-id>/inferref-run.json       per-step adapter run record
  inferref-scenario-run.json              scenario report
```

Directory names use the same slug + SHA-256 artifact key and containment rules
as Suite runs.

## 6.5 Status semantics

Per-step status values:

```text
pass
mismatch          numerical failure, valid domain result
unsupported       preflight rejected the step before process creation
error             adapter, process, or validation failure
```

Scenario status aggregation:

| Condition | Scenario status |
| --- | --- |
| every step `pass` | `pass` |
| at least one `mismatch`, no `error` | `fail` |
| at least one `error` | `error` |
| every step `unsupported`, none executed | `unsupported` |
| mixed `pass`/`unsupported`, none failed, `--allow-unsupported` | `partial` |
| any real failure without `--allow-unsupported` | `fail` |

Agent response maps scenario `fail` to envelope status `fail` and scenario
`error` to envelope status `error`.

---

# 7. Scenario Run Report

```json
{
  "format": "inferref-scenario-run",
  "format_version": "0.1",
  "status": "pass",
  "accepted": true,
  "scenario": {
    "id": "qwen35-gdn-prefill-decode",
    "source": "scenarios/qwen35-gdn"
  },
  "adapter": {
    "id": "inferref-sycl-xpu",
    "name": "inferref-sycl-xpu"
  },
  "state_mode": "reference",
  "compare_state": false,
  "steps": [
    {
      "id": "prefill",
      "status": "pass",
      "state_status": "not_applicable",
      "run": {},
      "inputs": {"cache": "scenario.inputs.prefill_kv"}
    }
  ],
  "outputs": ["logits"],
  "duration_ms": 123.45
}
```

`state_status` values:

```text
not_applicable      reference-state mode
ok                  engine state shape/dtype (and numeric when requested) ok
not_compared        engine-state, shape/dtype ok, numeric not requested
state_missing       engine-state, engine produced no state output
state_mismatch      numeric state divergence
state_shape_mismatch
state_dtype_mismatch
```

Every step embeds the complete `inferref-run.json` execution record so the
first-divergence report stays actionable.

---

# 8. CLI

## 8.1 `inferref scenario run`

```bash
inferref scenario run scenarios/kv-chain \
  --adapter sycl.adapter.json \
  --runs-dir runs/ \
  --state-mode reference|engine \
  [--compare-state] \
  [--allow-unsupported] \
  [--fail-fast] \
  [--atol F] [--rtol F] \
  [--strict-layout] [--first-failure] [--all-failures] \
  [--json]
```

Behavior:

- `--state-mode` defaults to `reference`;
- `--compare-state` is only meaningful with `engine-state`;
- `--fail-fast` raises on the first unexpected exception instead of recording
  `error`, matching Suite semantics;
- exit 0 on accepted status, 1 otherwise.

## 8.2 Agent operation

```bash
inferref agent run_scenario scenarios/kv-chain \
  --adapter sycl.adapter.json \
  --runs-dir runs/ --json
```

The operation returns the standard Agent envelope with `data` = scenario run
report, `status` mapping per section 6.5, and `next_actions` recommending the
first failed step.

---

# 9. Suite v0.2 Integration

## 9.1 Manifest

```json
{
  "format": "inferref-suite",
  "format_version": "0.2",
  "name": "xpu-corpus",
  "cases": [
    {
      "id": "rope-fp16",
      "kind": "testcase",
      "testcase": "cases/rope",
      "tags": ["rope"]
    },
    {
      "id": "kv-decode-chain",
      "kind": "scenario",
      "testcase": "scenarios/kv-chain",
      "tags": ["kv", "stateful"]
    }
  ]
}
```

Rules:

- `format_version` is `0.1` or `0.2`; writers emit `0.2`;
- `kind` defaults to `testcase` for 0.2 and is always `testcase` in 0.1;
- `kind` must be `testcase` or `scenario`;
- case IDs remain unique under portable case folding, independent of kind;
- path containment rules are unchanged;
- `load_suite` validates scenario cases with the scenario validator and
  testcase cases with the testcase validator.

## 9.2 `inferref suite run`

For a scenario case, `execute_adapter` is replaced by the scenario executor
with the same adapter and the case's runs root. The cell embeds the scenario
run report as its `run` value and derives cell status per section 6.5.

## 9.3 `inferref suite validate`

`schema_valid` and `runnable` now cover both kinds. `non_runnable_cases`
includes scenario cases with non-reproducible steps.

## 9.4 `inferref suite report`

The HTML report renders scenario cells with step status counts and the first
failed step; the JSON sidecar keeps the scenario run report intact.

---

# 10. MCP Transport

New tool: `inferref_run_scenario`.

Arguments:

```text
scenario:        string (required)
adapter:         string (required)
runs_root:       string (required)
state_mode:      "reference" | "engine" (default "reference")
compare_state:   boolean (default false)
atol, rtol:      number (optional)
ignore_stride:   boolean (default false)
strict_layout:   boolean (default false)
first_failure:   boolean (default true)
```

The scenario directory is resolved against `--read-root`; adapter against
read root; runs_root against `--write-root`. Payload paths inside the scenario
manifest and its testcases must remain contained. The existing symlink/
reparse-point caveat is unchanged.

`capabilities()` advertises the new operation and MCP tool name.

---

# 11. Security and Resource Controls

- Every step inherits the adapter's timeout, stream limits, artifact byte
  limits, and artifact file limits.
- Effective testcase trees live under the scenario runs root and are created
  with the existing containment helpers.
- State files are ordinary `.irtensor` files copied through contained paths;
  they are not a sandbox boundary.
- Scenario execution adds no new process or filesystem escape surface: the
  only spawned process is the adapter itself.

---

# 12. Compatibility

- Testcase 0.2 and adapter 0.2 are unchanged.
- Suite 0.1 remains readable; 0.2 is additive.
- Agent protocol 0.1 remains additive; `run_scenario` is a new operation.
- Scenario v0.1 is a new format; no existing artifact is reinterpreted.
- The torch-free core boundary is unchanged: manifest loading, validation,
  and scenario execution never import torch. Scenario fixtures used by tests
  are committed artifacts, not live traces.

---

# 13. Error Codes

| Code | Meaning |
| --- | --- |
| `scenario_invalid_manifest` | not JSON, wrong format/version |
| `scenario_invalid_id` | bad or duplicate scenario ID |
| `scenario_duplicate_step_id` | duplicate step ID |
| `scenario_bad_step_id` | step ID fails portable-ID rules |
| `scenario_testcase_invalid` | referenced testcase fails validation |
| `scenario_testcase_nonreproducible` | referenced testcase is not reproducible |
| `scenario_role_unknown` | bound role not in testcase or declaration |
| `scenario_role_kind_mismatch` | bound role kind differs |
| `scenario_source_undeclared` | input source not in `scenario.inputs`/`state` |
| `scenario_destination_undeclared` | output destination not in `state`/`outputs` |
| `scenario_state_uninitialized` | state read before initialization |
| `scenario_state_written_twice` | two outputs write one slot in one step |
| `scenario_output_unwritten` | declared scenario output never written |
| `scenario_path_escape` | testcase path escapes the scenario root |
| `scenario_state_shape_mismatch` | engine state shape differs from reference |
| `scenario_state_dtype_mismatch` | engine state dtype differs from reference |
| `scenario_state_mismatch` | numeric state comparison failed |
| `scenario_state_missing` | engine produced no tensor for a bound state output |
| `scenario_output_missing` | engine produced no tensor for a bound scenario output |
| `scenario_reference_invalid` | binding reference violates section 4.3 syntax |

---

# 14. Acceptance Criteria

1. A committed KV scenario (prefill + two decode steps) validates and runs in
   `reference-state` mode against a fixture adapter without torch, producing
   `inferref-scenario-run` v0.1.
2. `engine-state` mode feeds the engine's own cache forward; a fixture engine
   that corrupts `cache_out` in step 0 is reported as
   `state_shape_mismatch`/`state_dtype_mismatch`/`state_mismatch` at step 0
   when `--compare-state` is used, and as a downstream mismatch without it.
3. An unbound input role falls back to the testcase's embedded input.
4. `inferref suite run` executes a scenario cell and `suite report` renders
   step summaries.
5. `inferref_run_scenario` works through the MCP server with path policies
   enforced.
6. Scenario execution never imports torch; the core suite runs with torch
   blocked.
7. `--allow-unsupported` produces `partial` for mixed pass/unsupported
   scenarios and `unsupported` when every step is unsupported.

---

# 15. Implementation Tasks

## Task S1: Manifest schema and validation

Files:

- add `inferref/scenario/__init__.py`
- add `inferref/scenario/schema.py` (dataclasses: `Scenario`, `ScenarioStep`,
  `ScenarioState`, `ScenarioInput`, `ScenarioOutput`; load and structural
  validation)
- add `inferref/scenario/validate.py` (schema_valid + runnable split, reuses
  `validate_testcase`)
- add `inferref/cli/main.py` `scenario validate` subcommand
- tests in `tests/core/test_scenario_manifest.py`

Acceptance: all section 5 rules covered; CLI exit codes correct.

## Task S2: Executor and run report

Files:

- add `inferref/scenario/run.py` (effective testcase compilation, state
  management, replay modes, report)
- add `inferref/scenario/state.py` if the state store grows beyond run.py
- tests in `tests/core/test_scenario_run.py` with fixture adapters and
  committed scenario fixtures

Acceptance: section 6 and 7 behavior; engine-state corruption cases.

## Task S3: Agent and MCP integration

Files:

- `inferref/agent/service.py` (`run_scenario`)
- `inferref/agent/mcp_server.py` (`inferref_run_scenario`)
- `inferref/agent/protocol.py` if operation metadata is enumerated there
- `inferref/cli/main.py` `agent run_scenario`
- tests in `tests/agent/test_mcp_server.py` and core CLI tests

Depends on S2.

## Task S4: Suite 0.2 integration

Files:

- `inferref/suite/schema.py` (`kind`, 0.2 writer, 0.1 reader)
- `inferref/suite/run.py` (scenario cells)
- `inferref/suite/report.py` (scenario rendering)
- tests in `tests/core/test_suite.py`

Depends on S1 and S2; can run in parallel with S3 after S2 lands.

## Task S5: Fixtures

Files:

- `tests/fixtures/scenarios/kv-chain/` (manifest + chained committed
  testcases, e.g., derived from the XPU corpus KV cases)
- a fixture copy-adapter used by core tests (numpy-only)

Sequencing: S1 -> S2 -> S3/S4; S5 is incremental with S2.

---

# 16. Test Plan

| Area | Cases |
| --- | --- |
| Manifest | field rules, binding grammar, state init, path containment |
| Validation | all structural errors, runnable split, CLI policy |
| Execution | reference-state, engine-state, unbound fallback, output bindings |
| State | init, uninitialized read, double write, shape/dtype/numeric mismatch |
| Suite | 0.1 read, 0.2 write, scenario cells, report rendering |
| Agent | envelope status mapping, MCP args, path policy |
| Torch-free | core scenario tests run with torch hard-blocked |
