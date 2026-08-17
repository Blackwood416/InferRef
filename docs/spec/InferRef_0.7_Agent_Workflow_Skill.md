# InferRef 0.7: Agent Workflow Integration Skill

> **Status:** Draft / 0.7.0 scope

## 1. Problem statement

InferRef already has the machinery a coding agent needs: framework-neutral CLI
JSON, an optional MCP stdio transport, trusted engine adapters, testcase and
scenario execution, and a response envelope with `status`, `data`,
`diagnostics`, and `next_actions`.

What is missing is practical guidance for an external coding agent that is
asked to validate or fix an inference engine. The README explains how to
install InferRef and how InferRef's own blind repair evaluation works, but it
does not show an agent how to:

1. discover what InferRef knows about a trace or testcase;
2. extract the smallest engine boundary;
3. write or select a trusted adapter;
4. run a single engine or a stateful scenario;
5. interpret `ok` / `pass` / `fail` / `error` and `next_actions`;
6. iterate from a first divergence to a fixed and rerun engine.

The project should therefore ship a user-facing workflow guide plus a
distributable Codex skill. The skill is not a replacement for specs: it is the
procedural layer that makes the existing protocol usable inside an agent
session.

## 2. Target users

The primary audience is a coding agent working in an inference-engine
repository, not the InferRef maintainers:

- a CUDA / SYCL / ROCm / CPU engine developer;
- a coding agent asked to fix a kernel that currently fails an InferRef
  comparison;
- a model-integration engineer who wants a prefill -> decode scenario to run
  against their engine;
- a contributor adding a new semantic detector, executable contract, or
  corpus case.

The guide must work for both CLI-only agents and agents with the MCP server
configured. It must not assume the agent can read the full Trace IR or Agent
Protocol specs first.

## 3. Deliverable shape

```text
docs/
  AGENT_WORKFLOW.md
  spec/InferRef_0.7_Agent_Workflow_Skill.md
skills/
  inferref/
    SKILL.md
    agents/
      openai.yaml
    references/
      cli.md
      adapter.md
      protocol.md
      workflows.md
scripts/
  check_agent_workflow_skill.py
tests/
  docs/
    test_agent_workflow_skill.py
```

`docs/AGENT_WORKFLOW.md` is the human-facing user guide. `skills/inferref/` is
the distributable Codex skill. The validation script and pytest wrapper keep
the two artifacts honest: every command they mention must exist, and every
referenced file must exist.

## 4. Content requirements

### 4.1 `docs/AGENT_WORKFLOW.md`

The guide must be followable from a clean clone without reading another spec.
It should contain, in order:

1. **When to use this guide.** Validate an engine, extract a repro, debug a
   first divergence, add a corpus case, or run a stateful scenario.
2. **Install.** `uv pip install -e ".[torch,dev,agent]"` plus `inferref
   doctor` so the first step proves the runtime and accelerator state.
3. **Discover the protocol.** `inferref agent capabilities --json` and how to
   read the returned formats, operations, and MCP tool names.
4. **Configure MCP (optional).** Start `inferref-mcp` with `--read-root` and
   `--write-root`, then add a host-specific stdio entry. The guide must show a
   generic JSON snippet:

   ```json
   {
     "mcpServers": {
       "inferref": {
         "command": "<path-to-inferref-mcp>",
         "args": [
           "--read-root", "<workspace>",
           "--write-root", "<runs>"
         ]
       }
     }
   }
   ```

   It must state that exact paths are host policy and are not committed to the
   repository.
5. **Standard agent loop.**

   ```text
   discover -> context -> extract -> run / scenario -> compare -> fix -> rerun
   ```

   Each step needs the exact CLI and MCP equivalent.
6. **Inspect a trace.** `inferref agent context trace/ --json`, what
   `data.analysis`, `data.regions`, and `data.validation` mean, and what
   `next_actions` recommends.
7. **Extract a testcase.** `inferref agent extract` with exactly one
   `--region` or `--op-id`, named inputs/outputs, and `--contract` when the
   operation has an executable contract.
8. **Run an engine.** `inferref agent run testcase/ --adapter
   examples/engine_sim/rope_numpy.adapter.json --runs-dir runs --json`, plus
   `inferref agent compare` for an engine output that already exists.
9. **Run a scenario.** `inferref scenario validate` and `inferref agent
   run_scenario`, with `state_mode=reference` as the default and `engine` as
   the opt-in mode that feeds engine state forward.
10. **Interpret the envelope.** A short table:

    | status | meaning | agent action |
    | --- | --- | --- |
    | `ok` | discovery/inspection succeeded | read `data` and `next_actions` |
    | `pass` | validation/extraction/run succeeded | stop or move to the next case |
    | `fail` | domain found a mismatch or non-reproducible artifact | fix the first divergence and rerun |
    | `error` | adapter/process/protocol failure | inspect diagnostics, stderr, command, cwd |

11. **Common traps.** A concise list:

    - `fail` is not a crash; it is the expected result of a numerical mismatch.
    - Adapters are executable configuration and must be trusted.
    - Never reuse a run directory; InferRef always creates a fresh one.
    - `reproducible: true` in a manifest is not trusted until validation runs.
    - MCP roots are path-policy containment, not a sandbox.
    - Use environment variables for secrets; expanded argv is recorded.

12. **Worked end-to-end example.** Starting from
    `examples/mini_llama/run_trace.py`, then `region detect`, `agent extract`,
    and `agent run` against `examples/engine_sim/rope_numpy.py`.
13. **Where to go deeper.** Links to the Agent Protocol, Engine Adapter,
    Scenario, Contract Schema, and EXTENDING docs.

Acceptance: a new user can execute the loop without opening the spec docs;
all commands are verbatim CLI commands that exist in the current `main`.

### 4.2 Codex skill `skills/inferref`

The skill follows the Codex skill layout from `skill-creator`:

```text
skills/inferref/
  SKILL.md
  agents/openai.yaml
  references/
    cli.md
    adapter.md
    protocol.md
    workflows.md
```

`SKILL.md` requirements:

- YAML frontmatter with `name: inferref` and a `description` that triggers for
  inference-engine validation, trace inspection, testcase extraction, adapter
  execution, first-divergence debugging, and scenario runs.
- Body under 300 lines, focused on procedure, not marketing.
- The standard loop at the top, then a decision table for choosing between
  `context`, `extract_testcase`, `compare_outputs`, `run_engine`, and
  `run_scenario`.
- A references table:

  | file | when to read |
  | --- | --- |
  | `references/cli.md` | need exact CLI subcommand, flag, or MCP tool name |
  | `references/adapter.md` | create, edit, or debug an adapter JSON |
  | `references/protocol.md` | interpret an envelope, status, diagnostics, or `next_actions` |
  | `references/workflows.md` | follow a full workflow with commands |

`references/cli.md` must include, at minimum:

- `inferref doctor` and `inferref doctor --verify-plugins`;
- `inferref trace`, `inferref inspect`, `inferref analyze`, `inferref validate`;
- `inferref region detect/list/create/delete`;
- `inferref testcase extract`;
- `inferref contract list/show/validate`;
- `inferref agent capabilities/context/extract/compare/run/run_scenario`;
- `inferref scenario validate/run`;
- `inferref suite validate/run/report`;
- MCP tools: `inferref_capabilities`, `inferref_context`,
  `inferref_extract_testcase`, `inferref_compare_outputs`,
  `inferref_run_engine`, `inferref_run_scenario`.

`references/adapter.md` must contain:

- Engine Adapter v0.2 JSON shape;
- the four command placeholders `{testcase}`, `{output}`, `{adapter_dir}`,
  `{python}`;
- capability fields and per-contract capability subset rule;
- trust boundary and environment-variable guidance for secrets;
- one copy-paste minimal adapter example.

`references/protocol.md` must contain:

- the response envelope shape;
- status semantics;
- how to read `diagnostics` and `next_actions`;
- one `run_engine` pass/fail/error example.

`references/workflows.md` must contain three complete workflows:

1. **Validate an existing adapter against one testcase.**
2. **Extract a failing region from a trace and isolate it.**
3. **Run a KV scenario with `reference-state`, then with `engine-state` and
   `compare_state`.**

`agents/openai.yaml` must follow the current skill-creator UI metadata rules
and contain a human-facing display name, short description, and default
prompt.

Acceptance:

- `SKILL.md` can be copied to `~/.codex/skills/inferref` and is loadable.
- The skill never duplicates long spec text; it links to references.
- Every command in the skill exists in current InferRef.

### 4.3 README update

Add a short "Use with a coding agent" section between the current Agent
integration section and the blind repair evaluation. It must:

- link to `docs/AGENT_WORKFLOW.md`;
- link to `skills/inferref/`;
- show the one-line skill install command for the local Codex skill directory:

  ```powershell
  Copy-Item -Recurse skills/inferref "$HOME\.codex\skills\inferref"
  ```

- keep all existing installation and evaluation prose intact.

### 4.4 Validation script

Add `scripts/check_agent_workflow_skill.py`, stdlib-only, with:

1. Check that `docs/AGENT_WORKFLOW.md`, `skills/inferref/SKILL.md`, and all
   files referenced from the skill's references table exist.
2. Check that `SKILL.md` frontmatter contains `name: inferref` and a non-empty
   `description`.
3. Extract `inferref <subcommand>` and `inferref_<tool>` names from the guide,
   skill, and references, then verify each against the current CLI help:
   - `python -m inferref.cli.main --help`;
   - subcommand help for `agent`, `scenario`, `suite`, `contract`, `region`,
     and `testcase` as needed.
4. Check that no doc/skill file contains absolute local paths from the author
   machine (for example `E:\RiderProjects`) or credential-like values.
5. Exit non-zero on any failure with a stable, actionable message.

Add `tests/docs/test_agent_workflow_skill.py` that invokes the script and
asserts a clean result. The test must run in `tests/core` style, without
PyTorch and without MCP.

Manual release smoke (not CI):

```powershell
python -m inferref.cli.main agent capabilities --json
python -m inferref.cli.main scenario validate tests/fixtures/scenarios/kv-chain --json
```

### 4.5 Dev status

After implementation, add a point-in-time status report under
`docs/dev/status/2026-08-16-agent-workflow-skill.md` and update
`docs/dev/status/README.md`, following the existing convention.

## 5. Implementation tasks

### Task A1: Agent workflow guide

- Files: `docs/AGENT_WORKFLOW.md`
- Depends on: current CLI/MCP inventory (shared with A2)
- Outcome: section 4.1 is fully implemented.
- Verification:
  - run every CLI command in the guide against `main`;
  - `scripts/check_agent_workflow_skill.py` passes after A4 exists;
  - no absolute author paths or credentials.

### Task A2: Codex skill package

- Files:
  - `skills/inferref/SKILL.md`
  - `skills/inferref/agents/openai.yaml`
  - `skills/inferref/references/cli.md`
  - `skills/inferref/references/adapter.md`
  - `skills/inferref/references/protocol.md`
  - `skills/inferref/references/workflows.md`
- Depends on: current CLI/MCP inventory (shared with A1)
- Outcome: section 4.2 is fully implemented.
- Verification:
  - skill frontmatter is valid;
  - references are one level deep and linked from `SKILL.md`;
  - commands in references match current CLI;
  - skill installs cleanly into `~/.codex/skills/inferref`.

### Task A3: README pointer

- Files: `README.md`
- Depends on: A1, A2
- Outcome: section 4.3 is fully implemented.
- Verification:
  - README links resolve;
  - no duplicated guide content;
  - existing Agent integration and blind evaluation sections unchanged except
    for the inserted section.

### Task A4: Validation harness

- Files:
  - `scripts/check_agent_workflow_skill.py`
  - `tests/docs/test_agent_workflow_skill.py`
- Depends on: A1, A2, A3
- Outcome: section 4.4 is fully implemented.
- Verification:
  - `python -m pytest tests/docs/test_agent_workflow_skill.py -q` passes;
  - script fails when a referenced file or command is missing;
  - script remains stdlib-only.

### Task A5: Dev status

- Files:
  - `docs/dev/status/2026-08-16-agent-workflow-skill.md`
  - `docs/dev/status/README.md`
- Depends on: A1-A4
- Outcome: section 4.5 is fully implemented.
- Verification:
  - README status index points to the new report;
  - report records test count, CI state, XPU/GPU state, and known gaps.

## 6. Dispatch order

Wave 1 (parallel):

- A1 (workflow guide)
- A2 (skill package)

Wave 2 (parallel, after A1/A2):

- A3 (README pointer)
- A4 (validation harness)

Wave 3:

- A5 (dev status)

Each task is small enough for one worker. The review checklist for every task:

- files and links resolve;
- no invented CLI commands or MCP tools;
- no absolute author paths or secrets;
- ASCII only unless quoting an existing non-ASCII label;
- no unrelated refactors;
- test suite for touched areas passes.

## 7. Acceptance

The milestone is complete when:

1. A new user can follow `docs/AGENT_WORKFLOW.md` end-to-end on a clean
   checkout.
2. `skills/inferref/SKILL.md` is installable into a Codex skill directory and
   loads without errors.
3. README points to both artifacts.
4. `tests/docs/test_agent_workflow_skill.py` runs in CI without PyTorch or MCP.
5. A dev status report records what was shipped, what was verified, and what
   remains manual.
