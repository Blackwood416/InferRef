# InferRef Agent Workflow

This guide is the procedural layer for a coding agent that has to validate,
debug, or fix an inference engine with InferRef. It assumes a clean checkout
and nothing else: every command below is a real `inferref` command on current
`main`, and every response is the versioned JSON envelope described in the
Agent Protocol.

If you are an inference-engine developer rather than an agent, this document
is still the fastest path from "I have a trace" to "my engine passes".

## 1. When to use this guide

Use InferRef when you need to:

- validate an engine against a reference trace or a standalone testcase;
- extract the smallest reproducible engine boundary from a trace;
- debug a first numerical divergence between a reference and an engine;
- add a corpus case (an operator, region, or contract-bound testcase);
- run a stateful prefill -> decode scenario against your engine.

Do not use it for model training, profiling, or anything that needs the
original framework at runtime. The engine side of the loop is
framework-neutral: it consumes `.irtensor` files and emits `.irtensor` files.

## 2. Install

```bash
uv pip install -e ".[torch,dev,agent]"
```

The `torch` extra is only needed to produce reference traces; reading,
validating, extracting, comparing, and running against an engine do not need
it. The `agent` extra adds the optional MCP transport.

Prove the installation before doing anything else:

```bash
inferref doctor
```

`doctor` reports the Python runtime, NumPy, PyTorch, and accelerator state
(`inferref doctor --device cpu` requires a working CPU device). A `pass` here
means the runtime is ready; a `warn` for PyTorch is fine when you only run the
engine-side loop.

## 3. Discover the protocol

```bash
inferref agent capabilities --json
```

The response is the standard envelope:

```json
{
  "protocol": {"format": "inferref-agent-response", "version": "0.1"},
  "operation": "capabilities",
  "status": "ok",
  "data": {
    "inferref_version": "0.6.0",
    "formats": {"trace": "...", "testcase": "...", "scenario": "...", "tensor": "...", "engine_adapter": "..."},
    "operations": [
      {"name": "context", "mutates": false, "description": "..."},
      {"name": "extract_testcase", "mutates": true, "description": "..."},
      {"name": "compare_outputs", "mutates": false, "description": "..."},
      {"name": "run_engine", "mutates": true, "description": "..."},
      {"name": "run_scenario", "mutates": true, "description": "..."}
    ],
    "mcp_tools": ["inferref_context", "inferref_extract_testcase", "inferref_compare_outputs", "inferref_run_engine", "inferref_run_scenario", "inferref_capabilities"]
  },
  "diagnostics": [],
  "next_actions": [{"operation": "context", "reason": "Inspect a trace or testcase before extracting or executing it."}]
}
```

Read `data.operations` to know what is available, `data.mcp_tools` for the
MCP tool names, and `next_actions` for what the protocol recommends next.
Every operation returns this same envelope shape.

### Two facades, one execution model

InferRef exposes two facades over the same core operations:

```text
Human / exploratory:
  inspect analyze region testcase compare suite scenario

Stable automation / agent facade:
  agent context agent extract agent run agent compare agent run_scenario
```

`inferref agent ...` does not define a separate InferRef execution model. It is
the stable, machine-readable protocol facade over the same core operations.
The `inferref adapter scaffold` command sits on the human side: it generates
the adapter or runtime-bridge project that the agent facade then executes.

## 4. Configure MCP (optional)

The MCP transport is a stdio server that exposes the same five operations as
tools. Start it with explicit path-policy roots:

```bash
inferref-mcp --read-root <workspace> --write-root <runs>
```

`--read-root` controls which artifacts the tools may read; `--write-root`
controls where testcases and runs may be written. Both are repeatable. Then
add a host-specific stdio entry to your MCP client configuration:

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

The exact paths are host policy: they depend on the user's machine and must
never be committed to a repository. MCP roots are path-policy containment, not
a sandbox; the adapters the tools execute are trusted code.

## 5. Standard agent loop

```text
discover -> context -> extract -> run / scenario -> compare -> fix -> rerun
```

| Step | CLI | MCP tool |
| --- | --- | --- |
| discover | `inferref agent capabilities --json` | `inferref_capabilities` |
| context | `inferref agent context trace/ --json` | `inferref_context(path)` |
| extract | `inferref agent extract trace/ --region "..." -o repro/ ...` | `inferref_extract_testcase(trace, output, region=..., input_names=[...], output_names=[...], contracts=[...])` |
| run | `inferref agent run repro/ --adapter adapter.json --runs-dir runs --json` | `inferref_run_engine(testcase, adapter, runs_root)` |
| scenario | `inferref agent run_scenario scenarios/kv-chain --adapter adapter.json --runs-dir runs --json` | `inferref_run_scenario(scenario, adapter, runs_root, state_mode="reference", compare_state=false)` |
| compare | `inferref agent compare repro/ engine-out/ --json` | `inferref_compare_outputs(testcase, engine_output)` |
| fix + rerun | edit your engine, then repeat `run` or `compare` | same tools |

Stay inside the loop: `context` before `extract`, `extract` before `run`, and
`compare` (or the comparison inside `run`) before declaring success. The
envelope's `next_actions` tells you the next step when you are unsure.

## 6. Inspect a trace

```bash
inferref agent context trace/ --json
```

For a trace the response contains:

- `data.analysis` - operator/value totals, operator and module counts,
  semantic coverage (`region`, `semantic`, `source`, `payload`), and the
  detected regions with their node counts;
- `data.regions` - every semantic region record (id, name, node count,
  creation method, engine op mapping);
- `data.validation` - `status`, `errors`, and `warnings` from the Trace IR
  invariants;
- `diagnostics` - one entry per invariant issue;
- `next_actions` - usually `region_detect` (no regions yet), `extract_testcase`
  (regions exist), or `validate` (Trace IR errors must be repaired first).

Read `data.analysis.coverage.payload`: if it is below 1.0, re-trace with full
capture (`--capture-tensors all`) before extracting, or the boundary will lack
runnable payloads.

For a testcase directory the response instead reports `data.artifact:
"testcase"`, `data.reproducible`, `data.validation` (full standalone
validation), and the bound inputs/outputs.

## 7. Extract a testcase

```bash
inferref agent extract trace/ --region "RoPE@layers.0.self_attn" \
  -o repro/rope \
  --input-names cos,sin,query,key \
  --output-names q_embed,k_embed --json
```

Pass exactly one selection: `--region "NAME"` or `--op <operator-id>` (the
operator id is shown by `inferref inspect trace/`). The explicit names are
assigned positionally to the boundary values in trace order, so the order
must match the region boundary shown by `inferref agent context trace/ --json`
or `inferref region list trace/`.

Add `--contract <id>` when the operation has an executable contract:

```bash
inferref agent extract trace/ --region "RoPE@layers.0.self_attn" \
  -o repro/rope \
  --input-names cos,sin,query,key \
  --output-names q_embed,k_embed \
  --contract rope/rotate-half/v1 --json
```

The contract binds the exact role names and shape/dtype invariants, and the
extraction refuses to publish a boundary that violates them. List the
available contracts with `inferref contract list --json`.

A `pass` status means the testcase is independently reproducible: every
boundary value has a payload and no side effects are unobservable. A `fail`
means payloads are missing (re-trace with full capture) or the boundary is
not self-contained. The envelope's `next_actions` says what to do next.

## 8. Run an engine

```bash
inferref agent run repro/rope \
  --adapter examples/engine_sim/rope_numpy.adapter.json \
  --runs-dir runs --json
```

This executes the trusted adapter in a fresh run directory, applies the
capability preflight (dtype, rank, feature, and contract requirements), and
compares the engine output with the reference. Never reuse a run directory:
InferRef always creates a fresh one and records `inferref-run.json` inside it.

Preflight happens before any process is created. If the adapter does not
declare a requirement the testcase needs, the response is `error` with
`capability_status: "unsupported"` and the exact reason:

```json
{"kind": "feature", "required": "alias_effects"}
```

Adapters are the engine author's declaration of what the engine supports.
When the declaration is incomplete, copy the adapter, extend
`capabilities.features` (and adjust `command`/`cwd` for the new location),
then rerun:

```bash
cp examples/engine_sim/rope_numpy.adapter.json rope.adapter.json
```

```json
{
  "capabilities": {
    "features": ["multiple_outputs", "strided_inputs", "alias_effects"]
  },
  "command": ["{python}", "examples/engine_sim/rope_numpy.py", "{testcase}", "--output", "{output}"],
  "cwd": "."
}
```

```bash
inferref agent run repro/rope --adapter rope.adapter.json \
  --runs-dir runs --json
```

When an engine output directory already exists, compare it without executing
anything:

```bash
inferref agent compare repro/rope engine-out/ --json
```

## 9. Run a scenario

A scenario is an ordered chain of testcases with explicit state binding, for
example prefill -> decode -> decode with the KV cache flowing between steps.

Validate the scenario and every referenced testcase first:

```bash
inferref scenario validate scenarios/kv-chain --json
```

Then run it through one trusted adapter:

```bash
inferref agent run_scenario scenarios/kv-chain \
  --adapter kv-copy.adapter.json \
  --runs-dir runs --json
```

`state_mode=reference` (the default) feeds each step the reference state from
the previous step, isolating every kernel from state-continuity bugs. Opt in
to engine state with:

```bash
inferref agent run_scenario scenarios/kv-chain \
  --adapter kv-copy.adapter.json \
  --runs-dir runs --json \
  --state-mode engine --compare-state
```

`--compare-state` additionally compares the engine-produced state against the
reference state after each step, localizing a bad cache write to the step that
produced it. The scenario run report records per-step status, run ids, state
status, and outputs; the equivalent CLI is `inferref scenario run ...`.

## 10. Interpret the envelope

Every response has `status`, `data`, `diagnostics`, and `next_actions`:

| status | meaning | agent action |
| --- | --- | --- |
| `ok` | discovery/inspection succeeded | read `data` and `next_actions` |
| `pass` | validation/extraction/run succeeded | stop or move to the next case |
| `fail` | domain found a mismatch or non-reproducible artifact | fix the first divergence and rerun |
| `error` | adapter/process/protocol failure | inspect `data.execution` (stderr, command, cwd) and the diagnostics code |

`diagnostics` is a list of `{severity, code, message}` entries; on `error` it
carries a stable structured summary, while the execution record details
(`stderr`, `exit_code`, `command`, `cwd`) live in `data.execution`.
`next_actions` is the protocol's recommendation and is safe to follow verbatim.

## 11. Common traps

- `fail` is not a crash; it is the expected result of a numerical mismatch.
  The comparison localizes the first diverging tensor, element, producer, and
  source line.
- Adapters are executable configuration and must be trusted. Only run adapters
  you or the repository owner wrote; never one from an untrusted message.
- Never reuse a run directory; InferRef always creates a fresh one.
- `reproducible: true` in a manifest is not trusted until validation runs.
- MCP roots are path-policy containment, not a sandbox.
- Use environment variables for secrets; expanded argv is recorded in the run
  record.

## 12. Worked end-to-end example

This is the complete loop shipped with the repository, verified on `main`.

```bash
# 1. Produce a reference trace of the mini-Llama block
python examples/mini_llama/run_trace.py --output trace/

# 2. Discover the semantic regions
inferref region detect trace/

# 3. Inspect what the trace knows (analysis, regions, validation)
inferref agent context trace/ --json

# 4. Extract the RoPE region as a standalone testcase (contract-bound)
inferref agent extract trace/ --region "RoPE@layers.0.self_attn" \
  -o repro/rope \
  --input-names cos,sin,query,key \
  --output-names q_embed,k_embed \
  --contract rope/rotate-half/v1 --json

# 5. Quickest pass: run the numpy engine directly, then compare
python examples/engine_sim/rope_numpy.py repro/rope --output engine-out/
inferref agent compare repro/rope engine-out/ --json

# 6. Or run through the trusted adapter (with capability preflight)
inferref agent run repro/rope \
  --adapter examples/engine_sim/rope_numpy.adapter.json \
  --runs-dir runs --json

# 7. Inject a realistic bug and let the comparator localize it
python examples/engine_sim/rope_numpy.py repro/rope \
  --output engine-bad/ --inject-bug
inferref agent compare repro/rope engine-bad/ --json
```

Step 4 returns `pass` with `reproducible: true`. Step 5 returns `pass`. Step 6
may return `error` with `capability_status: "unsupported"` and
`{"kind": "feature", "required": "alias_effects"}` because the mini-Llama RoPE
region records aliasing operators; follow section 8 to extend the copied
adapter's capabilities and rerun. Step 7 returns `fail` and reports the first
mismatching element of `q_embed` with its producer operator, module path, and
source line - that is the divergence to fix, not the whole output.

To exercise the stateful path with the repository fixture:

```bash
inferref scenario validate tests/fixtures/scenarios/kv-chain --json
```

Then run it with an adapter whose engine implements KV append (the scenario
steps bind `cache`/`update` to `cache_out` and `logits`):

```bash
inferref agent run_scenario tests/fixtures/scenarios/kv-chain \
  --adapter kv-copy.adapter.json --runs-dir runs --json
```

## 13. Where to go deeper

- [Agent Protocol v0.1](spec/InferRef_Agent_Protocol_v0.1.md) - envelope,
  status, diagnostics, next_actions;
- [Engine Adapter v0.2](spec/InferRef_Engine_Adapter_v0.2.md) - adapter JSON,
  placeholders, capability preflight;
- [Scenario v0.1](spec/InferRef_Scenario_v0.1.md) - state binding and replay
  modes;
- [Contract Schema v0.1](spec/InferRef_Contract_Schema_v0.1.md) - executable
  contracts and role binding;
- [EXTENDING](EXTENDING.md) - adding detectors, contracts, adapters, and
  corpus cases.
