# InferRef

[![CircleCI](https://dl.circleci.com/status-badge/img/circleci/PuMzCg2wMRTLmqJJN1tfGW/9g5ZmC1DWg3UfH9Eu8AKTb/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/circleci/PuMzCg2wMRTLmqJJN1tfGW/9g5ZmC1DWg3UfH9Eu8AKTb/tree/main)

Reference execution tracing, testcase extraction, and numerical comparison for inference engine
development.

InferRef turns a PyTorch model execution into a **stable, machine-readable reference specification**
that a custom inference engine (CUDA / SYCL / ROCm / CPU) can be validated against — without the
engine side needing PyTorch, or Python at all.

Check the local frontend and accelerator before tracing:

```bash
inferref doctor                 # inventory the installation
inferref doctor --device xpu    # require an Intel XPU and test host capture
inferref doctor --device cuda   # require CUDA/ROCm-as-CUDA
inferref doctor --verify-plugins # explicitly import and validate detector plugins
```

Trace execution device metadata is inferred from the tensors observed at
runtime. `--device` remains available as an explicit manifest override.

Run a reusable testcase suite against an engine adapter:

```bash
inferref suite validate suite.json
inferref suite run suite.json --adapter sycl.adapter.json --adapter cpu.adapter.json --runs-dir runs/
inferref suite report runs/inferref-suite-run.json --output report.html
```

Adapter v0.2 declares its target device, dtypes, rank and effect support.
Testcase v0.2 records matching derived requirements, allowing InferRef to
return `unsupported` before starting an incompatible engine. Legacy 0.1
adapters and testcases remain readable.

InferRef 0.5 also includes an optional native C++/SYCL engine and a deterministic
XPU corpus. It reads `.irtensor` directly and has no Python/PyTorch engine-side
dependency. See [the Windows XPU/SYCL gate](docs/xpu-sycl.md).

```text
Model → Reference Trace → Trace IR → Testcase → Engine → Compare → First Divergence
```

Implements [InferRef_SPEC.md](docs/spec/InferRef_SPEC.md) §62 (MVP scope) and
[InferRef_Trace_IR_v0.1.md](docs/spec/InferRef_Trace_IR_v0.1.md).

## Install

```bash
uv pip install -e ".[torch,dev]"
```

The core package — Trace IR reader, `.irtensor` codec, comparator, testcase extraction, region
tools, and the `inspect`/`analyze`/`compare`/`validate` CLI commands — depends on **numpy only**.
PyTorch is required solely for the tracing frontend. This is enforced by the test suite: a
subprocess blocks `import torch` outright and then loads a trace, decodes payloads, extracts a
testcase and runs a comparison (Trace IR §57 acceptance criterion 10).

## Quick start

```bash
# 1. Produce a reference trace
python examples/mini_llama/run_trace.py --output trace/

# 2. Look at it
inferref inspect trace/ --limit 20
inferref analyze trace/

# 3. Find the semantic regions, then extract a standalone testcase
inferref region detect trace/
inferref testcase extract trace/ --region "RoPE@layers.0.self_attn" -o repro/rope \
    --input-names cos,sin,query,key --output-names q_embed,k_embed

# 4. Run your engine against the testcase, then compare
python examples/engine_sim/rope_numpy.py repro/rope --output engine-out/
inferref compare repro/rope engine-out/ --first-failure
```

`region detect` recognises Linear, RMSNorm, RoPE, Attention, SwiGLU and friends from the module
hierarchy and source functions the trace already records — no operator ids, no static export:

```text
Detected 30 semantic region(s), created 30:

    14x  Linear
     5x  RMSNorm
     3x  RoPE
     2x  Attention
     2x  RepeatKV
     2x  SwiGLU
     2x  TransformerBlock
```

Add `--inject-bug` to the engine step to see first-divergence reporting locate the exact element:

```text
First divergence:

  Tensor:    q_embed
  Value id:  47
  Producer:  #40 aten.add.Tensor
  Module:    layers.0.self_attn
  Region:    RoPE@layers.0.self_attn
  Source:    examples/mini_llama/model.py:77 in apply_rotary_pos_emb
  Shape:     [1, 4, 8, 16]
  DType:     float32

Metrics:
  max_abs_error:     2.333312
  max_rel_error:     42.437114
  mean_abs_error:    0.18578845
  rmse:              0.39833667
  cosine_similarity: 0.75503036
  mismatched:        447 / 512

First mismatching element:
  index:     [0, 0, 1, 0]
  reference: -0.52425408
  actual:    0.66629636
```

The producer operator, module path, region and source line are recorded into `testcase.json` at
extraction time, so the report stays actionable even though the engine never saw the model.

## Agent integration

InferRef 0.3 exposes one structured workflow through Python, CLI JSON, and an
optional local MCP stdio server. The CLI can discover its contract and execute a
trusted engine adapter without requiring an Agent to parse human-readable output:

```bash
inferref agent capabilities --json
inferref agent context trace/ --json
inferref agent extract trace/ --region "RoPE@layers.0.self_attn" \
  -o repro/rope --input-names cos,sin,query,key \
  --output-names q_embed,k_embed --json
inferref agent run repro/rope \
  --adapter examples/engine_sim/rope_numpy.adapter.json \
  --runs-dir inferref-runs --json
```

Every response has a versioned envelope with `status`, `data`, `diagnostics`, and
`next_actions`. A numerical mismatch is `fail`; an adapter process or output error
is `error`, so an Agent cannot confuse a kernel bug with a broken test invocation.
Each run receives a unique output directory and writes an `inferref-run.json`
execution record.

Install and start the MCP transport with:

```bash
pip install -e ".[agent]"
inferref-mcp --read-root E:/trusted/workspace --write-root E:/trusted/runs
```

For a local MCP host, configure the `inferref-mcp` executable from this virtual
environment as a stdio server. The tools are `inferref_capabilities`,
`inferref_context`, `inferref_extract_testcase`, `inferref_compare_outputs`, and
`inferref_run_engine`, plus `inferref_run_scenario` for stateful testcase
chains. Read and write roots are host policy, not tool arguments; requests
outside them are rejected. If omitted, both policies default to the server's
current working directory. These path checks prevent accidental and static
lexical/resolve-based escapes; they are not a strong sandbox boundary against a
same-user process racing directory symlinks or Windows reparse points.

Adapter JSON is executable configuration: InferRef uses an argv array and
`shell=False`, but the configured process must still be trusted. The complete
wire contract and threat boundary are in
[InferRef Agent Protocol v0.1](docs/spec/InferRef_Agent_Protocol_v0.1.md).

### Use with a coding agent

The [Agent workflow guide](docs/AGENT_WORKFLOW.md) is the practical loop for a
coding agent (or an engine developer) asked to validate or fix an engine:
discover the protocol, inspect a trace, extract the smallest testcase, run a
trusted adapter or a stateful scenario, and iterate from the first divergence
to a passing rerun. It is self-contained and assumes nothing beyond a clean
checkout.

The same loop ships as an installable Codex skill in
[`skills/inferref/`](skills/inferref/). Install it into the local Codex skill
directory with:

```powershell
Copy-Item -Recurse skills/inferref "$HOME\.codex\skills\inferref"
```

The skill is discovered through the normal Codex skill mechanism and links to
the workflow guide plus the protocol, adapter, and CLI references instead of
duplicating their content.

### Blind Agent repair evaluation

Evaluation v0.2 runs real coding Agents in independent disposable workspaces. A
workspace contains only `TASK.md` and the editable `engine.py`; the MCP proxy
uses `eval://` URIs and keeps reference arrays inside the host evaluator. The
candidate engine receives input-only `.irtensor` files. After the Agent exits,
the host silently reruns the final `engine.py` against the visible case and three
holdouts, so a historical PASS from an earlier patch cannot be combined with a
different final patch.

The Codex driver uses `--ignore-user-config` so user plugins and unrelated MCP
servers cannot enter the evaluation session; authentication remains available.
The audit file is atomically checkpointed with a valid terminal footer after
every tool call, because some Agent CLIs terminate MCP children without a graceful
stdio shutdown. Its validator also enforces the evaluation tool/operation/status
state machine and exact engine-run counter transitions. The footer detects
corruption and torn writes; it is not authentication against a same-user process
that can rewrite the complete stream.

Here, “blind” means candidate-input blind, oracle-isolated, and hidden from the
Agent workspace. It does not mean adversarially secret: holdout metadata is in the
public benchmark, the processes share one OS user, and hashes detect final-state
changes rather than reads or modify-then-restore behavior.

The manual dual-Agent gate uses fresh Codex and Claude sessions and requires both
to pass within four engine runs:

```bash
inferref agent evaluate examples/agent_eval/rope_sign/benchmark.json \
  --agents codex,claude \
  --report-dir .scratch/agent-eval/rope-sign \
  --public-attestation docs/dev/attestations/rope-sign.json \
  --json
```

When Claude Code must use a local provider configuration, pass it only at run
time with `--claude-settings /path/to/settings.json`. A custom provider can also
select its concrete model with `--claude-model MODEL`. The evaluator records model
identity as `requested_only` or `cli_self_reported`, including the evidence source
and whether a reported value matches the request. This is not provider-signed
model identity. The settings path and contents are never copied into the report
or benchmark manifest.

The private report contains bounded CLI output. A formal public attestation can
only be emitted by a fresh `python -I` worker; ordinary Python API runs and
injected drivers produce explicitly labelled development evidence. The worker
resolves each Agent command once, uses that exact command for `--version` and
execution, and hashes every executable component before version detection, after
version detection, and after the Agent exits. On Windows this binds `node.exe` and
`codex.js` separately for Codex, and the actual Claude executable for Claude.

Attestation v0.5 binds repository, runtime, distribution, evaluator source and
benchmark evidence before and after evaluation, plus the Agent command-chain
hashes, a path-free normalized argv policy with a verifiable digest, model
evidence level and request-satisfaction verdict, sealed MCP audit, final patch,
visible/holdout results, transcript hashes, duration, and available usage/cost.
When Claude Code uses `--claude-settings`, the settings file is captured
no-follow before and after the Agent run; the attestation publishes only size,
SHA-256 and an unchanged verdict, never the path or contents. A CLI that
self-reports a model different from the requested one fails the benchmark as an
`identity_policy_failure` and is refused by formal attestation.

The isolated worker's launch evidence is a canonical policy object (`-I -m`,
request transport, request byte SHA-256, Python executable hash) with a
recomputable `launch_policy_sha256`; the parent cross-checks the request hash
against the worker report. Agent argv is split into an instance digest and the
public path-free policy digest so third parties can verify the policy without
private paths or prompts.

It excludes reasoning, credentials, reference payloads, executable absolute
paths, and local settings paths. Full model event streams remain untracked.
Ordinary CI uses a deterministic fake Agent to test this evaluator and produces
development evidence; it does not spend model tokens or claim autonomous repair.
The machine-readable contract lives in
[`benchmark.json`](examples/agent_eval/rope_sign/benchmark.json).

This pilot proves protocol usability, not that InferRef diagnostics improve an
Agent over ordinary tests. That claim requires a controlled full-diagnostics vs
PASS/FAIL-only vs no-InferRef ablation using the same benchmark and models.

## Semantic analysis

Semantic labels are annotation over an authoritative physical trace, never a replacement for it
(SPEC §17). Detection is therefore explicit — either after the fact:

```bash
inferref region detect trace/ --dry-run          # see what it would create
inferref region detect trace/ --min-confidence 1.0   # only certain matches
inferref region detect trace/ --detector source_function
```

or during tracing:

```bash
inferref trace run_model.py -o trace/ --semantic-analysis
```

Three detectors ship today, scored per IR §32:

| Detector | Evidence | Confidence |
| --- | --- | --- |
| `module_type` | `torch.nn.Linear` and other built-ins | 1.00 — deterministic |
| `module_type` | class names like `Qwen3RMSNorm`, `LlamaAttention` | 0.90 — very strong |
| `source_function` | `apply_rotary_pos_emb`, `repeat_kv`, … | 0.95 — very strong |
| `cache_update` | cache source frame plus storage mutation | 0.95 — very strong |
| `cache_update` | cache source frame plus tensor concatenation | 0.90 — strong |

Regions nest and may overlap (IR §36): a `Linear@layers.0.self_attn.q_proj` sits inside
`Attention@layers.0.self_attn` inside `TransformerBlock@layers.0`. `inspect` shows the chain
innermost-first, and reports pick the most specific one:

```text
    #37 aten.neg.default(t43 [1,4,8,8] float32)
        at examples/mini_llama/model.py:68 in rotate_half
        semantic: RoPE(0.95) < Attention(0.90) < TransformerBlock(0.90)
```

Matching uses the whole source stack, so operators from an inlined helper land in their caller's
region — `rotate_half`'s slice/neg/cat operators belong to RoPE, which is what makes the detected
region a clean contiguous slice rather than one with holes.

### KV-cache mutation path

The static-cache example performs a three-token prefill followed by one decode
step, then checks both calls against uncached causal attention:

```bash
python examples/mini_llama/run_kv_cache_trace.py --output trace-kv/
inferref inspect trace-kv/ --operator aten.copy_.default
inferref region list trace-kv/
inferref testcase extract trace-kv/ --region "KVCacheUpdate@cache#1" -o repro/cache-decode
```

The two cache storages remain allocated while their immutable Trace Values
advance from version 0 to 1 (prefill) and 1 to 2 (decode). Semantic analysis
splits the repeated cache-module invocation into `KVCacheUpdate@cache#0` and
`KVCacheUpdate@cache#1`; both mutation regions produce standalone testcase
payloads without filename collisions between storage generations.

The real-model suite also runs Hugging Face's `LlamaForCausalLM` with both
`DynamicCache` and `StaticCache`, entirely from a tiny random config (no model
download). For `StaticCache`, each decoder layer records three writes per phase:
the cumulative length plus key/value `index_copy_` operations. The
`cache_update` detector uses the cache source frame together with those
physical effects, so it can safely recognise the generic HF method named
`update` without matching every `update()` in the program. It requires at least
two mutation/concatenation signals, consistent with key and value handling.
Extracted prefill and decode regions are replayed independently with NumPy and
compared exactly to their `.irtensor` references.

## Tracing your own model

```python
import inferref

with torch.no_grad(), inferref.trace(output="trace/", scope="model.layers.0",
                                     capture_tensors="all") as session:
    session.mark_input("input_ids", input_ids)
    session.mark_output("logits", model(input_ids).logits)
```

Or trace a script you do not control — module paths and `--scope` filtering come from PyTorch's
global module hooks, so the script needs no changes:

```bash
inferref trace run_model.py --scope model.layers.0 -o trace/ -- --batch 4
```

## CLI

| Command | Purpose |
| --- | --- |
| `inferref trace` | Run a script under tracing (SPEC §33) |
| `inferref inspect` | Operators, tensors, module paths, source locations (SPEC §34) |
| `inferref analyze` | Operator/region/payload coverage and signature counts (SPEC §25) |
| `inferref validate` | Check all ten Trace IR invariants (IR §48) |
| `inferref compare` | Testcase-vs-engine or trace-vs-trace, `--first-failure` (SPEC §35) |
| `inferref testcase extract` | Standalone operator or region testcase (SPEC §23) |
| `inferref testcase dedup` | Group executions into unique signatures (SPEC §24) |
| `inferref region detect` | Find semantic regions automatically (SPEC §17) |
| `inferref region create/list/delete` | Reference regions (SPEC §37) |
| `inferref contract list/show/validate` | Built-in and plugin executable contracts (Contract Schema v0.1) |
| `inferref agent capabilities/context` | Versioned discovery and artifact context |
| `inferref agent extract/compare/run` | Structured Agent and engine-adapter loop |
| `inferref scenario validate/run` | Ordered testcase chains with state binding (Scenario v0.1) |
| `inferref export` | Whole trace as one JSON document |

Every command supports `--json` for agent and CI consumption (SPEC §42). `compare` exits non-zero
on failure while still emitting a full JSON report.

## Engine side (no PyTorch, no Python)

`cpp/include/inferref/` is header-only and dependency-free — copy it into your engine tree, or
build the bundled tools:

```bash
# Windows (from a shell with the VS environment initialised)
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build

cpp/build/inferref_selftest            # binary-format self-test
cpp/build/inferref_compare ref.irtensor actual.irtensor
```

`inferref_compare` produces the same metrics as the Python comparator and returns 0 on PASS,
1 on FAIL, so it drops straight into a CI step.

## Layout

| Path | Depends on | Purpose |
| --- | --- | --- |
| `inferref/ir/` | stdlib | Trace IR v0.1 records, package I/O, validation |
| `inferref/tensor/` | numpy | `.irtensor` binary codec |
| `inferref/compare/` | numpy | Metrics, tolerance policy, layout diffing, reports |
| `inferref/testcase/` | numpy | Testcase projection & signature dedup |
| `inferref/region/` | stdlib | Region boundary derivation (IR §34) |
| `inferref/semantic/` | stdlib | Semantic detectors (SPEC §17, §56) |
| `inferref/inspect/` | stdlib | Text views and coverage analysis |
| `inferref/agent/` | stdlib + optional MCP | Agent envelope, engine adapter, MCP transport |
| `inferref/cli/` | stdlib | argparse CLI; imports torch only for `trace` |
| `inferref/frontend/pytorch/` | torch | Dispatcher-level runtime tracer |
| `cpp/include/inferref/` | — | Header-only C++ reader + comparator |

## Notes on the tracing implementation

Findings that shaped the frontend and constrain how it may be modified:

- **`Tensor._version` cannot detect mutation from inside `__torch_dispatch__`.** The version
  counter is bumped by the autograd layer, which sits *above* the dispatch mode, so it reads
  identically before and after the operator runs. Writes are instead derived from the operator
  schema and InferRef maintains its own storage versions (IR §15).
- **Schema writes must be bound by name as well as position.** `aten.add.out` declares its write
  on a `kwarg_only` argument, and `aten._foreach_add_` declares one on a `List[Tensor]`; binding
  by position into `args` alone loses both. Aliased writes are collected into an ordered set so a
  storage advances exactly one generation per operator.
- **`data_ptr` is recycled after a storage is freed.** The storage table is guarded with
  `StorageWeakRef` so two unrelated allocations never alias onto one `storage_id` (IR §14).
- **Tensor capture re-enters the dispatch mode.** `.contiguous()`/`.view()`/`.cpu()` are
  themselves ATen ops, so capture runs under a re-entrancy guard; `mark_input`/`mark_output`
  use the same guard since they touch tensors outside `__torch_dispatch__`.
- **Value identity and payload identity are separate.** A trace value is keyed on
  `runtime_object_id` as well as storage/version/layout, so the graph records what actually ran —
  without it, `y = x.detach()` would steal `x`'s identity from later consumers and fabricate a
  serial dependency. Payloads are keyed on content and layout only, so two distinct values still
  share one `.irtensor` (IR §39).

Payloads are written in canonical logical contiguous order (IR §20). A tensor's recorded `stride`
describes the *reference* tensor's layout for debugging (SPEC §29) and does not describe the
payload — which is why a stride difference is reported but is not a failure by default (SPEC §20).
Pass `--strict-layout` to enforce it.

## Tests

```bash
python -m pytest tests -q
python -m pytest tests/core -q     # no PyTorch required
```

The suite is hermetic — no downloads, no network — and split along the dependency boundary:

| Suite | Requires torch | Covers |
| --- | --- | --- |
| `tests/core` | No | Trace IR, `.irtensor` codec, comparator, validation, semantic detection |
| `tests/frontend` | Yes | Tracing semantics, testcases, regions, CLI, end-to-end |

`tests/core` is verified to run with `import torch` hard-blocked, which is how Trace IR §57
criterion 10 is enforced rather than assumed. All ten acceptance criteria are covered.

The core/frontend CPU matrix runs on CircleCI — core with no torch installed at all, the
frontend against torch 2.1 (the verified floor) and current, one leg against real `transformers`
model code, and the C++ reader on Linux and Windows — see
[.circleci/config.yml](.circleci/config.yml). GitHub Actions hosts the GPU, XPU, and nightly
gates in [.github/workflows/](.github/workflows/).

## License

Apache-2.0
