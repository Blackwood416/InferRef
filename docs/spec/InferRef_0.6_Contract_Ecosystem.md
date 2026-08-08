# InferRef 0.6 - Contract Ecosystem

> **Status:** Draft
> **Version:** 0.6.0 milestone
> **Project:** InferRef
> **Document type:** Product & Technical Specification

---

# 1. Summary

InferRef 0.6 makes executable contracts a first-class, pluggable ecosystem.

Today the semantic side already has a clean plugin mechanism
(`inferref.semantic_detectors`), but the contracts that define the engine ABI
(RMSNorm, RoPE, KV-cache) are hardcoded Python validators in
`inferref/testcase/contracts.py`. That is fine for four contracts and a
non-starter for dozens.

0.6 adds:

1. A contract registry reachable through an `inferref.contracts` entry point,
   with a lightweight declarative schema and a Python escape hatch.
2. Scenario: a stateful chain of testcases between the standalone testcase and
   the Suite matrix, with explicit state binding.
3. Documentation fixes: README test counts and CI pointers, plus
   `docs/EXTENDING.md`.

The same milestone also formalizes the roadmap for quantified/compound tensors,
contract-driven fuzzing with failure minimization, corpus building, and engine
debug probes. Those are specified here so later milestones inherit a coherent
design, but only items 1-3 are P0 for 0.6.0.

The milestone acceptance statement is:

```text
pip install inferref-qwen

inferref contract list              -> new GatedDeltaNet / Qwen RoPE / ...
inferref region detect              -> third-party detector
inferref testcase extract           -> third-party contract
inferref suite run                  -> own SYCL engine

without modifying InferRef core.
```

## Detailed specifications

Implementation-level specs for the P0 features:

- [InferRef Contract Schema v0.1](InferRef_Contract_Schema_v0.1.md)
- [InferRef Scenario v0.1](InferRef_Scenario_v0.1.md)
- [0.6 Docs Housekeeping](InferRef_0.6_Docs_Housekeeping.md)

---

# 2. Product Goals

1. External engine developers can add model-specific knowledge (detectors,
   contracts, corpus definitions) without forking or patching InferRef.
2. The engine ABI for one operation is described once, in one place, by the
   party that understands it, and is then consumed by extraction, validation,
   preflight, suite, agents, and fuzzing.
3. InferRef can express stateful execution (prefill -> decode -> decode) as a
   first-class artifact instead of a collection of unrelated testcases.
4. The 0.6 feature set is delivered without weakening the existing
   correctness, safety, and compatibility guarantees.

---

# 3. Non-Goals

0.6 does NOT:

- Build a general-purpose constraint language or universal model IR.
- Make InferRef a numerical reference library (fuzzing reference outputs come
  from an explicit reference source, not from core math).
- Require engines to reproduce ATen operator partitioning (probes are optional).
- Turn scenario execution into an OS sandbox or session daemon.
- Add model knowledge to core. Model knowledge belongs in packs.
- Change the existing Trace IR 0.1 or testcase 0.2 wire formats in a breaking
  way. Additive optional fields only.

---

# 4. Priority Model

| Priority | Feature | Milestone |
| --- | --- | --- |
| P0 | Contract plugin + lightweight schema | 0.6.0 |
| P0 | Scenario artifact and execution | 0.6.0 |
| P0 | README/CI accuracy + `docs/EXTENDING.md` | 0.6.0 |
| P1 | Quantized / compound tensor ABI | 0.7 |
| P1 | Contract-driven fuzzing + failure minimization | 0.7 |
| P1 | Corpus builder / coverage minimizer | 0.7 |
| P2 | Engine debug probe / checkpoint protocol | 0.8 |

XPU corpus expansion is a continuous thread, not a single feature. It consumes
the P0/P1 features in the order described in section 12.

---

# 5. P0 Feature 1: Contract Plugin + Lightweight Contract Schema

## 5.1 Goals

- Introduce `inferref.contracts` entry points with the same discovery quality
  as semantic detectors: duplicate-name detection, built-in shadowing
  rejection, explicit verification, and graceful degradation.
- Let a contract be declared with a small declarative schema covering the 90%
  case (role kinds, shape/dtype relations, effects) and a Python validator for
  the rest.
- Migrate the four built-in contracts onto the same registry path so there is
  one loading mechanism, not two.
- Keep unknown-contract forward compatibility: well-formed unknown IDs in
  existing testcases stay warnings, extraction remains strict.

## 5.2 Non-Goals

- A general constraint DSL with loops, arithmetic on values, or cross-tensor
  numeric semantics.
- Runtime YAML dependency in core (see decision D1).
- Changing the testcase 0.2 manifest, adapter 0.2 capabilities, or contract ID
  versioning rules.

## 5.3 Discovery

New entry-point group: `inferref.contracts`.

An entry point value is a factory returning an iterable of contract
descriptors. A descriptor is either:

- a dict matching the declarative schema (JSON-compatible), or
- a `ContractProfile` object returned by the plugin's own Python code.

Factory semantics mirror `inferref.semantic.base.SemanticDetector` loading:

- entry-point name must be unique across the installation;
- an entry point named after a built-in contract pack is rejected;
- loading must return only valid descriptors;
- discovery errors are surfaced by `inferref contract list --json` and
  `inferref doctor --verify-plugins`, and must not break core operations that
  do not need contracts.

The registry is the single lookup used by:

- `inferref testcase extract --contract`
- standalone testcase validation
- adapter preflight and `contract_capabilities`
- suite validation
- future fuzzing and corpus tooling

## 5.4 Schema

Canonical on-disk format is JSON (`*.contract.json`). A YAML authoring front
end MAY be provided as an optional extra that converts YAML to the canonical
JSON; the runtime never depends on PyYAML. See D1.

```json
{
  "format": "inferref-contract",
  "format_version": "0.1",
  "id": "swiglu/fused/v1",
  "description": "SwiGLU fusion over the last dimension",
  "inputs": {
    "x": {"kind": "tensor"},
    "gate": {"kind": "tensor"}
  },
  "outputs": {
    "y": {"kind": "tensor"}
  },
  "relations": [
    "y.shape == x.shape",
    "y.dtype == x.dtype",
    "gate.shape == x.shape"
  ],
  "features": ["multiple_outputs"],
  "effects": ["pure"]
}
```

### Field rules

| Field | Required | Rules |
| --- | --- | --- |
| `format` | yes | `inferref-contract` |
| `format_version` | yes | `0.1` |
| `id` | yes | versioned path, must be unique in the registry, stable across releases |
| `inputs` | yes | non-empty object; each role has `kind` (`tensor` for v0.1) |
| `outputs` | yes | non-empty object; roles have `kind` |
| `relations` | no | array of safe expressions, all must hold |
| `features` | no | subset of the existing adapter feature vocabulary |
| `effects` | no | controlled vocabulary: `pure`, `alias_effects`, `mutation_effects` |
| `description` | no | human-readable |

Contract input roles are required named roles; outputs are an exact observable
set. This preserves the adapter 0.2 semantics: extra non-contract inputs may be
ignored by an engine, outputs are the full ABI.

### Relation expression language

The expression language is deliberately tiny and side-effect-free. Supported:

- role references: `x`, `y`
- properties: `.shape`, `.dtype`, `.rank`, `.numel`
- index into shape: `x.shape[0]`, `x.shape[-1]`
- `len(x.shape)`
- operators: `==`, `!=`, `and`, `or`, `not`, parentheses

The evaluator parses the expression with an AST walker, never `eval` or
`exec`. A malformed expression is a contract schema error at load time.
Evaluation failure produces a stable issue message:

```text
relation 'y.shape == x.shape' failed (y.shape=[2,3,5], x.shape=[2,3,6])
```

## 5.5 Python escape hatch

For validators the schema cannot express, a descriptor may provide Python
callables with the same signatures as today's `ExecutableContract`:

```python
validate_inputs(inputs) -> list[str]
validate_outputs(outputs) -> list[str]
validate_relation(inputs, outputs) -> list[str]
```

A plugin may also return a `ContractProfile` directly instead of a schema
dict. The Python path is trusted code, exactly like an adapter or a semantic
detector plugin; `doctor --verify-contracts` loads and smoke-tests it.

## 5.6 Built-in migration

The four built-ins become the first built-in contract pack and are registered
through the same loader. Their behavior must not change:

- same contract IDs;
- same role names and output sets;
- same shape/relation/dtype validators;
- same `contract_requirements` role projection;
- same `unsupported` preflight reasons.

The current `EXECUTABLE_CONTRACTS` tuple and `REGISTRY` dict are replaced by
the registry, but `get_contract()` and `contract_boundary_issues()` remain the
stable API surface.

## 5.7 CLI surface

```bash
inferref contract list                 # built-ins + discovered plugins
inferref contract show swiglu/fused/v1
inferref contract validate path/to/contract.json
inferref doctor --verify-plugins       # now also verifies contract entry points
```

All commands support `--json`. `contract list` reports discovery errors
separately from valid contracts.

## 5.8 Acceptance criteria

1. `pip install` of a fixture third-party pack adds contracts to
   `inferref contract list` without core changes.
2. `testcase extract --contract <plugin-id>` binds roles, validates shapes, and
   refuses invalid boundaries exactly like built-ins.
3. Adapter `contract_capabilities` referencing a plugin contract preflights
   correctly and returns `unsupported` before process creation.
4. Built-in behavior is unchanged; the full existing suite stays green.
5. Discovery errors, duplicate names, and built-in shadowing are reported and
   never silently ignored.

---

# 6. P0 Feature 2: Scenario

## 6.1 Positioning

```text
Testcase   one semantic operation, one executable contract

Scenario   an ordered chain of testcases with explicit state binding

Suite      many independent Testcases/Scenarios x many Engines
```

Scenarios exist because the hardest inference-engine bugs live across steps:

```text
prefill -> decode #0 -> decode #1 -> decode #2
```

They are the formal home for:

- Static/Dynamic/Paged KV cache
- sliding-window cache
- speculative fork / rollback
- Mamba recurrent state
- Gated DeltaNet recurrent state
- causal conv state

## 6.2 Scenario manifest v0.1

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

Binding grammar:

- `scenario.inputs.<name>` - scenario-level input tensor
- `state.<name>` - named state slot
- `scenario.outputs.<name>` - scenario-level output tensor

## 6.3 Validation rules

1. Step IDs are unique and match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`.
2. Every referenced testcase passes standalone validation and is reproducible.
3. Every bound role exists in the referenced testcase with a matching kind.
4. A state slot read in step N must be initialized by an earlier step or by
   `state.<slot>.init`.
5. No state slot is written twice within one step.
6. Scenario inputs referenced by bindings must be declared in `inputs`;
   scenario outputs must be written by at least one step.
7. All payload paths remain contained below the scenario directory.

Unknown roles, cycles, duplicate writes, or uninitialized reads are structural
errors. The validator is torch-free and lives in core.

## 6.4 Execution model

The v0.1 executor compiles each step into an effective testcase at run time and
reuses the existing adapter ABI. No new adapter placeholders are required.

For each step:

1. Materialize the step testcase into a fresh run-local directory.
2. For every input binding, copy the bound tensor into the effective
   testcase's `inputs/` and patch the manifest.
3. Invoke the adapter with `{testcase}` = effective directory and `{output}` =
   a fresh step output directory, using the existing timeout/output/artifact
   controls.
4. Validate and compare the step outputs against the step's reference.
5. Apply output bindings: copy bound output tensors into state slots or the
   scenario output area.

Two replay modes:

- `reference-state` (default): each step receives the reference state tensor
  from the slot. This validates every kernel at every step without depending
  on the engine's state continuity. It is the mode used for corpus and
  semantic validation.
- `engine-state`: after a step, the engine's own produced state tensor
  replaces the slot and is fed to the next step. Optionally the engine state is
  compared against the reference state after each step, so a wrong cache write
  is localized to the step that produced it instead of surfacing as a
  downstream numeric failure.

The mode is selected by `--state-mode reference|engine` at scenario run time
and recorded in the scenario run report. Default is `reference`.

## 6.5 Suite and report integration

Suite format moves to 0.2 with an additive `kind` field:

```json
{
  "cases": [
    {"id": "rope-fp16", "kind": "testcase", "testcase": "cases/rope", "tags": ["rope"]},
    {"id": "kv-decode-chain", "kind": "scenario", "testcase": "scenarios/kv-chain", "tags": ["kv", "stateful"]}
  ]
}
```

Readers accept suite 0.1; writers emit 0.2. `kind` defaults to `testcase`.
`suite validate` validates scenarios structurally and for reproducibility;
`suite run` executes them; the run report records per-step status inside the
scenario cell and keeps the existing cell status semantics (pass/fail/
unsupported/partial).

## 6.6 Agent integration

Agent protocol 0.1 gains one operation:

- `run_scenario(testcase, adapter, runs_root, ...)` with MCP tool
  `inferref_run_scenario`.

Response envelope, status semantics, diagnostics, and next_actions are
unchanged. `capabilities()` advertises the new operation.

## 6.7 Acceptance criteria

1. A prefill + two decode KV scenario validates and runs against the native
   SYCL engine in `reference-state` mode, producing a scenario run report.
2. `engine-state` mode feeds the engine's own cache forward and detects a
   deliberately corrupted cache write at the producing step.
3. `suite run` executes a scenario cell and `suite report` renders it.
4. `inferref_mcp` exposes `inferref_run_scenario` with path policies enforced.
5. A Qwen3.5 GDN scenario is the stretch acceptance: it must be expressible
   with zero InferRef core changes once the GDN contract pack exists.

---

# 7. P0 Housekeeping: README and EXTENDING

- Remove hardcoded test counts from README; replace with a CI badge and a
  command that stays correct (`python -m pytest tests -q`).
- Correct CI pointers: CPU matrix now runs on CircleCI; `.github/workflows/`
  holds the GPU/XPU/nightly gates.
- Add `docs/EXTENDING.md` with four short recipes:

```text
How to add:
- semantic detector
- executable contract
- engine adapter
- corpus case
```

Acceptance: README contains no stale test counts or workflow references, and
EXTENDING.md walks a new contributor through one complete plugin example.

---

# 8. P1 Feature 3: Quantized / Compound Tensor

## 8.1 Design principle

Do not overload `dtype`. Keep the logical tensor as the comparison identity and
describe physical encoding as metadata:

```text
Logical Tensor
  float16 [4096, 4096]

  -> encoding ->

Physical Representation
  data:     uint8
  scales:   fp16
  group_size: 64
  codec:    "nf4/block/v1"
```

## 8.2 Shape of the change

Additive metadata only, no breaking change to testcase 0.2 or `.irtensor` v1:

```json
{
  "name": "weight",
  "dtype": "float16",
  "shape": [4096, 4096],
  "payload": "inputs/weight.data.irtensor",
  "encoding": {
    "codec": "nf4/block/v1",
    "group_size": 64,
    "components": {
      "scales": {"dtype": "float16", "payload": "inputs/weight.scales.irtensor"}
    }
  }
}
```

`dtype` and `shape` remain logical. `payload` remains the primary data
component. Readers that do not understand `encoding` can still compare shape
and dtype and can reject rather than misinterpret.

## 8.3 Codec plugins

New entry-point group `inferref.codecs`. A codec provides:

```python
decode(components) -> numpy logical tensor
encode(logical, params) -> components
```

First built-in slice: `nf4/block/v1` and `int8/group/v1`. Additional codecs are
plugins.

## 8.4 Comparator domains

- `logical` (default): decode both sides and run the ordinary numeric metrics.
- `physical`: exact packed-byte comparison.
- `both`: logical comparison plus physical equality as an additional report
  field.

Adapter capabilities gain `codecs`; preflight rejects an unsupported codec
before process creation with an `unsupported` result.

## 8.5 Scope guard

The first slice covers corpus-generated quantized testcases and adapter
validation, not full tracing of quantized models. Quantized tracing is a later
milestone and reuses the same metadata shape.

---

# 9. P1 Feature 4: Contract-Driven Fuzzing + Failure Minimization

## 9.1 Goal

From one contract, derive a bounded family of inputs, run an engine, and shrink
the first failure to a minimal standalone testcase:

```bash
inferref fuzz \
  --contract rope/rotate-half/v1 \
  --adapter myengine.json \
  --reference-adapter ref-numpy.json \
  --seed 42 \
  --budget 500 \
  --output fuzz-run/
```

## 9.2 Critical design decision: reference outputs

Fuzzing needs expected outputs. InferRef is not a numerical library, so the
reference must be explicit. One of the following is required:

- `--reference-adapter`: a second trusted engine adapter (differential
  fuzzing); recommended as the primary mode, because it works for any contract
  with no core math;
- contract schema `reference` entry: a trusted Python function provided by the
  contract plugin for small primitives;

No implicit or hidden reference exists. This is decision D2 and must be
resolved before implementation.

## 9.3 Generation

The contract may declare an optional `fuzz` profile in its schema:

- per-role shape bounds and edge dimension candidates (`1, 2, 7, 127, 128, 129`);
- dtype candidates;
- layout variants: contiguous, transposed view, sliced view, non-contiguous;
- relation-preserving value generation (for example matching `gate` and `x`
  shapes when the relation says so).

Without a profile, generation derives candidates from the relation language
and role kinds conservatively, and refuses to generate when a shape cannot be
satisfied.

## 9.4 Minimization

On the first failing candidate, shrink deterministically:

1. preserve the failure class (`mismatch` vs `error`);
2. prefer reducing dimensions that are not edge-defining;
3. keep the first-divergence signature (output role, element index, probe id
   when available) stable where possible;
4. respect a minimization budget inside the fuzz budget.

The final artifact is a normal standalone testcase whose `origin` records
`{"generator": "fuzz", "contract": "...", "seed": ...}`, plus an
`inferref-fuzz-run` v0.1 report with candidate counts, failure trace, and
minimization history.

---

# 10. P1 Feature 5: Corpus Builder / Coverage Minimizer

```bash
inferref corpus build \
  llama.trace qwen.trace gemma.trace \
  --output llm-core/ \
  --max-cases 100 \
  --seed 20260807
```

## 10.1 Pipeline

1. Load traces and enumerate contract-bindable execution instances (falling
   back to operator signatures for non-contract instances).
2. Cluster by a deterministic signature:

```text
contract
dtype
rank
shape family
layout
effect
semantic variant
```

Shape family uses a documented bucketing policy (for example canonical ratios
and bucketed per-axis sizes), not raw dimensions.
3. Select a minimum coverage set with a deterministic greedy set cover.
4. Verify every selected instance is reproducible; materialize testcases and a
   suite manifest.
5. Emit a coverage report mapping source instances to selected cases and
   reporting dropped counts per signature.

Acceptance: a fixture set of 25k runtime instances is reduced to a documented,
deterministic coverage set, and every selected case passes standalone
validation and suite validation.

---

# 11. P2 Feature 6: Engine Debug Probe / Checkpoint Protocol

## 11.1 Shape

Optional debug extension, never a requirement:

- Extraction gains `--probe-names`: selected trace values are published as
  reference probes in the testcase (additive optional `probes` array).
- An engine may emit `probes/<name>.irtensor` in its output directory.
- Comparison checks contract outputs first; when probes are present on both
  sides it also reports per-probe status and the first internal divergence:

```text
q          PASS
scores     PASS
softmax    FAIL  <- first internal divergence
out        not checked
```

Missing engine probes are skipped, not failed. Probe outputs count against the
existing artifact limits. Probe comparison is available through `compare`,
`run_engine`, scenario steps, and the agent envelope.

Acceptance: a fused contract with an injected internal bug reports the
internal probe divergence without requiring the engine to emit ATen-level
operators, and a probe-less engine run is byte-compatible with today's
behavior.

---

# 12. XPU Corpus Expansion Roadmap

The native SYCL corpus expands in dependency order. Each step exercises a new
architectural capability, not merely a new op:

| Order | Contract / region | New capability exercised | Depends on |
| --- | --- | --- | --- |
| 1 | SwiGLU | stateless multi-input fusion | P0 contract plugin |
| 2 | GQA / Attention | wide multi-output fused region | P0 contract plugin |
| 3 | Gated DeltaNet | recurrent state, cross-step relations | P0 scenario |
| 4 | MoE routing | routing/selection region | P0 scenario + P1 probe |
| 5 | NF4 / INT4 Linear | quantized ABI | P1 compound tensor |

Gated DeltaNet is the architectural stress test: if InferRef expresses a Qwen3.5
GDN testcase and scenario cleanly, the framework has crossed beyond ordinary
Transformer stateless kernels.

---

# 13. Sequencing

```text
0.6.0  contract plugins + schema
       scenario artifact + executor + suite/agent integration
       README + EXTENDING.md
       built-in contracts migrated, all existing tests green

0.7    compound tensor + first codecs
       contract fuzzing + minimization
       corpus builder

0.8    debug probe protocol
       GDN / MoE corpus depth
```

XPU cases for each capability land with the dependency that enables them, not
later.

---

# 14. Risks and Open Decisions

## D1. Schema file format

Recommendation: canonical JSON at runtime; YAML authoring as an optional extra.
Alternative: YAML becomes a core dependency (rejected: violates the numpy-only
core constraint).

## D2. Fuzzing reference source

Recommendation: `--reference-adapter` required for v0.1 fuzzing; contract
plugin Python reference functions optional for small primitives. Alternative:
infer a reference from contract plugins only (limits differential testing).

## D3. Scenario replay default

Recommendation: `reference-state` default, `engine-state` opt-in. Alternative:
default to engine-state for maximal fidelity at the cost of harder attribution.

## D4. Suite versioning

Recommendation: suite 0.2 with additive `kind`, readers accept 0.1.
Alternative: keep suite 0.1 and encode scenarios via tags (weaker validation).

## D5. Adapter ABI for scenarios

Recommendation: compiled effective testcases per step, zero adapter changes.
Alternative: new `{scenario}`/`{state}` placeholders (more engine control,
more surface area).

## D6. Trust model

Contract and codec plugins are trusted executable configuration, like adapters.
`doctor --verify-contracts` must load and smoke-test them. Formal attestation
for plugin execution is out of scope for 0.6.

## Risks

- Registry drift between semantic detectors, contracts, codecs, and packs;
  mitigated by one discovery pattern and `doctor` coverage.
- Scenario state mismatch causing hard-to-attribute failures; mitigated by
  optional per-step state comparison in `engine-state` mode.
- Quantized ABI growth if every codec adds bespoke fields; mitigated by the
  components map and codec registry.
- Fuzzing budget explosions; mitigated by hard budgets at generation and
  minimization.

---

# 15. Milestone Acceptance

Given an installed third-party pack (fixture `inferref-qwen` for CI), the
following must work with no InferRef core changes:

```text
inferref contract list                  -> plugin contracts present
inferref doctor --verify-plugins        -> contracts + detectors verified
inferref region detect                  -> third-party detector active
inferref testcase extract --contract    -> third-party contract bound
inferref suite run                      -> own SYCL engine passes
inferref scenario run                   -> stateful chain passes (fixture)
```

The full existing test suite stays green, and the torch-free core boundary is
unchanged.

---

# 16. Verification

```bash
python -m pytest tests -q
python -m pytest tests/core -q          # torch hard-blocked, unchanged

# P0 acceptance, fixture pack installed
inferref contract list --json
inferref scenario validate scenarios/kv-chain
inferref suite validate suite-v02.json
inferref suite run suite-v02.json --adapter sycl.adapter.json --runs-dir runs/
```
