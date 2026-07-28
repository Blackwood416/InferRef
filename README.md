# InferRef

Reference execution tracing, testcase extraction, and numerical comparison for inference engine
development.

InferRef turns a PyTorch model execution into a **stable, machine-readable reference specification**
that a custom inference engine (CUDA / SYCL / ROCm / CPU) can be validated against — without the
engine side needing PyTorch, or Python at all.

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

# 3. Carve out a semantic region and extract a standalone testcase
inferref region create trace/ --name RotaryEmbedding \
    --from-op 31 --to-op 48 --semantic RoPE
inferref testcase extract trace/ --region RotaryEmbedding -o repro/rope \
    --input-names cos,sin,query,key --output-names q_embed,k_embed

# 4. Run your engine against the testcase, then compare
python examples/engine_sim/rope_numpy.py repro/rope --output engine-out/
inferref compare repro/rope engine-out/ --first-failure
```

Add `--inject-bug` to the engine step to see first-divergence reporting locate the exact element:

```text
First divergence:

  Tensor:    q_embed
  Value id:  47
  Producer:  #40 aten.add.Tensor
  Module:    layers.0.self_attn
  Region:    RotaryEmbedding
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
| `inferref region create/list/delete` | Reference regions (SPEC §37) |
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
| `inferref/inspect/` | stdlib | Text views and coverage analysis |
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
python -m pytest tests -q          # 167 tests
python -m pytest tests/core -q     # 100 tests, no PyTorch required
```

The suite is hermetic — no downloads, no network — and split along the dependency boundary:

| Suite | Requires torch | Covers |
| --- | --- | --- |
| `tests/core` | No | Trace IR, `.irtensor` codec, comparator, validation |
| `tests/frontend` | Yes | Tracing semantics, testcases, regions, CLI, end-to-end |

`tests/core` is verified to run with `import torch` hard-blocked, which is how Trace IR §57
criterion 10 is enforced rather than assumed. All ten acceptance criteria are covered.

CI runs the core suite with no torch installed at all, the frontend against several torch
versions, and the C++ reader on Linux and Windows — see [.github/workflows/ci.yml](.github/workflows/ci.yml).

## License

Apache-2.0
