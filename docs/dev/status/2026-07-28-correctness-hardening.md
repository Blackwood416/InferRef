# InferRef Project Status

> **Updated:** 2026-07-28
> **Version:** 0.1.1 (unreleased)
> **Phase:** MVP complete (SPEC §64 Phases 0-3) + correctness hardening
> **Tests:** 167 passing (100 core / 67 frontend) + C++ self-test passing

Supersedes [2026-07-27-mvp-complete.md](2026-07-27-mvp-complete.md). Read that one for the
original MVP scope and architecture; this report covers what changed since.

---

## 1. What this round was

A design review raised two correctness issues in the PyTorch frontend. Both were reproduced
against the installed PyTorch before any code changed, and both were real. This round fixes them,
adds the tied-weight metadata that falls out of the second fix, and puts the whole thing under CI.

No new features. The MVP surface is unchanged.

---

## 2. Fixed: mutation binding only handled positional tensor arguments

### What was wrong

Writes were bound to runtime arguments by position into `args`, and required the bound value to be
a `Tensor`:

```python
for position in mutated_arg_positions(func):
    if position >= len(args):
        continue
    target = args[position]
    if not isinstance(target, torch.Tensor):
        continue
```

Three cases escaped, all confirmed by experiment:

| Case | Example | Why it escaped |
| --- | --- | --- |
| Keyword-only write | `aten.add.out` | Declares its write on `out`, which is `kwarg_only`; schema index 3 with `len(args) == 2`, so `position >= len(args)` skipped it |
| Container write | `aten._foreach_add_.Scalar` | Declares its write on `self`, typed `List[Tensor]`; the `isinstance` check rejected the list |
| Positional passed by name | `foo(self=x)` | Never looked at `kwargs` |

The first is not exotic — every `torch.add(a, b, out=c)` style call silently lost its mutation
record, meaning no storage version bump and no `effects.mutated_storages` entry.

A fourth, subtler issue: `written_storages` was a plain list, so two writable arguments aliasing
one storage would bump it twice in a single operator.

### What it is now

`iter_written_tensors(func, args, kwargs)` binds each schema-declared write to its runtime
argument by position *or* name, honouring `kwarg_only`, then walks nested lists, tuples and dicts
via `iter_tensors`. Storages are collected into an ordered set, so each advances exactly one
generation per operator.

Arguments left at their default are absent and correctly skipped.

### Covered by

`tests/frontend/test_trace_adversarial.py` — `test_keyword_only_write_is_detected`,
`test_container_write_is_detected`, `test_aliased_writes_bump_a_storage_once`,
`test_iter_written_tensors_binds_positional_arg_given_by_name`,
`test_iter_written_tensors_skips_defaulted_arguments`, `test_iter_tensors_walks_nested_containers`,
`test_pure_operator_records_no_mutation`.

---

## 3. Fixed: value identity was conflated with payload identity

### What was wrong

`value_key()` was `(storage_id, storage_version, dtype, shape, stride, storage_offset)` — no
`runtime_object_id`. Two different tensor objects over one storage at the same version and layout
therefore collapsed onto a single trace value.

Combined with output interning, which rebinds `_value_by_key[key]` to the freshly allocated output,
this rewrote lineage. Reproduced:

```python
y = x.detach()      # new object, same storage / version / layout
z = x + 1
```

```text
before:  #0 aten.detach.default  in=[1] out=[2]
         #1 aten.add.Tensor      in=[2] out=[3]      <-- add consumes detach's output
```

`add` recorded value 2 as its input, when at runtime it consumed `x` (value 1). The graph asserted
a serial `x -> detach -> add` dependency where the truth was two independent uses of `x`.

Numerically harmless — the bytes are the same — but wrong for anything that reasons about the
graph: region boundary derivation, causal analysis in first-divergence, and any consumer count.

### What it is now

Two explicitly separate keys, matching the IR's three-layer object/storage/value model:

| Key | Fields | Purpose |
| --- | --- | --- |
| `Identity.value_key` | `runtime_object_id`, `storage_id`, `storage_version`, dtype, shape, stride, offset | Trace value identity — the runtime graph is true |
| `Identity.payload_key` | content digest, dtype, shape, stride, offset | Payload deduplication — one file per distinct bytes+layout |

So two trace values can share one `.irtensor`:

```text
Value X ─┐
         ├─→ payload P
Value Y ─┘
```

Verified after the change:

```text
after:   #0 aten.detach.default  in=[1] out=[2]
         #1 aten.add.Tensor      in=[1] out=[3]      <-- correct
```

Parameter deduplication is unaffected: a parameter is the *same* object on every use, so it still
interns to one value. A `Linear` called twice yields 3 weight value records (the parameter plus one
transposed view per call — genuinely 3 distinct objects) sharing 2 payload files.

### Covered by

`test_lineage_is_not_rewritten_by_an_alias_only_operator`,
`test_distinct_objects_over_one_storage_are_distinct_values`,
`test_distinct_values_share_one_payload`, `test_repeated_use_of_one_object_interns_once`.

---

## 4. Added: tied weight metadata

`ParameterIndex` kept one canonical name per storage, so a tied embedding lost the fact that
`lm_head.weight` and `model.embed_tokens.weight` are one allocation.

Two changes:

- **`named_parameters(remove_duplicate=False)`.** The default deduplicates shared tensors, so a
  tied `lm_head` never reported its own name at all — the alias was invisible, not merely
  unrecorded.
- **`TensorValueRecord.qualified_names`**, a new optional field listing every name bound to the
  storage. Present only when there is more than one, so older readers are unaffected (IR §46).
  `qualified_name` remains the canonical single name.

Canonical-name selection is now first-registration-wins in both the object and storage tables; it
was last-wins in one and first-wins in the other, which gave a tied weight different canonical
names depending on whether it was reached directly or through a view.

Covered by `test_tied_weights_record_every_name`, `test_untied_parameters_have_no_alias_list`,
`test_tied_weight_names_survive_a_roundtrip`.

---

## 5. Added: CI, split along the dependency boundary

The project's central claim is that **the PyTorch frontend may break; the Trace IR may not.** The
test layout and CI now encode that.

```text
tests/
├── conftest.py          # no torch import at all
├── core/                # 100 tests — IR, codec, comparator, validation
│   └── conftest-free
└── frontend/            # 67 tests — tracing, testcases, regions, e2e, CLI
    └── conftest.py      # importorskip("torch")
```

`tests/core` is verified to run with torch hard-blocked via an import hook, not merely uninstalled.

### `.github/workflows/ci.yml`

| Job | Matrix | Notes |
| --- | --- | --- |
| `core` | {ubuntu, windows} × py{3.10, 3.12, 3.13} | Installs **no torch**; asserts `find_spec("torch") is None` before running |
| `frontend` | {ubuntu, windows} × py3.12 × torch{min, current}, plus py3.10/min and py3.13/current on ubuntu, plus nightly (non-blocking) | Probes the private APIs, runs the full suite, then runs the documented end-to-end loop **in both directions** |
| `cpp` | {ubuntu, windows} | Builds the header-only reader, runs its self-test, then checks the C++ and Python comparators agree on the same tensors |

The end-to-end steps assert the negative case explicitly: if `inferref compare` reports PASS for a
deliberately broken engine, the job fails. A comparator that never fails proves nothing.

Operator ids are no longer hardcoded in CI — `.github/scripts/rope_range.py` derives layer 0's RoPE
range from the trace, so editing the example model cannot silently break the workflow. Verified
locally to produce the same range (31..48) that was previously hardcoded.

### `.github/workflows/nightly.yml`

Weekly scheduled run against torch nightly. Separate from the PR matrix because a
`continue-on-error` job is one nobody looks at; a scheduled run shows red in the Actions tab.

### Not yet verified

**Neither workflow has been executed.** They are syntactically valid and the steps were rehearsed
locally on Windows, but the Linux legs, the `torch==2.1.*` floor, and the nightly leg are all
untested. Expect the first run to need adjustment.

The declared floor of `torch>=2.1` is a *claim*, not a measurement — the tracer depends on
`torch.utils._python_dispatch.TorchDispatchMode` and
`torch.multiprocessing.reductions.StorageWeakRef`, both private, plus
`named_parameters(remove_duplicate=...)` which this round newly requires. The `min` CI leg exists
to establish the real floor. No compatibility shim was added, deliberately: silent degradation on
an old torch would be worse than a clear failure.

---

## 6. Current state

### Tests

| Suite | Count | Requires torch |
| --- | ---: | --- |
| `tests/core` | 100 | No — verified with imports blocked |
| `tests/frontend` | 67 | Yes |
| C++ self-test | 30 checks | No |

All ten Trace IR §57 acceptance criteria remain covered; see the previous report for the mapping.

### Verified environment

Unchanged from the previous report: Windows 11, Python 3.13.11, torch 2.13.0+cpu, numpy 2.4.4,
MSVC 19.50. Linux/WSL, macOS and non-CPU devices remain unverified.

---

## 7. Known gaps

Carried forward from the previous report and still open:

- `inferref export` emits a single JSON document; the `.irtrace` archive of SPEC §38 is not
  implemented.
- No trace sets (SPEC §51).
- Trace-to-trace comparison aligns only by `execution_index`; IR §52's other fallback keys and
  heuristic-confidence reporting are absent.
- No quantised tensor support (SPEC §28).
- `--max-capture-elements` silently degrades large tensors to hash-only, making their testcases
  non-reproducible, with no warning at trace time.
- Source-function regions can be non-contiguous when a helper is inlined; use `--from-op/--to-op`.
- Tracing overhead is untuned; the source-mapping stack walk runs on every operator.
- Only one `TraceSession` may be active at a time; module hooks are global.

New this round:

- **Parameters interned before the first forward pass are not classified.** `ParameterIndex` is
  populated when the first root module is entered, so a tensor passed to `mark_input` beforehand
  gets `role="input"` and no `qualified_name`. Low impact — the tensors marked up front are usually
  activations, not parameters — but it is an ordering dependency, not a design.
- **CI is unexecuted.** See §5.
- `qualified_names` is populated by the PyTorch frontend but nothing consumes it yet: `inspect`
  does not display it and testcase extraction does not surface it.

---

## 8. Suggested next steps

The review that prompted this round proposed the following ordering, which supersedes the previous
report's list:

**0.1.1 (this round, plus)**
1. Run the CI and fix whatever the Linux and `torch==2.1` legs surface.
2. Surface `qualified_names` in `inspect` and in testcase manifests.

**0.2**
3. `torch.export` enrichment and runtime/static graph correlation (Phase 4).
4. Semantic detectors (Phase 5) — removes the manual `--from-op/--to-op` step, currently the least
   pleasant part of the workflow.
5. Smarter region creation building on the above.

**0.3**
6. Viewer (Phase 6).
7. Engine adapter protocol.
8. Agent integration / MCP (Phase 7).
9. Multi-model corpus (SPEC §45).

Deliberately *not* prioritised: semantic detection was the previous report's #3, but the review
argued correctness and compatibility infrastructure should land first. That is the right call — a
semantic detector built on a graph with rewritten lineage would have encoded the bug.

---

## 9. How to verify the current state

```bash
uv pip install -e ".[torch,dev]"

python -m pytest tests -q                       # 167 passed
python -m pytest tests/core -q                  # 100 passed, no torch needed

# The core suite really is torch-free:
python -c "
import sys
class B:
    def find_spec(s,n,p=None,t=None):
        if n.split('.')[0] in ('torch','torchvision','functorch'):
            raise ImportError('blocked')
for n in list(sys.modules):
    if n.split('.')[0] in ('torch','torchvision','functorch'): del sys.modules[n]
sys.meta_path.insert(0, B())
import pytest; raise SystemExit(pytest.main(['tests/core','-q']))
"

# End-to-end, with the operator range derived rather than hardcoded:
python examples/mini_llama/run_trace.py --output trace/
eval "$(python .github/scripts/rope_range.py trace/)"
inferref region create trace/ --name RotaryEmbedding \
    --from-op "$ROPE_FROM" --to-op "$ROPE_TO" --semantic RoPE
inferref testcase extract trace/ --region RotaryEmbedding -o repro/rope \
    --input-names cos,sin,query,key --output-names q_embed,k_embed

python examples/engine_sim/rope_numpy.py repro/rope --output engine-out/
inferref compare repro/rope engine-out/ --first-failure      # PASS, exit 0

python examples/engine_sim/rope_numpy.py repro/rope --output engine-bad/ --inject-bug
inferref compare repro/rope engine-bad/ --first-failure      # FAIL, exit 1
```
