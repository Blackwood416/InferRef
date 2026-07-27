# InferRef Project Status

> **Updated:** 2026-07-27
> **Version:** 0.1.0
> **Phase:** MVP complete (SPEC §64 Phases 0-3)
> **Tests:** 148 passing (Python) + C++ self-test passing

---

## 1. Summary

The MVP skeleton defined in [SPEC §62](../../spec/InferRef_SPEC.md) is implemented and the
end-to-end validation loop of SPEC §65 runs clean:

```text
reference trace -> extract testcase -> external implementation -> compare -> PASS/FAIL
```

Both directions of that loop are verified: a correct numpy engine reports PASS, and an engine with
an injected bug reports FAIL localised to the exact element, operator, module, region and source
line.

---

## 2. Phase status (SPEC §64)

| Phase | Goal | Status |
| --- | --- | --- |
| 0 — Prototype | Dispatcher-level tracing produces operator records | **Done** |
| 1 — Stable Trace Package | Trace IR + persistent tensor storage | **Done** |
| 2 — Comparator | `inferref compare`, `inferref testcase extract` | **Done** |
| 3 — Regions | ReferenceRegion, manual creation, region testcases | **Done** |
| 4 — Static Analysis | `torch.export`, decomposition, metadata enrichment | Not started |
| 5 — Semantic Analysis | Automatic RMSNorm / RoPE / SwiGLU detection | Not started |
| 6 — Viewer | Interactive graph and tensor inspection | Not started |
| 7 — Agent Integration | JSON diagnostics, MCP server, CI integration | JSON diagnostics done; rest not started |

---

## 3. Trace IR v0.1 acceptance criteria (IR §57)

All ten are implemented and covered by tests.

| # | Criterion | Test |
| --- | --- | --- |
| 1 | A normal `Linear` | `test_trace_semantics.py::test_linear_is_recorded` |
| 2 | A metadata-only `transpose` | `::test_transpose_is_metadata_only` |
| 3 | Two views sharing one storage | `::test_two_views_share_one_storage` |
| 4 | An in-place tensor mutation | `::test_inplace_mutation_creates_two_values` |
| 5 | A Transformer submodule call stack | `::test_module_stack_is_recorded` |
| 6 | Operator scalar/keyword arguments | `::test_scalar_and_keyword_arguments` |
| 7 | Source mapping | `::test_source_mapping_points_at_model_code` |
| 8 | A standalone operator testcase | `test_testcase_and_regions.py::test_extract_single_operator` |
| 9 | A multi-op RoPE reference region | `::test_single_layer_rope_region_has_four_inputs_two_outputs` |
| 10 | A trace readable without PyTorch | `test_read_without_torch.py` (6 tests) |

Criterion 10 is enforced rather than assumed: a subprocess installs an import hook that makes
`import torch` raise, then loads a trace, decodes payloads, extracts a testcase, runs a comparison
and drives the CLI.

---

## 4. What exists

### Python packages

| Path | LOC | Depends on | Contents |
| --- | ---: | --- | --- |
| `inferref/ir/` | 2260 | stdlib | IR records, value system, package I/O, all 10 §48 invariants |
| `inferref/tensor/` | 448 | numpy | `.irtensor` codec, dtype tables incl. bfloat16 |
| `inferref/compare/` | 1095 | numpy | Metrics, tolerance policy, layout diffing, reports |
| `inferref/testcase/` | 637 | numpy | Testcase projection, signature dedup |
| `inferref/region/` | 268 | stdlib | Boundary derivation (IR §34), region manager |
| `inferref/inspect/` | 493 | stdlib | Text views, coverage analysis |
| `inferref/cli/` | 655 | stdlib | argparse CLI, 8 commands |
| `inferref/frontend/pytorch/` | 1430 | torch | Dispatch mode, identity, modules, sources, params, capture, session, runner |

The dependency split is a hard architectural constraint, not a convention: everything except
`frontend/pytorch/` must remain importable without PyTorch.

### C++ (header-only, no dependencies)

| Path | LOC | Contents |
| --- | ---: | --- |
| `cpp/include/inferref/irtensor.hpp` | 508 | `.irtensor` reader/writer, dtype enum, fp16/bf16 software decode |
| `cpp/include/inferref/compare.hpp` | 262 | SPEC §27 metrics, SPEC §29 layout diff |
| `cpp/examples/compare_tensors/` | 149 | `inferref_compare ref.irtensor actual.irtensor` |
| `cpp/examples/selftest/` | 194 | Binary-format self-test |

### CLI

`trace`, `inspect`, `analyze`, `validate`, `compare`, `testcase {extract,dedup}`,
`region {list,create,delete}`, `export`. All support `--json` (SPEC §42). `compare` exits non-zero
on failure while still emitting a full JSON report.

### Examples

- `examples/mini_llama/` — hermetic RMSNorm + RoPE + GQA + SwiGLU reference model. Used by the
  test suite; no download, no network.
- `examples/engine_sim/rope_numpy.py` — numpy "engine" with an `--inject-bug` switch, used to
  demonstrate and test first-divergence reporting.
- `examples/hf_causal_lm/run_trace.py` — optional, needs the `hf` extra. Defaults to a randomly
  initialised tiny Llama so it still runs offline.

---

## 5. Verified environment

| | |
| --- | --- |
| OS | Windows 11 Pro 26200 |
| Python | 3.13.11 |
| PyTorch | 2.13.0+cpu |
| numpy | 2.4.4 |
| C++ | MSVC 19.50 (VS 18 Community), CMake 4.1.2, Ninja |

**Not yet verified:** Linux / WSL, macOS, CUDA/XPU devices, any PyTorch version other than 2.13.

---

## 6. Implementation decisions worth knowing

These were established by experiment against the installed PyTorch, and constrain how the frontend
may be modified.

### `Tensor._version` cannot detect mutation from inside `__torch_dispatch__`

The version counter is bumped by the autograd layer, which sits *above* the dispatch mode, so it
reads identically before and after the operator runs. Mutation is therefore derived from the
operator schema (`alias_info.is_write`), and InferRef maintains its own `storage_id -> version`
table (IR §15). This also makes invariant 7 (versions never decrease) true by construction.

### `data_ptr` is recycled after a storage is freed

A bare `data_ptr -> storage_id` table would eventually alias two unrelated allocations. The storage
table is guarded with `StorageWeakRef`; an expired entry causes a fresh `storage_id` to be issued
(IR §14).

### Tensor capture re-enters the dispatch mode

`.contiguous()`, `.view()` and `.cpu()` are themselves ATen ops. Capture runs under a thread-local
re-entrancy guard, and `mark_input`/`mark_output` acquire the same guard via `mode.paused()`
because they touch tensors while the mode is active but outside `__torch_dispatch__`.

### Operator outputs always allocate fresh value records

An alias-only operator such as `aten.detach` produces a tensor whose intern key matches its input.
Reusing the value id would give one value two producers and a self-loop in the dataflow graph.
Outputs therefore always allocate, with a per-operator memo so a tensor appearing twice in one
result still maps to a single value.

### Payload dedup keys on the full header, not just content bytes

A tensor and a reshaped view of it have identical canonical payloads but different `.irtensor`
headers, so they cannot share a file. The dedup key covers dtype, shape, stride and storage offset
as well as the content hash.

### Stride differences are reported but do not fail by default

Payloads hold canonical logical contiguous values (IR §20), and SPEC §20 explicitly does not
require an engine to reproduce PyTorch's operator or layout partitioning. A reference tensor that
was a transposed view will legitimately differ in stride from an engine's contiguous output while
being numerically identical — SPEC §29's canonical case. Shape and dtype must still match.
`--strict-layout` enforces stride and storage offset.

**This is a deliberate deviation from the original plan**, which treated stride mismatch as a
failure. That default made every honest engine report a spurious failure.

---

## 7. Known gaps and limitations

### Functional gaps

- **`inferref export` only emits a single JSON document.** The `.irtrace` archive format of
  SPEC §38 is not implemented.
- **No trace sets.** SPEC §51 (`model.irset/` grouping prefill/decode/vision traces) is not
  implemented; coverage analysis operates on one trace at a time.
- **Trace-to-trace comparison is aligned only by `execution_index`.** IR §52's other fallback keys
  (module path, source location, semantic region, shape signature) and heuristic-confidence
  reporting are not implemented. Two traces from different engines with different operator
  partitioning will not align.
- **No quantised tensor support.** SPEC §28 metadata (scale, zero point, group size, packing) is
  absent; only the dtype table's 15 stable types are handled.
- **`--max-capture-elements` degrades large tensors to hash-only**, which silently makes their
  testcases non-reproducible. The testcase manifest flags this, but there is no CLI warning at
  trace time.

### Behavioural notes

- **Source-function regions can be non-contiguous.** Selecting by function name misses operators
  issued by inlined helpers — `apply_rotary_pos_emb` calls `rotate_half`, whose operators carry the
  helper's source location. The resulting node set has holes and therefore extra boundary inputs.
  Use `--from-op/--to-op` for a clean boundary. Covered by
  `test_source_function_region_can_be_non_contiguous`.
- **Region boundary value names are positional by default** (`input_0`, `input_1`, …) unless the
  value is a parameter or buffer, in which case its qualified name is used. Pass
  `--input-names`/`--output-names` for legible testcases.
- **Tracing overhead is untuned.** SPEC §46 favours correctness over tracing speed and the MVP
  takes that literally; no benchmarking has been done, and the source-mapping stack walk runs on
  every operator.
- **Only one trace session may be active at a time**; `TraceSession` is not reentrant and the
  module hooks are global.

### Test coverage gaps

- No test for tracing on a non-CPU device.
- No test for a model with genuinely dynamic control flow.
- No test for concurrent/multi-threaded tracing.
- The C++ side is verified by its own self-test and by cross-checking metrics against Python on one
  tensor pair; there is no automated cross-language test in CI.

---

## 8. Suggested next steps

Roughly in order of value for the stated goal (making engine bring-up faster):

1. **Linux/WSL verification** — build the C++ side with GCC/Clang and run the Python suite there.
   The codec assumes little-endian and uses explicit byte assembly, so this should be mechanical.
2. **Trace a real model end to end** — run `examples/hf_causal_lm/run_trace.py` against an actual
   Qwen/Llama checkpoint and see what breaks at scale: trace size, opaque values (KV cache objects
   are the likely first casualty), and source-mapping overhead.
3. **Semantic detection (Phase 5)** — even a small set of pattern detectors for RMSNorm, RoPE and
   SwiGLU would remove the manual `--from-op/--to-op` step, which is currently the least pleasant
   part of the workflow.
4. **`.irtrace` archive + trace sets** — needed before reference traces can be stored as CI
   artifacts (SPEC §43, §51).
5. **Better trace-to-trace alignment** — required for the "engine trace vs reference trace" use
   case, as opposed to the per-testcase loop that works today.
6. **Static analysis (Phase 4)** — `torch.export` integration, to enrich source metadata and give a
   decomposed view alongside runtime truth.

---

## 9. How to verify the current state

```bash
uv pip install -e ".[torch,dev]"
python -m pytest tests/ -q                      # 148 passed

python examples/mini_llama/run_trace.py --output trace/
inferref analyze trace/
inferref region create trace/ --name RotaryEmbedding --from-op 31 --to-op 48 --semantic RoPE
inferref testcase extract trace/ --region RotaryEmbedding -o repro/rope \
    --input-names cos,sin,query,key --output-names q_embed,k_embed

python examples/engine_sim/rope_numpy.py repro/rope --output engine-out/
inferref compare repro/rope engine-out/ --first-failure     # PASS, exit 0

python examples/engine_sim/rope_numpy.py repro/rope --output engine-bad/ --inject-bug
inferref compare repro/rope engine-bad/ --first-failure     # FAIL, exit 1
```

C++ (Windows, from a shell with the VS environment initialised):

```bash
cmake -S cpp -B cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
cpp/build/inferref_selftest
cpp/build/inferref_compare repro/rope/reference/q_embed.irtensor engine-out/q_embed.irtensor
```
