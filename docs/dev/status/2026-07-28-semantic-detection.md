# InferRef Project Status

> **Updated:** 2026-07-28
> **Version:** 0.2.0 (unreleased)
> **Phase:** SPEC §64 Phase 5 (semantic analysis) delivered ahead of Phase 4
> **Tests:** 250 passing (149 core / 101 frontend) + C++ self-test passing
> **CI:** green on 14 jobs across Linux + Windows, torch 2.1 / current / nightly

Supersedes [2026-07-28-correctness-hardening.md](2026-07-28-correctness-hardening.md).

---

## 0. CI results — the two flagged risks are closed

Both open risks from the previous report have been resolved by measurement.

### The torch floor is now measured, not claimed

The first CI run failed on both `torch min` legs. Root cause was **not** a PyTorch API
incompatibility:

`Tensor.numpy()` crosses PyTorch's NumPy ABI bridge, and torch 2.1 is compiled against NumPy 1.x.
Since `pyproject.toml` declares no upper bound, pip installs NumPy 2.x alongside it and the call
raises `RuntimeError: Numpy is not available`. Every payload in the trace failed.

The failure was invisible, which was the more serious half: capture caught the exception and
returned metadata, so `--capture-tensors all` silently produced a trace that looked complete and
yielded testcases that could not run.

Both are fixed. Byte extraction now reads tensor memory directly through `ctypes` — verified
byte-for-byte on both torch versions across transposed views, offset and strided slices, bfloat16,
bool, empty and scalar tensors — which also removes NumPy from the tracer entirely. That is the
right layering anyway: extracting bytes is the writer's job, parsing them is the reader's. Capture
failures now surface as manifest warnings naming the exception.

A second, smaller finding: torch 2.1 dispatches `aten.detach` **twice** per `.detach()` call where
2.13 dispatches once. The tracer was right — two detaches really did dispatch — so the two
adversarial tests that counted them were made to assert the property that matters instead.

The matrix itself was also wrong: it scheduled py3.12 against torch 2.1, which ships cp38-cp311
wheels only, so those legs failed at install. It is now spelled out explicitly rather than
generated as a cross product, because the supported Python range differs per torch version.

**Verified pairings** (all green):

| Leg | Python | torch | numpy |
| --- | --- | --- | --- |
| min (ubuntu + windows) | 3.10 | 2.1.2+cpu | 2.2.6 |
| current (ubuntu ×2, windows) | 3.12 / 3.13 | 2.13.0+cpu | 2.5.1 |
| nightly (ubuntu) | 3.12 | 2.14.0.dev20260727 | 2.5.1 |

The `min` leg runs torch 2.1.2 against NumPy 2.2.6 — precisely the pairing that broke — so the fix
is proven by the exact case that failed. `torch>=2.1` is now a measurement.

### Detection validated on real Hugging Face code

The class-name rules (`LlamaRMSNorm` → RMSNorm) were assumptions about naming conventions. Run
against an actual `transformers` 5.14.1 Llama they hold:

| transformers class | Detected as |
| --- | --- |
| `LlamaDecoderLayer` | TransformerBlock |
| `LlamaAttention` | Attention |
| `LlamaRMSNorm` | RMSNorm |
| `LlamaMLP` | MLP |
| `LlamaRotaryEmbedding` | RoPE |
| `apply_rotary_pos_emb` (free function) | RoPE, via source stack |

**91.2% semantic coverage** on a model nobody wrote for InferRef, validating cleanly. The 16
unlabelled operators are top-level plumbing — position-id arithmetic and
`find_packed_sequence_indices` — not unrecognised kernels, and `analyze` lists them as the work
list exactly as SPEC §25 intends. A RoPE testcase extracts and is reproducible.

`tests/frontend/test_semantic_hf.py` covers this and skips without the `hf` extra; one CI leg
installs transformers so it stays covered.

---

## 1. What this round delivers

Automatic semantic region detection. The workflow's worst step is gone.

Before:

```bash
inferref inspect trace/ | grep apply_rotary_pos_emb   # hunt for operator ids
inferref region create trace/ --from-op 31 --to-op 48 # type two magic numbers
```

After:

```bash
inferref region detect trace/
#   14x  Linear    5x  RMSNorm    3x  RoPE    2x  Attention
#    2x  RepeatKV  2x  SwiGLU     2x  TransformerBlock
```

The detected `RoPE@layers.0.self_attn` is **exactly** the region the hand-written
`--from-op 31 --to-op 48` produced: same 18 operators, same 4 inputs, same 2 outputs. That
equivalence is asserted by a test, and the operator ids have been deleted from CI entirely.

---

## 2. The measurements that reordered this release

0.2 was planned as `torch.export → static/runtime correlation → semantic detectors`. Measuring the
first two before writing any code inverted that order.

### torch.export op-level correlation is 47.6%

`torch.export.export()` produces a *higher-level* graph than runtime dispatch:

| | static | runtime |
| --- | ---: | ---: |
| `aten.linear.default` | 14 | 0 |
| `aten.t` + `aten.mm` | 0 | 14 + 14 |
| `aten.matmul.default` | 4 | 0 |
| `aten.bmm.default` | 0 | 4 |

`run_decompositions()` does **not** converge — it produces 222 nodes against runtime's 206, and 16
operator kinds still disagree (`permute` vs `_unsafe_view`, `sigmoid` vs `silu`,
`arange.start_step` vs `arange.default`). Using `(module_path, source_line)` as a correlation key
covers **47.6%** of runtime operators.

> Op-level runtime↔static correlation is not a solved problem and cannot be the foundation of
> anything.

### Module-level correlation is 100%

26/26 module paths match; 206/206 runtime operators fall in a shared module. And that granularity
is exactly the 1-semantic-to-N-physical mapping InferRef exists to express (SPEC §20):
`q_proj` is one static `aten.linear.default` against four runtime operators.

### Semantic detection needs neither

`ModuleRecord.type` is already recorded at trace time — `examples.mini_llama.model.RMSNorm`,
`torch.nn.modules.linear.Linear`. So detection was built on runtime metadata alone, with **no
`torch.export` dependency and no IR schema change**: `Annotation`,
`OperatorRecord.annotations`, `RegionRecord.semantic` and the `semantic_pattern` creation method
were all already in the IR. `FORMAT_VERSION` stays `0.1`.

`torch.export` is consequently demoted to an optional module-level enrichment, deferred to 0.3.

---

## 3. Design: stack matching, not innermost-frame matching

The previous report listed "source-function regions can be non-contiguous" as a known limitation:
`apply_rotary_pos_emb` calls `rotate_half`, whose operators carry the *helper's* source location,
so selecting by function name left holes and picked up spurious region inputs.

Matching the **whole source stack** rather than `primary.function` fixes it, because the helper's
operators have the target function as a caller:

```text
ei=30..34  primary_fn=apply_rotary_pos_emb
ei=35..38  primary_fn=rotate_half          <-- matched via its caller
ei=39..47  primary_fn=apply_rotary_pos_emb
--- only gap: 47 -> 127, the layer 0/1 boundary ---
```

18 contiguous operators, split into one region per layer on the gap. The limitation is closed.

---

## 4. What exists

### `inferref/semantic/` — stdlib only

Detection runs on a loaded `TracePackage` and never imports torch, so it sits on the torch-free
side of the dependency split and is covered by `tests/core`.

| File | Purpose |
| --- | --- |
| `base.py` | `Detection` record, `SemanticDetector` protocol (SPEC §56), IR §32 confidence bands |
| `invocations.py` | Split a matched set into per-invocation contiguous runs |
| `module_type.py` | Label modules from their class |
| `source_function.py` | Label operator runs from the source stack |
| `registry.py` | Built-in detector registry (SPEC §55) |
| `run.py` | Orchestration: `detect()` / `apply_detections()` |

Detectors observe; only `run.py` writes. Physical trace truth is never modified (SPEC §68.3),
asserted by tests in both suites.

### Detectors and confidence (IR §32)

| Detector | Evidence | Confidence |
| --- | --- | --- |
| `module_type` | exact `torch.nn` class | 1.00 deterministic |
| `module_type` | class-name pattern (`Qwen3RMSNorm`, `LlamaAttention`) | 0.90 very strong |
| `source_function` | `apply_rotary_pos_emb`, `repeat_kv`, `scaled_dot_product_attention` | 0.95 very strong |

Recognised today: Linear, LayerNorm, RMSNorm, Embedding, Conv1d/2d/3d, GroupNorm, BatchNorm,
SiLU/GELU/ReLU/Softmax/Sigmoid/Tanh, RoPE, SwiGLU, GeGLU, MLP, Attention, GroupedQueryAttention,
TransformerBlock, RepeatKV, CausalMask.

### CLI

- `inferref region detect <trace>` with `--dry-run`, `--min-confidence`, `--detector`, `--replace`,
  `--json`
- `inferref trace ... --semantic-analysis` (SPEC §33), **off by default** — SPEC §17 requires
  semantic analysis to be optional and physical tracing to work without it
- `inferref inspect` shows the nesting chain innermost-first:
  `semantic: RoPE(0.95) < Attention(0.90) < TransformerBlock(0.90)`
- `inferref analyze` reports semantic coverage and, per SPEC §25, lists modules with no label —
  the work list for supporting a new model

### Also fixed this round

Region provenance in reports and testcase manifests picked whichever containing region came first
in the list. With nesting now routine that was arbitrary; both
`inferref/testcase/extract.py` and `inferref/compare/compare.py` now report the **smallest**
containing region, which is the informative one.

---

## 5. Results on the example model

```text
Coverage:
  Source mapping:   100.0%
  Region coverage:  100.0% (206/206 operators)
  Semantic coverage:100.0%
  Payload coverage: 100.0%
```

30 regions from 206 operators, all validating cleanly against IR §48. On a real
transformers Llama the same detectors reach 91.2% (see §0).

---

## 6. Known gaps

Carried forward and still open:

- `inferref export` emits a single JSON document; the `.irtrace` archive of SPEC §38 is not
  implemented.
- No trace sets (SPEC §51).
- Trace-to-trace comparison aligns only by `execution_index`.
- No quantised tensor support (SPEC §28).
- `--max-capture-elements` silently degrades large tensors to hash-only with no warning at trace
  time.
- Tracing overhead is untuned.
- Only one `TraceSession` at a time; module hooks are global.
- Parameters interned before the first forward pass are not classified.
Closed this round:

- ~~Source-function regions can be non-contiguous~~ — fixed by stack matching.
- ~~CI has never been executed~~ — green on 14 jobs; see §0.
- ~~The torch floor is a claim, not a measurement~~ — verified at 2.1.2; see §0.
- ~~Detection is unvalidated on a real HF model~~ — 91.2% coverage on transformers Llama; see §0.
- ~~Capture failures degrade silently~~ — they now surface as manifest warnings.

New this round:

- **No operator-pattern detector.** A model whose modules and helpers are unhelpfully named gets no
  labels. Deliberate: pattern matching over operator sequences has the highest false-positive risk
  and module + source function already covers mini-Llama and HF-style models.
- **No KV-cache detection.** SPEC §64 Phase 5 lists it, but the example model has no KV cache, so
  there is nothing to validate against. Needs the example extended first.
- **`RoPE@rotary` and `RoPE@layers.N.self_attn` are both labelled RoPE** but are different things:
  one builds the cos/sin tables, the other applies them. The vocabulary does not yet distinguish
  them.

---

## 7. Suggested next steps

1. **Extend the example model with a KV cache**, then add the Phase 5 KV-cache detector. This also
   gives the mutation-tracking work from 0.1.1 a realistic test — currently no example exercises a
   cache write end to end.
2. **Validate detection on a second architecture** (Qwen, Mistral, MoE). Llama confirms the
   conventions hold once; a mixture-of-experts model would test them properly.
3. `torch.export` module-level enrichment: symbolic shapes (SPEC §50) and a cross-check on
   semantic labels. Explicitly not op-level correlation.
4. `.irtrace` archive + trace sets (SPEC §38, §51), needed before reference traces can be CI
   artifacts.
5. Viewer, engine adapter protocol, MCP (0.3).

---

## 8. How to verify

```bash
uv pip install -e ".[torch,dev]"
python -m pytest tests -q                       # 250 passed
python -m pytest tests/core -q                  # 149 passed, no torch needed

python examples/mini_llama/run_trace.py --output trace/
inferref region detect trace/
inferref analyze trace/

inferref testcase extract trace/ --region "RoPE@layers.0.self_attn" -o repro/rope \
    --input-names cos,sin,query,key --output-names q_embed,k_embed

python examples/engine_sim/rope_numpy.py repro/rope --output engine-out/
inferref compare repro/rope engine-out/ --first-failure      # PASS, exit 0

python examples/engine_sim/rope_numpy.py repro/rope --output engine-bad/ --inject-bug
inferref compare repro/rope engine-bad/ --first-failure      # FAIL, exit 1
```

No operator ids appear anywhere in that sequence, which is the point.
