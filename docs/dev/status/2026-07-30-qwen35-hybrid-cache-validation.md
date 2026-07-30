# Status: Qwen3.5 hybrid-cache validation

**Date:** 2026-07-30

**Version:** 0.2 development after `0.2.0`

**Baseline:** commit `f248128` (green Windows/Linux compatibility matrix)

**Verified environment:** Windows, Python 3.13.11, PyTorch 2.13.0+cpu,
Transformers 5.14.1, eager attention, pure-PyTorch Gated DeltaNet fallback

## Outcome

InferRef now has model-level coverage for the hybrid cache used by
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B). The official
852,985,920-parameter checkpoint was loaded locally in FP32 and executed through
a four-token prefill followed by one-token decode. No vision inputs were supplied,
so the test exercised the checkpoint's real text backbone while retaining the
official tied language-model head and token embedding.

The model has 24 decoder layers: 18 Gated DeltaNet linear-attention layers and 6
full-attention layers. Its Transformers `DynamicCache` therefore contains two
different physical cache forms:

```text
linear attention (18 layers)
  conv state       [1, 6144, 4]
  recurrent state  [1, 16, 128, 128]

full attention (6 layers)
  keys              [1, 2, sequence, 256]
  values            [1, 2, sequence, 256]
```

FP32 cached decode matches an uncached five-token reference with maximum absolute
error `1.1444091796875e-05`; top-5 token ids and ordering are identical. BF16 was
also exercised: it retained the same top prediction, but prefill/decode's different
fallback operator paths produced maximum absolute logit error `0.125` and mean
absolute error approximately `0.01943`. The regression therefore uses FP32 and does
not encode an unjustifiably strict BF16 equality contract.

## Trace results

Two focused metadata-only traces keep the test diagnostic without retaining model
activations or weights:

| Scope | Operators | Values | Semantic regions | Package size | IR errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model.language_model.layers.0` (DeltaNet) | 1,120 | 1,209 | 30 | 2.406 MiB | 0 |
| `model.language_model.layers.3` (full attention) | 209 | 238 | 35 | 0.657 MiB | 0 |

The full-attention trace contains two key/value concatenations in prefill and two
in decode. The existing detector correctly produces two `KVCacheUpdate` detections.

The pure-PyTorch DeltaNet fallback is intentionally much more verbose. The prefill
expands its recurrent rule into scalar/tensor loops, producing 69 physical mutation
operators in the focused trace. Four mutations are persistent cache state:

```text
prefill  update_conv_state             conv storage       v0 -> v1
prefill  update_recurrent_state        recurrent storage  v0 -> v1
decode   torch_causal_conv1d_update     conv storage       v1 -> v2
decode   update_recurrent_state        recurrent storage  v1 -> v2
```

The remaining writes are algorithm-internal temporary state and remain visible as
physical IR effects. They are not relabelled as cache updates.

## Semantic correction

The real trace exposed a P1 omission in `CacheUpdateDetector`: it recognised only
generic `update()` calls and therefore missed every persistent linear-attention
state write. Detection now distinguishes:

- `KVCacheUpdate` for static/dynamic key-value history;
- `StateCacheUpdate` for recurrent and convolution state.

Named `update_conv_state()` and `update_recurrent_state()` helpers still require a
cache-source filename and a physical mutation. A direct
`torch_causal_conv1d_update()` is accepted outside `cache_utils.py` only when its
trace contains a real storage mutation. Generic `update()` retains the stronger
two-signal threshold, so one unrelated cache-file write remains rejected.

## Regression strategy

Normal CI must not download a 1.75 GB checkpoint. Two levels of coverage are now
provided:

1. `test_hf_hybrid_cache.py` builds a 24K-parameter, two-layer Qwen3.5 from config.
   It runs the real Transformers implementation and asserts numerical equivalence,
   cache-layer types, storage-version transitions, both semantic labels, and IR
   validation. This test runs whenever the installed Transformers exposes Qwen3.5.
2. `test_qwen35_08b.py` is opt-in via `INFERREF_QWEN35_MODEL`. It loads local
   official weights only (`local_files_only=True`), verifies parameter count, the
   18/6 layer split, tied weights, FP32 cache equivalence, both focused traces, all
   six cache detections, and both IR packages.

The tested local checkpoint is a ModelScope git/LFS clone of the official model.
Its weight object SHA-256 is
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`; its config is
semantically identical to Hugging Face revision
`2fc06364715b967f1860aea9cf38778875588b17`.

## Verification

```text
core semantic + tiny hybrid cache             66 passed
official Qwen3.5-0.8B opt-in integration       1 passed
full default suite                           284 passed, 1 skipped
```

Reproduction on Windows:

```powershell
python -m pytest tests/core/test_semantic.py tests/frontend/test_hf_hybrid_cache.py -q

$env:INFERREF_QWEN35_MODEL = "E:\RiderProjects\Aila\models\Qwen3.5-0.8B"
python -m pytest tests/frontend/test_qwen35_08b.py -q -rs

Remove-Item Env:INFERREF_QWEN35_MODEL
python -m pytest tests -q
```

## Remaining risks

- The official checkpoint run is CPU/eager only. CUDA fused DeltaNet,
  `causal-conv1d`, flash-linear-attention, vLLM paged cache, batching, and chunked
  prefill remain unverified.
- Metadata-only focused traces validate dataflow and mutation identity but cannot
  produce numerically replayable DeltaNet testcases. A selective full-capture
  policy for state boundaries is still needed before kernel extraction is useful.
- A single semantic `StateCacheUpdate` currently denotes both conv and recurrent
  state writes. If engine adapters need separate contracts, the schema should gain
  a cache-state subtype instead of inferring it from evidence strings.
- BF16 cached and uncached paths are semantically consistent but not elementwise
  close at the FP32 threshold. Cross-engine comparisons need dtype-aware tolerances.
- vLLM source support was inspected in `vllm/model_executor/models/qwen3_5.py`, but
  vLLM execution was not attempted on this CPU-only Windows environment.

## Suggested next step

Add CUDA coverage for the fused linear-attention path before broadening model
corpus work. In parallel, define a selective state-boundary capture policy so a
`StateCacheUpdate` region can become a reproducible engine testcase without storing
the entire 1,120-op fallback trace.
