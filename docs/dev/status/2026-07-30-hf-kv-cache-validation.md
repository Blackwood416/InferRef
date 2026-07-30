# Status: Real Hugging Face KV-cache validation

**Date:** 2026-07-30

**Version:** 0.2 development after `0.2.0`

**Baseline:** commit `ff66486` (semantic and capture hardening)

**Verified environment:** Windows, Python 3.13.11, PyTorch 2.13.0+cpu,
Transformers 5.14.1

## Outcome

InferRef now executes a real Hugging Face `LlamaForCausalLM` through a
three-token prefill and one-token decode using both `DynamicCache` and
`StaticCache`. The model is built from a tiny random `LlamaConfig`, so the test
uses upstream model/cache code but remains hermetic and downloads no weights or
tokenizer.

Both cached paths match a four-token uncached causal reference with maximum
absolute error below `1e-5` (observed approximately `8.94e-08`). Both return the
same cache object supplied by the caller and report a final sequence length of
four.

## Physical StaticCache result

The two-layer model produces twelve schema-declared mutations:

```text
                         prefill       decode
layer 0 cumulative len   v0 -> v1      v1 -> v2
layer 0 key cache        v0 -> v1      v1 -> v2
layer 0 value cache      v0 -> v1      v1 -> v2
layer 1 cumulative len   v0 -> v1      v1 -> v2
layer 1 key cache        v0 -> v1      v1 -> v2
layer 1 value cache      v0 -> v1      v1 -> v2
```

The concrete operators are four `aten.add_.Tensor` calls and eight
`aten.index_copy_.default` calls. Every write has exactly one storage effect,
and each of the six storages advances once per phase.

## Cache semantic detector

HF cache classes are plain Python objects and the method is named only
`update`, so module detection and global source-function matching cannot label
them safely. A new stdlib-only `CacheUpdateDetector` requires two signals:

1. a source frame whose basename denotes cache implementation code; and
2. at least two physical cache-like signals in the invocation: storage
   mutations or tensor concatenations, consistent with key/value handling.

Static mutation runs receive confidence `0.95`; dynamic concat runs receive
`0.90`. The detector creates four static regions, one per decoder layer and
phase:

```text
KVCacheUpdate@model.layers.0.self_attn#0
KVCacheUpdate@model.layers.0.self_attn#1
KVCacheUpdate@model.layers.1.self_attn#0
KVCacheUpdate@model.layers.1.self_attn#1
```

All four regions satisfy Trace IR invariants. Negative core tests prove that a
mutating function named `update` in `model.py`, and a cache-named file with only
one weak mutation signal, are not labelled.

## Testcase quality

All four static-cache regions extract with full input and output payloads. The
test does more than check file presence: a small framework-independent NumPy
executor replays the recorded `zeros`, `arange`, `add`, `add_` and
`index_copy_` sequence from `testcase.json`, then compares all three outputs
exactly against the extracted `.irtensor` references.

## Verification

```text
real HF KV-cache suite                         5 passed
core cache detector tests                     4 added
full suite                                  277 passed
core suite                                  161 passed
PyTorch 2.1 full suite                       267 passed, 2 HF modules skipped
```

The first GitHub-hosted run passed the dedicated current-Transformers leg; other
frontend legs correctly skipped the two optional HF modules. It exposed one
unrelated Windows runner change: pip 26.2 refuses `pip install --upgrade pip`
and requires `python -m pip install --upgrade pip`. The C++ cross-language job
now uses the module form consistently; a second matrix run verifies that
maintenance fix.

## Remaining risks

- This is real upstream model/cache code but a tiny random Llama, not downloaded
  pretrained weights or a tokenizer-driven generation loop.
- Only CPU, batch size one, contiguous static cache, and eager prefill/decode
  are covered. Sliding-window, hybrid, quantized, offloaded and paged caches are
  still untested.
- `DynamicCache` is numerically tested on the real model; its concat-based
  semantic path is tested in the framework-independent detector suite rather
  than a second full trace, to keep CI cost bounded.
- Cache source recognition depends on a cache-named source-file basename. This
  is intentionally conservative and may miss projects that hide cache updates
  in generically named files.
- The NumPy replay supports the physical StaticCache operator set observed in
  the supported Transformers version. An upstream implementation change will
  fail explicitly and becomes a prompt to extend the adapter, not a silent
  pass.

## Reproduction

```bash
python -m pytest tests/frontend/test_hf_kv_cache.py -q
python -m pytest tests/core/test_semantic.py -q
python -m pytest tests -q
```
