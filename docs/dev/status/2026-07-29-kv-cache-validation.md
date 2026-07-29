# Status: KV-cache validation and mutation effect edges

**Date:** 2026-07-29  
**Version:** 0.2 development after `0.2.0`  
**Baseline:** commit `1531dbe` (15 CI jobs green)  
**Tests:** 263 collected; full suite passes on local torch current; the eight
KV-cache tests also pass on torch 2.1.2 + NumPy 2.2.6.

## Outcome

InferRef now has a deterministic static KV-cache workload:

```text
three-token prefill       one-token decode
        │                        │
        ├─ key copy_  v0→v1      ├─ key copy_  v1→v2
        └─ value copy_ v0→v1     └─ value copy_ v1→v2
```

The cached outputs match an uncached four-token causal-attention reference to
PyTorch tolerance. The physical trace records four schema-declared mutations,
two storages, and one version increment per storage per cache call.

Semantic detection creates two regions without operator ids:

```text
KVCacheUpdate@cache#0    prefill update
KVCacheUpdate@cache#1    decode update
```

Both regions can be extracted as independently readable testcase packages.

## Bugs exposed by the workload

### 1. Post-mutation base reads had no producer

`copy_` returns its target view, but mutation changes the generation of every
alias over the backing storage. A later read through the full cache tensor was
therefore represented as a new value with `producer=None`, even though its
storage version could only have been created by the preceding mutation.

Consequences:

- graph input inference treated post-write cache state as an external input;
- region boundary derivation added spurious post-state inputs;
- cache-update testcases described the wrong causal contract.

`Graph.derived_links()` now treats `effects.mutated_storages.version_after` as
an effect output. It only fills values lacking an explicit producer, so a later
slice/view keeps its real producing operator. Validation and region boundary
derivation use the same link model.

### 2. Testcase payload names collided across storage generations

Several immutable values over one buffer all inherit a qualified name such as
`cache.key_cache`. Testcase extraction reused that name and overwrote earlier
`.irtensor` files while still reporting `reproducible=true`.

Automatically derived names are now uniquified (`cache_key_cache`,
`cache_key_cache_2`, ...). Explicit duplicate names are rejected as user
errors instead of silently overwriting files. Tests read every emitted payload
back and compare its header shape with `testcase.json`.

### 3. Repeated source-function calls needed stable names

Prefill and decode invoke the same cache module and source function. The source
detector now adds ordinals only when the same semantic and module scope repeat;
different layer paths remain cleanly named without unnecessary suffixes.

## What is tested

- cached prefill/decode numerically equals uncached attention;
- capacity overflow fails before writing;
- four `copy_` operators produce exactly `v0→v1` and `v1→v2` transitions;
- post-generation base values receive mutation effect producers;
- all Trace IR invariants hold;
- module and source-function detectors recognise `KVCacheUpdate`;
- prefill/decode become separate semantic regions;
- both regions have four truthful external inputs;
- testcase payload names are unique and every payload decodes correctly;
- the path passes on torch current and the verified 2.1.2 floor.

## Remaining risks

- The example is a **static contiguous cache**, not paged attention, a dynamic
  growable cache, sliding-window cache, quantized cache, or cache offload.
- Hugging Face cache classes are often plain objects rather than `nn.Module`s,
  and their method may be named only `update`. The current conservative
  detector intentionally does not label every function named `update`; real HF
  cache mutation detection still needs effect-based evidence or a cache-object
  adapter.
- Extracted mutation regions are self-contained, but the NumPy engine simulator
  does not yet execute cache-update testcases and compare their multiple output
  views.
- Mutation effect producers are inferred from `(storage_id, version_after)`.
  This is deterministic and validated, but deserves corpus testing against
  operators that mutate several aliases or return complex containers.

## Reproduction

```bash
python examples/mini_llama/run_kv_cache_trace.py --output trace-kv/
inferref validate trace-kv/
inferref region list trace-kv/
python -m pytest tests/frontend/test_kv_cache.py -q
python -m pytest tests -q
```
