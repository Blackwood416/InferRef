# Status: Testcase and effect hardening

**Date:** 2026-07-30

**Version:** 0.2 development after `0.2.0`

**Baseline:** commit `d9868c4` (foundation reviewer hardening)

**Verified environment:** Windows, Python 3.13.11, PyTorch 2.13.0+cpu,
Transformers 5.14.1

## Scope

This is the final CPU-only foundation pass before beginning 0.3 work. It follows
the previous storage-identity and rebinding fixes into the two consumers that
must preserve those facts: IR validation and standalone testcase projection.
CUDA remains deliberately deferred because the development machine has no CUDA
device.

An independent read-only reviewer reported no P0 issues. Its reliable findings
covered metadata-only writes, mutation testcase projection, effect validation,
payload-validator/codec disagreement, and several codec/package edge cases. All
confirmed findings were reproduced and addressed in this pass rather than merely
recorded as future work.

## Correctness fixes

### Mutation-only operator testcases

Operator extraction previously projected only the Python result. An operator
such as `aten._foreach_add_.Scalar` has a writable `Tensor[]` argument but
returns `None`; even when post-mutation tensor states existed in the trace, the
generated testcase declared no outputs and could incorrectly claim to be
reproducible.

Operator extraction now uses all values physically produced by the operator,
including storage-generation effect outputs. A traced tensor-list mutation is
covered end to end: two pre-state inputs are loaded, the operation is replayed,
and both exported post-state references equal `input + scalar`.

If an externally supplied storage is mutated but the trace contains no captured
post-state value, extraction records an `unobservable_mutations` entry and marks
the testcase non-reproducible. It no longer emits a vacuous testcase that cannot
validate the side effect.

### Effect metadata in testcase manifests

Each projected operator now carries its mutation and alias effects in
`testcase.json`. Engine adapters and agents can therefore distinguish an
ordinary tensor result from a state transition without reopening the complete
trace.

The projection also contains a compact `values` table for every tensor referenced
by its boundary or nodes. This is required for a write-through-view followed by a
read-through-base: the post-mutation base value is produced by a storage effect,
not by the previous operator's Python result, so an adapter needs its layout,
storage id, and storage version to materialise the internal alias.

### Alias and record validation

Validation now rejects:

- duplicate operator, value, module, source, region, or storage ids before
  reindexing can hide them;
- duplicate storage mutations or alias effects within one operator;
- mutation effects disconnected from every tensor input;
- unknown alias relationship names;
- alias endpoints that are not an input/output of the declaring operator;
- `same_object` claims with conflicting runtime object ids;
- `shared_storage` or `view` claims whose values do not share a storage id.

These checks remain additive under the existing ten Trace IR invariants and do
not require PyTorch.

Mutation transitions must now start at the observed high-water version and advance
exactly one generation. A malformed `v7 -> v9` effect can no longer manufacture a
disconnected future storage state.

### Object metadata versus storage mutation

Schema-declared writes do not always change storage bytes. Operators such as
`transpose_`, `squeeze_`, `unsqueeze_`, `as_strided_`, same-allocation `set_`, and
`resize_` change Tensor object metadata. Treating them as storage writes advanced
every alias to a fictitious new content generation and assigned false producers.

The frontend now distinguishes this bounded set of metadata-only writes. Their
output remains a fresh immutable TraceValue with the new layout and a same-object
alias, while other objects sharing the storage retain their existing version and
lineage.

### Binary codec/validator agreement

Payload validation now checks the complete `.irtensor` metadata contract rather
than only byte count and shape: format version, header size, dtype code, canonical
flag, reserved field, logical element count, stride, and storage offset must agree
with the TensorValue record.

The codec rejects negative dimensions and shape/numel disagreement before NumPy
reshape, and `write_array()` normalises big-endian arrays to the required
little-endian wire representation. Hash-only testcase outputs produce a structured
missing-reference comparison instead of a `TypeError`. Saving a loaded package to
a new root now copies its deduplicated payloads, preserving a valid round trip.

### Testcase and payload path containment

Boundary labels are display data, not filesystem paths. Explicit names such as
`../escape`, `nested/output`, or Windows device name `CON` are now retained in
the manifest but mapped to deterministic safe payload filenames.

Trace payload references are resolved through a package-root containment check.
Validation rejects absolute or escaping paths, and testcase extraction refuses
to copy them even if the referenced external file exists. Trace comparison uses
the same checked resolver.

## Verification

```text
codec/effect/testcase/validator focused suites 175 passed
official Qwen3.5-0.8B CPU integration           1 passed
full default suite                           314 passed, 1 skipped
```

Reproduction:

```powershell
python -m pytest tests/core/test_irtensor_codec.py `
  tests/core/test_ir_and_validate.py `
  tests/core/test_compare.py `
  tests/frontend/test_testcase_and_regions.py `
  tests/frontend/test_trace_adversarial.py -q

$env:INFERREF_QWEN35_MODEL = "E:\RiderProjects\Aila\models\Qwen3.5-0.8B"
python -m pytest tests/frontend/test_qwen35_08b.py -q -rs
Remove-Item Env:INFERREF_QWEN35_MODEL

python -m pytest -q
```

## Remaining risks

- CUDA fused DeltaNet, paged KV cache, and vLLM execution remain unverified.
- A storage rebind is still represented by same-object aliasing plus changed
  input/output storage ids; Trace IR v0.1 has no explicit rebind effect.
- Testcase manifests describe physical operators and effects but do not yet
  define a general engine-adapter execution protocol for every structured value.
- Symlink-safe path containment is enforced through resolved paths, but trace
  packages are still local trusted artifacts rather than a hardened archive
  ingestion format.

## 0.3 readiness

The CPU/Core foundation is ready for 0.3 planning. The next work should define
the engine-adapter protocol around the now-truthful testcase manifest, then add
viewer/agent surfaces over stable IR rather than expanding frontend semantics
first.
