# Status: Foundation reviewer hardening

**Date:** 2026-07-30

**Version:** 0.2 development after `0.2.0`

**Baseline:** commit `c26e960` (Qwen3.5 hybrid-cache validation)

**Verified environment:** Windows, Python 3.13.11, PyTorch 2.13.0+cpu,
Transformers 5.14.1

## Scope

A read-only reviewer subagent audited the Qwen3.5 changes and the underlying
identity, mutation, semantic-region, and compatibility foundations. CUDA was
explicitly excluded because the available machine has no CUDA device; this review
therefore concentrated on framework-independent IR truth and CPU-dispatch behavior.

The reviewer reported no P0 findings. It produced four implementation findings and
two test-strength findings. Two implementation findings independently matched bugs
already reproduced during the main-agent audit, providing useful confirmation rather
than duplicate speculative work.

## Correctness fixes

### Empty storage identity

Every independent empty PyTorch storage reports `data_ptr() == 0`. The previous
`data_ptr -> storage_id` map consequently merged unrelated empty tensors while both
were alive. Mutating one advanced the apparent storage version of all of them, and
zero-sized parameter views could be classified under the wrong qualified name.

Non-empty allocations remain keyed by pointer so a real reallocation stays visible.
Only pointer-zero storages now use the underlying storage object's `_cdata` identity,
still guarded by `StorageWeakRef`. Parameter/buffer indexing uses the same key.
When a live StorageImpl moves to another pointer, its stale pointer slot is removed so
allocator reuse cannot map a future unrelated allocation back to the historical id.

Regression coverage proves that:

- two independent `torch.empty(0)` tensors receive different storage ids;
- mutating one leaves the other's version at zero;
- views of two empty parameters classify under their respective names.

### Runtime object storage rebinding

Writable storages were previously resolved only before dispatch. For
`resize_()` that reallocates, the trace therefore bumped the old allocation while
the same runtime object appeared as an output over a new storage at version zero.
`set_()` had the same issue and additionally retained only the `same_object` alias,
dropping its alias to the source storage.

The dispatcher now retains every writable tensor object and compares its pre-call
and post-call storage identity:

```text
same storage after call
  -> storage mutation, version advances once

different storage after call
  -> object/storage rebind, old storage is not mutated
  -> output starts on the new storage at its current version
  -> same_object alias records runtime-object continuity
```

Alias detection no longer stops after a `same_object` match. It records that object
continuity plus at most one canonical storage relationship, so `set_(source)` can
express both `same_object` to its old self value and `shared_storage` to the source
without adding every duplicate alias argument to semantic region boundaries.

Tests cover reallocating resize, resize within an existing allocation, set-based
rebinding, and an `out=` tensor resized from empty storage. This is deliberately an
IR v0.1 encoding choice: there is no dedicated storage-rebind effect yet, so object
continuity plus changed input/output storage ids is the truthful representation.

### Mutation high-water validation

Invariant 7 previously observed storage versions only through operator input/output
tensor values. A custom cache operation could declare `storage 1: v0 -> v1`, return
`None`, and then read a stale version-zero value without validation failure.

Validation now follows physical execution order inside each operator:

```text
input values -> mutation effects -> output values
```

Mutation `version_after` advances the storage high-water mark even when no tensor is
returned. A later stale read, a stale output, or a mutation whose `version_before`
falls below the observed generation is rejected.

### Opaque operator ids versus execution order

Semantic detection promised execution-order sorting but used the minimum operator
id as its tie-break. External frontends may legally emit execution order
`op:100 -> op:1`; operator ids are references, not timestamps.

Detection now derives the first known `execution_index` from each proposal. A
reverse-id core test locks this frontend-independent behavior.

## Qwen3.5 test strengthening

The official 0.8B opt-in test now records cache length before decode and checks both
states independently:

```text
prefill cache length = 4
decode returns the same cache object
final cache length = 5
```

It no longer accepts cache detections based on count alone. The test asserts exact
linear/full-attention scopes, chronological ordinal assignment, contiguous node
runs, mutation/concat evidence, successful region creation, and validated derived
boundaries.

## Verification

```text
official Qwen3.5 + focused foundation suites   146 passed
full default suite                             292 passed, 1 skipped
```

Reproduction:

```powershell
$env:INFERREF_QWEN35_MODEL = "E:\RiderProjects\Aila\models\Qwen3.5-0.8B"
python -m pytest tests/frontend/test_qwen35_08b.py `
  tests/frontend/test_trace_adversarial.py `
  tests/core/test_ir_and_validate.py `
  tests/core/test_semantic.py -q

Remove-Item Env:INFERREF_QWEN35_MODEL
python -m pytest tests -q
```

## Remaining risks

- CUDA fused DeltaNet remains unverified by design on this machine. CPU correctness
  work should continue independently; CUDA can be added later on suitable hardware.
- `_cdata` is a private PyTorch storage identity. It is used only for the pointer-zero
  ambiguity, and the existing torch 2.1/current/nightly CI matrix is the compatibility
  guard. A public storage-object token should replace it if PyTorch exposes one.
- IR v0.1 has mutation and alias effects but no explicit storage-rebind effect. The
  current encoding is truthful, but a future schema may make rebinding directly
  queryable instead of inferred from same-object values with different storage ids.
- Alias validation checks referenced values but does not yet verify that
  `shared_storage`/`view` relationships agree with the values' storage ids and layouts.
  That is the next useful validator hardening target.

## Suggested next step

Strengthen alias-effect validation, then inspect testcase extraction for
storage-rebound operators so an engine adapter receives both the old object boundary
and the new allocation dependency.
