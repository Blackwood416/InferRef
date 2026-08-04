# InferRef 0.5.1 — evidence and contract hardening

## Closed issues

- Suite logical IDs can no longer control output paths; portable ID validation,
  collision checks, hash-backed artifact keys, and final containment are all
  enforced.
- The native gate now proves Intel GPU execution through `gpu_selector_v`, an
  Intel vendor check, required Level Zero backend, fixed device selector, and
  per-case machine-readable evidence.
- Adapter/testcase v0.2 supports optional versioned executable contracts, so an
  unsupported operation is rejected before engine process creation.
- Suite numerical status is separate from `--allow-unsupported` acceptance;
  matrix cells isolate expected infrastructure/configuration exceptions.
- Native capability claims were reduced to the tested `multiple_outputs`
  feature; operation coverage is expressed through four explicit contracts.
- C++ float16/bfloat16 encoding now covers subnormal, infinity, NaN, overflow,
  and round-to-nearest-even boundaries.
- RMSNorm, rotate-half RoPE, and KV-cache inputs have explicit Python and native
  shape-contract validation.
- `doctor` discovers plugins without importing them unless
  `--verify-plugins` is passed, and invalid device strings return structured
  failures.

## Acceptance evidence

- Python: `470 passed, 8 skipped` on Windows/Python 3.13.11.
- Native codec: MSVC rebuild and CTest selftest passed, including float16 and
  bfloat16 edge vectors.
- Native XPU gate: 13/13 positive Suite cells and 3/3 deliberately corrupted
  negative cases passed.
- Every native cell emitted evidence for `Intel(R) Arc(TM) A770 Graphics`,
  vendor `Intel(R) Corporation`, backend `ext_oneapi_level_zero`, driver
  `1.15.37669`.
- Packaging: `inferref-0.5.1-py3-none-any.whl` and
  `inferref-0.5.1.tar.gz` built successfully.
