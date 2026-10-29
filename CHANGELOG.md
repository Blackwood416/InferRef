# Changelog

User-facing changes belong here. Historical implementation evidence remains in
[development status](docs/dev/status/README.md).

## Unreleased — 0.12.x M4

### Added

- Agent evaluation benchmarks `nf4_block_scale_offset` (flat NF4 block scales
  crossing row boundaries) and `kv_cache_index_stride` (indexed KV-cache write
  at the wrong decode position), each with visible and hidden holdout cases and
  deterministic fake-Agent repair coverage.
- `inferref agent evaluate --diagnostic-mode {full,status_only}`. `status_only`
  reduces MCP context/execution responses to PASS/FAIL, artifact names, output
  role names, and execution status without tensor metrics, element coordinates,
  shapes, or execution logs while preserving the sealed audit state machine and
  attestation binding.
- `inferref agent evaluate --codex-config <json>` and `--codex-model <model>` for
  an isolated Codex gateway. The JSON object is converted to Codex `-c`
  overrides without a shell, so `--ignore-user-config` can remain in force; the
  public argv policy records override keys, not their values.

### Fixed

- `inferref_context` reports actual input dtypes for non-float tensors (uint8
  NF4 codes, int32 group size and KV index) instead of hard-coding float32.
- The new Agent benchmark tasks no longer reveal their root cause. The KV
  benchmark uses a cache-layout ambiguity, with a two-head holdout that fails
  the baseline and passes the repaired engine.
- `status_only` includes output role names without exposing metrics or
  coordinates.
- Agent `argv_policy` binds the selected `diagnostic_mode`, so full and
  status-only runs have distinct public policy digests.
- `AlignmentRunner` preflights quantized and per-contract requirements against
  the extracted testcase, handles executor `unsupported` results without an
  `output` key, and no longer leaves empty prepared run directories.
- Scaffolded runtime-bridge projects parse `--batch-manifest` and call
  `RunBridgeBatch`; the skill CLI reference includes `--diagnostic-mode`.

## 0.12.0 — 2026-10-22

0.12.0 completes the P0 infrastructure milestones M1–M3: native C++ consumption of
quantized testcases, execution-grade low-bit hardware contracts, and
hierarchical whole-model alignment with batch execution. M4 (multi-bug Agent
benchmark and `--diagnostic-mode`) is P1 and is deferred to 0.12.x.

### Added

- C++ testcase `0.4` reader and quantization parameter ABI: `Testcase::Load`
  accepts `0.1`–`0.4`, parses the top-level `quantization_parameters` list by
  `value_id`, validates `QuantizationSpec` (`scheme`, `granularity`, nullable
  `axis`/`group_size`/`zero_point_value_id`, NF4-only `codebook`), and fails
  closed on path traversal, missing or duplicate parameters, header/manifest
  mismatch, missing scale/zero-point/codebook references, and unreferenced
  parameters. `DequantizeInputToFloat32` reproduces the Python reference
  bit-for-bit for `affine-int8/v1`, `affine-int4/v1`, `affine-uint4/v1` and
  `nf4/v1`; committed Python/C++ goldens compare float32 bit patterns, not
  tolerances. No Python or numpy dependency is added to the C++ reader.
- Runtime bridge quantized dispatch: `DebugInvokeEx`/`RunBridge` receive
  `const Testcase &` so an adapter can query `HasInputQuantization`,
  `InputQuantization` and `QuantizationParameter`. A legacy two-argument bridge
  that receives a `0.4` testcase exits with `kBridgeUnsupportedQuantization` and
  a diagnostic instead of silently ignoring scales. The adapter scaffolder emits
  the same queries for `0.4` testcases and derives `capabilities.features` from
  the testcase requirements.
- Hardware quantization contracts `linear/affine-int8-bias/v1`,
  `linear/affine-w4a16-bf16/v1` and `linear/nf4-w4a16-bf16/v1` (float32
  dequantization and accumulation with a single nearest-even bfloat16 output
  conversion), plus the core region contracts `swiglu/mlp/v1` and
  `attention/sdpa-gqa/v1` with shape, dtype, rank and `heads % kv_heads`
  validation.
- `linear/w8a8-int32-accum/v1`: int8 activations and weights with exact int32
  accumulation per reduction group, float32 scale epilogue, no bias in v1,
  explicit overflow rejection, and a numpy-only reference validated against an
  independent integer oracle.
- W8A8 adapter feature `quantization/w8a8-int32-accum/v1` merged into derived
  requirements. Adapter preflight fails closed before launch when an adapter
  declares the contract but omits the W8A8 token, including adapters that
  provide no per-contract capabilities.
- Hierarchical step trace: `compile_step_trace` records one prefill/decode step,
  every decoder layer, each sub-block (`input_layernorm`, `self_attn`,
  `post_attention_layernorm`, `mlp`) and the leaf projections as deterministically
  ordered nested regions inside a single `TracePackage`, with optional explicit
  quantization descriptors on leaf projections.
- Serial-prefix binary-search narrowing with the deterministic reason
  `"binary search midpoint of failing serial chain"`; a single faulty layer in a
  28-layer chain is isolated within the documented evaluation bound instead of
  one evaluation per layer.
- C++ batch manifest (`inferref-batch` `1.0`, `--batch-manifest`): ordered
  `{id, testcase_dir, output_dir}` items execute in one bridge process, each
  item writes its own `manifest.json`, and `batch-results.json` records every
  outcome. A failing item is recorded and the batch continues; the process
  exits non-zero at the end.
- Adapter capability `batch_execution`, the `execute_adapter_batch` helper and
  the `AlignmentRunner` batch path: `run(limit=K)` executes `K > 1` candidates
  in one adapter launch when the adapter declares the feature, and otherwise
  falls back to sequential execution with an explicit diagnostic.

### Fixed

- Guard the batch alignment path with the same quantized-comparison validation
  as sequential execution. An unevaluable comparator now aborts before launch
  and is reported as an error without false `fail` evidence.
- Map batch results by request order so duplicate boundary ids cannot share a
  run record or output directory, and honor `first_failure` overrides.

### Compatibility

- Trace IR (`0.1`/`1.0`), `.irtensor` formats 1/2, testcase versions `0.1`–`0.4`
  and Engine Adapter `0.2` are wire-compatible with 0.11.0; the new capability
  tokens are additive.
- The C++ testcase reader now accepts `0.4`. A native adapter that only
  implements the legacy two-argument bridge must keep rejecting quantized
  testcases; the bridge reports `kBridgeUnsupportedQuantization` rather than
  dropping quantization metadata.
- `linear/affine-w4a16/v1` and `linear/nf4-w4a16/v1` remain float16-activation
  contracts. bfloat16 activations and outputs require the new `*-bf16/v1` IDs.
- `linear/w8a8-int32-accum/v1` is an integer-dot-product contract, not
  `linear/affine-int8/v1`. An adapter must declare the W8A8 token together with
  the granularity tokens it implements; an adapter that declares only the
  contract no longer inherits adapter-wide capabilities and is rejected during
  preflight.
- Batch execution is opt-in and used only for `K > 1`. One process emits every
  item's artifacts, so an adapter's per-testcase timeout and artifact budget
  apply per item; `execute_adapter_batch` accepts an explicit batch timeout.

### Known limits

- **Serial-prefix only.** The binary-search evaluation bound assumes a runner
  that fails a serial prefix of a failing chain, as specified in
  [SPEC §7.2](docs/spec/InferRef_0.12_Core_Infrastructure.md). With a
  containment-only runner, which fails only the exact faulty child region, a
  passing midpoint is not a valid prefix bound and the bound must not be
  generalized.
- **M4 deferred to 0.12.x.** The multi-bug Agent benchmark cases
  (`nf4_block_scale_offset`, `kv_cache_index_stride`) and
  `inferref agent evaluate --diagnostic-mode` are not part of 0.12.0.
- Linux and remote CI passed for 0.12.0: CircleCI **16/16** branch jobs and
  **7/7** `v0.12.0` tag-qualification jobs, plus the GitHub Actions Ubuntu
  package build. The tag and GitHub Release carry the wheel/sdist; PyPI was not
  published. See the [0.12.0 release notes](docs/releases/0.12.0.md).

## 0.11.0 — 2026-09-21

### Added

- Whole-model causal autoregressive generation scenario compilation and multi-step
  alignment workflow (`examples/quantization/qualify_whole_model_generation.py`),
  verifying prompt prefill and autoregressive decode steps as self-contained testcases
  and InferRef Scenario v0.1.
- Intel Arc A770 GPU (Level Zero / SYCL) end-to-end execution qualification with
  Aila Qwen3 NF4 engine adapter, verifying 100% token sequence parity across multi-step
  generation, cosine similarity > 0.9998, `inferref-sycl-device` v0.1 device evidence,
  bug injection detection, and Alignment Orchestrator verification (`session_verified`).
- Robust UTF-8 with BOM (`utf-8-sig`) decoding for engine adapter JSON manifests.


## 0.11.0rc1 — 2026-09-18

Prerelease for CPU quantization artifact/reference workflows. Whole-model
generation, GPU execution and vendor fused-kernel correctness are not qualified.

### Added

- Native `float8_e4m3fn` and `float8_e5m2` tensor payloads (append-only wire codes
  16/17), numpy-only software decoding, C++ reader support, and PyTorch capture
  coverage. Exact numerical comparison is the default; this is not a scaled
  quantization contract, and existing adapters do not gain FP8 kernel support.
- Exhaustive FP8 encoding tests and Python/C++ binary roundtrip checks in CI.
- Standalone affine INT8 quantization descriptors, numpy-only per-tensor/
  per-axis/per-group reference arithmetic, float32 Linear evaluation, and
  dependency-sensitive fingerprints.
- Quantized trace 1.0 / testcase 0.4 integration, standalone parameter extraction,
  scheme/granularity-aware adapter preflight, and the `linear/affine-int8/v1`
  contract with an independent numpy adapter and Agent/Suite/Alignment coverage.
- Quantized trace comparison defaults to each side's own dequantized values;
  `compare --quantization-mode raw_codes` explicitly selects raw-code comparison.
  Testcase quantized reference outputs require float32 engine outputs.
- Packed INT4/UINT4 (codes 18/19, tensor format 2), bit/payload-size helpers,
  low-nibble-first canonical packing, C++ decode/write support and wire goldens.
- Explicit affine W4A16 and NF4 W4A16 contracts with float32 accumulation and
  final float16 rounding. NF4 carries its codebook and block scales as immutable
  tensor dependencies. A pinned bitsandbytes 0.48.2 non-nested uint8/float16,
  blocksize-64 host profile importer converts high-nibble-first source packing;
  unknown/incomplete/nested states are rejected.
- Separate bitsandbytes 0.50.1 QuantState object/host-projection bridge, with
  one-level double-quantized scale expansion and explicit float16/bfloat16/
  float32 source-output rounding. A read-only safetensors audit compares real
  weights with the actual bitsandbytes CPU runtime and exports a portable case.
  The old 0.48.2 flat profile remains unchanged; core decoding stays numpy-only.

### Fixed

- Restore trace-vs-trace CLI comparison; apply numeric comparison configuration
  and reject custom testcase-only comparators with a structured JSON usage error.
- Keep automatic alignment execution consistent with adapter-aware planning.
- Provide a ready-to-run alias-aware adapter for the README Agent example.
- Separate suite execution failures from unsupported exit-policy rejections.
- Preserve rank-0 array shape in the `write_array` convenience writer.
- Bind quantized trace identity to code/parameter bytes, recheck long-lived
  alignment runners before testcase reuse, and skip source-blocked candidates.
- Include explicit quantization descriptors in operator dedup signatures so
  identically shaped UINT4 tensors with affine and NF4 semantics stay distinct.

### Compatibility

- Suite run reports now use format version **0.3**. `counts.failed` counts actual
  failures/errors, not unsupported-only rejections. Use `counts.rejected` for the
  old policy-rejection count, or `accepted` for a CI gate. `counts.partial` counts
  scenarios containing only passing and unsupported steps. Exit codes are unchanged.
- Rendered suite report JSON uses version **0.2** and records
  `source_run_format_version`; old suite run versions 0.1/0.2 remain readable.
- Plain trace/testcase writer defaults and adapter/suite manifest schemas remain
  unchanged. Explicit quantized artifacts require trace 1.0 or testcase 0.4;
  older readers reject these versions. Future trace 1.x minors are not implicitly
  accepted. FP8 extends the dtype table; old readers reject its new dtype codes.
- Legacy formats reject core `quantization` metadata. Adapters must explicitly
  declare scheme/granularity features; legacy undeclared adapters fail closed.
  The C++ testcase reader continues to reject quantized testcase 0.4.
- Byte-addressable tensor writers retain format 1; packed dtypes require format
  2. `dtype_itemsize` rejects sub-byte types; use `dtype_payload_nbytes` instead.
  Python packed TensorView data is unpacked logical codes; C++ data remains
  packed bytes. Neither frontend auto-capture nor hardware low-bit kernels are
  implied by the new reader and reference support.

### Documentation and distribution

- Add documentation and contribution entry points, native dtype limitations,
  an inline typing marker, and distribution build/release-asset checks.
- PyPI publication still requires maintainer ownership and an explicit publishing
  decision; no PyPI release is implied by these changes.

## 0.10.0 — 2026-09-04

- Add Alignment Orchestrator sessions, boundary planning, immutable evidence,
  fixture tracking, first-divergence narrowing, and JSON/Markdown reporting.
- Expose alignment through the CLI, Agent API, and MCP.
- Resolve trusted symlinked engine executables on POSIX.

[Release](https://github.com/Blackwood416/InferRef/releases/tag/v0.10.0)

## Earlier releases

See [GitHub releases](https://github.com/Blackwood416/InferRef/releases) and
[the dated development records](docs/dev/status/README.md). This changelog does
not retroactively claim complete coverage of pre-0.10 changes.
