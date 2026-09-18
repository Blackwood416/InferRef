# Changelog

User-facing changes belong here. Historical implementation evidence remains in
[development status](docs/dev/status/README.md).

## Unreleased

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
