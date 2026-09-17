# Changelog

User-facing changes belong here. Historical implementation evidence remains in
[development status](docs/dev/status/README.md).

## Unreleased

### Added

- Native `float8_e4m3fn` and `float8_e5m2` tensor payloads (append-only wire codes
  16/17), numpy-only software decoding, C++ reader support, and PyTorch capture
  coverage. Exact numerical comparison is the default; this is not a scaled
  quantization contract, and existing adapters do not gain FP8 kernel support.
- Exhaustive FP8 encoding tests and Python/C++ binary roundtrip checks in CI.

### Fixed

- Restore trace-vs-trace CLI comparison; apply numeric comparison configuration
  and reject custom testcase-only comparators with a structured JSON usage error.
- Keep automatic alignment execution consistent with adapter-aware planning.
- Provide a ready-to-run alias-aware adapter for the README Agent example.
- Separate suite execution failures from unsupported exit-policy rejections.
- Preserve rank-0 array shape in the `write_array` convenience writer.

### Compatibility

- Suite run reports now use format version **0.3**. `counts.failed` counts actual
  failures/errors, not unsupported-only rejections. Use `counts.rejected` for the
  old policy-rejection count, or `accepted` for a CI gate. `counts.partial` counts
  scenarios containing only passing and unsupported steps. Exit codes are unchanged.
- Rendered suite report JSON uses version **0.2** and records
  `source_run_format_version`; old suite run versions 0.1/0.2 remain readable.
- Trace IR, testcase, adapter, and suite manifest schemas are unchanged.
  FP8 extends the dtype code table; old readers reject the new dtype codes.

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
