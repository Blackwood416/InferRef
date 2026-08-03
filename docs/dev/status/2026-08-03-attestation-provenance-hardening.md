# Agent attestation provenance hardening

Date: 2026-08-03

## Outcome

Agent evaluation evidence now distinguishes numerical correctness from runner
provenance. A formal public attestation can only come from InferRef's built-in
Codex/Claude CLI path, and it binds the repository, benchmark, evaluator source,
and runtime environment used across the complete evaluation interval.

## Formal versus development evidence

- `evaluate_benchmark(..., public_attestation=..., driver=...)` is rejected before
  creating a report directory. An injected callback can no longer claim that a
  configured Codex or Claude CLI produced the candidate.
- Every evaluation records `runner_mode`. Built-in CLI runs use `builtin_cli`;
  injected CI/test callbacks use `custom_driver`.
- Private evaluations still receive a redacted `attestation.json`, but custom
  drivers are explicitly labelled `attestation_level: development`.
- `inferref-agent-evaluation-attestation` v0.3 reserves
  `attestation_level: formal` for the built-in runner with clean and unchanged
  repository/evaluator/benchmark evidence.

## Evidence interval

- Repository evidence is captured before Agent launch and after all Agents exit.
  Formal publication requires the same commit and status evidence at both ends,
  with both snapshots clean.
- The benchmark is parsed from one no-follow byte read. Its attestation digest is
  frozen from those exact bytes, and the source is hashed again at the end to
  detect replacement or mutation during evaluation.
- The imported `inferref` source manifest is compared before and after as before;
  repository, source-tree, or benchmark changes also fail overall host integrity.
- Runtime evidence records Python implementation/version, OS/platform/machine,
  architecture, byte order, NumPy and InferRef versions, MCP SDK version, import
  mode, and installed-distribution `RECORD` digests when available. Absolute local
  import paths are not published.

## Audit semantics

The sealed JSONL validator now checks an explicit evaluation state machine in
addition to framing and digest integrity:

- tool names and operations have a fixed one-to-one mapping;
- statuses are constrained by each operation;
- only an actually executed `inferref_run_engine` may increment `engine_runs`;
- an executed run increments exactly once and requires successful capabilities
  and context calls;
- rejected sequence/path/budget calls do not consume a run;
- non-run tools cannot alter the counter.

Out-of-order calls remain valid recorded evidence when they carry the expected
rejection; candidate assessment then classifies the integrity attempt. The footer
SHA-256 is corruption/torn-write detection, not keyed authentication against a
same-user process able to rewrite the complete stream.

## Verification

- Focused Agent evaluator tests: `53 passed`.
- Full Python suite: `426 passed, 5 skipped`.
- Ruff format/lint on changed evaluator and test modules: clean.
- `uv build`: wheel and source distribution succeeded.

## Deliberately deferred

- Parent-directory path resolution still has a same-user TOCTOU window. Existing
  lexical, resolved-path, entry-type, no-follow final-file, process-tree, and
  reparse checks are suitable for the documented non-adversarial local pilot, not
  a strong OS sandbox. Closing it requires handle-relative traversal (`openat2`
  or directory fds on Linux, directory handles/final-path and reparse validation
  on Windows).
- The audit digest is not a MAC or signature. Formal remote attestations would
  need a host-held signing key and a stronger process isolation model.
- KV-cache detector v1 remains deliberately precision-first; scored evidence for
  packed, fused, or page-table caches remains future detector work.

## Reproduction

```powershell
.venv\Scripts\python -m pytest tests/agent/test_repair_evaluation.py -q
.venv\Scripts\python -m pytest -q
uv build
```
