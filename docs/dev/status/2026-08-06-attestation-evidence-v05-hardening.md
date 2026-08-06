# Agent attestation v0.5 — external settings, model identity, launch/argv policy

## Closed issues

- Claude `--claude-settings` content is now bound to the attestation. The file is
  captured no-follow before and after each Claude run, and again at benchmark
  level between the first and last Agent. The published record only contains
  `present`, `kind`, `size`, `sha256`, `after_sha256`, `after_kind`,
  `after_size`, `after_file_id`, and `unchanged`; paths and contents are never
  published. Symlinks/reparse points are rejected, and delete-and-recreate or
  inode replacement with identical bytes is still flagged as changed. A Claude
  run without settings records an explicit `present: false`.
- Model identity now separates evidence validity from request satisfaction. A
  `cli_self_reported` model that does not match the requested model classifies
  the candidate as `identity_policy_failure`, fails the benchmark even when the
  numerical results pass, and is refused by formal attestation.
- Worker `launch_policy_sha256` is now a recomputable digest of a canonical
  policy object (`-I -m`, request transport, request schema version, exact
  request byte SHA-256) instead of an opaque hash of `sys.orig_argv`. The parent
  independently hashes the request bytes it wrote and rejects a worker report
  whose evidence does not match. The worker also hashes its own Python
  executable (name, size, SHA-256).
- POSIX shebang resolution honors `env -S` and direct interpreter arguments.
  `#!/usr/bin/env -S node --no-warnings` and `#!/usr/bin/python3 -s` now resolve
  interpreter flags as launch arguments while hashing only executable file
  components.
- Agent argv is split into `argv_instance_sha256` (full instance command) and a
  path-free, prompt-free `argv_policy` with a verifiable `argv_policy_sha256`
  that third parties can recompute from the published policy.
- Runner/worker validators are strict: lowercase-hex SHA-256, non-negative
  sizes, `error: null`, exact component role sequences, recomputed digest
  checks, and structured validation of external-file evidence.

## Acceptance evidence

- Python full suite: `493 passed, 9 skipped` on Windows/Python 3.13.11.
- New coverage includes: settings change/symlink/delete-recreate/inode
  replacement, benchmark-level settings drift, model mismatch classification and
  formal refusal, canonical worker launch policy, parent/worker request-hash
  cross-check, `env -S`/direct-shebang/quoted-arg resolution, path-free argv
  policy, and malformed component/role/digest rejection.
- `inferref-agent-evaluation-attestation` format bumped to v0.5; report carries
  `agent_model_request_satisfied`, `claude_settings_unchanged`,
  `external_files`, and `worker_evidence` for parent cross-checking.

## Decisions

- `model_identity_policy` is fixed at `reported_must_match_if_available` for
  now; benchmark-level policy configuration can be added later without a format
  break.
- External-file `unchanged` requires the same file identity (`st_dev:st_ino`) in
  addition to identical bytes, so recreation is distinguishable from a stable
  file.
- Worker `python_executable` evidence is published, but the worker Python binary
  identity is not yet part of the formal refusal condition; it is recorded for
  auditability and symmetric with Agent executable hashing.

## Known gaps

- Path resolution between capture and use still has the documented TOCTOU
  window of a same-user process; this remains accidental-path-escape protection,
  not an OS sandbox boundary.
- The audit digest is corruption/torn-write detection, not a keyed
  authenticator.
- Direct-shebang argument tests run only on POSIX CI; Windows CI covers the
  `env` and `env -S` forms.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest tests\agent\test_repair_evaluation.py -q
.\.venv\Scripts\python.exe -m pytest -q
```
