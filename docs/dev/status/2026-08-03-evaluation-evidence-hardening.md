# Evaluation evidence and container-alias hardening

Date: 2026-08-03

## Outcome

The Agent pilot evidence chain now fails closed and binds the implementation that
actually produced a verdict. Trace IR alias detection also walks tensor containers
symmetrically on the input and output sides.

## Evidence-chain changes

- A formal `--public-attestation` now requires a clean Git worktree at a concrete
  commit. Dirty or unavailable Git state is rejected before an Agent is launched.
- Attestation format v0.2 records repository dirty/status/diff hashes and a
  deterministic manifest plus aggregate SHA-256 for the complete imported
  `inferref` package source tree. This binds the runtime source bytes instead of a
  manually maintained import closure.
- `report_json_sha256` replaces the misleading `private_report_sha256`; it hashes
  exactly `report.json`.
- Every Agent run snapshots all package source files before and after execution.
  Formal publication is refused if the source-tree manifest changes.
- MCP audit JSONL uses continuous global call indices, monotonic engine-run counts,
  and a terminal footer containing the record count and cumulative SHA-256. Any
  malformed line, invalid schema, missing footer, torn tail, or digest/count
  mismatch is an `infrastructure_failure`.
- Each tool call atomically checkpoints all records plus a valid footer. This was
  required by a real Codex run: Codex terminates its MCP child directly after the
  turn, so a footer written only from Python `finally` was not reliable.
- Codex starts with `--ignore-user-config`; otherwise user plugins and global MCP
  servers can hide the evaluation-only InferRef server. A zero-call MCP probe no
  longer creates/reserves the audit path before the real connection starts.

## Workspace integrity

Workspace snapshots now record each entry's kind, size, mode, content hash, and
link target without following symlinks or Windows reparse points. Directories are
included, so adding an empty directory is observable. Protected entries and the
editable engine must remain regular files; final engine reads use no-follow checks
and verify the opened file identity.

Regression coverage includes same-content symlink replacement, an engine symlink,
empty-directory creation, and a Windows junction. These checks constrain final
state. They still cannot detect a same-user process that reads files or modifies
and restores them between snapshots.

## Trace IR alias completeness

Input and output pairing now uses one recursive walker for `TensorRef`, list,
tuple/namedtuple, and dict structures. A custom `Tensor(a)[] -> Tensor(a)[]` test
proves `same_object` and view relationships survive container boundaries; a nested
dict/list unit test covers deeper structures.

`aten._foreach_add_.Scalar` itself has a void dispatcher result even though the
Python convenience wrapper returns its input list, so its physical Trace IR can
only assert the two storage mutations. It is not used as a false output-alias
oracle.

## C++ agreement

The cross-language CI leg now compares both `q_embed` and `k_embed`, verifies the
known q-only injected failure while k still passes, then corrupts k independently
and requires the C++ comparator to reject it.

## Verification

- Python: `414 passed, 5 skipped`
- Focused Agent evaluation hardening: `41 passed`
- Ruff on changed Python files: format and lint clean
- Package build: wheel and sdist succeeded
- Windows C++: MSVC 19.50 configure/build and `irtensor_selftest` passed

## Deliberately deferred

- MCP root containment remains accidental path-escape protection, not a strong
  same-user sandbox. Closing the check/use race requires handle-relative access:
  `openat2`/directory fds on Linux and final-handle/reparse validation on Windows.
- KV-cache detector v1 remains the high-precision rule. A scored v2 detector for
  packed/fused/page-table caches should be designed separately rather than
  weakening v1.

## Real lifecycle findings

The first formal v0.2 attempt against clean commit `535af43` produced a legitimate
`infrastructure_failure`: Claude/DeepSeek passed, while Codex could not initialize
the evaluation MCP server because user-level configuration remained active. The
redacted failure attestation is retained under `docs/dev/attestations` rather than
rewritten as a pass. After config isolation and crash-consistent audit checkpoints,
an isolated real Codex rerun passed with one engine run, final visible PASS, and
all three holdouts PASS.

A fresh strict run from clean commit `7927d9f` then passed 2/2:

- Codex `gpt-5.6-sol`: one engine run; final visible and 3/3 holdouts PASS;
- Claude Code / `deepseek-v4-flash[1m]`: two engine runs; final visible and 3/3
  holdouts PASS;
- both audit streams valid; repository dirty `false`; evaluator source unchanged;
- source-tree SHA-256:
  `4e6dfb6408b39abbfb2fb51eacb1f7eac72cac12aa81a5f78090a232c8606967`;
- public attestation SHA-256:
  `f65314dc256277bc291e85b07631a43c4d3cac0c9e959eac9be78ef68562b93`.

Public evidence:

- [failed infrastructure attempt at `535af43`](../attestations/2026-08-03-rope-dual-agent-535af43-codex-mcp-failure.json)
- [strict 2/2 PASS at `7927d9f`](../attestations/2026-08-03-rope-dual-agent-7927d9f.json)
