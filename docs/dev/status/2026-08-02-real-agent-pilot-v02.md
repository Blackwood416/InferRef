# Real Agent repair pilot v0.2

Date: 2026-08-02

## State

The blind RoPE repair evaluator is implemented and locally verified. The
deterministic harness passes, and a fresh Codex process completed the real repair.
The required dual-Agent manual gate is **not yet green**: Claude Code reached its
configured providers, but both Anthropic Opus and the DeepSeek Flash compatibility
route were unavailable due to repeated HTTP 429 responses. This is recorded as an
`infrastructure_failure`, not an Agent repair failure.

## What exists

- `inferref-agent-evaluation` format v0.2 and
  `inferref agent evaluate <benchmark> --agents ... --report-dir ...`.
- Independent disposable workspaces containing only `TASK.md` and `engine.py`.
- An evaluation-only MCP proxy exposing the standard capabilities, context, and
  run-engine tool names through opaque `eval://` identifiers.
- Host-memory reference tensors; candidate staging contains input payloads and
  output metadata only.
- A hard four-run budget enforced by the MCP server, including active rejection of
  the fifth attempt.
- Silent host evaluation of head dimensions 4 and 12 plus a changed batch/head/
  sequence/seed case.
- Workspace, staged-input, benchmark, runner, oracle, and adapter integrity hashes.
- Codex and Claude Code drivers with fresh sessions, bounded transcripts, process
  tree cleanup, model/CLI metadata, usage data when available, and local Claude
  provider overrides that are never copied into reports.
- Fixed failure classes: infrastructure, Agent, integrity, and overfit.

## Real manual results

### Codex

- Requested model: `gpt-5.6-sol`
- CLI: `codex-cli 0.146.0`
- Result: PASS
- Agent duration: 36.395 seconds
- Visible engine runs: 1 of 4
- Required MCP order: capabilities, context, run-engine
- Hidden holdouts: 3/3 PASS
- Changed path: `engine.py` only

The patch corrected the half-split rotation generally, rather than specializing
the visible head dimension.

### Claude Code providers

- CLI: Claude Code 2.1.116
- Anthropic Opus: ten HTTP 429 retries, zero tokens, zero MCP calls
- DeepSeek Flash through the Anthropic-compatible Claude Code route: ten HTTP 429
  retries, zero tokens, zero MCP calls
- Classification: `infrastructure_failure`

The DeepSeek route followed the provider's documented Claude Code integration,
where Flash is selected through the Haiku/default-Haiku mapping. The provider
accepted the route but rejected the request for capacity/rate-limit reasons.
The local user settings file was also tested via a byte-for-byte temporary swap;
it was restored to its original SHA-256 after the run. Claude Code's noninteractive
`--print` mode did not consume credentials from that default location, while an
explicit settings override did reach the provider.

No raw model transcript, credentials, reference tensor, or full reasoning record is
tracked by Git. Manual artifacts remain under ignored `.scratch/` directories.

## Verification

```text
pytest: 389 passed, 5 skipped
build:  inferref-0.3.0 sdist and wheel succeeded
ruff:   new evaluator, proxy, drivers, CLI delta and tests clean
```

CI remains deterministic and model-free. It verifies successful general repair,
visible-only overfit rejection, oracle isolation, run-budget enforcement, required
tool order, protected-file and staged-input tampering, timeout/malformed output,
and Codex/Claude JSON event parsing.

## Known gaps

- The real 2/2 gate remains open until a fresh Claude Code provider request reaches
  inference and passes visible plus all three hidden cases.
- This benchmark demonstrates constrained repair protocol usefulness, not an OS
  sandbox or broad autonomous repair capability.
- Only the RoPE numerical repair is in the corpus. Build/execution and KV-cache
  repair cases remain deferred.
- Raw JSONL is intentionally local, so there is no public manual-run artifact link
  for this first attempt. Required GitHub checks cover only the deterministic
  evaluator and anti-cheating suite.

## Reproduce

```bash
inferref agent evaluate examples/agent_eval/rope_sign/benchmark.json \
  --agents codex,claude \
  --report-dir .scratch/agent-eval/rope-sign \
  --json
```

For a local Claude Code provider configuration:

```bash
inferref agent evaluate examples/agent_eval/rope_sign/benchmark.json \
  --agents claude \
  --claude-settings /path/to/settings.json \
  --claude-model haiku \
  --report-dir .scratch/agent-eval/rope-sign-provider \
  --json
```

The dual gate must be rerun from new workspaces after provider recovery; the Codex
result from this snapshot must not be combined with a later Claude-only run and
called a 2/2 pass.
