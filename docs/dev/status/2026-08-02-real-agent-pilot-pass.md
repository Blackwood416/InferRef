# Real dual-Agent repair pilot PASS

Date: 2026-08-02

## State

The first blind dual-Agent RoPE repair pilot passed its strict 2/2 manual gate.
Codex and Claude Code ran in one evaluator invocation from separate pristine
workspaces. Each independently repaired `engine.py`, passed the visible case
within the four-run budget, and passed all three host-only holdouts.

This supersedes the earlier same-day infrastructure-failure snapshot. No oracle,
threshold, holdout, prompt, or run budget was changed between that attempt and
this successful run; only the user-managed Claude provider configuration changed.

## Configuration isolation

- The repository contains no `.claude/settings*.json`.
- A physically generated Agent workspace contained exactly `TASK.md` and
  `engine.py`; it contained no `.claude` directory.
- The evaluator did not pass `--claude-settings` or `--claude-model` for the
  successful run. Claude Code used the user's default configuration.
- The global settings SHA-256 was unchanged before and after evaluation:
  `8B5F3EFD1692199FFB48F345B6E93B162532984C1A73772A1DD8113F07212488`.

## Results

### Codex

- Requested model: `gpt-5.6-sol`
- CLI: `codex-cli 0.146.0`
- Agent duration: 50.710 seconds
- Visible engine runs: 2 of 4
- Hidden holdouts: 3/3 PASS
- Changed paths: `engine.py`
- Protected workspace/host changes: none
- Staged-input tampering and tool-order violations: none

### Claude Code with DeepSeek

- Benchmark driver alias: `opus`
- Resolved provider model: `deepseek-v4-flash[1m]`
- CLI: Claude Code 2.1.116
- Agent duration: 25.984 seconds
- Visible engine runs: 2 of 4
- Hidden holdouts: 3/3 PASS
- Changed paths: `engine.py`
- Protected workspace/host changes: none
- Staged-input tampering and tool-order violations: none
- Reported provider cost: USD 0.160514

Both patches made the general half-rotation sign correction. Neither specialized
the implementation to the visible head dimension.

## Acceptance

```text
required Agent passes: 2
actual Agent passes:   2
visible cases:         2/2 PASS
hidden holdouts:       6/6 PASS
integrity violations: 0
overfit failures:      0
result:                PASS
```

Raw Agent JSONL, reference tensors, credentials, and full reasoning remain under
ignored local paths and are not committed. The machine-readable manual report is
stored locally at `.scratch/rope-sign-dual-manual-config-final/report.json`.

## Repository verification

- Local suite: 389 passed, 5 skipped.
- Wheel and sdist build succeeded.
- [CI run 30751401285](https://github.com/Blackwood416/InferRef/actions/runs/30751401285):
  19/19 jobs passed.

## Boundary of the claim

This proves that two real coding-Agent frontends can independently use InferRef's
opaque MCP diagnostics to complete this constrained RoPE numerical repair. It does
not prove a general OS sandbox, arbitrary repository repair, build-system repair,
or KV-cache repair. Those require additional benchmark corpus entries.

## Reproduce

```bash
inferref agent evaluate examples/agent_eval/rope_sign/benchmark.json \
  --agents codex,claude \
  --report-dir .scratch/rope-sign-dual-next \
  --json
```

The report directory must be fresh, and a later run must again obtain 2/2 within
one invocation rather than combining independent historical runs.
