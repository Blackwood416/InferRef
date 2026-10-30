# Real-Agent matrix — 2026-10-30

- Scope: first controlled `full` vs `status_only` matrix for the three M4
  benchmark contracts.
- Evaluator: `main` at `e812b3d` with a clean worktree.
- Model path: both Agent CLIs through the local `magpie` gateway and the
  configured `opencode-go` DeepSeek V4.1 Flash model.
- Evidence level: development reports only. No public attestation, no repeats,
  and no `no-InferRef` arm.

## 1. Design

Six fresh evaluations: `rope_sign`, `nf4_block_scale_offset` and
`kv_cache_index_stride`, each in `full` and `status_only` mode. Every evaluation
started both Codex and Claude in fresh sessions with the same model, prompt,
benchmark, `max_engine_runs=4` budget and no formal worker.

The command shape was:

```powershell
inferref agent evaluate <benchmark>/benchmark.json `
  --agents codex,claude `
  --report-dir <fresh-dir> `
  --codex-config <isolated-gateway.json> `
  --codex-model <gateway-model> `
  --claude-settings <ephemeral-settings.json> `
  --claude-model <gateway-model> `
  --diagnostic-mode <full|status_only> `
  --json
```

## 2. Results

All six reports passed: two of two Agents passed, every visible and holdout case
passed, both audits were valid, and every report had
`repository_unchanged=true`, `host_integrity_passed=true` and
`agent_model_request_satisfied=true`.

| Benchmark | Mode | Codex engine runs | Claude engine runs | Wall time | Result |
| --- | --- | ---: | ---: | ---: | --- |
| `rope_sign` | `full` | 1 | 1 | 96.7 s | pass |
| `rope_sign` | `status_only` | 2 | 1 | 95.3 s | pass |
| `nf4_block_scale_offset` | `full` | 3 | 3 | 200.9 s | pass |
| `nf4_block_scale_offset` | `status_only` | 4 | 2 | 583.2 s | pass |
| `kv_cache_index_stride` | `full` | 1 | 1 | 98.1 s | pass |
| `kv_cache_index_stride` | `status_only` | 1 | 1 | 168.5 s | pass |

All holdouts passed for all 12 Agent executions. The `nf4_block_scale_offset`
`status_only` Codex run used the full four-run budget; it did not fail, but had
no spare engine runs.

An initial `rope_sign` pair was invalidated because this report's release-note
edit changed the worktree during the run (`repository_unchanged=false`). Both
rope rows above are clean reruns after the worktree was restored.

## 3. Observations

- **`kv_cache_index_stride` is saturated.** Both modes and both Agents solved it
  on the first visible engine run. It cannot provide a useful
  full-versus-status-only contrast in its current form.
- **`rope_sign` is easy.** One or two engine runs were enough. It is useful as a
  smoke/control, not as the primary difficulty signal.
- **`nf4_block_scale_offset` is the only non-trivial cell.** It required two to
  four engine runs, and `status_only` did not reduce effort: Codex used more
  runs and the wall time was longer. This is one observation, not evidence that
  diagnostics hurt; the mode × Agent × run variance is not estimated.
- **No diagnostic-benefit claim.** The matrix has one run per cell. Repeats,
  randomized ordering, fixed token/time accounting and a no-InferRef arm are
  still required before comparing modes.

## 4. Raw evidence

Reports and the generated summaries were kept outside the repository at:

```text
%TEMP%\inferref-agent-matrix-20261030-205901\
```

Useful files:

- `matrix-summary.json` and `matrix-summary.csv` — all six evaluations; the two
  invalid rope rows are marked by `repository_unchanged=false`.
- `rope-rerun-summary.json` — the two clean rope reruns.
- `report-<benchmark>-<mode>/report.json` — per-run reports.
- `run-<benchmark>-<mode>.log` — CLI output.

These paths are temporary workstation artifacts, not release evidence.

## 5. Next steps

1. Repeat `nf4_block_scale_offset` `full` vs `status_only` at least three times
   with fresh sessions and randomized order; report median and range.
2. Harden `kv_cache_index_stride` or remove it from the primary ablation. It
   currently measures one-run repair, not diagnostic use.
3. Keep `rope_sign` as a control and smoke case.
4. Define the separate `no-InferRef` arm before making a diagnostic-benefit
   claim, because `status_only` still exposes MCP tools and artifact names.
5. Use `--public-attestation` only after the final matrix and a clean,
   reproducible run configuration are fixed.
