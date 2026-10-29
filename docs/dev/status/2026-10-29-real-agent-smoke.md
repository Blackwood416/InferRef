# Real-Agent smoke — 2026-10-29

- Scope: confirm that Codex and Claude can both drive the M4 evaluation through
  the local `magpie` model gateway.
- Release status: development evidence only; **not** a formal public attestation
  and **not** part of the `v0.12.0` release.
- Evaluator base: `ae5d64d` (the smoke ran with the same change set before it was
  committed)

## 1. Environment

| Item | Value |
| --- | --- |
| Codex CLI | 0.165.0 |
| Claude Code | 2.1.301 |
| Gateway | `http://127.0.0.1:3425` (`magpie`) |
| Model used by both Agents | `opencode-go/<configured gateway model>` |
| Report location | temporary directory outside the repository |
| Attestation | none; ordinary development run |

The gateway was reached with:

- Claude: an ephemeral `--claude-settings` file containing `ANTHROPIC_BASE_URL`
  and the gateway token, plus `--claude-model`.
- Codex: `--codex-config` with `model_provider=magpie`, the magpie `/v1`
  `base_url`, `wire_api=responses`, and the gateway bearer token, plus
  `--codex-model`. The harness keeps `--ignore-user-config`, so user plugins and
  unrelated MCP servers stay out of the session.

Neither token nor gateway settings were copied into the report or this document.
The exact model identifier is intentionally shown only as
`opencode-go/<configured gateway model>`; the gateway's model list identifies it
as the 1,000,000-token, 384,000-output DeepSeek V4.1 Flash entry.

## 2. Runs

Both runs selected `--agents codex,claude` and required two Agent passes.

### 2.1 `nf4_block_scale_offset`, `full` mode

Command shape:

```powershell
inferref agent evaluate examples/agent_eval/nf4_block_scale_offset/benchmark.json `
  --agents codex,claude `
  --report-dir <temp>/report `
  --codex-config <temp>/codex-config.json `
  --codex-model opencode-go/MODEL `
  --claude-settings <temp>/claude-settings.json `
  --claude-model opencode-go/MODEL `
  --diagnostic-mode full --json
```

Result: **report pass**, duration about 5.4 minutes.

| Agent | Agent status | Engine runs | Visible | Holdouts | Audit |
| --- | --- | ---: | --- | --- | --- |
| Codex | pass | 3 | pass | pass, pass, pass | valid |
| Claude | pass | 3 | pass | pass, pass, pass | valid |

`agent_model_request_satisfied=true`; repository, evaluator source, benchmark
source, runtime, Claude settings, command chains, audit integrity, and protected
paths were all unchanged.

### 2.2 `kv_cache_index_stride`, `status_only` mode

Same command with the KV benchmark and `--diagnostic-mode status_only`.

Result: **report pass**, duration about 1.6 minutes.

| Agent | Agent status | Engine runs | Visible | Holdouts | Audit |
| --- | --- | ---: | --- | --- | --- |
| Codex | pass | 1 | pass | pass, pass, pass | valid |
| Claude | pass | 2 | pass | pass, pass, pass | valid |

`agent_model_request_satisfied=true` and host integrity checks passed.

## 3. Branch CI

The gateway-support change set was committed as `ae5d64d` and pushed to `main`
with no tag:

- CircleCI workflow `7014dfbf-17d4-40e9-99b3-eb258e74b853`: **16/16 success**
- GitHub Actions `Package distributions` run `40902247581`: **success**
  (`release-assets` skipped for a push event)

## 4. Observations

- The local gateway, both Agent CLIs, the MCP tool bridge, sealed audit, model
  request checks, and host-integrity checks work end to end with real models.
- Codex needed the explicit `--codex-config`/`--codex-model` path because its
  evaluation driver uses `--ignore-user-config`. Claude needed only an ephemeral
  settings file and model override.
- Codex repaired NF4 after three engine runs and Claude after three; neither
  solved it from the prompt at the first run. This is consistent with the
  anti-spoiler task rewrite, but one seed is not enough for a claim.
- The KV `status_only` smoke was too easy to be discriminating: Codex passed on
  its first engine run and Claude on its second. This benchmark should not be
  used as the primary full-vs-status-only contrast without further redesign or
  additional cases.
- These runs are smoke tests, not the planned matrix: there was one run per
  benchmark/mode, no repeat, no no-InferRef arm, no public attestation, and no
  statistical claim about diagnostic benefit.

## 5. Next steps

1. Run the controlled matrix with matched prompts, models, budgets, and fresh
   sessions for both `full` and `status_only`.
2. Decide whether the KV benchmark needs a harder seeded bug or whether the
   primary ablation should use `nf4_block_scale_offset` and `rope_sign`.
3. Define the separate no-InferRef arm before comparing diagnostic modes.
