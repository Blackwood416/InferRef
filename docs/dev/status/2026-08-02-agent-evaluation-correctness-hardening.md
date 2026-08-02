# Agent evaluation correctness hardening

Date: 2026-08-02

## State

The two P1 evaluation-contract gaps found after the first real pilot are fixed.
The final candidate, rather than any historical MCP run, now determines visible
and holdout acceptance. The benchmark `success` object is a parsed and enforced
policy. A redacted, independently inspectable attestation for the post-fix real
2/2 run is committed with this snapshot.

## Final-candidate acceptance

Historical `inferref_run_engine` records are retained only to prove that the Agent
used InferRef and reached a visible PASS within its four-run interaction budget.
After the Agent exits, the host:

1. reads the final `engine.py` once;
2. records its SHA-256;
3. creates an independent execution copy from those frozen bytes for final visible;
4. creates three more independent copies from the same bytes for the holdouts;
5. requires final visible and holdout verdicts according to the success policy.

The regression test covers the exact adversarial sequence that motivated this
change: engine A obtains a historical visible PASS, engine B then passes all three
holdouts but fails visible. The evaluator now returns `agent_failure`.

## Success policy

`EvaluationSuccessPolicy` parses and validates:

- `required_agent_passes`;
- `visible_status` (`pass` in format v0.2);
- `all_holdouts_pass`;
- `protected_paths_unchanged`.

The overall threshold and per-Agent acceptance use these values. Tests prove that
changing `required_agent_passes` changes the overall result, disabling the holdout
requirement changes overfit gating, and malformed policy values are rejected.

## Public attestation

The evaluator always writes a redacted `attestation.json` beside the private
report and can atomically write a second fresh copy with
`--public-attestation PATH`. The public schema includes:

- source commit and benchmark/evaluator/private-report hashes;
- requested/resolved model and CLI version;
- required MCP call sequence and engine-run count;
- final engine SHA-256 and unified patch;
- final visible and holdout verdicts;
- protected-file hashes;
- raw transcript/stderr hashes and available usage/cost.

It excludes transcript text, reasoning, credentials, reference payloads, prompts,
and absolute local paths.

The first committed attestation is:

- [`2026-08-02-rope-dual-agent-e63492e.json`](../attestations/2026-08-02-rope-dual-agent-e63492e.json)
- code commit: `e63492eacc48b3e2f43d5cdadd58b1dc3f4a7730`
- benchmark SHA-256: `84ef708f9be345b1674555c64125b4633444eb74c71f411d98c1c64d0f4f6c8e`
- attestation SHA-256: `d0749d546fd8575e0d29629ed1785b6034e525772dbf4a8a9528da2cd6f71450`

## Post-fix real 2/2 run

### Codex

- requested model: `gpt-5.6-sol`
- CLI: `codex-cli 0.146.0`
- Agent duration: 39.465 seconds
- historical engine runs: 1 of 4
- frozen final visible: PASS
- frozen holdouts: 3/3 PASS

### Claude Code with DeepSeek

- benchmark alias: `opus`
- resolved model: `deepseek-v4-flash[1m]`
- CLI: Claude Code 2.1.116
- Agent duration: 20.714 seconds
- historical engine runs: 2 of 4
- frozen final visible: PASS
- frozen holdouts: 3/3 PASS
- reported cost: USD 0.147905

Both final engines have SHA-256
`b54e4093af5addc24735031ffc3ed39abb6d1e6f71f9b75fb17e27cd5e490a66`.
There were no protected-path, staged-input, run-budget, or tool-order violations.

## Verification

```text
evaluator tests: 25 passed
Core + Agent:    262 passed
full suite:      396 passed, 5 skipped
build:           wheel and sdist succeeded
```

## Scope and next experiment

“Blind” now explicitly means candidate-input blind, oracle-isolated, and hidden
from the Agent workspace. Holdout metadata is public, all processes share an OS
user, and final-state hashes cannot detect reads or modify-then-restore behavior.
This is not adversarial secrecy or an OS sandbox.

The Pilot proves that real Agents can use the protocol; it does not establish the
incremental value of structured numerical diagnostics. The next experiment should
run the same RoPE task and models under three conditions:

1. full structured first-divergence diagnostics;
2. PASS/FAIL-only InferRef output;
3. ordinary test execution without InferRef.

Compare success, engine runs, first-correct-patch round, tokens, wall time, and
overfit rate. This ablation should precede expanding the repair corpus.
