# Formal attestation v0.4 identity-bound PASS

Date: 2026-08-03

## Outcome

The first real attestation v0.4 run passed from clean commit
`0cac0a5a12af9c0abe590090f52b02d692997f30`. The supported public path launched
a fresh Python-isolated worker, bound the local Agent executable chains, preserved
explicit model evidence levels, and passed the strict 2/2 numerical gate.

## Gate result

| Agent | CLI | Model evidence | Visible runs | Final visible | Holdouts | Command chain |
| --- | --- | --- | ---: | --- | --- | --- |
| Codex | `codex-cli 0.146.0` | `gpt-5.6-sol`, `requested_only` | 1 | PASS | 3/3 PASS | unchanged |
| Claude Code | `2.1.116 (Claude Code)` | `deepseek-v4-flash[1m]`, `cli_self_reported`, match | 1 | PASS | 3/3 PASS | unchanged |

Both records state `provider_verified: false`. The result does not upgrade a model
request or CLI event into provider-certified identity.

## Worker and executable evidence

- runner mode: `isolated_builtin_cli_worker`;
- worker Python isolated: `true`;
- worker entry: `inferref.agent.evaluation_worker`;
- repository, evaluator, benchmark, runtime and command chains: unchanged;
- Codex runtime `node.exe` SHA-256:
  `f13ac3ca23248dc389507e8fe38c34489ab7edb3e6d6700eb6da6a0b7e128eaf`;
- Codex entry `codex.js` SHA-256:
  `134063e133f0b4244fa3b251acf973d4fe4b4aeeacbdc135211bf480f59f1477`;
- Claude executable `claude.exe` SHA-256:
  `41c82dc1988225938a7d588b157aef3cd3c2fd62ff0661731132404d5bf17258`.

Each component had identical before-version, after-version and after-execution
evidence. Public records contain basenames and hashes, not executable paths.

## Artifact provenance

- benchmark SHA-256:
  `84ef708f9be345b1674555c64125b4633444eb74c71f411d98c1c64d0f4f6c8e`;
- evaluator source-tree SHA-256:
  `efcedfdd69742a06069fe50716627004914083334c38e44c5ca128b2c3afc345`;
- private `report.json` SHA-256:
  `58629b3bfa4e67c83ed3594eb44582d949ec6d8d4ee934788bdc18b29fa530cd`;
- public attestation SHA-256:
  `7d9c21a80c9071808e578e7a6ce08c31ee3a8e3dc8561f9205ad70387556ad4a`.

Public evidence:

- [formal v0.4 attestation](../attestations/2026-08-03-rope-dual-agent-0cac0a5.json)
- [GitHub Actions run 30824062848](https://github.com/Blackwood416/InferRef/actions/runs/30824062848) — 19/19 jobs passed

The committed artifact passed a scan for absolute local paths, `.scratch`, local
provider settings names, prompt/stdout/stderr fields and reasoning markers. Raw
model streams remain private and are represented by hashes only.

## Interpretation

This artifact provides strong evidence for the exact local CLI bytes and command
policy used to produce two independently validated candidate patches. It does not
prove the CLI bytes are vendor-signed, does not provider-verify the remote model,
and is not itself digitally signed. Those remain separate supply-chain and remote
identity layers.
