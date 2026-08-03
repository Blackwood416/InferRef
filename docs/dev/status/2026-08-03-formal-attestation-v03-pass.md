# Formal attestation v0.3 dual-Agent PASS

Date: 2026-08-03

## Outcome

The first real `inferref-agent-evaluation-attestation` v0.3 run passed its strict
dual-Agent gate from clean commit `08b8801fc8c9ce3c56895fdf4932c74fd1d911e6`.
This run exercised the built-in CLI path; no custom evaluation driver was used.

## Gate result

| Agent | Requested/resolved model | CLI | Visible runs | Final visible | Holdouts | Audit |
| --- | --- | --- | ---: | --- | --- | --- |
| Codex | `gpt-5.6-sol` / not reported by CLI | `codex-cli 0.146.0` | 1 | PASS | 3/3 PASS | valid |
| Claude Code | `deepseek-v4-flash[1m]` | `2.1.116 (Claude Code)` | 2 | PASS | 3/3 PASS | valid |

The overall policy passed 2/2. Repository, imported evaluator source, and loaded
benchmark bytes were unchanged across the evaluation interval. Both final engine
patches were silently rerun against the visible case and all three holdouts.

## Provenance

- attestation level: `formal`;
- runner mode: `builtin_cli`;
- Python: CPython 3.13.11 on Windows AMD64;
- NumPy: 2.5.1;
- InferRef: 0.3.0, imported from repository source;
- MCP SDK: 2.0.0;
- benchmark SHA-256:
  `84ef708f9be345b1674555c64125b4633444eb74c71f411d98c1c64d0f4f6c8e`;
- evaluator source-tree SHA-256:
  `265a35910000fac0d2640f9d58bf0a671d806c023cf3c30c9cfdb06a5485dbb0`;
- private `report.json` SHA-256:
  `d5701e201ad3280e50b3135608178c0cdd1cf31fc1230806f81610e20afabd54`;
- public attestation SHA-256:
  `1610faaf29dc2123a9fe3fb1dbb66f98fff5c1718d6877de80fd1c29f7ea4c7b`.

Public evidence:

- [formal v0.3 attestation](../attestations/2026-08-03-rope-dual-agent-08b8801.json)
- [GitHub Actions run 30817866233](https://github.com/Blackwood416/InferRef/actions/runs/30817866233) — 19/19 jobs passed for the attested commit

The private JSONL transcripts remain under the ignored `.scratch` directory and
are represented publicly only by hashes. The public artifact contains no complete
reasoning, credentials, provider settings, reference payloads, or absolute local
paths.

## Scope

This result proves that, under the documented local-pilot threat model, the
built-in Agent runner produced two independently validated RoPE repairs while the
bound host evidence remained unchanged. It does not prove an OS sandbox,
adversarial secrecy, or a general repair-success advantage over ordinary tests.
The next empirical step remains the structured-diagnostics/PASS-FAIL/no-InferRef
ablation.
