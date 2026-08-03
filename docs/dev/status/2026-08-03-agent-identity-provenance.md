# Agent identity provenance hardening

Date: 2026-08-03

## Outcome

Attestation format v0.4 separates result correctness from Agent identity evidence.
Formal publication now runs in a fresh isolated Python worker and binds the exact
local Agent executable chain used for version detection and execution. Model
identity is represented with an explicit evidence level rather than an ambiguous
resolved-model string.

## Isolated formal worker

- `evaluate_benchmark(..., public_attestation=...)` delegates to a new
  `python -I -m inferref.agent.evaluation_worker` process.
- Ordinary in-process API runs, including the built-in driver path, produce only
  development evidence. Custom drivers remain unable to request public formal
  output.
- The worker requires Python isolated mode before reading its request and records
  `python_isolated`, its fixed entry module, and a launch-policy SHA-256.
- Formal eligibility requires valid worker evidence in addition to the clean and
  unchanged repository, evaluator, benchmark, runtime, and Agent command evidence.

This prevents parent-process monkeypatches of `run_agent_cli()` from carrying into
the supported formal path. It is process isolation for provenance, not an OS
sandbox or a signature preventing arbitrary same-user JSON fabrication.

## Executable-chain binding

Each Agent command is resolved exactly once. The same frozen command prefix is
used for `--version` and the Agent run. Component bytes are hashed:

1. before version detection;
2. after version detection;
3. after Agent execution.

Formal publication requires all three manifests to match. Public runner evidence
contains component role, basename, size and SHA-256 plus an argv-policy digest and
version output; it contains no absolute executable path.

On the verified Windows host:

- Codex resolves to separate `node.exe` (`runtime`) and `codex.js` (`cli_entry`)
  components;
- Claude Code resolves to the actual `claude.exe` component.

Replacing a component during execution is covered by a deterministic regression
test and invalidates runner evidence.

## Model evidence

Public Agent records now contain:

```json
{
  "requested": "gpt-5.6-sol",
  "reported": null,
  "evidence_source": "unavailable",
  "matches_request": null,
  "identity_level": "requested_only",
  "provider_verified": false
}
```

Claude `system/init` events and supported Codex events produce
`cli_self_reported`; an absent Codex model event remains `requested_only`.
Neither level is interpreted as provider-signed identity, and formal status does
not silently upgrade it.

## Runtime interval

Runtime/distribution evidence is captured before and after Agent execution.
Attestation v0.4 publishes both snapshots and `unchanged`; a changed runtime
invalidates host integrity and formal publication.

## Verification

- Focused Agent evaluation tests: `62 passed`.
- Full Python suite: `435 passed, 5 skipped`.
- Ruff format and lint on changed Python files: clean.
- `uv build`: wheel and source distribution succeeded, including the worker
  module.
- Local executable probe resolved and hashed Codex's two-component chain and
  Claude Code's native executable using the same version commands as evaluation.

## Remaining trust boundaries

- Executable hashes identify the local bytes that ran; they do not prove those
  bytes are vendor-issued. Supply-chain verification would require trusted vendor
  signatures, package provenance, or a published digest allowlist.
- `cli_self_reported` model identity is not provider verification. A future
  provider-signed receipt can add `provider_verified` without changing the
  requested/self-reported distinction.
- The public JSON is not independently signed. Its current persistence trust is
  the Git commit history; DSSE/Sigstore or GitHub artifact attestations remain
  future work.
- Parent-directory TOCTOU and same-user audit rewriting remain the documented
  non-adversarial local-pilot limitations.
