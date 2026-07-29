# Status: Semantic and capture hardening

**Date:** 2026-07-29

**Version:** 0.2 development after `0.2.0`

**Baseline:** commit `18285d5` (static KV-cache validation)

**Tests:** 268 passed locally; 157 core tests remain PyTorch-independent.

## Outcome

Four review findings were confirmed and fixed before expanding the real-model
KV-cache corpus.

### Source detector configuration is real

`SourceFunctionDetector(patterns=...)` now passes its instance patterns into
function matching. Previously the public field was ignored and detection
always used the global defaults. A custom-pattern regression test uses a name
that is absent from the built-in table, so it cannot pass accidentally.

### Equal physical boundaries may carry different semantics

Semantic deduplication now keys on `(semantic name, sorted node ids)`. Duplicate
proposals for the same interpretation still collapse to the highest confidence,
while `MLP` and `SwiGLU` proposals over the same nodes both survive. This
matches the Trace IR rule that semantic regions may overlap and nest; physical
execution truth is unchanged.

### Capture downgrade is explicit and actionable

When `--capture-tensors all` exceeds `--max-capture-elements`, each affected
value records:

```json
{
  "mode": "hash",
  "requested_mode": "full",
  "degraded_reason": "max_capture_elements",
  "limit": 1000000,
  "logical_numel": 4194304
}
```

Capture exceptions similarly record `requested_mode` and
`degraded_reason="capture_error"`; the exception type and message remain in
manifest determinism warnings.

Testcase extraction preserves the old `missing_payloads` id list and adds
`missing_payload_details`. CLI and generated README diagnostics identify the
boundary name, requested/retained mode, configured limit and actual tensor
size. This is additive: traces without the new fields continue to load.

### Identity documentation matches implementation

The PyTorch identity module now documents the actual value key, including
`runtime_object_id`, and distinguishes same-object value interning from
content/layout payload deduplication.

### CI runtime warnings removed

`actions/checkout` and `actions/setup-python` now use v6, whose JavaScript
runtime is Node.js 24. The workflow matrices and test commands are unchanged.

## Verification

```text
targeted semantic / IR / capture / testcase suite   143 passed
full suite                                           268 passed
core suite                                           157 passed
torch 2.1 capture / testcase subset                   44 passed
```

The GitHub-hosted CI matrix still needs to run once on this commit to verify
the action-major upgrades and all supported PyTorch combinations.

## Remaining risks

- Real Hugging Face KV-cache mutation remains the next high-value corpus test.
- `max_capture_elements` avoids writing a full payload but still materialises
  canonical bytes to retain a full-content hash; it is a disk cap, not a peak
  memory cap.
- Capture failure details in testcase metadata use a stable reason code; the
  exception text remains package-level to avoid copying environment-specific
  error strings into every value.
- Multi-label identical boundaries are represented as separate regions. A
  future multi-annotation region schema could reduce duplication, but would be
  a semantic schema decision rather than a deduplication tweak.
- The repository does not currently declare Ruff or another static linter, so
  this round is verified by tests and serialization round-trips rather than a
  lint gate.

## Next step

Run the complete GitHub Actions matrix, then add a real Hugging Face cache
prefill/decode trace. Prefer an effect-backed adapter or detector over matching
the generic method name `update` globally.
