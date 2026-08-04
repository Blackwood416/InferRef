# InferRef 0.4.0 — pluggable semantics, capabilities, and suites

## Delivered

- explicit PyPA entry-point discovery for third-party semantic detectors;
- engine adapter v0.2 capabilities and testcase v0.2 requirements;
- pre-execution structured `unsupported` results;
- backward-compatible reading of adapter/testcase 0.1;
- split adapter lifecycle modules for schema, execution, process policy,
  artifact policy, and run records;
- `inferref-suite` v0.1 with containment-safe validation and sequential runs;
- required no-download Tiny Llama and Tiny Qwen3.5 tests on Intel XPU.

## XPU findings

Both models execute and trace successfully on the local Intel Arc A770 with
PyTorch 2.13.0+xpu and Transformers 5.14.1. Tiny Llama creates two CPU scalar
control tensors inside its XPU layer trace, so automatic execution metadata
correctly reports `mixed` rather than hiding the cross-device execution.

Tiny Qwen3.5 covers both recurrent/state cache mutation and conventional KV
cache concatenation during prefill/decode without downloading weights.
