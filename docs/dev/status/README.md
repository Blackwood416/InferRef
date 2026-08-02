# Development status reports

Point-in-time snapshots of where the project stands: what works, what is deliberately deferred,
what is known to be broken or untested, and what to do next.

## Convention

One file per snapshot, named `YYYY-MM-DD-<slug>.md`. Snapshots are **not** edited after the fact —
write a new one instead, so the history stays readable. The newest entry at the top of the index
below is the current state.

Each report should cover:

- phase status against [SPEC §64](../../spec/InferRef_SPEC.md)
- acceptance criteria against [Trace IR §57](../../spec/InferRef_Trace_IR_v0.1.md)
- what exists, with enough detail to orient someone who has not read the code
- verified environment, and explicitly what is *not* verified
- implementation decisions that constrain future changes
- known gaps and limitations, including test coverage gaps
- suggested next steps
- commands to reproduce the claimed state

The gaps section matters most. A status report that only lists accomplishments is not useful for
planning.

## Index

| Date | Report | State |
| --- | --- | --- |
| 2026-08-02 | [Agent boundary hardening](2026-08-02-agent-boundary-hardening.md) | MCP roots; testcase validator; streamed limits + Job/process-group cleanup; HF/GPU CI; 360 tests |
| 2026-08-02 | [Agent repair evaluation](2026-08-02-agent-repair-evaluation.md) | Bounded RoPE repair benchmark; CLI/MCP FAIL→one-file fix→PASS; 328 tests |
| 2026-08-02 | [Agent integration vertical slice](2026-08-02-agent-integration-v0.3.md) | Agent envelope; trusted adapter runner; CLI + MCP v2; 325 tests |
| 2026-07-30 | [Testcase and effect hardening](2026-07-30-testcase-effect-hardening.md) | Metadata/content split; self-contained effects; codec agreement; 314 tests |
| 2026-07-30 | [Foundation reviewer hardening](2026-07-30-foundation-review-hardening.md) | Empty-storage identity; storage rebinding; mutation validation; opaque op ordering; 292 tests |
| 2026-07-30 | [Qwen3.5 hybrid-cache validation](2026-07-30-qwen35-hybrid-cache-validation.md) | Official 0.8B weights; DeltaNet state + full-attention KV cache; 284 tests |
| 2026-07-30 | [Real Hugging Face KV-cache validation](2026-07-30-hf-kv-cache-validation.md) | Real Llama Dynamic/StaticCache prefill+decode; effect-backed cache detector; 277 tests |
| 2026-07-29 | [Semantic and capture hardening](2026-07-29-semantic-capture-hardening.md) | Configurable source detection; multi-label boundaries; explicit capture degradation; 268 tests |
| 2026-07-29 | [KV-cache validation](2026-07-29-kv-cache-validation.md) | Static prefill/decode cache; mutation effect edges; 263 tests |
| 2026-07-28 | [Semantic detection](2026-07-28-semantic-detection.md) | Automatic region detection; operator ids gone from the workflow; 235 tests |
| 2026-07-28 | [Correctness hardening](2026-07-28-correctness-hardening.md) | Mutation binding + value/payload identity fixed; CI added (unexecuted); 167 tests |
| 2026-07-27 | [MVP complete](2026-07-27-mvp-complete.md) | SPEC §64 Phases 0-3 done; 148 tests passing; Windows only |
