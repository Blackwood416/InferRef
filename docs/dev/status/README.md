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
| 2026-08-21 | [v0.9.2 core-linux lint gate fix](2026-08-21-v092-core-linux-lint-fix.md) | Ruff policy + lint fixes; core-linux green; 749 tests |
| 2026-08-21 | [v0.9.1 core-linux MCP test fix](2026-08-21-v091-semantic-validation-fix.md) | Skip optional-MCP comparator test on core-only installs; Linux core 504 passed |
| 2026-08-21 | [v0.9.0 Semantic Validation](2026-08-21-v090-semantic-validation.md) | Comparison Spec v0.1, comparator plugins, multi-output comparator, agent summary, region recommend, doctor details; 749 tests |
| 2026-08-21 | [v0.8.0 Adapter DX and v0.8.1 CI hardening](2026-08-21-v080-adapter-dx-and-v081-ci.md) | C++ testcase helper + adapter scaffold + runtime bridge; Windows Python via uv; NMake C++ CI; 22/22 green |
| 2026-08-18 | [v0.7.0 CI retrigger and upstream compatibility fixes](2026-08-18-v070-ci-upstream-compat.md) | Re-triggered lost v0.7.0 CI; hardened Qwen3.5 latest test and Python 3.10 CLI parsing |
| 2026-08-16 | [Agent workflow guide and Codex skill](2026-08-16-agent-workflow-skill.md) | AGENT_WORKFLOW guide + inferref Codex skill; stdlib validation harness; README pointer; 667 tests; docs-only milestone |
| 2026-08-08 | [v0.6.0 contract ecosystem and CI/XPU gate](2026-08-08-v060-contract-ecosystem.md) | Contract registry + Scenario v0.1; Suite/Agent/CLI integration; 658 tests; Windows/XPU gates green; tag pipelines deduplicated |
| 2026-08-07 | [Contract dtype ABI and CircleCI cost tiers](2026-08-07-dtype-abi-and-circleci-cost-tiers.md) | Required dtype metadata; duplicate contract rejection; CircleCI 3-tier Windows 7→3; numpy-only cpp-windows; on_fail artifacts |
| 2026-08-07 | [Gate safety and contract ABI details](2026-08-07-gate-safety-and-contract-abi-details.md) | Safe evidence-dir deletion; scalar epsilon; per-contract subset; KV dtype relation; JSON control chars |
| 2026-08-07 | [Contract observable outputs and atomic extraction](2026-08-07-contract-outputs-and-atomic-extraction.md) | Exact output role sets; input/output relation validators; staging+atomic publish; single contract per testcase; 529 tests |
| 2026-08-06 | [Executable contract profiles and per-contract capability](2026-08-06-executable-contract-profiles.md) | Contract registry/role binding; per-contract preflight; native JSON parser; fresh XPU gate dirs; suite validate runnable split |
| 2026-08-06 | [Agent attestation v0.5 hardening](2026-08-06-attestation-evidence-v05-hardening.md) | Claude settings binding; model identity policy; canonical worker launch policy; strict validators; 493 tests |
| 2026-08-03 | [Formal attestation v0.4 identity-bound PASS](2026-08-03-formal-attestation-v04-pass.md) | Python-isolated worker; bound Codex/Claude executable chains; explicit model evidence; 2/2 PASS |
| 2026-08-03 | [Agent identity provenance hardening](2026-08-03-agent-identity-provenance.md) | Isolated formal worker; executable-chain hashes; model evidence levels; runtime interval; 435 tests |
| 2026-08-03 | [Formal attestation v0.3 dual-Agent PASS](2026-08-03-formal-attestation-v03-pass.md) | Built-in runner; clean/unchanged evidence; Codex + DeepSeek/Claude Code 2/2; CI 19/19 |
| 2026-08-03 | [Agent attestation provenance hardening](2026-08-03-attestation-provenance-hardening.md) | Formal built-in runner only; before/after repository binding; audit FSM; runtime evidence; 426 tests |
| 2026-08-03 | [Evaluation evidence and container-alias hardening](2026-08-03-evaluation-evidence-hardening.md) | Source manifest; crash-consistent audit; recursive aliases; formal v0.2 Agent gate 2/2; 414 tests |
| 2026-08-02 | [Agent evaluation correctness hardening](2026-08-02-agent-evaluation-correctness-hardening.md) | Frozen final visible+holdouts; enforced success policy; public attestation; post-fix real 2/2 |
| 2026-08-02 | [Real dual-Agent repair pilot PASS](2026-08-02-real-agent-pilot-pass.md) | Codex + DeepSeek/Claude Code 2/2; visible 2/2; hidden 6/6; no integrity violations |
| 2026-08-02 | [Real Agent repair pilot v0.2](2026-08-02-real-agent-pilot-v02.md) | Blind harness complete; Codex PASS; Claude provider limited; gate 1/2; CI 19/19 |
| 2026-08-02 | [Adversarial input and resource hardening](2026-08-02-adversarial-input-hardening.md) | Total testcase validation; bounded artifact traversal; Windows fixed HF; 374 tests; 19/19 Actions |
| 2026-08-02 | [Agent boundary CI green](2026-08-02-agent-boundary-ci-green.md) | 18/18 Actions jobs; Windows/Linux path/process controls; HF 4.40 + Qwen 5.14/latest |
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
