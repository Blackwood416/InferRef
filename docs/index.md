# InferRef documentation

## Start here

- [Installation, quick start, and current limitations](../README.md)
- [Agent workflow and capability preflight](AGENT_WORKFLOW.md)
- [Alignment sessions and planning](spec/InferRef_0.10_Alignment_Orchestrator.md)
- [Suite execution and count semantics](spec/InferRef_Suite_v0.1.md)
- [Building and distributing packages](DISTRIBUTION.md)

## Build an integration

- [Extension guide](EXTENDING.md) and [semantic plugins](semantic-plugins.md)
- [Engine adapter contract](spec/InferRef_Engine_Adapter_v0.2.md)
- [Comparator plugins](spec/InferRef_0.9_Semantic_Validation.md)
- [Native FP8 and quantization implementation plan](spec/InferRef_Quantization.md)
- [Stateful scenarios](spec/InferRef_Scenario_v0.1.md)
- [Agent protocol](spec/InferRef_Agent_Protocol_v0.1.md)
- [XPU/SYCL guide](xpu-sycl.md)

## Reference and project history

- [Trace IR and frozen wire types](spec/InferRef_Trace_IR_v0.1.md)
- [Overall specification and roadmap](spec/InferRef_SPEC.md)
- [0.12 core infrastructure plan](spec/InferRef_0.12_Core_Infrastructure.md),
  [execution plan](spec/InferRef_0.12_Execution_Plan.md), and
  [W8A8 contract freeze](spec/InferRef_0.12_W8A8_Contract.md)
- [Changelog](../CHANGELOG.md) and [contribution guide](../CONTRIBUTING.md)
- [Dated development status](dev/status/README.md)

Specifications may include future work: a roadmap entry is not a supported
runtime capability. Dated status/attestation files preserve historical versions
and test environments; consult the README, changelog, and current API contracts
for present behavior rather than treating old snapshots as installation advice.
