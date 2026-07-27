# InferRef Specification

> **Status:** Draft  
> **Version:** 0.1  
> **Project:** InferRef  
> **Document type:** Architecture & Product Specification

---

## 1. Overview

**InferRef** is a reference execution, tracing, testcase extraction, and numerical comparison framework for inference engine development.

Its primary goal is to transform a model implementation—initially PyTorch / Hugging Face Transformers—into a **stable, machine-readable reference specification** that can be consumed by custom inference engines, kernel test harnesses, and coding agents.

InferRef is designed to solve a common bottleneck in inference engine development:

> Writing or iterating a kernel is often fast; proving that the kernel matches the reference implementation is slow.

Today, adapting a new model frequently requires engineers to repeatedly:

1. inspect `modeling_*.py`;
2. trace call chains manually;
3. infer tensor layouts and broadcasting semantics;
4. dump intermediate tensors;
5. write one-off comparison scripts;
6. rerun the entire model;
7. locate the first numerical divergence;
8. repeat the process for the next model or architecture.

InferRef turns this workflow into a reusable pipeline:

```text
Model
  ↓
Reference Trace
  ↓
Trace IR
  ↓
Testcase / Region Extraction
  ↓
Engine Execution
  ↓
Automatic Numerical Comparison
  ↓
First Divergence Report
```

The project is intended to serve both humans and coding agents.

---

# 2. Goals

InferRef SHALL provide the following capabilities.

## 2.1 Reference execution tracing

InferRef SHALL capture the actual runtime behavior of a model, including:

- operator execution order;
- operator arguments;
- tensor inputs and outputs;
- shape;
- stride;
- dtype;
- device;
- storage offset;
- contiguity;
- scalar arguments;
- module ownership;
- source location;
- producer / consumer relationships.

The reference implementation is initially PyTorch.

---

## 2.2 Static graph inspection

InferRef SHOULD extract a normalized static representation of supported model regions when possible.

For the PyTorch frontend, this may use facilities such as:

- `torch.export`;
- FX graphs;
- ATen-level graph representations;
- decomposition passes.

Static graph information SHALL be complementary to runtime traces rather than treated as an authoritative replacement for runtime execution.

---

## 2.3 Stable Trace IR

InferRef SHALL define its own versioned intermediate representation.

The Trace IR SHALL NOT expose PyTorch internal objects directly.

This is required so that:

- recorded traces remain usable across framework versions;
- engine-side tooling does not depend on PyTorch;
- future frontends can be added;
- PyTorch tracing implementation details can change independently.

---

## 2.4 Tensor testcase extraction

InferRef SHALL be able to extract standalone testcases from a model execution.

A testcase MAY represent:

- one primitive operator;
- one logical operator;
- a group of primitive operators;
- one semantic region;
- one fused engine kernel boundary.

Each testcase SHALL contain enough information to reproduce reference outputs without rerunning the original model.

---

## 2.5 Numerical comparison

InferRef SHALL support automatic numerical comparison between reference outputs and inference-engine outputs.

The comparator SHOULD support at least:

- exact equality;
- absolute error;
- relative error;
- mean absolute error;
- root mean square error;
- maximum absolute error;
- maximum relative error;
- cosine similarity;
- NaN / Inf detection.

---

## 2.6 First-divergence debugging

InferRef SHALL provide a mode that identifies the earliest point where engine execution diverges from reference execution.

A typical report should look like:

```text
PASS #411
PASS #412
PASS #413
FAIL #414

Reference node:
  aten.mul.Tensor

Semantic region:
  RotaryEmbedding

Source:
  transformers/models/.../modeling_xxx.py:527

Input A:
  MATCH

Input B:
  MATCH

Output:
  MISMATCH

Max abs error:
  0.21484375

First mismatch:
  index = [0, 3, 29, 41]

Reference:
  0.718750

Actual:
  0.691406
```

---

## 2.7 Coding-agent friendly workflow

InferRef SHOULD expose all core functionality through non-interactive CLI commands and stable machine-readable output.

A coding agent SHOULD be able to perform the following loop without manually reading the whole model implementation:

```text
inspect failure
    ↓
modify kernel
    ↓
build
    ↓
run isolated testcase
    ↓
receive PASS / FAIL
```

This feedback loop is one of the core design objectives of InferRef.

---

# 3. Non-Goals

The initial versions of InferRef SHALL NOT attempt to:

- become a general-purpose deep learning compiler;
- replace ONNX;
- replace TorchInductor;
- replace PyTorch Profiler;
- optimize model execution;
- generate production kernels automatically;
- force custom engines to reproduce PyTorch's operator partitioning;
- define one universal high-level neural network IR;
- guarantee bitwise-identical execution across all hardware.

InferRef is primarily a **reference, debugging, interoperability, and validation system**.

---

# 4. Terminology

## 4.1 Physical Operator

A **Physical Operator** is an operator actually observed in the reference execution.

Examples:

```text
aten.mm.default
aten.mul.Tensor
aten.add.Tensor
aten.slice.Tensor
aten.cat.default
```

Physical operators represent runtime truth.

---

## 4.2 Semantic Operator

A **Semantic Operator** is a higher-level interpretation attached to one or more physical operators.

Examples:

```text
Linear
RMSNorm
RoPE
SwiGLU
GroupedQueryAttention
GatedDeltaRule
```

Semantic annotations SHALL NOT replace physical operators.

They are advisory metadata intended for:

- humans;
- viewers;
- coding agents;
- region grouping;
- engine mappings.

---

## 4.3 Reference Region

A **Reference Region** is a named subgraph with explicit inputs and outputs.

For example:

```text
Region: RotaryEmbedding

Inputs:
  query
  key
  cos
  sin

Reference nodes:
  #411 - #429

Outputs:
  query_embed
  key_embed
```

A region may map to one engine kernel even when its reference implementation contains many operators.

---

## 4.4 Trace Tensor

A **Trace Tensor** is a first-class object in the Trace IR.

A tensor SHALL be independently identifiable from the operator that produced it.

This allows InferRef to represent the computation as a true dataflow graph.

---

# 5. High-Level Architecture

```text
                       +----------------------+
                       |      Model Input     |
                       +----------+-----------+
                                  |
                                  v
                 +----------------------------------+
                 |         Frontend Adapter         |
                 |                                  |
                 |  PyTorch / Transformers          |
                 |  future: JAX / ONNX Runtime ...  |
                 +----------------+-----------------+
                                  |
                 +----------------+-----------------+
                 |                                  |
                 v                                  v
        +------------------+              +------------------+
        |  Runtime Tracer  |              |  Static Analyzer |
        |                  |              |                  |
        | dispatch trace   |              | export / FX      |
        | real tensors     |              | static graph     |
        | runtime order    |              | source metadata  |
        +--------+---------+              +--------+---------+
                 |                                 |
                 +---------------+-----------------+
                                 |
                                 v
                       +-------------------+
                       |     Trace IR      |
                       +---------+---------+
                                 |
            +--------------------+----------------------+
            |                    |                      |
            v                    v                      v
     +-------------+      +-------------+      +----------------+
     | Tensor Store|      | Regionizer  |      | Source Mapping |
     +------+------+      +------+------+      +-------+--------+
            |                    |                      |
            +--------------------+----------------------+
                                 |
                                 v
                       +-------------------+
                       | Trace Package     |
                       +---------+---------+
                                 |
             +-------------------+-------------------+
             |                   |                   |
             v                   v                   v
      +-------------+     +-------------+     +--------------+
      | Inspector   |     | Comparator  |     | Test Extractor|
      +-------------+     +------+------+     +--------------+
                                |
                                v
                       +------------------+
                       | Engine Adapter   |
                       +------------------+
```

---

# 6. Frontend Architecture

InferRef SHALL use a frontend abstraction.

A frontend converts one framework's model execution into InferRef Trace IR.

Possible frontends:

```text
inferref.frontend.pytorch
inferref.frontend.onnxruntime
inferref.frontend.jax
```

Only the PyTorch frontend is required for the MVP.

---

# 7. PyTorch Frontend

## 7.1 Runtime tracing

The PyTorch frontend SHOULD use dispatcher-level interception where practical.

Candidate mechanisms include:

- `TorchDispatchMode`;
- dispatcher hooks;
- PyTorch debugging facilities;
- framework-supported operator tracing APIs.

The runtime tracer SHALL preserve the real execution order.

---

## 7.2 Module tracing

Module-level tracing SHOULD be supported as an additional hierarchy.

Example:

```text
model
└── layers.0
    ├── input_layernorm
    ├── self_attn
    │   ├── q_proj
    │   ├── k_proj
    │   ├── v_proj
    │   └── o_proj
    └── mlp
```

This hierarchy is useful for navigation but SHALL NOT be used as the only tracing granularity.

---

## 7.3 Static analysis

Where supported, the PyTorch frontend SHOULD capture:

- exported graph;
- ATen graph;
- node metadata;
- module stack;
- source-function stack;
- stack traces;
- symbolic shapes;
- decomposition information.

Static information MAY be unavailable for some dynamic models.

Failure to export a graph SHALL NOT prevent runtime tracing.

---

# 8. Trace IR

The Trace IR is the central architectural component of InferRef.

It SHALL be:

- framework-independent;
- versioned;
- serializable;
- human-inspectable;
- machine-friendly;
- forward-extensible.

---

# 9. Trace Manifest

Every trace package SHALL contain a manifest.

Example:

```json
{
  "format": "inferref-trace",
  "format_version": "0.1",
  "created_by": "InferRef 0.1.0",
  "frontend": "pytorch",
  "framework_version": "2.x",
  "model": {
    "name": "example-model",
    "revision": null
  },
  "capture": {
    "device": "cpu",
    "tensor_policy": "selected",
    "source_mapping": true
  }
}
```

The manifest SHOULD also contain environment information.

Examples:

- operating system;
- Python version;
- backend;
- device name;
- framework versions;
- Transformers version;
- model revision;
- random seed;
- precision mode.

---

# 10. Operator Record

A physical operator record MAY look like:

```json
{
  "id": 382,
  "kind": "operator",
  "op": "aten.mm.default",

  "inputs": [
    {
      "kind": "tensor",
      "tensor_id": 192
    },
    {
      "kind": "tensor",
      "tensor_id": 193
    }
  ],

  "outputs": [
    {
      "kind": "tensor",
      "tensor_id": 194
    }
  ],

  "module_path": "model.layers.17.self_attn.q_proj",

  "source": {
    "file": "modeling_example.py",
    "line": 418,
    "function": "forward"
  },

  "semantic": {
    "name": "Linear",
    "confidence": 1.0
  }
}
```

---

# 11. Tensor Record

A tensor record SHOULD include:

```json
{
  "id": 194,

  "dtype": "float16",

  "shape": [1, 128, 3584],

  "stride": [458752, 3584, 1],

  "device": "cpu",

  "storage_offset": 0,

  "contiguous": true,

  "requires_grad": false,

  "producer": 382,

  "consumers": [383],

  "data": {
    "storage": "external",
    "path": "tensors/194.irtensor"
  }
}
```

---

# 12. Tensor Metadata

At minimum, InferRef SHOULD capture:

- dtype;
- rank;
- dimensions;
- stride;
- storage offset;
- logical byte size;
- storage byte size;
- device;
- contiguity;
- scalar status;
- producer;
- consumers.

Optional metadata MAY include:

- minimum;
- maximum;
- mean;
- variance;
- norm;
- histogram;
- hash;
- NaN count;
- Inf count.

---

# 13. Tensor Storage Format

InferRef SHOULD define a framework-independent tensor file format.

Suggested extension:

```text
.irtensor
```

A possible structure:

```text
+-------------------------+
| magic                   |
+-------------------------+
| version                 |
+-------------------------+
| dtype                   |
+-------------------------+
| rank                    |
+-------------------------+
| shape[]                 |
+-------------------------+
| stride[]                |
+-------------------------+
| storage_offset          |
+-------------------------+
| logical_size            |
+-------------------------+
| storage_size            |
+-------------------------+
| flags                   |
+-------------------------+
| raw data                |
+-------------------------+
```

The format SHOULD be easy to implement in:

- C++;
- Rust;
- C#;
- Python.

---

# 14. Tensor Capture Policies

Dumping every tensor may require excessive memory and storage.

InferRef SHALL support configurable tensor capture policies.

Examples:

```text
none
metadata-only
hash-only
outputs
selected
region
all
```

Example CLI:

```bash
inferref trace model.py \
  --capture-tensors selected \
  --scope model.layers.0
```

---

# 15. Deduplication

InferRef SHOULD avoid unnecessarily storing duplicate tensor data.

Possible strategies:

- content hashes;
- shared storage tracking;
- parameter references;
- immutable tensor reuse;
- testcase-level deduplication.

Large model parameters SHOULD NOT be duplicated for every operator invocation.

---

# 16. Source Mapping

InferRef SHOULD map runtime operators to source-level context.

A source mapping MAY contain:

```json
{
  "file": "transformers/models/example/modeling_example.py",
  "line": 527,
  "function": "apply_rotary_pos_emb",

  "source_text":
    "q_embed = (q * cos) + (rotate_half(q) * sin)"
}
```

Source mapping SHOULD support:

- direct runtime stack mapping;
- static graph metadata;
- module stack;
- source function stack.

InferRef SHOULD prefer framework-provided metadata where available.

---

# 17. Semantic Annotation

Semantic annotation SHOULD be implemented as a separate analysis pass.

Example:

```json
{
  "region": "RotaryEmbedding",
  "confidence": 0.97,
  "detector": "builtin.rope.pattern.v1"
}
```

Detectors MAY use:

- module names;
- Python source functions;
- graph patterns;
- operator sequences;
- shape patterns;
- parameter identities.

Semantic analysis SHALL be optional.

Physical tracing SHALL remain usable without it.

---

# 18. Reference Regions

Reference Regions are a major feature of InferRef.

A region SHALL contain:

- region ID;
- name;
- node set;
- external inputs;
- external outputs;
- source mapping;
- semantic label;
- optional engine mapping.

Example:

```json
{
  "id": "region_17",
  "name": "RotaryEmbedding",

  "nodes": [411, 412, 413, 414, 415, 416],

  "inputs": [
    201,
    202,
    203,
    204
  ],

  "outputs": [
    211,
    212
  ]
}
```

---

# 19. Region Extraction

Regions may be created by:

1. explicit user selection;
2. module boundaries;
3. source-function boundaries;
4. semantic pattern matching;
5. graph selection;
6. engine mapping definitions.

Example:

```bash
inferref region create trace/ \
  --from-op 411 \
  --to-op 429 \
  --name RotaryEmbedding
```

---

# 20. Engine Mapping

InferRef SHALL NOT require engine operators to match reference operators one-to-one.

An engine kernel MAY correspond to multiple physical reference operators.

Example:

```text
Reference:

slice
neg
cat
mul
mul
add

Engine:

SYCLRotaryEmbeddingKernel
```

Mapping:

```json
{
  "engine_op": "SYCLRotaryEmbeddingKernel",
  "reference_region": "region_17"
}
```

This abstraction is required for fused inference engines.

---

# 21. Engine Adapter

InferRef SHOULD define a small engine-side adapter contract.

The adapter SHALL NOT require a dependency on PyTorch.

Conceptually:

```cpp
struct TraceTensorView
{
    DataType dtype;

    std::span<const int64_t> shape;
    std::span<const int64_t> stride;

    std::span<const std::byte> data;
};

struct EngineResult
{
    std::vector<OwnedTensor> outputs;
};
```

An adapter may expose:

```cpp
EngineResult Run(
    const ReferenceTestCase &testCase
);
```

---

# 22. Engine Output Protocol

For maximum interoperability, engines SHOULD also be allowed to emit output tensors without linking InferRef as a library.

Example directory:

```text
engine-output/
├── manifest.json
├── tensor_194.irtensor
├── tensor_202.irtensor
└── tensor_219.irtensor
```

This allows language-independent comparison.

---

# 23. Testcase Extraction

InferRef SHALL support extracting an isolated testcase.

Example:

```bash
inferref testcase extract trace/ \
  --op 419 \
  --output repro/419
```

Result:

```text
repro/419/
├── testcase.json
├── input_0.irtensor
├── input_1.irtensor
├── output_0.irtensor
└── README.md
```

A coding agent should be able to receive this directory and work without reopening the complete model.

---

# 24. Testcase Deduplication

Repeated operators SHOULD be grouped into shape / layout signatures.

A signature may include:

```text
operator
dtype
rank
shape
stride
broadcast pattern
transpose state
scalar arguments
```

Example:

```text
226 MM executions
↓
11 unique MM signatures
```

This enables automatic generation of compact kernel test suites.

---

# 25. Model Coverage Analysis

InferRef SHOULD provide a command such as:

```bash
inferref analyze trace/
```

Possible output:

```text
Model: Example-4B

Physical operator coverage:
  97.2%

Semantic region coverage:
  84.1%

Unsupported patterns:
  GatedDeltaRule
  Specialized RMSNorm layout
  Dynamic cache update
```

This allows developers to estimate the work needed to support a new model.

---

# 26. Comparator

The comparator SHALL compare:

- shapes;
- ranks;
- dtypes;
- tensor values;
- optional strides;
- optional metadata.

Comparison policies SHALL be configurable.

---

# 27. Numerical Metrics

Suggested metrics:

```text
exact_match
max_abs_error
max_rel_error
mean_abs_error
rmse
cosine_similarity
nan_count
inf_count
mismatch_count
```

A tolerance policy MAY look like:

```json
{
  "float16": {
    "atol": 0.002,
    "rtol": 0.002
  },

  "float32": {
    "atol": 1e-5,
    "rtol": 1e-5
  }
}
```

---

# 28. Quantized Tensor Comparison

Quantized inference paths require additional metadata.

InferRef SHOULD support:

- quantization type;
- scale;
- zero point;
- block size;
- group size;
- packing layout;
- logical dtype;
- physical storage dtype.

Example:

```json
{
  "logical_dtype": "nf4",
  "storage_dtype": "uint8",
  "group_size": 64
}
```

Reference comparison MAY operate on either:

- packed representation;
- dequantized values;
- both.

---

# 29. Layout Debugging

InferRef SHOULD treat tensor layout as a first-class debugging concern.

The report SHOULD clearly distinguish:

```text
shape mismatch
stride mismatch
storage-offset mismatch
value mismatch
```

Example:

```text
Tensor values are identical when interpreted logically,
but engine stride differs from reference stride.
```

This is particularly important for:

- transpose;
- permute;
- view;
- reshape;
- cache layouts;
- attention layouts.

---

# 30. First Divergence Search

InferRef SHOULD support:

```bash
inferref compare ref.trace engine.trace \
  --first-failure
```

The implementation SHOULD stop reporting downstream mismatches once the earliest causally meaningful divergence is identified.

Future versions MAY implement graph-aware divergence analysis.

---

# 31. Trace-to-Trace Comparison

InferRef SHOULD compare two complete traces.

Use cases include:

- engine vs reference;
- PyTorch version A vs B;
- CPU vs GPU;
- eager vs compiled;
- FP32 vs FP16;
- regression testing.

---

# 32. CLI

Suggested top-level interface:

```text
inferref
├── trace
├── inspect
├── analyze
├── compare
├── testcase
├── region
└── export
```

---

# 33. `inferref trace`

Example:

```bash
inferref trace run_model.py \
  --scope model.layers.0 \
  --output trace/
```

Possible options:

```text
--scope
--exclude
--capture-tensors
--device
--max-ops
--source-map
--static-graph
--decompose
--semantic-analysis
```

---

# 34. `inferref inspect`

Example:

```bash
inferref inspect trace/
```

MVP behavior:

- textual tree;
- operator listing;
- tensor metadata;
- source location;
- producer / consumer links.

Future behavior:

```bash
inferref inspect trace/ --web
```

This may start a local viewer.

---

# 35. `inferref compare`

Examples:

```bash
inferref compare \
  reference.trace \
  engine.trace
```

or:

```bash
inferref compare \
  testcase/ \
  engine-output/
```

Useful options:

```text
--first-failure
--atol
--rtol
--metric
--ignore-stride
--json
```

---

# 36. `inferref testcase`

Examples:

```bash
inferref testcase extract trace/ \
  --op 419 \
  --output testcase/
```

```bash
inferref testcase dedup trace/ \
  --operator aten.mm.default
```

---

# 37. `inferref region`

Examples:

```bash
inferref region list trace/
```

```bash
inferref region create trace/ \
  --from-op 411 \
  --to-op 429 \
  --name RotaryEmbedding
```

```bash
inferref testcase extract trace/ \
  --region RotaryEmbedding
```

---

# 38. Trace Package Layout

Suggested layout:

```text
trace/
├── manifest.json
├── graph.json
├── modules.json
├── regions.json
├── sources.json
│
├── tensors/
│   ├── 000001.irtensor
│   ├── 000002.irtensor
│   └── ...
│
├── static/
│   ├── graph.json
│   └── decomposed_graph.json
│
└── reports/
    └── summary.json
```

Files MAY later be packed into a single archive format.

Suggested archive extension:

```text
.irtrace
```

---

# 39. Viewer

The viewer is not mandatory for the first MVP but is highly desirable.

The viewer SHOULD display:

```text
Module
  ↓
Semantic Region
  ↓
Physical Operators
  ↓
Tensors
```

For each tensor:

- shape;
- stride;
- dtype;
- statistics;
- producer;
- consumers;
- optional values.

For each operator:

- source code;
- module path;
- inputs;
- outputs;
- semantic annotation.

---

# 40. Graph Visualization

The viewer SHOULD allow switching between:

### Physical graph

```text
aten.mul
aten.cat
aten.add
```

### Semantic graph

```text
RotaryEmbedding
Attention
SwiGLU
```

### Module graph

```text
TransformerBlock
SelfAttention
MLP
```

These represent different views of the same trace.

---

# 41. Coding Agent Workflow

A recommended agent workflow:

```text
1. inferref analyze model.trace

2. identify missing / failing region

3. inferref testcase extract ...

4. agent modifies kernel

5. build engine test

6. run isolated testcase

7. inferref compare ...

8. repeat until PASS
```

The agent should not need to repeatedly inspect the entire model source.

---

# 42. Machine-Readable Diagnostics

All major CLI commands SHOULD support JSON output.

Example:

```bash
inferref compare ref/ actual/ --json
```

Result:

```json
{
  "status": "fail",

  "first_failure": {
    "operator_id": 419,
    "tensor_id": 242,

    "metrics": {
      "max_abs_error": 0.21484375,
      "cosine_similarity": 0.99871
    }
  }
}
```

This is important for agentic tooling and CI.

---

# 43. CI Integration

InferRef SHOULD support CI workflows.

Example:

```text
Pull Request
   ↓
Build inference engine
   ↓
Run InferRef testcase suite
   ↓
Compare reference tensors
   ↓
Pass / Fail
```

Reference traces MAY be stored as versioned artifacts.

---

# 44. Regression Testing

InferRef SHOULD support frozen reference packages.

Example:

```text
QwenX/
├── rope/
├── rmsnorm/
├── attention/
├── mlp/
└── cache_update/
```

This allows kernel changes to be tested against multiple model-derived real-world inputs.

---

# 45. Multi-Model Test Corpus

A long-term InferRef feature MAY provide a way to build a corpus of unique operator and region signatures across models.

For example:

```text
RMSNorm
├── Llama
├── Qwen
├── Gemma
└── Mistral

RoPE
├── standard
├── interleaved
├── partial
└── scaled
```

This would make model compatibility testing significantly more systematic.

---

# 46. Performance Considerations

Tracing can introduce significant overhead.

InferRef SHALL favor correctness over tracing speed.

However, the following optimizations SHOULD be considered:

- metadata-only mode;
- hash-only mode;
- selective tensor dumping;
- asynchronous host copies;
- parameter deduplication;
- tensor compression;
- per-region capture;
- capture count limits.

---

# 47. Device Considerations

InferRef SHOULD allow reference execution on:

- CPU;
- CUDA;
- XPU;
- other PyTorch-supported devices.

However, deterministic reference generation MAY prefer CPU or well-defined execution settings when practical.

InferRef SHALL record the device environment in the manifest.

---

# 48. Determinism

InferRef SHOULD record:

- random seed;
- inference/eval mode;
- autocast state;
- dtype;
- deterministic settings where applicable.

The tool SHOULD warn when:

- dropout is enabled;
- stochastic operators are detected;
- reference execution is non-deterministic.

---

# 49. Large Model Handling

Large models create several practical problems:

- parameter duplication;
- large activation dumps;
- slow device-to-host copies;
- huge traces.

InferRef SHOULD support tracing selected model scopes.

Example:

```bash
inferref trace run.py \
  --scope model.layers.17
```

It SHOULD also allow capture of only one invocation of repeated modules.

---

# 50. Dynamic Models

Some model behaviors depend on:

- sequence length;
- cache state;
- batch size;
- control flow;
- multimodal inputs.

A trace SHALL describe one concrete execution.

InferRef SHOULD NOT claim that a single trace fully specifies all model behavior.

Instead, multiple traces MAY form a trace set:

```text
prefill_1_token
prefill_128_tokens
decode_1_token
decode_with_cache
batch_4
vision_prefill
```

---

# 51. Trace Sets

A Trace Set groups executions for one model.

Example:

```text
model.irset/
├── manifest.json
├── prefill_128.irtrace
├── decode.irtrace
└── vision.irtrace
```

Coverage analysis SHOULD operate across trace sets.

---

# 52. Model Adaptation Workflow

A recommended workflow for supporting a new model:

```text
Step 1
Run representative reference inputs.

Step 2
Generate trace set.

Step 3
Analyze physical and semantic coverage.

Step 4
Identify unsupported regions.

Step 5
Extract isolated testcases.

Step 6
Implement / adapt engine kernels.

Step 7
Validate each region.

Step 8
Validate end-to-end model execution.
```

---

# 53. API Design Principles

InferRef SHOULD prefer:

- stable public schemas;
- simple file formats;
- language independence;
- explicit versioning;
- deterministic output;
- composable tools.

InferRef SHOULD avoid making the engine side depend on Python.

---

# 54. Python API

A future Python API MAY look like:

```python
from inferref import trace

with trace(
    output="trace/",
    scope="model.layers.0"
):
    output = model(**inputs)
```

More advanced:

```python
session = inferref.TraceSession(...)

with session:
    model(**inputs)

session.save()
```

---

# 55. Plugin Architecture

InferRef SHOULD eventually support plugins.

Possible plugin classes:

```text
Frontend
SemanticDetector
TensorCodec
Comparator
RegionDetector
EngineAdapter
ViewerExtension
```

---

# 56. Semantic Detector API

Conceptual interface:

```python
class SemanticDetector:
    def detect(
        self,
        graph,
        trace
    ) -> list[SemanticRegion]:
        ...
```

Detectors SHALL NOT modify physical trace truth.

---

# 57. Security and Privacy

Reference traces may contain:

- user prompts;
- model activations;
- weights;
- embeddings;
- proprietary source paths.

InferRef SHOULD clearly warn users that trace packages may contain sensitive data.

Future versions MAY support:

- input redaction;
- source-path redaction;
- parameter exclusion;
- encryption;
- trace sanitization.

---

# 58. Licensing Considerations

InferRef itself SHOULD avoid embedding model source code into trace packages unless explicitly requested.

By default, source mapping SHOULD prefer:

```text
file path
line number
function name
```

Embedding source text SHOULD be optional.

This reduces trace size and potential licensing concerns.

---

# 59. Error Handling

InferRef SHALL distinguish:

```text
trace error
static export error
tensor capture error
semantic analysis error
comparison failure
engine execution failure
```

For example:

```text
WARNING:
Static export failed.

Runtime trace is still valid.

Reason:
Unsupported dynamic control flow.
```

---

# 60. Compatibility Strategy

PyTorch-facing code SHALL be isolated in the PyTorch frontend.

The rest of the project SHALL depend only on InferRef interfaces.

```text
PyTorch version changes
        ↓
PyTorch frontend changes
        ↓

Trace IR remains stable
Comparator remains stable
Engine adapters remain stable
Existing traces remain readable
```

This is a core architectural requirement.

---

# 61. Proposed Repository Layout

```text
InferRef/
├── inferref/
│   ├── frontend/
│   │   └── pytorch/
│   │
│   ├── ir/
│   ├── trace/
│   ├── tensor/
│   ├── compare/
│   ├── region/
│   ├── semantic/
│   ├── testcase/
│   ├── cli/
│   └── viewer/
│
├── cpp/
│   ├── include/
│   ├── tensor/
│   ├── compare/
│   └── examples/
│
├── tests/
├── examples/
├── docs/
└── tools/
```

---

# 62. MVP Scope

The MVP SHOULD be deliberately narrow.

## Required

### PyTorch runtime tracer

Capture:

- operator order;
- tensor metadata;
- source mapping;
- module mapping.

### Trace IR

Support:

- manifest;
- operators;
- tensors;
- sources.

### Tensor dump

A simple framework-independent format.

### CLI

Required commands:

```text
inferref trace
inferref inspect
inferref compare
inferref testcase extract
```

### Comparator

Support standard floating-point metrics.

### Isolated testcase extraction

Allow one runtime operator to be reproduced independently.

---

# 63. MVP Non-Requirements

The MVP does NOT need:

- web viewer;
- automatic semantic region detection;
- automatic fusion matching;
- ONNX frontend;
- distributed tracing;
- graph database;
- sophisticated compression;
- automatic engine invocation;
- quantization-aware comparison;
- model compatibility database.

These can be added after the core trace pipeline is stable.

---

# 64. Recommended Development Phases

## Phase 0 — Prototype

Goal:

Prove that dispatcher-level tracing can produce usable operator-level records.

Deliver:

```text
operator
inputs
outputs
shape
dtype
stride
source
```

---

## Phase 1 — Stable Trace Package

Goal:

Define Trace IR and persistent tensor storage.

Deliver:

```text
manifest.json
graph.json
sources.json
tensors/*.irtensor
```

---

## Phase 2 — Comparator

Goal:

Enable isolated engine validation.

Deliver:

```text
inferref compare
inferref testcase extract
```

---

## Phase 3 — Regions

Goal:

Support fused engine kernels.

Deliver:

```text
ReferenceRegion
manual region creation
region testcase extraction
```

---

## Phase 4 — Static Analysis

Goal:

Merge runtime and static graph knowledge.

Deliver:

```text
torch.export integration
decomposition
source metadata enrichment
```

---

## Phase 5 — Semantic Analysis

Goal:

Recognize common inference patterns.

Initial targets:

```text
Linear
RMSNorm
LayerNorm
RoPE
SwiGLU
Attention
KV Cache Update
```

---

## Phase 6 — Viewer

Goal:

Provide interactive graph and tensor inspection.

---

## Phase 7 — Agent Integration

Goal:

Make InferRef an ideal backend for coding agents.

Possible interfaces:

```text
JSON diagnostics
MCP server
structured testcase API
CI integration
```

---

# 65. Suggested First End-to-End Prototype

A practical first milestone:

### Target

One Transformer block.

### Reference

A simple Hugging Face causal language model.

### Capture

```text
one block
one prefill execution
CPU reference
FP32 or FP16
```

### Expected output

```text
trace/
├── manifest.json
├── graph.json
├── sources.json
└── tensors/
```

### Validation

Implement one external operator test:

```text
RMSNorm
or
RoPE
```

and prove:

```text
reference trace
   ↓
extract testcase
   ↓
custom C++ implementation
   ↓
InferRef compare
   ↓
PASS / FAIL
```

If this loop works cleanly, the fundamental architecture is validated.

---

# 66. Success Criteria

InferRef can be considered successful when adding support for a new model changes from:

```text
read model source
trace manually
write temporary scripts
guess tensor semantics
debug whole model
```

to:

```text
run InferRef
inspect unsupported regions
extract testcase
implement kernel
compare
```

A stronger success criterion is:

> A coding agent can fix a numerical mismatch using an isolated InferRef testcase without needing to inspect the complete model source or rerun the full model for every iteration.

---

# 67. Long-Term Vision

InferRef may eventually become a common reference layer between model frameworks and inference engines.

Conceptually:

```text
PyTorch / Transformers
JAX
ONNX Runtime
Other Frameworks
        │
        ▼
     InferRef
        │
        ├── reference traces
        ├── real model testcases
        ├── numerical validation
        ├── semantic regions
        └── compatibility analysis
        │
        ▼
CUDA engines
SYCL engines
ROCm engines
CPU engines
mobile engines
custom accelerators
```

The long-term value of InferRef is therefore not merely tracing.

Its role is to provide a reproducible answer to the question:

> **“What exactly must an inference engine reproduce for this model, at this execution point, with these inputs?”**

---

# 68. Core Design Principles

The following principles SHOULD guide future development.

### 1. Runtime truth comes first

Always preserve what actually executed.

### 2. Tensor layout is first-class

Shape alone is insufficient.

### 3. Semantic information is annotation

Never replace physical execution truth with inferred semantics.

### 4. Regions are the bridge to fused engines

Inference engines should not be forced to copy PyTorch's operator boundaries.

### 5. Trace IR must outlive framework APIs

Framework-specific behavior belongs in frontends.

### 6. Testcases must be portable

C++ engines should not need PyTorch.

### 7. Agent feedback must be structured

Every important result should be available in machine-readable form.

### 8. Debug the first divergence

Downstream errors are usually symptoms.

### 9. Real model inputs are valuable test vectors

InferRef should turn model executions into reusable kernel test corpora.

### 10. Start small

A reliable primitive tracer and comparator are more valuable than an ambitious but fragile universal graph system.

---

# Appendix A — Example Trace

```text
Model:
  ExampleLM

Scope:
  model.layers.0.self_attn

#121 aten.mm.default
  module:
    self_attn.q_proj

  input:
    tensor_81
    [1, 128, 3584]
    fp16

  output:
    tensor_82
    [1, 128, 3584]
    fp16

#122 aten.view.default

#123 aten.transpose.int

#124 aten.mul.Tensor
  semantic:
    RotaryEmbedding

#125 aten.cat.default

#126 aten.mul.Tensor

#127 aten.add.Tensor

...
```

---

# Appendix B — Example Comparison Report

```text
InferRef Comparison

Reference:
  qwen-prefill.irtrace

Engine:
  myengine-prefill.irtrace

Summary:
  Compared tensors: 382
  Passed:           381
  Failed:             1

First divergence:

  Tensor:
    242

  Reference producer:
    #419 aten.mul.Tensor

  Module:
    model.layers.0.self_attn

  Semantic region:
    RotaryEmbedding

  Source:
    modeling_example.py:527

  Shape:
    [1, 28, 128, 128]

  DType:
    float16

Metrics:

  max_abs_error:
    0.21484375

  mean_abs_error:
    0.004921

  cosine_similarity:
    0.99871

First mismatching element:

  index:
    [0, 3, 29, 41]

  reference:
    0.718750

  actual:
    0.691406

Upstream status:

  #418 PASS
  #417 PASS
  #416 PASS
```

---

# Appendix C — Example Agent Loop

```text
Agent input:

  testcase/
  ├── testcase.json
  ├── query.irtensor
  ├── cos.irtensor
  ├── sin.irtensor
  └── reference_output.irtensor

Task:

  Fix RotaryEmbeddingKernel.

Validation command:

  myengine-test testcase/
  inferref compare testcase/ output/

Completion condition:

  InferRef reports PASS.
```

---

**End of specification.**
