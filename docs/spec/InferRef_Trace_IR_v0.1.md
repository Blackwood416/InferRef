# InferRef Trace IR v0.1 Specification

> **Project:** InferRef  
> **Document:** Trace IR Specification  
> **IR Version:** 0.1  
> **Status:** Draft / MVP Contract  
> **Audience:** InferRef frontend developers, engine adapter authors, tooling authors, coding agents

---

## 1. Purpose

The InferRef Trace IR is the stable data contract between:

```text
Reference Framework Frontend
        ↓
     Trace IR
        ↓
+-------+--------+---------+----------+
|                |                    |
Inspector     Comparator         Test Extractor
                                     |
                                     ↓
                              Inference Engine
```

The Trace IR describes **what actually happened during one concrete reference execution**.

It is deliberately separate from PyTorch, Hugging Face Transformers, `torch.export`, FX, ATen implementation objects, and any particular inference engine.

The primary design objective is:

> A trace produced by one InferRef frontend should remain inspectable, comparable, and consumable without importing the original framework.

---

# 2. Design Principles

## 2.1 Runtime truth is authoritative

The runtime trace records the concrete operator execution that occurred.

Static graphs, semantic labels, module names, and source mappings are annotations around that execution.

They MUST NOT replace runtime truth.

---

## 2.2 Values are first-class entities

Operators do not "own" tensors.

Instead:

```text
Value
  ↓
Operator
  ↓
Value
```

The IR therefore models a dataflow graph rather than a flat list of operator logs.

---

## 2.3 Tensor layout is first-class

A tensor is not sufficiently described by:

```text
dtype + shape + bytes
```

The IR MUST be able to represent:

- shape;
- stride;
- storage offset;
- storage identity;
- object identity;
- aliasing;
- views;
- mutation versions.

This is necessary for debugging:

- `view`;
- `reshape`;
- `transpose`;
- `permute`;
- KV caches;
- in-place updates;
- fused kernels.

---

## 2.4 Trace values are immutable snapshots

Runtime tensor objects may be mutated.

InferRef trace values MUST NOT be.

Every value referenced by the graph represents the state observed at a particular execution point.

For an in-place operation:

```text
tensor object A, storage S, version 3
        ↓
     inplace op
        ↓
tensor object A, storage S, version 4
```

the input and output MUST be represented as different immutable trace values even though they originated from the same runtime tensor object.

---

## 2.5 Physical and semantic representations are separate

The Trace IR records physical operators such as:

```text
aten.mul.Tensor
aten.mm.default
aten.slice.Tensor
```

Semantic information such as:

```text
RoPE
RMSNorm
SwiGLU
Attention
```

is optional annotation.

Semantic inference MUST NOT rewrite or erase the physical trace.

---

## 2.6 Framework internals must not leak into the stable contract

Framework-specific objects MAY appear in debug extensions, but the core IR MUST use InferRef-defined types and strings.

For example, the stable IR should store:

```json
"dtype": "float16"
```

rather than a pickled `torch.dtype` object.

---

# 3. Scope of v0.1

Trace IR v0.1 defines:

- trace manifest;
- execution records;
- operators;
- tensor values;
- scalar values;
- container values;
- storage identity;
- tensor object identity;
- tensor state versions;
- source mappings;
- module stack metadata;
- semantic annotations;
- reference regions;
- tensor payload references;
- trace package layout.

It does NOT attempt to define:

- a universal neural-network compiler IR;
- symbolic execution semantics;
- gradient graphs;
- training semantics;
- optimization passes;
- device kernel binaries.

---

# 4. Trace Package

A persisted trace is called a **Trace Package**.

Directory representation:

```text
example.irtrace/
├── manifest.json
├── graph.json
├── sources.json
├── regions.json
├── storages.json
├── tensors/
│   ├── v00000001.irtensor
│   ├── v00000002.irtensor
│   └── ...
└── extensions/
```

A future packed representation MAY use the same logical structure inside an archive.

The `.irtrace` suffix refers to the logical InferRef trace package, not necessarily a particular archive encoding.

---

# 5. Versioning

Every trace MUST specify:

```json
{
  "format": "inferref-trace",
  "format_version": "0.1"
}
```

Version rules:

- patch-compatible additions MAY introduce optional fields;
- readers MUST ignore unknown optional fields;
- required semantic changes require an IR version change;
- framework frontend versions are independent of IR versions;
- tensor binary format versioning is independent of graph IR versioning.

---

# 6. Identifier Model

All persisted entities use stable IDs within one trace.

Recommended ID namespaces:

```text
op:1
value:42
storage:7
object:11
source:4
region:3
module:9
```

JSON MAY store numeric IDs when the entity type is implied by the field.

Example:

```json
{
  "producer": 382,
  "value_id": 194
}
```

IDs only need to be unique within their own namespace.

---

# 7. Execution Order

Every runtime operator MUST have an `execution_index`.

Example:

```json
{
  "id": 382,
  "execution_index": 381
}
```

Requirements:

- indices start at zero;
- indices increase monotonically;
- execution order is preserved even if the dataflow graph contains branches;
- `execution_index` is the canonical ordering for first-divergence analysis.

Operator `id` and `execution_index` SHOULD normally be correlated but MUST NOT be assumed identical.

---

# 8. Core Graph Model

The runtime graph consists of:

```text
ValueRecord
OperatorRecord
ValueRecord
```

An operator receives ordered positional and keyword arguments, then produces one arbitrary structured result.

Example:

```python
out = op(x, y, dim=1)
```

is represented as:

```text
Operator
├── positional_args
│   ├── TensorValueRef(x)
│   └── TensorValueRef(y)
├── keyword_args
│   └── dim = ScalarValue(1)
└── result
    └── TensorValueRef(out)
```

---

# 9. Value System

The Trace IR SHALL support the following core value kinds:

```text
tensor
scalar
none
string
list
tuple
dict
opaque
```

The value system exists because PyTorch operators do not only receive tensors.

---

# 10. ValueRef

A generic value reference MAY be encoded as:

```json
{
  "kind": "tensor",
  "value_id": 194
}
```

Scalar example:

```json
{
  "kind": "scalar",
  "dtype": "int64",
  "value": 1
}
```

None:

```json
{
  "kind": "none"
}
```

Tuple:

```json
{
  "kind": "tuple",
  "items": [
    {"kind": "tensor", "value_id": 194},
    {"kind": "scalar", "dtype": "int64", "value": 4}
  ]
}
```

Dictionary:

```json
{
  "kind": "dict",
  "items": [
    {
      "key": {"kind": "string", "value": "dim"},
      "value": {"kind": "scalar", "dtype": "int64", "value": 1}
    }
  ]
}
```

---

# 11. Scalar Representation

Supported stable scalar dtype names SHOULD include:

```text
bool
int8
int16
int32
int64
uint8
uint16
uint32
uint64
float16
bfloat16
float32
float64
complex64
complex128
```

Framework-specific scalar kinds MAY be stored through extensions.

Special floating-point values MUST have a JSON-safe representation.

Recommended:

```json
{
  "kind": "scalar",
  "dtype": "float32",
  "encoding": "special",
  "value": "nan"
}
```

Allowed special strings:

```text
nan
+inf
-inf
```

---

# 12. Tensor Value Record

A tensor value represents one immutable tensor state at one point in the execution.

Example:

```json
{
  "id": 194,
  "kind": "tensor",

  "dtype": "float16",

  "shape": [1, 128, 3584],
  "stride": [458752, 3584, 1],

  "device": {
    "type": "cpu",
    "index": null
  },

  "storage_offset_elements": 0,

  "logical_numel": 458752,
  "logical_nbytes": 917504,

  "runtime_object_id": 51,
  "storage_id": 7,
  "storage_version": 3,

  "contiguous": true,

  "producer": 382,
  "consumers": [383],

  "capture": {
    "mode": "full",
    "payload": "tensors/v00000194.irtensor"
  }
}
```

---

# 13. Runtime Tensor Object Identity

`runtime_object_id` identifies the framework tensor object observed during execution.

It exists to distinguish:

```text
same Python tensor object
```

from:

```text
different tensor objects that share storage
```

This ID is diagnostic metadata.

It MUST NOT be interpreted as a portable memory address.

---

# 14. Storage Identity

`storage_id` identifies aliased backing storage within the trace.

Example:

```text
A = original tensor
B = A.view(...)
C = A.transpose(...)
```

may produce:

```text
A.runtime_object_id = 10
B.runtime_object_id = 11
C.runtime_object_id = 12

A.storage_id = 3
B.storage_id = 3
C.storage_id = 3
```

This makes view relationships observable.

---

# 15. Storage Version

`storage_version` represents the logical mutation generation of a storage.

Before mutation:

```text
storage_id = 3
storage_version = 5
```

After mutation:

```text
storage_id = 3
storage_version = 6
```

All immutable Trace Tensor Values referencing the pre-mutation state retain version `5`.

This mechanism is essential for:

- KV-cache writes;
- in-place normalization;
- masked fills;
- scatter updates;
- custom mutable cache structures.

---

# 16. Storage Record

`storages.json` MAY contain:

```json
{
  "id": 7,
  "device": {
    "type": "cpu",
    "index": null
  },
  "storage_dtype": "float16",
  "observed_versions": [0, 1, 2, 3],
  "notes": null
}
```

The storage record describes identity.

Actual snapshot bytes remain attached to tensor values or storage snapshots.

---

# 17. Views

A view SHOULD be represented through metadata rather than by assuming a physical copy.

Example:

```text
input:
  storage_id: 9
  shape:  [2, 3]
  stride: [3, 1]

output:
  storage_id: 9
  shape:  [3, 2]
  stride: [1, 3]
```

If no storage mutation occurred:

```text
input.storage_version == output.storage_version
```

This gives an engine developer enough information to identify metadata-only transformations.

---

# 18. Tensor Capture Modes

Each tensor value MAY have one of the following capture modes:

```text
none
metadata
hash
sample
full
```

### `none`

Only graph identity is retained.

### `metadata`

Shape/layout/dtype metadata is retained.

### `hash`

Metadata plus one or more hashes.

### `sample`

Metadata plus selected values.

### `full`

Complete logical tensor payload is stored.

---

# 19. Tensor Hash Record

Example:

```json
{
  "capture": {
    "mode": "hash",
    "hashes": [
      {
        "algorithm": "xxh3-128",
        "domain": "logical-contiguous-bytes",
        "value": "..."
      }
    ]
  }
}
```

Hash domain MUST be explicit.

Possible domains:

```text
logical-contiguous-bytes
physical-storage-range
dequantized-fp32
```

InferRef MUST NOT compare hashes generated over different domains as equivalent.

---

# 20. Tensor Payload Semantics

For v0.1, a `.irtensor` payload SHOULD represent the tensor's **logical value in canonical contiguous order**, unless explicitly marked otherwise.

This avoids requiring every engine-side reader to reconstruct arbitrary framework storage layouts.

The graph metadata still preserves the original stride and storage relationships for debugging.

A future version MAY additionally support physical storage snapshots.

---

# 21. `.irtensor` v0.1

Recommended binary structure:

```text
Header
Metadata
Payload
```

Logical header fields:

```text
magic                  "IRTN"
tensor_format_version  uint16
header_size            uint16
dtype                   uint16
flags                   uint32
rank                    uint16
reserved                uint16
logical_numel           uint64
payload_nbytes          uint64
shape[rank]             int64[]
stride[rank]            int64[]
storage_offset          int64
payload                 byte[]
```

All integer endianness SHALL be little-endian in v0.1.

Payload SHALL contain canonical logical tensor values unless a flag declares another encoding.

---

# 22. Operator Record

Example:

```json
{
  "id": 382,
  "execution_index": 381,
  "kind": "operator",

  "namespace": "aten",
  "op": "mm",
  "overload": "default",
  "canonical_name": "aten.mm.default",

  "positional_args": [
    {"kind": "tensor", "value_id": 192},
    {"kind": "tensor", "value_id": 193}
  ],

  "keyword_args": {},

  "result": {
    "kind": "tensor",
    "value_id": 194
  },

  "source_id": 27,

  "module_stack": [
    1,
    8,
    31
  ],

  "annotations": []
}
```

---

# 23. Operator Naming

For PyTorch physical operators, recommended canonical naming is:

```text
namespace.op.overload
```

Examples:

```text
aten.mm.default
aten.add.Tensor
aten.slice.Tensor
```

Other frontends MUST use their own namespaces.

Examples:

```text
onnx.Add.v14
jax.lax.dot_general
```

InferRef itself MUST NOT assume all operator namespaces are ATen.

---

# 24. Mutation Metadata

An operator MAY explicitly record mutation effects:

```json
{
  "effects": {
    "mutated_storages": [
      {
        "storage_id": 7,
        "version_before": 3,
        "version_after": 4
      }
    ]
  }
}
```

This metadata SHOULD be produced whenever the frontend can reliably determine mutation.

---

# 25. Alias Metadata

An operator MAY record alias relationships:

```json
{
  "effects": {
    "aliases": [
      {
        "output_value_id": 202,
        "input_value_id": 194,
        "relationship": "view"
      }
    ]
  }
}
```

Allowed initial relationships:

```text
same_object
shared_storage
view
unknown_alias
```

---

# 26. Source Record

`sources.json` contains deduplicated source mappings.

Example:

```json
{
  "id": 27,

  "primary": {
    "file": "transformers/models/example/modeling_example.py",
    "line": 527,
    "function": "apply_rotary_pos_emb"
  },

  "stack": [
    {
      "file": "...",
      "line": 527,
      "function": "apply_rotary_pos_emb"
    },
    {
      "file": "...",
      "line": 612,
      "function": "forward"
    }
  ],

  "source_text": null
}
```

Embedding source text is OPTIONAL.

---

# 27. Source Path Policy

The trace manifest MUST declare path handling:

```json
{
  "source_policy": {
    "path_mode": "relative",
    "embed_source_text": false
  }
}
```

Suggested modes:

```text
absolute
relative
redacted
```

---

# 28. Module Records

Modules are stored separately and referenced by IDs.

Example:

```json
{
  "id": 31,
  "path": "model.layers.17.self_attn.q_proj",
  "type": "torch.nn.Linear",
  "parent_id": 28
}
```

The stable IR treats `type` as an informational string.

---

# 29. Module Stack

An operator MAY contain:

```json
"module_stack": [1, 8, 28, 31]
```

Ordered:

```text
outermost → innermost
```

This preserves nested model context.

---

# 30. Source Function Stack

Framework frontends MAY attach function-level annotations:

```json
{
  "extensions": {
    "pytorch": {
      "source_fn_stack": [
        "torch.nn.Linear"
      ]
    }
  }
}
```

This is intentionally frontend-specific unless a future cross-framework abstraction emerges.

---

# 31. Annotation Model

Annotations are optional derived metadata.

Example:

```json
{
  "type": "semantic",
  "name": "Linear",
  "confidence": 1.0,
  "detector": "inferref.semantic.pytorch.linear.v1"
}
```

Another example:

```json
{
  "type": "role",
  "name": "q_proj",
  "confidence": 0.99
}
```

---

# 32. Semantic Confidence

Recommended interpretation:

```text
1.00        deterministic mapping
0.90–0.99   very strong inference
0.70–0.89   likely
0.50–0.69   weak
<0.50       normally omit
```

Consumers MUST NOT treat semantic confidence as execution truth.

---

# 33. Reference Region

A Reference Region defines a subgraph boundary.

Example:

```json
{
  "id": 17,
  "name": "RotaryEmbedding",

  "node_ids": [
    411,
    412,
    413,
    414,
    415,
    416
  ],

  "inputs": [
    201,
    202,
    203,
    204
  ],

  "outputs": [
    211,
    212
  ],

  "semantic": {
    "name": "RoPE",
    "confidence": 0.97
  },

  "source_ids": [42],

  "creation": {
    "method": "manual"
  }
}
```

---

# 34. Region Boundary Rules

A region input is a value that:

- is consumed by a node inside the region;
- is produced outside the region or is an external graph input.

A region output is a value that:

- is produced inside the region;
- is consumed outside the region or selected as a trace output.

This formal definition allows regions to be extracted automatically.

---

# 35. Region Creation Methods

Initial method strings:

```text
manual
module
source_function
semantic_pattern
engine_mapping
graph_selection
```

---

# 36. Nested Regions

Regions MAY overlap or nest.

For v0.1, consumers MUST NOT assume a strict region tree.

Example:

```text
TransformerBlock
└── Attention

RoPE
```

A RoPE region may sit inside Attention, while another diagnostic region may overlap different boundaries.

---

# 37. Graph Inputs and Outputs

`graph.json` MUST identify external trace inputs and outputs.

Example:

```json
{
  "inputs": [
    {
      "name": "hidden_states",
      "value": {
        "kind": "tensor",
        "value_id": 1
      }
    }
  ],

  "outputs": [
    {
      "name": "output",
      "value": {
        "kind": "tensor",
        "value_id": 929
      }
    }
  ]
}
```

---

# 38. Parameters and Buffers

Tensor values MAY be classified:

```text
activation
parameter
buffer
input
output
constant
unknown
```

Example:

```json
{
  "role": "parameter",
  "qualified_name": "model.layers.0.self_attn.q_proj.weight"
}
```

This enables parameter deduplication and better test extraction.

---

# 39. Parameter Deduplication

A parameter SHOULD be captured once per immutable state.

Operators reference the same value or storage identity rather than creating repeated payloads.

For a 4B model, this is required to keep trace size manageable.

---

# 40. Constants

Non-tensor constants used by operators SHOULD be encoded directly in argument values when practical.

Large constants MAY be externalized.

---

# 41. Opaque Values

Unsupported runtime values MAY be represented as:

```json
{
  "kind": "opaque",
  "type": "SomeFrameworkType",
  "repr": "<redacted>",
  "portable": false
}
```

Opaque values are diagnostic only.

A testcase that depends on an opaque, non-portable value MUST report that it cannot yet be independently reproduced.

---

# 42. Manifest

Example:

```json
{
  "format": "inferref-trace",
  "format_version": "0.1",

  "inferref_version": "0.1.0",

  "frontend": {
    "name": "pytorch",
    "version": "0.1"
  },

  "reference_framework": {
    "name": "pytorch",
    "version": "2.x"
  },

  "model": {
    "name": "ExampleLM",
    "revision": null
  },

  "execution": {
    "mode": "inference",
    "device": "cpu"
  },

  "capture": {
    "tensor_policy": "selected",
    "source_mapping": true,
    "module_mapping": true
  },

  "source_policy": {
    "path_mode": "relative",
    "embed_source_text": false
  }
}
```

---

# 43. Environment Metadata

The manifest MAY contain:

```json
{
  "environment": {
    "python": "3.x",
    "os": "Windows",
    "architecture": "x86_64",
    "transformers": "...",
    "device_name": "...",
    "driver": "..."
  }
}
```

This metadata is useful for reproducibility but SHALL NOT affect IR parsing.

---

# 44. Determinism Metadata

Recommended:

```json
{
  "determinism": {
    "seed": 1234,
    "training": false,
    "grad_enabled": false,
    "autocast": false,
    "warnings": []
  }
}
```

---

# 45. Extension Namespace

Framework- or implementation-specific information SHALL live under:

```json
"extensions": {
  "pytorch": {},
  "inferref.experimental": {}
}
```

Extension fields MUST NOT be required for basic parsing.

---

# 46. Forward Compatibility

Readers MUST:

- ignore unknown object fields;
- reject unsupported major IR semantics cleanly;
- preserve unknown extension fields when rewriting a trace when practical.

Writers MUST NOT repurpose an existing field with incompatible semantics.

---

# 47. Minimal Valid Trace

A minimal v0.1 trace requires:

```text
manifest
graph inputs
graph outputs
operator list
value records
```

Tensor payloads, source mapping, regions, and semantic annotations are optional.

---

# 48. Recommended Validation Invariants

An IR validator SHOULD verify:

1. every referenced operator exists;
2. every referenced tensor value exists;
3. execution indices are unique and monotonic;
4. producer references are consistent;
5. consumer references are consistent;
6. tensor rank matches `shape` and `stride` lengths;
7. storage versions never decrease in execution order;
8. region node references exist;
9. region boundary inputs/outputs are graph-consistent;
10. full-capture payload byte counts match dtype and logical element count.

---

# 49. Example: View

Reference:

```python
y = x.transpose(1, 2)
```

Possible IR:

```text
value 10:
  object = 4
  storage = 2
  storage_version = 0
  shape = [1, 128, 64]
  stride = [8192, 64, 1]

op 20:
  aten.transpose.int

value 11:
  object = 5
  storage = 2
  storage_version = 0
  shape = [1, 64, 128]
  stride = [8192, 1, 64]
```

Important result:

```text
new object
same storage
same storage version
different view metadata
```

---

# 50. Example: In-Place Mutation

Reference:

```python
cache[:, :, pos] = value
```

Possible conceptual trace:

```text
input cache value:
  value_id = 300
  object = 80
  storage = 44
  storage_version = 7

mutation operator

output cache state:
  value_id = 305
  object = 80
  storage = 44
  storage_version = 8
```

The trace retains both states.

---

# 51. Example: Fused Engine Mapping

Reference region:

```text
#411 slice
#412 neg
#413 cat
#414 mul
#415 mul
#416 add
```

Region:

```text
RotaryEmbedding
```

Engine:

```text
SYCLRotaryEmbeddingKernel
```

The engine only needs to reproduce the region boundary contract:

```text
region inputs → region outputs
```

It does NOT need to expose six engine-side operators.

---

# 52. Comparison Identity

Trace comparison SHOULD primarily align entities using explicit mapping.

Fallback alignment MAY use:

```text
execution_index
canonical operator name
module path
source location
semantic region
tensor role
shape signature
```

Heuristic alignment MUST report confidence.

---

# 53. Testcase Projection

A testcase is a projection of the Trace IR.

For an operator testcase:

```text
operator external inputs
        ↓
operator
        ↓
operator result
```

For a region testcase:

```text
region external inputs
        ↓
region subgraph
        ↓
region external outputs
```

The testcase SHOULD include only required payloads and metadata.

---

# 54. Testcase Manifest

Example:

```json
{
  "format": "inferref-testcase",
  "format_version": "0.1",

  "origin": {
    "trace": "example.irtrace",
    "region_id": 17
  },

  "name": "RotaryEmbedding",

  "inputs": [
    {
      "name": "query",
      "value_id": 201,
      "payload": "inputs/query.irtensor"
    }
  ],

  "outputs": [
    {
      "name": "query_embed",
      "value_id": 211,
      "payload": "reference/query_embed.irtensor"
    }
  ]
}
```

---

# 55. IR v0.1 Decisions

The following decisions are intentionally fixed for the MVP:

### Decision 1

Trace values are immutable execution-point snapshots.

### Decision 2

Runtime object identity and storage identity are distinct.

### Decision 3

Storage mutations create new storage versions.

### Decision 4

Tensor payloads default to canonical logical contiguous values.

### Decision 5

Original layout metadata is retained separately.

### Decision 6

Physical operators remain authoritative.

### Decision 7

Semantic regions are optional annotations.

### Decision 8

Framework-specific metadata lives behind frontend or extension boundaries.

---

# 56. Deferred Questions for v0.2+

The following are explicitly deferred:

- physical storage byte snapshots;
- sparse tensors;
- nested tensors;
- distributed tensors;
- device-resident trace packages;
- zero-copy tensor payloads;
- quantized binary payload standardization;
- symbolic dimensions;
- control-flow graph nodes;
- asynchronous event/timeline representation;
- custom accelerator memory spaces;
- cross-trace stable tensor identifiers.

---

# 57. Acceptance Criteria for Trace IR v0.1

The IR design is sufficient for MVP implementation when it can correctly represent all of the following:

1. a normal `Linear`;
2. a metadata-only `transpose`;
3. two distinct views sharing one storage;
4. an in-place tensor mutation;
5. a Transformer submodule call stack;
6. operator scalar/keyword arguments;
7. source mapping;
8. a standalone operator testcase;
9. a multi-op RoPE reference region;
10. a trace readable without PyTorch installed.

---

# Appendix A — Compact Example

```json
{
  "operators": [
    {
      "id": 1,
      "execution_index": 0,
      "canonical_name": "aten.mm.default",

      "positional_args": [
        {"kind": "tensor", "value_id": 1},
        {"kind": "tensor", "value_id": 2}
      ],

      "keyword_args": {},

      "result": {
        "kind": "tensor",
        "value_id": 3
      },

      "source_id": 1,
      "module_stack": [1, 2, 3]
    }
  ],

  "values": [
    {
      "id": 1,
      "kind": "tensor",
      "dtype": "float16",
      "shape": [128, 3584],
      "stride": [3584, 1],
      "runtime_object_id": 1,
      "storage_id": 1,
      "storage_version": 0
    },

    {
      "id": 2,
      "kind": "tensor",
      "dtype": "float16",
      "shape": [3584, 3584],
      "stride": [3584, 1],
      "runtime_object_id": 2,
      "storage_id": 2,
      "storage_version": 0
    },

    {
      "id": 3,
      "kind": "tensor",
      "dtype": "float16",
      "shape": [128, 3584],
      "stride": [3584, 1],
      "runtime_object_id": 3,
      "storage_id": 3,
      "storage_version": 0,
      "producer": 1
    }
  ]
}
```

---

**End of InferRef Trace IR v0.1 Specification**
