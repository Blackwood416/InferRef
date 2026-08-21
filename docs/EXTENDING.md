# Extending InferRef

This page is the "where do I plug in my thing" guide for external engine
developers. InferRef ships four extension points: semantic detectors,
executable contracts, engine adapters, and corpus cases.

For stateful chains of testcases (prefill → decode), see
[InferRef Scenario v0.1](spec/InferRef_Scenario_v0.1.md).

## 1. Add a semantic detector

**Goal:** teach `inferref region detect` to recognise a construct your model
uses.

**Files to create:** a Python module exposing a zero-argument factory, and a
`pyproject.toml` entry point in the `inferref.semantic_detectors` group. The
entry-point name and `detector.name` must match, and the name must not shadow a
built-in detector.

**Minimal example:**

```toml
[project.entry-points."inferref.semantic_detectors"]
my_gate = "my_pack.detectors:create_detector"
```

```python
from inferref.semantic.base import SemanticDetector, Detection

class MyGateDetector:
    name = "my_gate"

    def detect(self, package) -> list[Detection]:
        """Return every construct this detector recognises in the trace."""
        # Inspect package.graph.ops_in_execution_order() and
        # package.source(op.source_id).stack, then return Detection records
        # with contiguous node_ids. See inferref/semantic/source_function.py
        # for a complete reference implementation.
        return []

def create_detector() -> SemanticDetector:
    return MyGateDetector()
```

**Verify:**

```bash
inferref doctor --verify-plugins
inferref region detect trace/ --detector my_gate
```

If the same pack also ships executable contracts, document them with
[InferRef Contract Schema v0.1](spec/InferRef_Contract_Schema_v0.1.md).

## 2. Add an executable contract

**Goal:** define the engine ABI for one semantic operation so extraction,
validation, and adapter preflight all share the same input/output roles and
invariants.

**Files to create:** a Python module returning contract descriptors, an
optional standalone `.contract.json` schema file, and a `pyproject.toml` entry
point in the `inferref.contracts` group.

**Minimal example (SwiGLU-style):**

```toml
[project.entry-points."inferref.contracts"]
my_pack = "my_pack.contracts:build"
```

```python
def build():
    return [{
        "format": "inferref-contract",
        "format_version": "0.1",
        "id": "swiglu/fused/v1",
        "description": "SwiGLU fusion over the last dimension",
        "inputs": {"x": {"kind": "tensor"}, "gate": {"kind": "tensor"}},
        "outputs": {"y": {"kind": "tensor"}},
        "relations": [
            "y.shape == x.shape",
            "y.dtype == x.dtype",
            "gate.shape == x.shape",
        ],
        "features": ["multiple_outputs"],
        "effects": ["pure"],
    }]
```

**Verify:**

```bash
inferref contract list
inferref contract validate contracts/swiglu.contract.json
inferref testcase extract trace/ --region "SwiGLU@layers.0.mlp" \
    -o repro/swiglu --input-names x,gate --output-names y \
    --contract swiglu/fused/v1
```

The full schema, relation grammar, and CLI contract are in
[InferRef Contract Schema v0.1](spec/InferRef_Contract_Schema_v0.1.md).

## 3. Add an engine adapter

**Goal:** run your engine against a standalone testcase or a whole suite,
with capability preflight before any process is started.

**Files to create:** one `inferref-engine-adapter` v0.2 JSON file.

**Minimal example:**

```json
{
  "format": "inferref-engine-adapter",
  "format_version": "0.2",
  "name": "my-engine",
  "target_device": "cuda",
  "capabilities": {
    "device_types": ["cuda"],
    "dtypes": ["float32", "float16"],
    "max_rank": 5,
    "features": ["multiple_outputs"],
    "contracts": ["swiglu/fused/v1"],
    "contract_capabilities": {
      "swiglu/fused/v1": {
        "dtypes": ["float32"],
        "max_rank": 4,
        "features": ["multiple_outputs"]
      }
    }
  },
  "command": ["my_engine", "--testcase", "{testcase}", "--output", "{output}"]
}
```

InferRef expands exactly four placeholders in `command` — `{testcase}`,
`{output}`, `{adapter_dir}`, and `{python}` — and starts the process without a
shell, so the adapter JSON is trusted executable configuration. Per-contract
capabilities must be subsets of the global declaration; a contradictory file is
rejected at load time, and mismatches become `unsupported` before launch.

For a C++ engine, generate the whole project instead of writing the JSON by
hand:

```bash
# Standard mode: edit the RunYourEngine body and return named outputs.
# Output loading, writing, and manifest.json are already wired.
inferref adapter scaffold repro/swiglu --language cpp --output adapter/

# Runtime-bridge mode: edit only the DebugInvoke callback; the bridge resolves
# the region key and handles loading, output writing, and manifest.json.
inferref adapter scaffold repro/swiglu --language cpp --runtime-bridge \
  --output bridge/
```

The scaffold derives `capabilities` from `derive_requirements()`, validates the
generated `adapter.json` with the same loader used by `inferref agent run`, and
writes a compilable `main.cpp` against the header-only C++ testcase helper and
runtime bridge. See
[InferRef 0.8 Adapter DX](spec/InferRef_0.8_Adapter_DX.md) for the C++ API and
the scaffold output layout.

**Verify:**

```bash
inferref agent run repro/swiglu --adapter my_engine.adapter.json \
    --runs-dir runs --json
inferref suite run suite.json --adapter my_engine.adapter.json --runs-dir runs
```

See [InferRef Engine Adapter v0.2](spec/InferRef_Engine_Adapter_v0.2.md) for
the full wire contract.

## 4. Add a corpus case

**Goal:** contribute a reusable, reproducible testcase to a corpus suite.

**Files to create:** a standalone testcase directory with `testcase.json`,
`inputs/*.irtensor`, and `reference/*.irtensor`, plus a suite manifest entry.

**Minimal layout:**

```text
corpus/my-corpus/
  suite.json
  cases/swiglu-fp32/
    testcase.json
    inputs/x.irtensor
    inputs/gate.irtensor
    reference/y.irtensor
```

Suite manifest entry:

```json
{
  "format": "inferref-suite",
  "format_version": "0.1",
  "name": "my-corpus",
  "cases": [
    {"id": "swiglu-fp32", "testcase": "cases/swiglu-fp32", "tags": ["swiglu"]}
  ]
}
```

**Verify:**

```bash
inferref suite validate suite.json
inferref suite run suite.json --adapter my_engine.adapter.json --runs-dir runs
```

Suite manifest rules, execution order, and run status semantics are in
[InferRef Suite v0.1](spec/InferRef_Suite_v0.1.md).
