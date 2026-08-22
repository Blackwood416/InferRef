# InferRef 0.8.0: Adapter DX

> **Status:** Draft / 0.8.0 scope, revised after spec review

## 1. Scope decision

The original 0.8 request combined two unrelated product surfaces:

1. semantic / task-level comparison;
2. adapter and runtime integration efficiency.

Code review confirmed that the two have no shared format, core plumbing, or
fixture. This revision splits the release:

- **0.8.0 = Adapter DX.** Add the missing C++ testcase helper, an adapter
  scaffolder, and a generic runtime bridge. This is pure increment over the
  existing adapter ABI and can ship without a format bump.
- **0.9.0 = Semantic Validation.** Comparison Spec, comparator plugins,
  multi-output comparison, policy precedence, agent summary mode, region
  recommendation, and doctor details. This release gets a deliberate format
  bump so old readers cannot silently ignore comparison policies.

This document is the 0.8.0 spec. Section 10 records the corrected 0.9
preconditions so the review findings are not lost.

## 2. Design goal

The YOLO26 / Aila experiment showed that the numerical harness works across a
new model family, but a coding agent still spends too much time writing
per-region C++ adapter glue.

0.8.0 goal:

> A runtime should integrate InferRef once and then execute every future
> region through the same adapter.

The runtime does not need to know about `testcase.json`, `.irtensor` headers,
or the engine output manifest. Those details live in shared C++ helpers.

## 3. Non-goals for 0.8.0

- No semantic comparison, task metrics, or comparator plugins.
- No comparison policy stored in testcase or suite.
- No `--json-summary`, region recommendation, or doctor hardware expansion.
- No change to Trace IR, testcase, suite, adapter, or agent protocol versions.

## 4. Terminology

| Term | Meaning |
| --- | --- |
| Testcase Helper | C++ API that loads a testcase and writes engine outputs |
| Adapter Scaffold | Generated adapter project from one testcase |
| Runtime Bridge | One adapter binary that dispatches any region into an existing runtime |
| DebugInvoke | Runtime callback: region name + named inputs -> named outputs |
| TFCM | Time-to-First-Correct-Model |

The word `semantic` remains reserved for region detection in this project.
Comparator work uses `comparator`, not `semantic`.

## 5. P0: C++ testcase helper

### 5.1 Motivation

`irtensor.hpp` provides low-level read/write and `json.hpp` provides JSON
parsing, but a real engine still has to duplicate testcase manifest handling
for every adapter. `testcase.hpp` is the missing high-level layer.

### 5.2 Public API

Add `cpp/include/inferref/testcase.hpp`, header-only and dependency-free:

```cpp
namespace inferref
{

class Testcase
{
public:
    static Testcase Load(const std::string &testcase_dir);

    std::string RegionName() const;
    std::vector<std::string> InputNames() const;
    std::vector<std::string> OutputNames() const;

    IRTensor Input(const std::string &name) const;
    std::map<std::string, IRTensor> Inputs() const;

    void SetOutputDir(const std::string &output_dir);
    void WriteOutput(const std::string &name, const IRTensor &tensor);
    void Finish();
};

} // namespace inferref
```

Decisions fixed by this API:

- `Input(name)` returns a loaded `IRTensor`, not a path. The caller never
  needs to know where payloads live.
- `Inputs()` returns a map keyed by role name. The runtime bridge and
  scaffold both use the named form, never positional vectors.
- `SetOutputDir(output_dir)` redirects output tensors and `manifest.json` to
  the Engine Adapter v0.2 `{output}` directory, creating it if needed. It
  defaults to the testcase directory; adapters and the bridge always set it.
- `WriteOutput(name, tensor)` writes one output file into the output directory
  and does not rewrite `manifest.json`.
- `Finish()` writes `manifest.json` once and is idempotent.
- `manifest.json` in the engine output directory remains optional for
  `inferref compare`; `Finish()` exists for runtimes that want explicit
  output metadata.

### 5.3 Validation

- `testcase.hpp` has a C++ self-test covering load, input read, output write,
  `Finish()` idempotence, and missing-role errors.
- It must compile with only the existing header-only `irtensor.hpp` and
  `json.hpp`.

## 6. P0: Adapter scaffolder

### 6.1 CLI

```bash
inferref adapter scaffold testcase/ \
  --language cpp \
  --output adapter/
```

Output:

```text
adapter/
├── CMakeLists.txt
├── adapter.json
├── main.cpp
└── README.md
```

### 6.2 Generated `main.cpp`

The generated file compiles immediately. The only user edit is the body of
`RunYourEngine`.

```cpp
#include <inferref/testcase.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

std::map<std::string, inferref::IRTensor> RunYourEngine(
    const std::string &region_name,
    const std::map<std::string, inferref::IRTensor> &inputs)
{
    // TODO: implement engine dispatch.
    (void)region_name;
    (void)inputs;
    throw std::runtime_error("RunYourEngine is not implemented");
}

int main(int argc, char **argv)
{
    std::string testcase_dir;
    std::string output_dir;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--testcase" && i + 1 < argc)
            testcase_dir = argv[++i];
        else if (arg == "--output" && i + 1 < argc)
            output_dir = argv[++i];
    }
    if (testcase_dir.empty() || output_dir.empty())
    {
        std::cerr << "usage: inferref_adapter --testcase DIR --output DIR\n";
        return 2;
    }

    try
    {
        auto testcase = inferref::Testcase::Load(testcase_dir);
        testcase.SetOutputDir(output_dir);
        auto inputs = testcase.Inputs();
        auto outputs = RunYourEngine(testcase.RegionName(), inputs);

        for (const auto &name : testcase.OutputNames())
            testcase.WriteOutput(name, outputs.at(name));
        testcase.Finish();
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "inferref_adapter: " << error.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "inferref_adapter: unknown non-standard exception\n";
        return 1;
    }
}
```

Why this shape:

- argv follows the Engine Adapter v0.2 command template, not positional
  arguments.
- `argc` / missing-flag errors return a non-zero exit code.
- `RunYourEngine` is declared with a concrete signature and a throwing stub,
  so the file compiles before the user edits it.
- Outputs are bound by role name: `RunYourEngine` returns a
  `std::map<std::string, inferref::IRTensor>` keyed by output role, so
  `outputs.at(name)` compiles and cannot accidentally reorder outputs.
- `SetOutputDir(output_dir)` directs output tensors and `manifest.json` to the
  adapter's `{output}` directory instead of the testcase directory.
- The dispatch loop is wrapped in `try`/`catch` (with a `catch (...)` fallback)
  so a throwing engine stub (or a runtime error) exits non-zero immediately
  with a message, matching the bridge's error behavior instead of surfacing an
  unhandled exception.

### 6.3 Generated `adapter.json`

The generated adapter must be a complete, valid Engine Adapter v0.2 file:

```json
{
  "format": "inferref-engine-adapter",
  "format_version": "0.2",
  "name": "scaffolded-adapter",
  "target_device": "cpu",
  "capabilities": {
    "device_types": ["cpu"],
    "dtypes": ["float16"],
    "max_rank": 4,
    "features": ["multiple_outputs"],
    "contracts": ["conv/silu/v1"]
  },
  "command": [
    "{adapter_dir}/inferref_adapter",
    "--testcase",
    "{testcase}",
    "--output",
    "{output}"
  ],
  "timeout_seconds": 60,
  "max_output_chars": 65536
}
```

Rules:

- `target_device` must be present and its base must appear in
  `capabilities.device_types`.
- `features` are copied from `derive_requirements()` output. The legal set is
  `alias_effects`, `multiple_outputs`, `mutation_effects`, `strided_inputs`;
  `multiple_inputs` is not a valid feature and must not be generated.
- `contracts` is copied from the testcase when present.
- The user edits `name`, `target_device`, device list, and the command
  executable; the rest is generated.

### 6.4 Generated CMake

The generated `CMakeLists.txt` supports an external include root:

```cmake
set(INFERREF_CPP_INCLUDE "" CACHE PATH "Path to InferRef cpp/include")
target_include_directories(inferref_adapter PRIVATE ${INFERREF_CPP_INCLUDE})
```

The README explains both options: point `INFERREF_CPP_INCLUDE` at the InferRef
checkout, or copy `cpp/include/inferref` into the engine tree.

## 7. P0: Runtime bridge

### 7.1 One bridge, many regions

The runtime bridge keeps the existing adapter ABI:

```bash
inferref-bridge \
  --testcase testcase/ \
  --output output/
```

The binary name is `inferref-bridge` in the reference implementation. A
downstream runtime may name its own binary anything it wants, for example
`aila-inferref-bridge.exe`.

### 7.2 Bridge API

Add `cpp/include/inferref/bridge.hpp`:

```cpp
using DebugInvoke = std::function<std::map<std::string, inferref::IRTensor>(
    const std::string &region_name,
    const std::map<std::string, inferref::IRTensor> &inputs)>;
```

Outputs are a map keyed by role name. This matches Engine Adapter v0.2's exact
output-role contract and cannot accidentally lose names.

### 7.3 Reference implementation

Add `cpp/examples/runtime_bridge/main.cpp`:

1. parse `--testcase` / `--output`;
2. load the testcase with `testcase.hpp`;
3. resolve the dispatch key:
   - `origin.region` when present;
   - otherwise `origin.contract` or `contracts[0]`;
   - otherwise fail with `missing_region`;
4. call `DebugInvoke(key, inputs)`;
5. write every declared output by role name;
6. call `Finish()`.

The reference bridge contains no model knowledge. A runtime registers one
`DebugInvoke` callback and supports future regions without writing another
adapter.

### 7.4 Scaffolder bridge mode

```bash
inferref adapter scaffold testcase/ \
  --language cpp \
  --runtime-bridge
```

This mode generates `main.cpp` that delegates to `DebugInvoke` instead of
requiring the user to rewrite output handling.

## 8. P1: CLI positioning docs

Update README, `docs/AGENT_WORKFLOW.md`, and `docs/EXTENDING.md` with:

- `inferref adapter scaffold` under the new `adapter` command group;
- the Human / Agent facade distinction:

  ```text
  Human / exploratory:
    inspect analyze region testcase compare suite scenario

  Stable automation / agent facade:
    agent context agent extract agent run agent compare agent run_scenario
  ```

Key statement:

> `inferref agent ...` does not define a separate InferRef execution model. It
> is the stable, machine-readable protocol facade over the same core
> operations.

## 9. Compatibility

0.8.0 does not bump any wire format:

- Trace IR stays `0.1`.
- Testcase stays `0.2`.
- Suite stays `0.2`.
- Engine Adapter stays `0.2`.
- Agent Protocol stays `0.1`.
- `.irtensor` stays v1.

New code is additive: new C++ headers, one new CLI command, and one reference
bridge example.

## 10. Deferred 0.9: Semantic Validation preconditions

The following review corrections are recorded here and must be applied before
0.9 implementation starts.

### 10.1 Comparison format bump

When `comparison` is added:

- testcase format must move to `0.3` when the field is present;
- suite format must move to `0.3` when a case carries `comparison`;
- `TESTCASE_READ_VERSIONS` / suite read versions add the new version;
- old readers reject the new version instead of silently ignoring it;
- the spec must document that scenario already rejects unknown versions
  strictly while testcase/suite currently accept known versions.

### 10.2 Comparator plugin protocol

- Add `validate_config(config)` to the plugin protocol. It raises
  `invalid_comparison_config` before any engine process starts.
- `missing` outputs are short-circuited by core; the plugin is not called and
  missing is counted as failure.
- Entry-point name is the full comparator ID. A loaded plugin's `.id` must
  equal its entry-point name, so discovery does not require import.
- Comparator IDs are resolved only against the comparator registry, even when
  they look like contract IDs.

### 10.3 Terminology

- Use `comparator` for comparison work: `inferref.comparators`,
  `examples/comparators/`, `data.comparator`.
- Keep `semantic` for region detection and `inferref.semantic_detectors`.

### 10.4 Policy precedence

Define the full tolerance lattice before implementation:

```text
CLI --atol / --rtol
  > suite case comparison
  > testcase comparison
  > CLI --tolerance file (per-dtype defaults)
  > comparator / dtype defaults
```

`suite run` currently has no tolerance or comparison policy plumbing; that is
a new T1 work item, not an existing capability.

### 10.5 Agent summary

- `--json-summary` and `--json` are mutually exclusive.
- Text mode with `--json-summary` prints the summary as text or returns an
  error; the spec must choose one.
- Exit codes are unchanged.
- `next_actions` keeps the existing `{operation, reason}` object shape.
- Summary metrics must be unambiguous: numeric run outputs numeric metrics,
  comparator runs output comparator metrics, not both in the same object.

### 10.6 TFCM

- 0.8 ships a runbook that produces the first TFCM record.
- 0.8 does not claim a comparison against a nonexistent 0.7.x baseline.

## 11. Acceptance criteria

All items below were closed by InferRef 0.8.0/0.8.1 and are covered by the
C++ selftests and `tests/core/test_adapter_scaffold.py`.

### 11.1 `testcase.hpp`

- [x] `Testcase::Load` reads a testcase without PyTorch.
- [x] `Input(name)` returns a loaded `IRTensor`.
- [x] `WriteOutput` writes output files.
- [x] `Finish()` writes `manifest.json` exactly once.
- [x] C++ self-test passes.

### 11.2 Adapter scaffolder

- [x] `inferref adapter scaffold` generates the four files.
- [x] Generated `main.cpp` compiles without user edits.
- [x] Generated `adapter.json` passes `EngineAdapter.load()`.
- [x] Generated features are members of the adapter feature whitelist.
- [x] Generated capabilities match `derive_requirements()` output.

### 11.3 Runtime bridge

- [x] `bridge.hpp` exists and uses named outputs.
- [x] `cpp/examples/runtime_bridge/main.cpp` compiles.
- [x] A stub `DebugInvoke` runs a fixture testcase end to end.
- [x] Missing region dispatch returns `missing_region`.

### 11.4 Docs

- [x] README documents `inferref adapter scaffold`.
- [x] `docs/AGENT_WORKFLOW.md` shows the Human / Agent facade split.
- [x] `docs/yolo-aila-080.md` records the first TFCM run.

## 12. Implementation tasks

### A1: C++ testcase helper

- Files: `cpp/include/inferref/testcase.hpp`, `cpp/examples/testcase_selftest/main.cpp`
- Outcome: section 5 API is implemented.
- Verify: C++ self-test passes on Linux and Windows.

### A2: Adapter scaffolder CLI

- Files: `inferref/adapter/`, `inferref/cli/main.py` `adapter` command
- Outcome: section 6 output tree, generated `main.cpp`, `adapter.json`,
  `CMakeLists.txt`, `README.md`.
- Verify: generated project compiles; adapter validates; capabilities derive
  from `derive_requirements()`.

### A3: Runtime bridge

- Files: `cpp/include/inferref/bridge.hpp`, `cpp/examples/runtime_bridge/main.cpp`
- Outcome: section 7 API and reference implementation.
- Verify: stub bridge runs a fixture testcase.

### A4: Bridge scaffold mode

- Files: `inferref/adapter/` scaffold bridge mode
- Outcome: `--runtime-bridge` generated project delegates to `DebugInvoke`.
- Verify: bridge-mode project compiles and runs a fixture.

### A5: Docs and TFCM runbook

- Files: `README.md`, `docs/AGENT_WORKFLOW.md`, `docs/EXTENDING.md`,
  `docs/yolo-aila-080.md`
- Outcome: section 8 and first TFCM record.
- Verify: links resolve; runbook can be followed on the YOLO26 workflow.

## 13. Dispatch order

Wave 1:

- A1 (testcase helper)

Wave 2 (parallel after A1):

- A2 (scaffolder)
- A3 (runtime bridge)

Wave 3:

- A4 (bridge scaffold mode)

Wave 4:

- A5 (docs and TFCM runbook)

## 14. Open decisions

### D1: `Input(name)` return type

Fixed: return `IRTensor` by value. The bridge and scaffold never need raw
paths.

### D2: Region dispatch fallback

Recommended: `origin.region` -> `origin.contract` / `contracts[0]` ->
`missing_region`.

### D3: Generated executable path

Recommended: `{adapter_dir}/inferref_adapter` for scaffold mode and
`{adapter_dir}/inferref_bridge` for bridge mode, both user-editable.

## 15. Release theme

```text
0.7: "我的 engine 是否数值复现了 reference？"

0.8: "如何让已有 inference runtime 只集成一次 InferRef？"
```

One-sentence goal:

> InferRef 0.8.0 reduces the cost of connecting real inference runtimes to
> extracted reference testcases, without changing any existing wire format.
