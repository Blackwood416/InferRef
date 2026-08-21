# InferRef 0.9.0: Semantic Validation

> **Status:** Draft / 0.9.0 scope

## 1. Scope

0.8 delivered Adapter DX and the 0.8.1 quality review confirmed the scaffold
and bridge loop. 0.9 moves InferRef from tensor-level comparison to
task-level semantic equivalence.

0.9 also starts with the small 0.8.1 review fixes that protect the scaffold
template and make the generated standard-mode adapter as safe as bridge mode.

The `semantic` term remains reserved for region detection. All comparison work
in this spec uses `comparator`.

## 2. Design goals

1. Define a versioned Comparison Spec that can be attached to testcases and
   suite cases, overridden by the CLI, and recorded in run artifacts.
2. Define a comparator plugin protocol where task knowledge lives outside
   core and plugin discovery does not require importing every plugin.
3. Support multi-output comparison, such as object detection's
   `(boxes, scores, classes)` artifact set.
4. Keep the old numeric comparison as the default and make old readers reject
   comparison-bearing artifacts instead of silently ignoring them.

## 3. Non-goals

- Dataset-level evaluation (mAP, WER, CER, ...) remains external.
- Image decoding, preprocessing, NMS, and serving are out of scope.
- No new Trace IR or `.irtensor` versions.
- Comparator plugins are trusted same-process Python; no sandbox or timeout is
  promised in 0.9.

## 4. Terminology

| Term | Meaning |
| --- | --- |
| Comparison Spec | Versioned comparison policy attached to a testcase or suite case |
| Effective Comparison | Resolved policy after CLI, suite, testcase, and defaults |
| Comparator Plugin | Entry-point-backed implementation of one comparison |
| ArtifactSet | Named output roles made available to a comparator |
| Conditional Format | Testcase/suite version chosen by whether `comparison` is present |

## 5. Preflight fixes from the 0.8.1 quality review

These tasks are merged before semantic P0 work so scaffold DX does not regress
while the comparator machinery is added.

### F1: Shared output-contract prevalidation

Current state:

- `RunBridge` validates the full output role contract before writing any file
  (`cpp/include/inferref/bridge.hpp`).
- Generated standard-mode `main.cpp` writes `outputs.at(name)` without
  prevalidation, so a missing role produces `std::out_of_range` and leaves
  partial output files.

Required change:

- Add `Testcase::WriteOutputs(
    const std::map<std::string, IRTensor> &,
    const std::string &caller_label = "engine")` to
  `cpp/include/inferref/testcase.hpp`.
- `WriteOutputs` validates every declared output role is present, rejects
  undeclared roles, and only then writes files. No partial output files are
  left on a contract error.
- `RunBridge` and the generated standard-mode `main.cpp` both call
  `WriteOutputs`; bridge passes `"DebugInvoke"`, scaffold passes
  `"RunYourEngine"`.
- Missing-role diagnostics use the caller label and role name, for example:

  ```text
  RunYourEngine did not return output role 'z'
  ```

### F2: Generated-project compile test

Current state:

- `tests/core/test_adapter_scaffold.py` only checks generated strings.
- The C++ CI job compiles `cpp/`, not scaffold output.

Required change:

- Add a `@pytest.mark.slow` test that scaffolds a fixture testcase into
  `tmp_path`, detects CMake and a C++ compiler, configures and builds the
  generated project, and fails if the build fails.
- Skip with a clear reason when no compiler/CMake is available, but run it in
  the C++ CI job where both are guaranteed.
- Cover both standard mode and bridge mode.

### F3: Path corrections

Update the 0.8 spec and dev status:

- `examples/runtime_bridge/main.cpp` -> `cpp/examples/runtime_bridge/main.cpp`
- any other stale `examples/` paths under `docs/spec/InferRef_0.8_Adapter_DX.md`
  and `docs/dev/status/2026-08-21-v080-adapter-dx-and-v081-ci.md`

### F4: EXTENDING standard-mode description

Current `docs/EXTENDING.md` says standard mode requires writing output
handling yourself, which contradicts the generated `main.cpp` and README.

Required change:

```text
Standard mode: edit the RunYourEngine body and return named outputs.
Output loading, writing, and manifest.json are already wired.
```

### F5: Structured adapter failure codes (optional P2)

Current `execute_adapter` maps any nonzero exit to `execution_error`, so the
bridge's `kBridgeMissingRegion` distinction only appears in stderr.

Optional 0.9 P2:

- Record `exit_code` and a normalized `adapter_error_code` in the run record.
- Map bridge codes: `1 -> bridge_error`, `2 -> bridge_usage`,
  `3 -> bridge_missing_region`.
- Keep the Agent envelope status `error` unchanged.

## 6. P0: Comparison Spec v0.1

### 6.1 Wire format

```json
{
  "format": "inferref-comparison",
  "format_version": "0.1",
  "comparator": "tensor/numeric/v1",
  "config": {
    "atol": 0.001,
    "rtol": 0.01,
    "strict_layout": false
  },
  "outputs": {
    "boxes": {
      "comparator": "vision/object-detection/v1",
      "config": {
        "box_format": "xyxy",
        "matching": "iou",
        "min_iou": 0.99,
        "class_exact": true,
        "score_atol": 0.002,
        "ordering": "ignore"
      }
    }
  }
}
```

`outputs.<name>.comparator` is optional. An output may override only `config`:

```json
{
  "outputs": {
    "y": {
      "config": {
        "atol": 0.0001
      }
    }
  }
}
```

### 6.2 Conditional format version

When `comparison` is present:

- testcase `format_version` becomes `0.3`;
- suite `format_version` becomes `0.3` when a case carries `comparison`;
- `TESTCASE_READ_VERSIONS` and suite read versions include the new version;
- old readers reject `0.3` instead of silently running numeric defaults.

This is intentional. The version is content-dependent:

- a testcase without `comparison` remains `0.2`;
- a testcase with `comparison` is `0.3`.

Implementation note:

- `inferref/testcase/extract.py` currently writes the `TESTCASE_FORMAT_VERSION`
  constant unconditionally. The writer must choose `0.2` or `0.3` based on
  whether `comparison` is present.
- There is no suite writer component today: suite manifests are hand-authored
  JSON, and `Suite.to_dict()` only embeds a suite into run reports. Suite-side
  conditional versioning is therefore enforced by the reader validator, not
  by a writer.
- Reader rule, enforced in `validate_testcase` and the suite validator:

  > `comparison` present with `format_version` lower than `0.3` is a hard
  > error, for example `comparison_requires_0_3`.

- Scenario already rejects unknown versions strictly; the 0.9 spec keeps that
  behavior and documents the asymmetry.

### 6.3 Effective comparison and precedence

Tolerance is two independent axes, matching `TolerancePolicy`:

Axis A - per-dtype default table, each layer replaces the previous table:

```text
suite case config.per_dtype
  > testcase config.per_dtype
  > CLI --tolerance file
  > DEFAULT_TOLERANCES
```

Axis B - scalar overrides, applied after Axis A:

```text
CLI --atol / --rtol
  > suite case config.atol / config.rtol
  > testcase config.atol / config.rtol
  > none
```

This removes the misleading "CLI loses to artifact" ordering. `--tolerance`
is a default-table replacement; `--atol` / `--rtol` are scalar overrides that
always win.

Every run records:

```json
{
  "effective_comparison": {
    "comparator": "tensor/numeric/v1",
    "config": {
      "atol": 0.001,
      "rtol": 0.01,
      "strict_layout": false
    },
    "per_output": {},
    "sources": {
      "comparator": "testcase",
      "config.atol": "cli",
      "config.rtol": "cli",
      "config.per_dtype": "testcase"
    }
  }
}
```

`sources` maps each resolved field to its contributing layer. A scalar
`source` is not enough because CLI overrides can apply to one field while the
comparator and per-dtype table come from the testcase.

### 6.4 Validation

- Unknown comparator IDs are rejected before any engine process starts.
- `validate_config` runs before engine launch and raises
  `invalid_comparison_config`.
- Missing output roles are short-circuited by core and count as failure.
- `comparison` with `format_version` below `0.3` is a hard validation error in
  both testcases and suites.

## 7. P0: Comparator plugin protocol

### 7.1 Entry point

```toml
[project.entry-points."inferref.comparators"]
"vision/object-detection/v1" = "inferref_vision:DetectionComparator"
```

The entry-point name is the full comparator ID. A loaded plugin's `.id` must
equal its entry-point name. Discovery therefore never requires importing the
plugin module, matching the semantic detector and contract registries.

### 7.2 Python protocol

```python
class ComparatorPlugin:
    id: str

    def validate_config(self, config: dict[str, Any]) -> None:
        """Raise on invalid config before any engine process starts."""

    def compare(
        self,
        reference: ArtifactSet,
        actual: ArtifactSet,
        config: dict[str, Any],
    ) -> ComparatorResult:
        ...
```

`ArtifactSet` maps output role name to artifact path:

```python
ArtifactSet = Mapping[str, Artifact]

@dataclass(frozen=True)
class Artifact:
    name: str
    path: Path
```

Tensor artifacts are `.irtensor` files. Plugins decode with
`inferref.tensor.codec.read`; core does not force a numpy-only API.

`ComparatorResult`:

```python
@dataclass(frozen=True)
class ComparatorResult:
    status: str                    # pass | fail | error
    comparator: str
    metrics: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    first_failure: dict[str, Any] | None = None
```

`missing` is not a plugin status. Core handles missing roles before calling
the plugin.

### 7.3 Registry rules

- Built-in `tensor/numeric/v1` wraps the current numeric comparator.
- Duplicate comparator IDs are rejected.
- Built-ins cannot be overridden.
- `inferref comparator list/show` mirrors `inferref contract list/show`.
- `inferref doctor --verify-plugins` reports comparator plugin status.
- A plugin exception is caught and converted to `error`; the suite continues.
- Comparators are trusted same-process Python, documented as such.
- **Reserved numeric config keys**: Numeric defaults (`per_dtype`, `strict_layout`, `ignore_stride`, `atol`, `rtol`) are reserved for numerical policy resolution. Non-numeric custom comparators receive cleaned configs with these numeric keys stripped during static validation (`validate_config`) and runtime execution (`compare`).

### 7.4 Multi-output comparison

The comparator receives the complete output role set:

```text
reference/
├── boxes.irtensor
├── scores.irtensor
└── classes.irtensor

actual/
├── boxes.irtensor
├── scores.irtensor
└── classes.irtensor
```

The object-detection comparator interprets `(boxes, scores, classes)` as one
record set, not three independent tensors. Variable-length records are handled
by the plugin through shapes or masks; no new tensor encoding is required.

## 8. P0: Suite, scenario, and agent integration

### 8.1 Suite

- Suite case gains optional `comparison`.
- `run_suite` gains tolerance/comparison policy plumbing; today it has none.
- `effective_comparison` appears in per-cell run records and the suite report.
- `SuiteCase` dataclass and `to_dict()` must preserve `comparison` so reports
  do not silently drop it.
- `Suite.to_dict()` must preserve the source suite manifest's actual
  `format_version` instead of hardcoding `SUITE_FORMAT_VERSION`, so audit
  records do not report a 0.1 suite as 0.2.

### 8.2 Scenario

- Scenario steps inherit comparison policy plumbing.
- Step-level comparison overrides are allowed when needed.
- Scenario run records include `effective_comparison`.

### 8.3 Agent protocol and MCP

- `compare_outputs` and `run_engine` accept optional `comparator` and
  `comparison_config`.
- MCP tools expose the same arguments as JSON values.
- The response envelope remains additive; comparator metrics live in
  `data.comparator`, not `data.semantic`.
- `status` semantics remain `pass` / `fail` / `error`.
- `capabilities()` no longer reports a single testcase version. It reports:

  ```json
  {
    "testcase": {
      "format": "inferref-testcase",
      "writes": ["0.2", "0.3"],
      "reads": ["0.1", "0.2", "0.3"]
    }
  }
  ```

## 9. P1: Agent JSON summary mode

### 9.1 Format

New `inferref-agent-summary` v0.1:

```json
{
  "format": "inferref-agent-summary",
  "format_version": "0.1",
  "status": "fail",
  "operation": "run_engine",
  "first_failure": {
    "output": "boxes",
    "message": "unmatched detection at index 3"
  },
  "metrics": {
    "reference_count": 17,
    "actual_count": 17,
    "matched": 16,
    "min_iou": 0.91
  },
  "artifacts": {
    "run_record": "runs/abc/inferref-run.json"
  },
  "next_actions": [
    {
      "operation": "inspect_first_divergence",
      "reason": "Fix the first unmatched detection and rerun"
    }
  ]
}
```

Rules:

- `--json-summary` and `--json` are mutually exclusive.
- `--json-summary` itself selects JSON output; it is not routed through the
  text renderer.
- Exit codes are unchanged.
- `next_actions` keeps the existing `{operation, reason}` object shape.
- Numeric runs emit numeric metrics; comparator runs emit comparator metrics,
  never both in the same object.
- Full data stays in `inferref-run.json`.

## 10. P1: Region preview and recommendation

### 10.1 `region list --details`

Fields:

| Field | Meaning |
| --- | --- |
| `operators` | number of operators |
| `inputs` | boundary input count |
| `outputs` | boundary output count |
| `activation_bytes` | boundary activation bytes |
| `parameter_bytes` | estimated parameter bytes |
| `largest_tensor` | largest boundary tensor |
| `payload_coverage` | boundary payload completeness |
| `reproducible` | whether extraction is runnable |
| `mutation` | whether the region has mutation effects |
| `semantic_confidence` | detector confidence when present |

Each region reports its own boundary bytes. Nested region byte counts are not
summable; the list provides no totals, so users cannot accidentally double
count overlapping regions.

### 10.2 Parameter heuristic

Use a checkable definition:

- a boundary input whose `value.producer` is absent is counted as a parameter
  candidate;
- all other boundary inputs are activation;
- the result is labeled `estimated` in JSON.

### 10.3 `region recommend`

Deterministic scoring:

```text
+ payload complete
+ reproducible
+ semantic confidence
+ small boundary
+ reasonable node count
- excessive parameter size
- excessive I/O count
- opaque value
- unresolved mutation
```

Warnings are emitted for large boundaries.

## 11. P1: Doctor hardware details

Extend `DoctorCheck.details` with best-effort hardware information:

- device name and ID;
- driver version;
- total and available memory;
- PyTorch XPU/CUDA runtime;
- oneAPI / SYCL runtime;
- oneDNN version when discoverable;
- installed plugin versions.

These fields are informational and never become hard PASS requirements. Text
output shows the device name instead of only device count.

## 12. Compatibility

0.9 changes wire formats deliberately:

- Testcase `0.3` only when `comparison` is present; `0.2` remains for legacy
  artifacts.
- Suite `0.3` only when a case carries `comparison`.
- Engine Adapter remains `0.2`.
- Agent Protocol remains `0.1`.
- Trace IR and `.irtensor` remain unchanged.

Old 0.8.x readers reject `0.3` comparison-bearing testcases/suites. They do
not silently downgrade to numeric comparison.

## 13. Acceptance criteria

### 13.1 Preflight fixes

- [ ] `WriteOutputs` validates before writing and leaves no partial output.
- [ ] Standard mode and bridge mode use the same output validation.
- [ ] Missing-role diagnostics include the caller label (`RunYourEngine` /
  `DebugInvoke`) and the missing role name.
- [ ] Slow scaffold compile test exists and runs in C++ CI.
- [ ] `cpp/examples/runtime_bridge` paths are correct in specs/status.
- [ ] `docs/EXTENDING.md` standard-mode description is accurate.

### 13.2 Comparison spec

- [ ] Comparison Spec v0.1 validates.
- [ ] Conditional testcase/suite `0.3` is implemented.
- [ ] Old readers reject `0.3`.
- [ ] `comparison_requires_0_3` is enforced by validators.
- [ ] Effective comparison is recorded with per-field `sources`.
- [ ] Two-axis tolerance is implemented as specified.
- [ ] Suite and scenario carry comparison policy.
- [ ] `Suite.to_dict()` preserves source `format_version`.

### 13.3 Comparator plugins

- [ ] Entry-point name equals comparator ID.
- [ ] `validate_config` rejects bad config before engine launch.
- [ ] Multi-output comparison works.
- [ ] Missing outputs are short-circuited.
- [ ] Plugin exceptions do not crash a suite.
- [ ] `tensor/numeric/v1` matches current numeric behavior.

### 13.4 Agent UX

- [ ] `--json-summary` works for run/compare/scenario.
- [ ] Full artifact retains all details.
- [ ] Summary keeps the envelope `next_actions` shape.
- [ ] `capabilities()` reports testcase `reads` / `writes` instead of a single
  version.

### 13.5 Region / doctor

- [ ] `region list --details` exists.
- [ ] `region recommend` exists.
- [ ] Doctor hardware details are best-effort and non-gating.

## 14. Implementation tasks

### C1: 0.8.1 preflight fixes

- Files:
  - `cpp/include/inferref/testcase.hpp`
  - `cpp/include/inferref/bridge.hpp`
  - `inferref/adapter/scaffold.py`
  - `tests/core/test_adapter_scaffold.py`
  - `docs/EXTENDING.md`
  - `docs/spec/InferRef_0.8_Adapter_DX.md`
  - `docs/dev/status/2026-08-21-v080-adapter-dx-and-v081-ci.md`
- Outcome: F1-F4 are implemented; F5 is designed but optional.

### C2: Comparison spec and policy

- Files:
  - `inferref/comparison/`
  - `inferref/cli/main.py`
  - `inferref/testcase/extract.py`
  - `inferref/testcase/validate.py`
  - `inferref/suite/schema.py`
  - `inferref/suite/run.py`
  - `inferref/scenario/`
  - `inferref/agent/`
- Outcome: sections 6 and 8.

### C3: Comparator registry and numeric comparator

- Files:
  - `inferref/comparators/`
  - `inferref/doctor.py`
  - `inferref/cli/main.py`
- Outcome: section 7 registry plus `tensor/numeric/v1`.
- C2 depends on C3 because policy resolution must reject unknown comparator
  IDs before execution.

### C4: Multi-output comparator

- Files:
  - `tests/fixtures/comparator_pack/`
  - `examples/comparators/object_detection.py`
- Outcome: section 7.4 and fixture-driven acceptance.

### C5: Agent summary mode

- Files:
  - `inferref/agent/`
  - `inferref/cli/main.py`
- Outcome: section 9.

### C6: Region preview and recommendation

- Files:
  - `inferref/region/`
  - `inferref/cli/main.py`
- Outcome: section 10.

### C7: Doctor hardware details

- Files:
  - `inferref/doctor.py`
  - `inferref/frontend/pytorch/accelerator.py`
- Outcome: section 11.

### C8: Structured adapter failure codes (P2, optional)

- Files:
  - `inferref/agent/adapter.py`
  - `cpp/include/inferref/bridge.hpp`
- Outcome: F5.

## 15. Dispatch order

Wave 1:

- C1 (preflight fixes)

Wave 2 (after C1):

- C3 (comparator registry)

Wave 3 (after C3):

- C2 (comparison spec and policy)

Wave 4 (after C2):

- C4 (multi-output comparator)
- C5 (agent summary)

Wave 5:

- C6 (region UX)
- C7 (doctor details)

Wave 6 (optional):

- C8 (structured adapter failure codes)

## 16. Open decisions

### D1: Comparator trust boundary

Recommend documenting comparators as trusted same-process code. CI uses only
deterministic fixture plugins; no timeout or subprocess isolation is promised
in 0.9.

### D2: Region parameter classification

Recommend `value.producer is None` as the only estimated parameter rule.

## 17. Release theme

```text
0.8: "如何让已有 inference runtime 只集成一次 InferRef？"

0.9: "我的 engine 是否在任务语义上复现了 reference？"
```

One-sentence goal:

> InferRef 0.9.0 extends validation from tensor equivalence to semantic
> equivalence without allowing old readers to silently downgrade a declared
> comparison policy.
