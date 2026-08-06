# Executable contract profiles, per-contract capability, native JSON parsing

## Closed issues

- Contracts are now registered profiles with fixed input/output role names and
  shape validators, not just ID labels. Extraction resolves the profile, binds
  the extracted boundary to the contract roles, validates bound tensor shapes,
  and refuses to publish when the contract is unknown, roles cannot be bound,
  or shapes are invalid. Standalone validation reuses the same registry and
  emits a non-blocking warning for well-formed but unknown contract IDs.
- Adapter capability is refined per contract via `contract_capabilities`
  (`dtypes`, `max_rank`, `features` per contract), checked against
  role-derived per-contract requirements before any engine process starts.
  `contracts` now has three states: missing (unchecked), `[]` (strictly zero
  contracts supported), and a non-empty list. Testcase `contracts` remains
  non-empty whenever present.
- The native SYCL engine reads the testcase manifest with a dependency-free
  JSON parser (`cpp/include/inferref/json.hpp`) instead of string scanning.
  String escapes, unicode escapes, duplicate keys, whitespace, and field order
  are covered by the C++ self-test, including manifests whose text merely
  mentions `"contracts"`.
- The XPU gate requires a fresh evidence directory per run (workflow scopes it
  by run ID/attempt; local runs refuse to reuse unless
  `-CleanEvidenceDirectory`), and supports `-ExpectedDeviceNameRegex` so the
  A770 qualification gate fails when the queue falls back to a different GPU.
- `kv-cache/indexed-update/v1` bounds are checked on the float value before any
  `size_t` conversion, and the target + update sum is checked without unsigned
  overflow. NaN, infinity, negative, non-integer, huge finite, out-of-range,
  and overflow cases are unit-tested in the C++ self-test.
- `inferref suite validate` reports `schema_valid` and `runnable` separately
  with `non_runnable_cases`; the CLI fails non-runnable suites by default and
  accepts them only with `--allow-nonreproducible`.
- `inferref suite report` rejects non-`.html` outputs so a `report.json`
  argument cannot overwrite the JSON sidecar.

## Acceptance evidence

- Python: `495 passed, 9 skipped` before this round; new coverage adds the
  contract registry, extractor role binding, per-contract preflight, suite
  validate runnability, and report collision tests.
- C++ MSVC self-test passes, including the new JSON parser and KV index bounds
  checks.
- The 13-case XPU corpus was checked against the new per-contract capability
  declaration and passes every derived requirement.

## Decisions

- Unknown-but-well-formed contract IDs stay valid in existing testcases with a
  warning (forward compatibility); extraction, which is where the profile is
  requested, is strict.
- Per-contract requirements are derived from the tensors bound to contract
  roles, so a multi-contract testcase is not penalized by unrelated dtypes;
  unknown contracts fall back to testcase-wide requirements.
- The C++ engine reads the top-level `contracts` field first, then
  `requirements.contracts`, matching the extractor output and the Python
  validator.

## Known gaps

- The contract profile is shape/role based; dtype and value-level semantics
  (for example rotation convention) still live in the engine implementation and
  are covered by the hardware gate, not by the registry.
- The index contract still uses a float32 index tensor for corpus compactness;
  an integer-typed index contract can be added as a future v2.
- `contract_capabilities` features are checked against role-derived features;
  node-effect-derived per-contract features are conservatively inherited from
  the testcase-wide requirements.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest tests\core\test_contracts.py tests\core\test_adapter_capabilities.py -q
ctest --test-dir cpp/build --output-on-failure
```
