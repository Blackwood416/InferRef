# InferRef Engine Adapter and Testcase Requirements v0.2

## Purpose

Adapter v0.2 lets InferRef reject an unsupported testcase before launching an
engine. Testcases remain device-neutral; deployment device selection belongs to
the adapter or Suite run.

## Adapter

```json
{
  "format": "inferref-engine-adapter",
  "format_version": "0.2",
  "name": "native-sycl",
  "target_device": "xpu",
  "capabilities": {
    "device_types": ["xpu"],
    "dtypes": ["float32", "float16", "bfloat16"],
    "max_rank": 5,
    "features": ["multiple_outputs"],
    "contracts": ["rope/rotate-half/v1"]
  },
  "command": ["engine", "--testcase", "{testcase}", "--output", "{output}"]
}
```

`device_types` and `dtypes` are non-empty unique string arrays. `max_rank` is a
non-negative integer. Defined feature names are `multiple_outputs`,
`strided_inputs`, `alias_effects`, and `mutation_effects`. The base device type
of `target_device` must occur in `device_types`.

Adapter JSON is trusted executable configuration. InferRef expands only
`{testcase}`, `{output}`, `{adapter_dir}`, and `{python}`, and starts the command
without a shell.

## Testcase requirements

Writers emit `inferref-testcase` v0.2 with:

```json
{
  "requirements": {
    "dtypes": ["float16"],
    "max_rank": 4,
    "features": ["multiple_outputs"]
  }
}
```

The extractor derives these fields from all tensor records, input layout,
output cardinality, and recorded alias/mutation effects. A testcase does not
require a device type.

Before process creation, each required dtype and feature must be declared and
`requirements.max_rank` must not exceed the adapter limit. Failure produces a
structured `unsupported` result with `execution: null`.

### Versioned executable contracts

`contracts` is an optional migration field on both adapter capabilities and
testcase requirements. Contract identifiers are versioned paths. When both
sides declare the field, every testcase contract must be supported or preflight
returns `unsupported` without creating a process. When a testcase declares a
contract and a v0.2 adapter omits `contracts`, execution retains compatibility
but the run records `capability_status: unchecked`.

The initial native contracts are:

- `rmsnorm/last-dim/v1`: normalize over a non-empty last dimension; weight
  numel equals that dimension.
- `rope/rotate-half/v1`: positive even rotary dimension, split-half rotation,
  and matching `[sequence, rotary_dim]` cosine/sine inputs.
- `kv-cache/append/v1`: append update tensors on the penultimate sequence axis;
  all other dimensions match.
- `kv-cache/indexed-update/v1`: replace one or more positions on the penultimate
  sequence axis using one non-negative scalar index.

Contracts come from an explicit extraction profile (`testcase extract
--contract ...`) or an engine mapping, never from a Suite tag or testcase name.
The `contracts` field becomes mandatory in the next adapter/testcase format.

Readers continue to accept v0.1 testcases by deriving requirements at load time.
Adapter v0.1 has capability status `unchecked` and retains its historical
execution behavior. Writers emit only v0.2.
