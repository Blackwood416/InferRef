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
    "contracts": ["rope/rotate-half/v1"],
    "contract_capabilities": {
      "rope/rotate-half/v1": {
        "dtypes": ["float32", "float16"],
        "max_rank": 4,
        "features": ["multiple_outputs"]
      }
    }
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

### Contract capability states

`contracts` has three distinct states on an adapter:

- missing: capability is `unchecked` for contract-declaring testcases and
  historical execution behavior is retained;
- `[]`: the adapter strictly supports zero executable contracts, so any
  contract-declaring testcase is `unsupported` before process creation;
- a non-empty list: exactly those versioned contract IDs are claimed.

Testcase `contracts` (or `requirements.contracts`) remains non-empty whenever
present.

### Per-contract capabilities

An adapter-wide `dtypes`/`max_rank`/`features` declaration is an upper bound,
not a proof that every contract supports every combination.
`contract_capabilities` refines this per contract:

```json
{
  "contract_capabilities": {
    "rmsnorm/last-dim/v1": {
      "dtypes": ["float32", "float16", "bfloat16"],
      "max_rank": 3
    },
    "rope/rotate-half/v1": {
      "dtypes": ["float32", "float16"],
      "features": ["multiple_outputs"]
    }
  }
}
```

Every key must be declared in `contracts`, and each value may carry `dtypes`,
`max_rank`, and `features` (all optional, each using the same rules as the
global fields). Preflight derives per-contract requirements from the tensors
bound to the contract roles, so a float16 RoPE case is rejected by an adapter
that only declares float32 RoPE even when its global `dtypes` list is wider.
Mismatches are reported as `contract_dtype`, `contract_max_rank`, or
`contract_feature` issues before any engine process starts.

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

### Contract registry and role binding

Each known contract is a registered profile with fixed input/output role names
and a shape validator, not just an ID label:

| contract | inputs | outputs |
| --- | --- | --- |
| `rmsnorm/last-dim/v1` | `x`, `weight`, `epsilon` | `y` |
| `rope/rotate-half/v1` | `query`, `key`, `cos`, `sin` | `q_embed`, `k_embed` |
| `kv-cache/append/v1` | `cache`, `update` | `cache_out` |
| `kv-cache/indexed-update/v1` | `cache`, `update`, `index` | `cache_out` |

Extraction refuses to publish a testcase when a requested contract is not in
the local registry, when the extracted boundary cannot be bound to the
contract's role names, or when the bound tensors fail the contract shape
validator. Standalone validation of an existing testcase applies the same
profile when the contract is known; a well-formed but unknown contract ID
produces a non-blocking warning so forward compatibility is preserved.

The `contracts` field becomes mandatory in the next adapter/testcase format.

Readers continue to accept v0.1 testcases by deriving requirements at load time.
Adapter v0.1 has capability status `unchecked` and retains its historical
execution behavior. Writers emit only v0.2.
