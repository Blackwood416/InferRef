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
global fields). Per-contract values must be subsets of the global declaration:
`contract_capabilities[C].dtypes ⊆ capabilities.dtypes`,
`contract_capabilities[C].features ⊆ capabilities.features`, and
`contract_capabilities[C].max_rank ≤ capabilities.max_rank`; a contradictory
adapter file is rejected at load time. Preflight derives per-contract
requirements from the tensors bound to the contract roles, so a float16 RoPE
case is rejected by an adapter that only declares float32 RoPE even when its
global `dtypes` list is wider. Mismatches are reported as `contract_dtype`,
`contract_max_rank`, or `contract_feature` issues before any engine process
starts.

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
  numel equals that dimension; epsilon is exactly one scalar value.
- `rope/rotate-half/v1`: positive even rotary dimension, split-half rotation,
  and matching `[sequence, rotary_dim]` cosine/sine inputs.
- `kv-cache/append/v1`: append update tensors on the penultimate sequence axis;
  all other dimensions match.
- `kv-cache/indexed-update/v1`: replace one or more positions on the penultimate
  sequence axis using one non-negative scalar index. For both KV contracts,
  `cache`, `update`, and `cache_out` dtypes must match.

Contracts come from an explicit extraction profile (`testcase extract
--contract ...`) or an engine mapping, never from a Suite tag or testcase name.

### Contract registry and role binding

Each known contract is a registered profile with fixed input/output role names
and shape validators, not just an ID label:

| contract | inputs | outputs |
| --- | --- | --- |
| `rmsnorm/last-dim/v1` | `x`, `weight`, `epsilon` | `y` |
| `rope/rotate-half/v1` | `query`, `key`, `cos`, `sin` | `q_embed`, `k_embed` |
| `kv-cache/append/v1` | `cache`, `update` | `cache_out` |
| `kv-cache/indexed-update/v1` | `cache`, `update`, `index` | `cache_out` |

Contract outputs are an **exact role set**. Contract inputs are **required
named roles**: a testcase must provide every input role, but may carry
additional non-contract input tensors that an engine is allowed to ignore. This
is deliberate — a region may expose auxiliary context beyond the contract's
operand set — so third-party adapter authors should treat inputs as a
capability projection and outputs as the full observable ABI.

Extraction refuses to publish a testcase when a requested contract is not in
the local registry, when the extracted boundary cannot be bound to the
contract's role names, or when the bound tensors fail the contract shape
validator. Standalone validation of an existing testcase applies the same
profile when the contract is known; a well-formed but unknown contract ID
produces a non-blocking warning so forward compatibility is preserved.

### Observable outputs are exact and relationally validated

For a known contract, standalone validation and extraction both enforce an
**exact observable output set**: the testcase must provide precisely the
contract's output roles. A RoPE testcase that keeps `q_embed` but drops
`k_embed` is invalid even when every input is legal, so a PASS cannot hide a
broken half of the contract.

Each profile also validates output shapes and input-output relations:

- `rmsnorm/last-dim/v1`: `y.shape == x.shape` and matching dtypes.
- `rope/rotate-half/v1`: `q_embed.shape == query.shape`,
  `k_embed.shape == key.shape`, and matching dtypes per branch.
- `kv-cache/append/v1`: `cache_out` shares rank, non-sequence dimensions, and
  width with `cache`; its sequence length equals
  `cache.sequence + update.sequence`.
- `kv-cache/indexed-update/v1`: `cache_out.shape == cache.shape`.

### Exactly one executable contract per testcase

v0.2 restricts a testcase to **exactly one** executable contract. The `contracts`
array must contain a single unique ID when present, extraction rejects repeated
`--contract` values, and the native engine refuses multiple supported
contracts. Composite operations (for example a fused RMSNorm→RoPE region)
should be expressed as a dedicated composite contract rather than an ambiguous
multi-contract array.

### Atomic extraction publish

Extraction builds the payloads, manifest, contract binding, and standalone
validation inside a hidden staging directory and only then promotes it into
place. A failed extraction never partially overwrites an existing testcase:
the output directory must not exist unless `--force` is passed, and `--force`
replaces it via a backup-and-swap so a failure leaves the previous testcase
intact. The extractor runs the same `validate_testcase` used by Suite and
engine execution before publishing.

The guarantee is rename-based atomic publication under normal process
execution. A hard crash between the two renames of a `--force` replacement (old
directory moved to a `.bak-*` sibling, staging moved into place) can leave a
`.bak-*`/`.tmp-*` pair without a final target; startup recovery for that window
is not provided by the extractor.

The `contracts` field becomes mandatory in the next adapter/testcase format.

Readers continue to accept v0.1 testcases by deriving requirements at load time.
Adapter v0.1 has capability status `unchecked` and retains its historical
execution behavior. Writers emit only v0.2.
