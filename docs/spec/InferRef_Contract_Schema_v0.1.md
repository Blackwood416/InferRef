# InferRef Contract Schema v0.1 Specification

> **Project:** InferRef
> **Document:** Executable Contract Registry and Schema
> **Contract Schema Version:** 0.1
> **Status:** Draft / 0.6.0 scope
> **Audience:** InferRef core maintainers, contract pack authors, worker agents

---

# 1. Purpose

This specification defines:

1. the `inferref.contracts` entry-point protocol for third-party executable
   contracts;
2. the `inferref-contract` v0.1 declarative schema;
3. the relation expression language used for shape/dtype invariants;
4. the Python escape hatch for validators the schema cannot express;
5. the registry, CLI, and integration points that replace the hardcoded
   `inferref/testcase/contracts.py` table.

The four built-in contracts (RMSNorm, RoPE, KV append, KV indexed update) are
migrated onto this registry with identical behavior.

---

# 2. Design Principles

1. A contract is the engine ABI for one semantic operation. It fixes input
   role names, the exact observable output set, and shape/dtype invariants.
2. The schema covers the common case. Python is the escape hatch, not the
   default.
3. Third-party model knowledge lives in installed packages, never in core.
4. Discovery is safe by default: malformed or duplicate plugins are reported,
   never silently loaded, and never break operations that do not need them.
5. The torch-free core boundary is unchanged. Contract loading, validation,
   and preflight never import torch.
6. Existing wire formats (testcase 0.2, adapter 0.2) do not change.

---

# 3. Entry-Point Discovery

## 3.1 Group

Entry-point group: `inferref.contracts`.

```toml
[project.entry-points."inferref.contracts"]
qwen = "inferref_qwen.contracts:build"
```

## 3.2 Factory protocol

The entry-point value is a factory. Calling it returns an iterable of contract
descriptors. Each descriptor is either:

- a dict matching the `inferref-contract` schema (section 5), or
- an `ExecutableContract` instance returned by the plugin's own Python code.

```python
def build() -> list[dict | ExecutableContract]:
    return [
        {
            "format": "inferref-contract",
            "format_version": "0.1",
            "id": "swiglu/fused/v1",
            "description": "SwiGLU fusion over the last dimension",
            "inputs": {
                "x": {"kind": "tensor"},
                "gate": {"kind": "tensor"},
            },
            "outputs": {
                "y": {"kind": "tensor"},
            },
            "relations": [
                "y.shape == x.shape",
                "y.dtype == x.dtype",
                "gate.shape == x.shape",
            ],
            "features": ["multiple_outputs"],
            "effects": ["pure"],
        },
    ]
```

A single-contract plugin returns an iterable containing one descriptor.

## 3.3 Loading rules

1. Built-in contracts are always registered first (section 11).
2. Plugin discovery is lazy and cached; it happens when the registry is first
   consulted for a non-built-in ID or when the CLI/doctor asks for the full
   list.
3. Entry-point names are unique per installed distribution. A name colliding
   with another entry point is an `error` entry, not loaded.
4. An entry-point name equal to a built-in pack name is rejected as a
   built-in shadow and is reported, not loaded.
5. A factory that raises, returns a non-iterable, or returns an invalid
   descriptor marks that plugin `error` with a stable message. Other plugins
   and built-ins still load.
6. Two discovered contracts with the same ID are both rejected with
   `contract_duplicate_id`. The ID must not resolve ambiguously.
7. A contract whose `id` is equal to a built-in ID is rejected as a built-in
   shadow.

## 3.4 Verification

`inferref doctor --verify-plugins` loads every discovered contract entry point
and smoke-tests it:

- factory call succeeds;
- every descriptor validates against the schema or `ExecutableContract`
  requirements;
- contract IDs are unique and versioned;
- relation expressions parse.

The doctor report gains a `contracts` section with per-plugin status:
`discovered`, `loaded`, or `error` plus a stable message.

---

# 4. Contract ID

Contract IDs are versioned paths:

```text
rmsnorm/last-dim/v1
rope/rotate-half/v1
kv-cache/append/v1
kv-cache/indexed-update/v1
swiglu/fused/v1
```

Rules:

- at least two `/`-separated segments;
- the final segment is `v` followed by one or more digits;
- every other segment matches `^[a-z0-9][a-z0-9-]*$`;
- IDs are case-sensitive, lowercase by convention;
- an ID is immutable once published; behavior changes require a new version.

---

# 5. Schema Format

The canonical on-disk format is JSON with `.contract.json` extension. A YAML
authoring front end MAY be provided later as an optional extra; the runtime
never depends on PyYAML.

```json
{
  "format": "inferref-contract",
  "format_version": "0.1",
  "id": "swiglu/fused/v1",
  "description": "SwiGLU fusion over the last dimension",
  "inputs": {
    "x": {"kind": "tensor"},
    "gate": {"kind": "tensor"}
  },
  "outputs": {
    "y": {"kind": "tensor"}
  },
  "relations": [
    "y.shape == x.shape",
    "y.dtype == x.dtype",
    "gate.shape == x.shape"
  ],
  "features": ["multiple_outputs"],
  "effects": ["pure"]
}
```

## 5.1 Top-level fields

| Field | Required | Type | Rules |
| --- | --- | --- | --- |
| `format` | yes | string | exactly `inferref-contract` |
| `format_version` | yes | string | exactly `0.1` |
| `id` | yes | string | section 4 rules |
| `description` | no | string | non-empty |
| `inputs` | yes | object | non-empty role map, section 5.2 |
| `outputs` | yes | object | non-empty role map, section 5.2 |
| `relations` | no | string array | section 9 expressions; may be empty |
| `features` | no | string array | section 7 vocabulary |
| `effects` | no | string array | section 7 vocabulary |

Unknown top-level fields are ignored by readers (forward compatibility).

## 5.2 Role records

Each role record is an object:

```json
{
  "kind": "tensor"
}
```

v0.1 supports exactly `"kind": "tensor"`. Quantized and scalar kinds arrive
with the compound-tensor milestone and use a schema minor version.

Role name rules:

- matches `^[A-Za-z_][A-Za-z0-9_]*$`;
- unique within `inputs` and unique within `outputs`;
- an input role and an output role MUST NOT share a name in v0.1. The relation
  language cannot disambiguate the boundary, so sharing is rejected at load
  time with `contract_schema_invalid` on both the schema and Python paths.
  Contracts that genuinely need one logical tensor on both sides must use
  distinct role names (for example `cache` / `cache_out`).

## 5.3 Boundary semantics

Input and output semantics match adapter 0.2 exactly:

- every contract input role is required: a testcase missing a bound input
  role is invalid;
- a testcase MAY carry additional non-contract input tensors that an engine is
  allowed to ignore;
- contract outputs are an exact observable set: a testcase with a missing or
  unexpected output role is invalid.

---

# 6. Python Escape Hatch

## 6.1 `ExecutableContract`

The existing frozen dataclass remains the runtime contract object:

```python
@dataclass(frozen=True)
class ExecutableContract:
    id: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    validate_inputs: InputValidator
    validate_outputs: OutputValidator | None = None
    validate_relation: BoundaryValidator | None = None
    description: str = ""
    features: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
```

Two optional fields are added with defaults so existing constructions do not
change: `features` and `effects`. They use the vocabularies in section 7.

## 6.2 Validator contracts

```python
def validate_inputs(inputs: Mapping[str, dict]) -> list[str]: ...
def validate_outputs(outputs: Mapping[str, dict]) -> list[str]: ...
def validate_relation(
    inputs: Mapping[str, dict], outputs: Mapping[str, dict]
) -> list[str]: ...
```

Each returns a list of human-readable issue strings. An empty list means no
issues. Validators must be pure, deterministic, and torch-free.

## 6.3 Mixing schema and Python

A plugin may load a schema dict and then attach Python validators by returning
an `ExecutableContract` directly. A schema-only descriptor always receives the
default generated validators (relation evaluator plus role/dtype checks from
section 5.3). Plugins may not add Python callables to a plain schema dict;
the dict form is declarative only.

---

# 7. Features and Effects

## 7.1 Feature vocabulary

The `features` field uses the existing adapter feature names:

```text
multiple_outputs
strided_inputs
alias_effects
mutation_effects
```

`features` declares additive requirements merged into the contract-derived
requirements during preflight. It is an upper bound the author asserts; it is
not derived from the schema.

## 7.2 Effect vocabulary

The `effects` field is a convenience declaration and maps onto the feature
vocabulary:

| `effects` entry | Equivalent `features` entry |
| --- | --- |
| `pure` | none |
| `alias_effects` | `alias_effects` |
| `mutation_effects` | `mutation_effects` |

Rules:

- `effects` may be absent or empty;
- `pure` must not coexist with `alias_effects` or `mutation_effects`;
- `features` and `effects` are merged into one effective set during registry
  loading; duplicates are removed.

The `pure` rule is checked on the **merged effective set**, not just inside
`effects`: a descriptor declaring `features: ["mutation_effects"]` together
with `effects: ["pure"]` is also rejected.

The four built-in contracts declare no `features` or `effects` after
migration, preserving their current derived behavior exactly.

---

# 8. Derived Requirements

`contract_requirements(manifest, contract_id)` keeps its current semantics:

- for a known contract, derive dtype/rank/feature requirements from the tensors
  bound to contract roles and merge contract-declared features;
- for an unknown contract, fall back to testcase-wide derived requirements;
- role names absent from the manifest fall back conservatively to
  testcase-wide requirements.

This is the function consumed by adapter preflight, so plugin contracts get
the same `unsupported` behavior before any engine process starts.

The missing-role fallback is intentionally stronger than the pre-registry
table: if **any** declared contract role is absent from the manifest, the
whole testcase-wide requirement set is used (the old table derived from the
roles that happened to be bound and only fell back when none were found). This
is a deliberate conservative change so preflight cannot miss requirements
hidden in unbound roles.

---

# 9. Relation Expression Language

## 9.1 Grammar

```text
expr        := or_expr
or_expr     := and_expr ("or" and_expr)*
and_expr    := not_expr ("and" not_expr)*
not_expr    := "not" not_expr | "(" expr ")" | comparison
comparison  := operand ("==" | "!=") operand
operand     := path | length | integer | string
path        := NAME ("." attr)? ("[" index "]")?
attr        := "shape" | "dtype" | "rank" | "numel"
length      := "len" "(" NAME "." "shape" ")"
index       := integer
integer     := "-"? DIGIT+
string      := '"' [^"]* '"'
```

Examples:

```text
y.shape == x.shape
y.dtype == x.dtype
gate.shape == x.shape
y.shape[0] == x.shape[0]
y.rank == 2
y.numel == x.numel
len(y.shape) == len(x.shape)
not (y.dtype == "float32")
```

## 9.2 Evaluation semantics

Evaluation is a restricted AST walk, never `eval` or `exec`:

| Expression | Value |
| --- | --- |
| `x.shape` | `list[int]` |
| `x.shape[i]` | dimension at index `i`; negative indexes allowed |
| `x.rank` | `len(x.shape)` |
| `x.numel` | product of shape dimensions |
| `x.dtype` | string |
| `len(x.shape)` | rank |
| integers | Python `int` |

The `==` and `!=` comparisons compare the resolved values. `and`, `or`, and
`not` short-circuit with boolean semantics.

Not supported in v0.1: arithmetic, `>`/`<`, string functions, role sets,
conditionals. Complex invariants use the Python escape hatch.

## 9.3 Error behavior

- A relation that does not parse is a schema error at load time:
  `contract_relation_syntax` with the expression and parse reason.
- A relation that references an undeclared role is a schema error.
- A relation that references a declared role of non-tensor kind is a schema
  error.
- A relation that indexes a bare role (`x[0]`) or applies more than one index
  (`x.shape[0][1]`) is a `contract_relation_syntax` error at load time; only
  `NAME.shape[i]` may be indexed, with a single integer.
- At validation time, a false relation produces:

```text
relation 'y.shape == x.shape' failed (y.shape=[2,3,5], x.shape=[2,3,6])
```

The message is deterministic and includes both operands.

## 9.4 Generated validators

A schema-only contract generates:

- `validate_inputs`: role/dtype presence checks from section 5.3 plus every
  relation that only reads input roles;
- `validate_outputs`: output role checks from section 5.3;
- `validate_relation`: the remaining relations.

The boundary validation order stays the same as today: dtype presence, input
issues, output issues, then relation issues.

---

# 10. Registry API

New package: `inferref/contracts/`.

```text
inferref/contracts/
  __init__.py        public API
  schema.py          inferref-contract schema parsing and validation
  relations.py       relation parser and evaluator
  registry.py        built-in + entry-point discovery, lookup
  builtin.py         the four migrated built-in contracts
```

Public functions:

```python
get_contract(contract_id: str) -> ExecutableContract | None
contract_list() -> list[ContractEntry]            # deterministic order
load_contract_file(path: str | Path) -> ExecutableContract
verify_contracts() -> list[ContractPluginStatus]
contract_input_issues(contract_id, inputs) -> list[str]
contract_boundary_issues(contract_id, inputs, outputs) -> list[str]
contract_requirements(manifest, contract_id) -> dict
```

`inferref/testcase/contracts.py` becomes a compatibility shim re-exporting:

```python
ExecutableContract
EXECUTABLE_CONTRACTS
REGISTRY
get_contract
contract_input_issues
contract_boundary_issues
contract_requirements
```

The shim is deprecated but must keep working so existing tests and third-party
callers do not break.

---

# 11. Registry Semantics

1. Built-ins are registered in `inferref/contracts/builtin.py` and always
   present, in sorted ID order.
2. Plugin entry points are discovered through `importlib.metadata` using the
   same compatibility branch as `inferref.semantic.registry` (Python 3.10 and
   3.11+).
3. Deterministic list order: built-ins first, then plugins sorted by
   distribution name, entry-point name, and contract ID.
4. The registry cache is process-local. `verify_contracts()` always performs a
   fresh load.
5. Unknown contract IDs keep their current behavior: warnings in standalone
   validation, strict errors in extraction, conservative testcase-wide
   preflight requirements.

---

# 12. CLI

## 12.1 `inferref contract list`

```text
Contract                               Source          Status
------------------------------------   -------------   --------
kv-cache/append/v1                     builtin         loaded
kv-cache/indexed-update/v1             builtin         loaded
rmsnorm/last-dim/v1                    builtin         loaded
rope/rotate-half/v1                    builtin         loaded
gated-deltanet/step/v1                 inferref-qwen   loaded
qwen-2.5/rope/v2                       inferref-qwen   error
```

JSON:

```json
{
  "format": "inferref-contract-list",
  "contracts": [
    {
      "id": "rope/rotate-half/v1",
      "source": "builtin",
      "distribution": null,
      "entry_point": null,
      "status": "loaded"
    }
  ],
  "errors": [
    {
      "entry_point": "qwen",
      "distribution": "inferref-qwen",
      "message": "duplicate contract id 'x/y/v1'"
    }
  ]
}
```

Exit code is 0 even when plugins have errors; errors are data. This mirrors the
semantic detector discovery philosophy.

## 12.2 `inferref contract show ID`

Renders the resolved `ExecutableContract` as schema/`to_dict()` JSON. Exit 1
with `contract_unknown` if not found.

## 12.3 `inferref contract validate PATH`

Loads one `.contract.json` file without entry points:

```json
{
  "format": "inferref-contract-validation",
  "status": "pass",
  "contract": {
    "id": "swiglu/fused/v1",
    "inputs": ["x", "gate"],
    "outputs": ["y"],
    "relations": 3
  }
}
```

Exit 0 on pass, 1 on fail. Failures carry `contract_schema_invalid` issues.

`validate` accepts either a single descriptor object or a JSON array of
descriptors; duplicate IDs inside one file are rejected with
`contract_duplicate_id`. The public `load_contract_file(path)` API (section 10)
loads a single object only and raises `contract_schema_invalid` for arrays;
the CLI shares its validation logic with the public
`validate_contract_file(path)` helper so the two cannot drift.

All three commands support `--json`.

---

# 13. Integration Points

| Call site | Current file | Change |
| --- | --- | --- |
| Testcase extraction | `inferref/testcase/extract.py` | import `get_contract` from `inferref.contracts`; behavior unchanged |
| Standalone validation | `inferref/testcase/validate.py` | same lookup through new registry |
| Requirements/preflight | `inferref/testcase/requirements.py`, `inferref/agent/adapter.py` | unchanged behavior; merge contract-declared features in `contract_requirements` |
| Suite validation | `inferref/suite/schema.py` | unchanged for testcases |
| Doctor | `inferref/doctor.py` | add `contracts` verification section to `--verify-plugins` |
| CLI | `inferref/cli/main.py` | add `contract` subcommand |
| Agent capabilities | `inferref/agent/service.py` | unchanged; contract IDs are opaque strings |

No testcase, adapter, or trace wire format changes.

---

# 14. Error Codes

New stable codes:

| Code | Meaning |
| --- | --- |
| `contract_schema_invalid` | schema file or descriptor is malformed |
| `contract_duplicate_id` | same ID from two sources |
| `contract_shadows_builtin` | plugin ID/name collides with built-in |
| `contract_entry_point_error` | factory raised or returned invalid iterable |
| `contract_unknown` | requested ID not in registry |
| `contract_relation_syntax` | relation expression does not parse |
| `contract_relation_role` | relation references unknown role/kind |
| `contract_relation_failed` | relation evaluated false at validation time; standalone testcase validation emits this code for schema-derived relations (built-in Python validators keep `contract_shape_invalid`) |

Existing codes (`contract_dtype`, `contract_max_rank`, `contract_feature`,
`contract_output_missing`, `contract_unexpected_output`) keep their meaning.

---

# 15. Compatibility

- Testcase 0.2 and adapter 0.2 remain byte-compatible.
- `inferref/testcase/contracts.py` remains importable.
- Built-in behavior is unchanged; existing tests must pass untouched.
- A well-formed but unknown contract ID in an existing testcase still
  validates with a warning.
- The torch-free core boundary is unchanged: `inferref/contracts/` depends on
  stdlib only.

---

# 16. Acceptance Criteria

1. A fixture distribution `inferref-qwen` with two valid contracts and one
   intentionally broken entry point installs; `inferref contract list --json`
   shows valid entries `loaded` and the broken one `error`.
2. `inferref contract validate` accepts the SwiGLU example and rejects a
   malformed schema, a bad relation expression, and a duplicate ID.
3. `testcase extract --contract swiglu/fused/v1` binds roles and refuses a
   boundary that violates `y.shape == x.shape`.
4. An adapter declaring `contract_capabilities["swiglu/fused/v1"]` preflights
   correctly and returns `unsupported` with `contract_dtype`/
   `contract_feature` reasons before process creation when requirements are
   not met.
5. The four built-in contracts produce identical validation and preflight
   results before and after migration.
6. `inferref doctor --verify-plugins` reports contract plugin status without
   importing torch.
7. Unknown contract IDs behave exactly as today.

---

# 17. Implementation Tasks

## Task C1: Registry package and built-in migration

Files:

- add `inferref/contracts/__init__.py`
- add `inferref/contracts/schema.py`
- add `inferref/contracts/relations.py`
- add `inferref/contracts/registry.py`
- add `inferref/contracts/builtin.py`
- replace `inferref/testcase/contracts.py` with a compatibility shim
- update imports in `inferref/testcase/extract.py`, `validate.py`,
  `requirements.py`, and any other direct importers

Acceptance: existing `tests/core/test_contracts.py`,
`tests/core/test_adapter_capabilities.py`, and extraction tests pass without
modification.

## Task C2: CLI and doctor

Files:

- `inferref/cli/main.py`: add `contract` subcommand
- `inferref/doctor.py`: extend `--verify-plugins`
- `tests/core/test_contract_registry.py` for CLI/doctor

## Task C3: Declarative features in preflight

Files:

- `inferref/testcase/requirements.py` or the registry's
  `contract_requirements`: merge contract-declared features
- tests asserting merged preflight behavior

## Task C4: Fixture pack and acceptance tests

Files:

- `tests/fixtures/contract_pack/` or a test-local synthetic distribution
- tests for entry-point discovery, duplicate/shadow rejection, factory
  errors, relation evaluator, and the SwiGLU extraction/preflight flow

Sequencing: C1 -> C2 -> C3 -> C4. C3 is independent of C2 and may run in
parallel once C1 lands.

---

# 18. Test Plan

All new tests live in `tests/core/` and run without torch, with the existing
import-blocked core verification.

| Area | Cases |
| --- | --- |
| Schema | required fields, role rules, feature/effect vocabulary, unknown-field tolerance |
| Relations | every operator, negative index, short-circuit, syntax errors, role errors |
| Discovery | fake distribution entry points, duplicate names, shadowing, factory failures |
| CLI | list/show/validate JSON and exit codes |
| Doctor | contracts section, no torch import |
| Preflight | declared-feature merge, plugin contract unsupported paths |
| Migration | built-in parity on all existing corpus testcases |
