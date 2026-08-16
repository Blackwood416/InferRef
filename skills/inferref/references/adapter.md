# Engine adapter v0.2

An adapter is executable JSON configuration that maps one trusted engine binary
to the InferRef testcase ABI. InferRef passes the command directly to the OS
with `shell=False`; shell syntax is never executed.

## JSON shape

```json
{
  "format": "inferref-engine-adapter",
  "format_version": "0.2",
  "name": "rope-numpy",
  "target_device": "cpu",
  "capabilities": {
    "device_types": ["cpu"],
    "dtypes": ["float32", "float16", "bfloat16"],
    "max_rank": 8,
    "features": ["multiple_outputs", "strided_inputs"]
  },
  "command": [
    "{python}",
    "rope_numpy.py",
    "{testcase}",
    "--output",
    "{output}"
  ],
  "cwd": ".",
  "timeout_seconds": 60,
  "max_output_chars": 65536
}
```

Required: `format`, `format_version`, `name`, `target_device`, `capabilities`,
and a non-empty string-array `command`. Optional: `cwd`, `environment`,
`timeout_seconds`, `max_output_chars`, `max_artifact_bytes`,
`max_artifact_files`.

A working copy-paste example lives at
`examples/engine_sim/rope_numpy.adapter.json` (engine:
`examples/engine_sim/rope_numpy.py`). It reads `.irtensor` inputs, computes the
reference operation, and writes `.irtensor` outputs.

## Placeholders

| Placeholder | Value |
| --- | --- |
| `{testcase}` | Absolute standalone testcase directory. |
| `{output}` | Absolute, fresh per-run engine output directory. |
| `{adapter_dir}` | Absolute directory containing the adapter JSON. |
| `{python}` | Absolute interpreter running InferRef. |

The command/environment must reference `{testcase}` and `{output}`. InferRef also
sets `INFERREF_TESTCASE` and `INFERREF_OUTPUT` in the child environment. Escape
literal braces as `{{` and `}}`.

## Capabilities

`capabilities` declares what the engine supports before a process is started:

- `device_types` - e.g. `cpu`, `cuda`, `xpu`.
- `dtypes` - stable InferRef dtype names the engine supports.
- `max_rank` - maximum tensor rank.
- `features` - optional subset of `multiple_outputs`, `strided_inputs`,
  `alias_effects`, `mutation_effects`.
- `contracts` - optional list of versioned contract IDs the engine supports.
- `contract_capabilities` - per-contract refinements. Each refinement's
  `dtypes`, `max_rank`, and `features` must be a subset of the global
  capabilities, and every refined contract must appear in `contracts`.

A mismatch between a testcase's derived requirements and these declarations
produces `unsupported` before process creation.

## Engine output contract

Read `testcase.json` from `{testcase}` and decode boundary tensors from the
payload paths (`.irtensor` header plus canonical contiguous payload). Write one
`.irtensor` per declared output using the output role name:
`<name>.irtensor`, `outputs/<name>.irtensor`, or `tensor_<value_id>.irtensor`.
An optional `manifest.json` with an `outputs` array of `{name, payload}` entries
is also accepted.

## Trust boundary

An adapter names a process to execute and is therefore trusted code. The runner
is timeout/output/artifact controlled, not a general-purpose sandbox. Never put
secrets in command arguments: the expanded argv is recorded. Pass secrets
through environment variables instead; environment keys are recorded but values
are redacted. Only expose adapters approved by the user or engine workspace.
