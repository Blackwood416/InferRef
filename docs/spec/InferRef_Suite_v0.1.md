# InferRef Suite and Run Matrix v0.1

## Suite manifest

```json
{
  "format": "inferref-suite",
  "format_version": "0.1",
  "name": "xpu-corpus",
  "cases": [
    {"id": "rope-fp16", "testcase": "cases/rope", "tags": ["rope", "decode"]}
  ]
}
```

Case IDs match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`, cannot use Windows
reserved names or trailing dots/spaces, and must remain unique after portable
case folding. Testcase paths are relative, must remain below the Suite
directory after canonical resolution, and are validated with the standalone
testcase validator before execution.

Logical IDs are never used directly as filesystem paths. Adapter and case run
directories use slug plus SHA-256 artifact keys, followed by a final containment
check below `runs_dir`.

## Execution

`inferref suite run` accepts one or more repeated `--adapter` arguments. Every
case is run against every adapter in deterministic case-major order. Unless
`--allow-unsupported` is set, mismatch, execution error, and unsupported all
fail the run. The option is intended only for capability inventory and changes
the CLI acceptance/exit policy, not the numerical status.

Run status is `pass` only when every cell executes and passes, `partial` when
at least one cell passes and the remainder are allowed unsupported cells,
`unsupported` when no cell executes and every cell is unsupported, and `fail`
when any real failure occurs. `accepted` and `exit_code_policy_satisfied` record
the separate CLI policy decision.

Expected per-cell configuration and validation exceptions are isolated as
`infrastructure_error`, allowing other engines/cases to finish. `--fail-fast`
restores exception-first behavior for development.

The writer emits `inferref-suite-run` v0.3. It includes stable adapter IDs,
per-case `results`, complete engine run records, aggregate cell counts, and
overall status. A single-adapter run retains the v0.1 convenience `adapter` and
per-case `run` fields.

`inferref suite report RUN --output report.html` writes a self-contained HTML
case-by-engine matrix and a sibling JSON `inferref-suite-report` v0.2. Cells
include status, maximum absolute error, first divergence, duration, and
unsupported reasons when available.

### Count semantics and migration from run v0.2

Counts are per case-by-adapter cell (a whole scenario counts as one cell):

- `total`: all attempted cells.
- `pass`: cells whose execution entirely passed.
- `unsupported`: cells with only unsupported outcomes.
- `partial`: scenario cells containing only passing and unsupported steps.
- `failed`: cells with numerical mismatches, execution/configuration errors, or
  other actual failures. Unsupported-only policy rejection is excluded.
- `rejected`: cells not accepted by the current exit-code policy (the old
  v0.2 meaning of `failed`).

`pass + unsupported + partial + failed == total`; `rejected` is a separate
policy count, not an additional outcome bucket. Scenario counts inspect step
outcomes even when their strict-policy aggregate status is `fail`. A scenario
containing both unsupported steps and a mismatch/error still counts as failed.

For 13 unsupported cells, strict mode produces `unsupported: 13, failed: 0,
rejected: 13` and a nonzero exit code. With `--allow-unsupported`, only `rejected`
and the acceptance/status fields change. Gate CI on `accepted`,
`exit_code_policy_satisfied`, or `counts.rejected`, not solely `counts.failed`.

The renderer accepts run versions 0.1, 0.2, and 0.3. Its JSON sidecar preserves
the input counts and records `source_run_format_version`, so old policy counts
are not silently reinterpreted. Suite **manifest** versions are unchanged.

## Validation

`inferref suite validate` separates structural validity from runnability:

```json
{
  "schema_valid": true,
  "runnable": false,
  "non_runnable_cases": ["case-x"]
}
```

`schema_valid` requires every referenced testcase to pass standalone structural
validation. `runnable` additionally requires every testcase to be reproducible.
By default the CLI fails a schema-valid suite that is not runnable;
`--allow-nonreproducible` keeps the CLI successful for corpus inventory while
still reporting `runnable: false` and the offending case IDs.

The report HTML output must end in `.html`; the sibling JSON sidecar is derived
by replacing the extension, so `report.json` as an output name is rejected
instead of silently overwriting the sidecar.
