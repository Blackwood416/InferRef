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

Case IDs are unique. Testcase paths are relative, must remain below the Suite
directory after canonical resolution, and are validated with the standalone
testcase validator before execution.

## Execution

`inferref suite run` accepts one or more repeated `--adapter` arguments. Every
case is run against every adapter in deterministic case-major order. Unless
`--allow-unsupported` is set, mismatch, execution error, and unsupported all
fail the run. The option is intended only for capability inventory.

The writer emits `inferref-suite-run` v0.2. It includes stable adapter IDs,
per-case `results`, complete engine run records, aggregate cell counts, and
overall status. A single-adapter run retains the v0.1 convenience `adapter` and
per-case `run` fields.

`inferref suite report RUN --output report.html` writes a self-contained HTML
case-by-engine matrix and a sibling JSON `inferref-suite-report` v0.1. Cells
include status, maximum absolute error, first divergence, duration, and
unsupported reasons when available.
