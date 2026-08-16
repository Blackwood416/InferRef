# Agent protocol envelopes

Every operation returns one self-describing envelope:

```json
{
  "protocol": {"format": "inferref-agent-response", "version": "0.1"},
  "operation": "run_engine",
  "status": "pass",
  "data": {},
  "diagnostics": [],
  "next_actions": []
}
```

## Status semantics

| Status | Meaning | Agent action |
| --- | --- | --- |
| `ok` | discovery/inspection succeeded | read `data` and `next_actions` |
| `pass` | validation/extraction/execution succeeded | stop or move to the next case |
| `fail` | domain found a mismatch or non-reproducible artifact | fix the first divergence and rerun |
| `error` | adapter/process/protocol failure | inspect `data.execution` (stderr, command, cwd) and the diagnostics code |

`fail` is a valid domain result, not a protocol error. `error` always carries at
least one structured diagnostic.

## `diagnostics` and `next_actions`

- `diagnostics` - array of `{severity, code, message}` records explaining a
  failure. Match on the stable `code`, not the message text.
- `next_actions` - array of `{operation, reason, ...}` records recommending the
  next protocol call. Follow the first entry unless the local context says
  otherwise.

## `run_engine` examples

Pass:

```json
{"operation": "run_engine", "status": "pass", "data": {"status": "pass", "comparison": {"status": "pass"}}, "next_actions": []}
```

Fail (numerical mismatch):

```json
{"operation": "run_engine", "status": "fail", "data": {"status": "mismatch", "comparison": {"first_failure": {"name": "q_embed"}}}, "next_actions": [{"operation": "modify_engine", "reason": "Fix the first comparison divergence and rerun this adapter."}]}
```

Error (process failure):

```json
{"operation": "run_engine", "status": "error", "diagnostics": [{"severity": "error", "code": "execution_error", "message": "Engine adapter did not complete successfully."}]}
```

## Scenario reports

`run_scenario` puts an `inferref-scenario-run` v0.1 report in `data`. Scenario
statuses are `pass`, `fail`, `error`, `partial`, and `unsupported`; the envelope
maps `pass` to `pass`, `fail` to `fail`, and everything else to `error`. Each
step embeds the complete adapter run record, `state_status` (`ok`,
`not_applicable`, `not_compared`, `state_missing`, `state_shape_mismatch`,
`state_dtype_mismatch`, `state_mismatch`), and input/output binding maps. Read
`data.steps` to find the first non-pass step.
