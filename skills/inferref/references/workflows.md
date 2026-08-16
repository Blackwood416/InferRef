# Workflows

## 1. Validate an existing adapter against one testcase

```powershell
inferref agent context cases/rope --json
inferref agent run cases/rope --adapter examples/engine_sim/rope_numpy.adapter.json --runs-dir runs --json
```

Interpret the envelope:

- `pass` - the adapter agrees with the reference; move on.
- `fail` - read `data.comparison.first_failure`, fix the engine, rerun.
- `error` - read `data.execution` (stderr, exit code, command, cwd) and
  `data.adapter`.

To compare an output directory that already exists, skip the run:

```powershell
inferref agent compare cases/rope engine-out/ --json
```

## 2. Extract a failing region from a trace and isolate it

Record a trace with semantic analysis:

```powershell
inferref trace examples/mini_llama/run_trace.py -o trace/ --semantic-analysis --json
```

Find the region that contains the failing kernel:

```powershell
inferref analyze trace/ --json
inferref inspect trace/ --json
inferref region detect trace/ --dry-run --json
inferref region detect trace/ --json
```

Project the region into a standalone testcase (replace `<region>` with a
detected name, and pass `--contract` when the operation has one):

```powershell
inferref agent extract trace/ --region <region> -o cases/<region> --input-names query,key,cos,sin --output-names q_embed,k_embed --contract rope/rotate-half/v1 --json
```

Isolate the failure against the numpy reference engine:

```powershell
inferref agent run cases/<region> --adapter examples/engine_sim/rope_numpy.adapter.json --runs-dir runs --json
```

If the extracted testcase is not reproducible, re-trace with full capture
(`--capture-tensors full`) and re-extract.

## 3. Run a KV scenario: reference state, then engine state

Validate the committed fixture first:

```powershell
inferref scenario validate tests/fixtures/scenarios/kv-chain --json
```

Run with reference state (default) - each step receives reference state, so a
failure is localised to one kernel:

```powershell
inferref agent run_scenario tests/fixtures/scenarios/kv-chain --adapter <kv-adapter.json> --runs-dir runs --state-mode reference --json
```

Run with engine state and numeric state comparison - engine state feeds the next
step, and a corrupted cache write is reported at the producing step
(`state_status` becomes `state_mismatch` or `state_shape_mismatch` /
`state_dtype_mismatch` and the chain stops):

```powershell
inferref agent run_scenario tests/fixtures/scenarios/kv-chain --adapter <kv-adapter.json> --runs-dir runs --state-mode engine --compare-state --json
```

The adapter must declare capabilities covering every step's derived
requirements (the fixture steps are float32, rank <= 4, with
`multiple_outputs` on the prefill step). Read `data.steps[].state_status` and
`data.steps[].run` to localise the first divergence.

## Deeper docs

- `docs/spec/InferRef_Agent_Protocol_v0.1.md`
- `docs/spec/InferRef_Scenario_v0.1.md`
- `docs/spec/InferRef_0.6_Contract_Ecosystem.md`
- `docs/EXTENDING.md`
