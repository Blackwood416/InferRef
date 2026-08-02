# Status: Agent repair evaluation

**Date:** 2026-08-02

**Version:** `0.3.0`

**Baseline:** commit `15d41b5` (Agent and MCP integration)

## Outcome

InferRef now has a repeatable repair benchmark rather than only callable Agent
tools. The benchmark starts from a plausible RoPE half-rotation sign/order defect
in a NumPy engine and proves this bounded loop:

```text
MCP capability discovery
  -> testcase context
  -> engine run: FAIL
  -> structured first divergence
  -> one permitted source edit
  -> fresh engine run: PASS
  -> protected artifacts unchanged
```

The fixture is under `examples/agent_eval/rope_sign/`. Its machine-readable
manifest declares the task, workspace template, trusted setup command, required
MCP tools, the sole editable path (`engine.py`), a four-run budget, expected
baseline failure, and successful terminal response.

## Isolation model

`prepare.py --workspace ...` creates a disposable directory containing only:

- the candidate `engine.py`;
- the trusted adapter;
- the Agent task;
- a deterministic standalone testcase.

The reference generator remains in the repository outside the candidate workspace.
The evaluation snapshots the adapter, task, testcase manifest, and every tensor
payload before Agent execution and verifies their combined digest afterward.

This is integrity checking, not an OS sandbox. A production evaluation host must
still enforce workspace roots and process permissions.

## Diagnostic sufficiency

The deliberately broken engine exits successfully, so InferRef must distinguish
the result from an invocation problem. CLI and MCP both report `status: fail` with:

- first failing tensor `q_embed`;
- producer `aten.add.Tensor`;
- `RoPE@agent-eval` region and reference source function;
- reference shape `[1, 2, 4, 8]` and `float32` dtype;
- maximum absolute/relative error and cosine similarity;
- 60 mismatches among 64 elements;
- first mismatching logical index and reference/actual values;
- `modify_engine` as the next action.

After the bounded engine correction, the second MCP call reports `pass`. Its run
directory differs from the failed run, retaining the stale-output guarantee.

## What the automated test proves

The test applies a known minimal one-file repair after asserting the complete
diagnostic contract. This proves that the fixture is deterministic, solvable, and
that InferRef carries enough information through MCP for the repair loop.

It is not yet an independent blind-model benchmark: the automated test contains
the oracle edit. The next evidence level is to give the prepared workspace and
task to a separate Agent process that cannot read `prepare.py` or the test source,
then record its tool transcript, changed paths, iterations, and outcome.

## Verification commands

```text
repair evaluation                             3 passed
full local suite                            328 passed, 1 skipped
```

```powershell
python -m pytest tests/agent/test_repair_evaluation.py -q
python examples/agent_eval/rope_sign/prepare.py `
  --workspace .scratch/agent-eval-rope
inferref agent run .scratch/agent-eval-rope/testcase `
  --adapter .scratch/agent-eval-rope/adapter.json `
  --runs-dir .scratch/agent-eval-rope-runs --json
```

The final command is expected to return exit code 1 before repair. A baseline PASS
is a benchmark failure.

## Next steps

1. Run the fixture through a separate blind Agent and persist a sanitized transcript.
2. Add an evaluation host that enforces edit paths and iteration budgets rather
   than checking them only after completion.
3. Add a second benchmark for an execution/build failure, ensuring Agents do not
   edit numerical code when the adapter itself is broken.
4. Add a stateful KV-cache repair benchmark after the simpler RoPE loop is stable.
