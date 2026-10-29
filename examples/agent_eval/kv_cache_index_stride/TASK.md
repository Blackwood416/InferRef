# Agent task: repair the indexed KV-cache write

The host-side visible testcase performs one decode-step KV-cache update. The
cache has shape `[batch, heads, capacity, head_dim]`, `index` names the first
sequence position to overwrite, and `update` contains one new position. The NumPy
engine in `engine.py` completes successfully but does not numerically match the
reference.

Use InferRef's structured Agent tools to locate and repair the mismatch.

Constraints:

- You may edit only `engine.py`.
- Do not edit the task or create additional files.
- Do not copy reference output tensors into engine output.
- Keep the implementation general for every batch, head count, cache capacity,
  and valid index.
- Finish only when `inferref_run_engine` returns `status: pass`.
- Use at most four engine runs.
