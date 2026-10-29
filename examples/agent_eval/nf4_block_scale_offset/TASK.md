# Agent task: repair NF4 block-scale dequantization

The host-side visible testcase contains NF4 codes, block scales, a codebook, and
the block size. The NumPy engine in `engine.py` completes successfully but does
not numerically match the reference.

Use InferRef's structured Agent tools to locate and repair the mismatch.

Constraints:

- You may edit only `engine.py`.
- Do not edit the task or create additional files.
- Do not copy reference output tensors into engine output.
- Keep the implementation general for every positive row and column count.
- Finish only when `inferref_run_engine` returns `status: pass`.
- Use at most four engine runs.
