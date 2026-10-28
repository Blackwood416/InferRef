# Agent task: repair NF4 block-scale dequantization

The host-side visible testcase contains NF4 codes, one scale per flat block, a
codebook, and the block size. The NumPy engine in `engine.py` completes
successfully but does not numerically match the reference.

The engine currently treats block scales as if they were laid out per row. The
logical tensor is flattened before NF4 blocks are assigned, so a block can span a
row boundary. Only the first row happens to use the correct scales.

Use InferRef's structured Agent tools to locate and repair the mismatch.

Constraints:

- You may edit only `engine.py`.
- Do not edit the task or create additional files.
- Do not copy reference output tensors into engine output.
- Keep the implementation general for every positive row and column count.
- Finish only when `inferref_run_engine` returns `status: pass`.
- Use at most four engine runs.
