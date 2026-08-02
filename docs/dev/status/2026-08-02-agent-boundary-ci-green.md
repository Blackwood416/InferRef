# Status: Agent boundary hardening CI green

**Date:** 2026-08-02

**Version:** `0.3.0`

**Code commits:** `5a253f3`, `c72163c`, `72cffda`

## Outcome

The pre-Agent-test hardening slice is complete. GitHub Actions run
[`30745928776`](https://github.com/Blackwood416/InferRef/actions/runs/30745928776)
finished with all 18 jobs successful across Windows and Linux.

The first remote run exposed a real POSIX-only bug: resolving the virtualenv
`sys.executable` symlink selected the base interpreter, so adapter children lost
their installed numpy/InferRef packages. The runner now preserves the venv
executable path and cleans its POSIX process group after normal exit as well as
timeouts. A WSL regression subset passed before the fix was pushed.

The first Transformers-floor run also did useful work: it showed that 4.40's
StaticCache behavior is not the same contract as 5.14. The final matrix now
states and enforces two separate claims:

- Transformers 4.40: real Llama tracing and semantic detection;
- Transformers 5.14.1/latest: modern Llama cache and Qwen3.5 hybrid-cache
  behavior.

Capability assertions ensure required tests cannot become green through a
module-level skip. The fixed 4.40 local environment passed 347 tests with nine
explicitly scoped skips; the current 5.14 cache/hybrid subset passed all 13
tests.

## Gate status

- unified path containment and MCP host roots: green on Windows/Linux;
- testcase validator and header-only payload validation: green;
- stdout/stderr flood, artifact growth, timeout-child, and normal-exit child
  cleanup: green;
- Core without PyTorch: green on Python 3.10/3.12/3.13 and both OSes;
- Agent+MCP without PyTorch: green;
- C++ reader and Python/C++ comparison agreement: green on both OSes;
- PyTorch 2.1/current/nightly frontend matrix: green (nightly remains advisory);
- generic HF floor, fixed Qwen3.5 floor, and latest Transformers: green;
- manual self-hosted CUDA workflow: defined but not executed.

## Next decision

The repository is ready for a real blind Agent repair test within configured MCP
roots. Formal release tags and versioned 0.1/0.2/0.3 golden assets remain a
separate release task and should precede declaring the exchange format frozen.
