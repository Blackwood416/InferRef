# Status: Agent boundary hardening

**Date:** 2026-08-02

**Version:** `0.3.0`

**Parent baseline:** commit `776faf4` (repair protocol evaluation harness)

**Verified environment:** Windows, Python 3.13.11, CPU-only

## Outcome

The 0.3 Agent surface now has an explicit filesystem and execution boundary
before a blind Agent is allowed to exercise it:

- trace, testcase, engine-output, and MCP paths share containment primitives;
- relative payload paths cannot escape through `..`, absolute syntax, symlinks,
  or Windows junctions;
- MCP read/write roots are host startup policy and are not controlled by a tool
  request;
- comparison and adapter execution share a formal standalone testcase validator;
- `reproducible: true` is treated as a declaration, not proof;
- `.irtensor` metadata validation is header-only and does not materialise large
  payloads;
- adapter stdout/stderr are streamed to hard-limited files;
- timeout, stream overflow, and artifact growth terminate the process tree;
- the adapter is documented as timeout/output-controlled, not as a general
  bounded-execution sandbox.

## Testcase validation contract

`validate_testcase()` reports structure validity and independently computed
reproducibility. It checks format/version, unique IDs and boundary names,
payload containment/existence, codec header versus manifest metadata, node/value
references, alias references, mutation storage/version records, and output
producer references.

This intentionally preserves three states:

1. valid and reproducible: comparison and engine execution are allowed;
2. valid but non-reproducible, such as hash-only boundaries: context remains
   inspectable but execution is blocked;
3. structurally invalid: context returns a protocol error and all consumers
   reject the testcase.

## Adapter controls

`max_output_chars` is now enforced as a per-stream byte ceiling while the child
is running. `max_artifact_bytes` adds active run-directory monitoring. POSIX
starts a new session and terminates the process group; Windows starts a new
process group and uses `taskkill /T /F` for tree cleanup. Tests cover stdout
flooding, a timeout child that attempts to outlive its parent, and continuous
large-artifact writes. Windows uses a kill-on-close Job Object; POSIX uses a
dedicated session/process group. Descendants are also cleaned up when the direct
child exits normally.

The logs remain available as `inferref-stdout.log` and
`inferref-stderr.log`; `inferref-run.json` records observed bytes, limits, paths,
termination status, and comparison data.

## Compatibility CI

The HF matrix no longer conflates all Transformers support:

- `transformers==4.40.*` asserts and runs real Llama tracing/semantic coverage;
- `transformers==5.14.1` is the fixed Qwen3.5-tested floor;
- floating latest gives upstream early warning;
- capability assertions fail before pytest, so missing Qwen3.5 classes cannot
  turn into a green module-level skip.

A manual self-hosted GPU workflow now covers FP16/BF16 payload capture, CUDA
views and mutation, pinned CPU memory, CPU/GPU storage identity separation, and
a non-default CUDA device when two GPUs are available. This workflow has not
been executed on the local CPU-only machine.

The reported Actions Node.js warning is already obsolete in the current tree:
checkout and setup-python use v6. No compatibility downgrade was added.

## Verification

```text
full local suite                         360 passed, 5 skipped
adapter adversarial + validator subset    15 passed
codec/validator/agent/compare subset       89 passed
WSL Agent/path cross-platform subset       24 passed, 1 skipped
Torch 2.1 + Transformers 4.40 floor       347 passed, 9 skipped
Transformers 5.14 cache/hybrid subset      13 passed
changed-file Ruff F/I checks                 passed
workflow YAML parsing                        passed
```

Four skips are the new CUDA contract on a CPU-only host; the existing official
Qwen3.5 weight test remains opt-in. Windows symlink and junction containment
tests passed locally. Linux containment and all new CI matrix legs still require
the remote Actions run. Modern Llama cache-shape and Qwen3.5 hybrid-cache tests
are explicitly scoped to the 5.14.1 tested floor; Transformers 4.40 has a
different StaticCache calling and mutation contract.

## Remaining trust boundary

- An approved adapter is trusted executable configuration. MCP roots constrain
  InferRef file parameters; they do not sandbox the adapter's own filesystem,
  network, or subprocess access.
- CPU, memory, GPU memory, syscall, and network quotas are not implemented.
- Artifact monitoring terminates observed growth but is not an atomic disk
  quota; a container/filesystem quota is required against hostile writers.
- The checked-in repair test proves the protocol/evaluation harness using a
  known oracle edit. It does not prove autonomous repair ability.

## Deferred release work

Formal `v0.1`/`v0.2`/`v0.3` golden artifacts and release tags remain a separate
release task. That task should freeze trace/testcase/tensor fixtures, test newer
readers against older assets, assert explicit rejection of unsupported future
versions, and compare Python/C++ verdicts. A `v0.3.0` tag should only be created
after the new Windows/Linux Actions matrix is green.
