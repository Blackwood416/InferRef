# Status: Adversarial input and resource hardening

**Date:** 2026-08-02

**Version:** `0.3.0`

**Code commit:** `4eba2a1`

## Outcome

The post-review correctness slice is complete. GitHub Actions run
[`30747593127`](https://github.com/Blackwood416/InferRef/actions/runs/30747593127)
finished with all 19 jobs successful.

Standalone testcase validation is now a total operation for malformed effect
JSON. Non-array `aliases` and `mutated_storages` values are normalized after a
structured error is recorded, malformed nested entries remain diagnostics, and
integer references reject booleans and unhashable JSON values safely. The tests
cover integer, string, object, null, nested malformed values, and 500
deterministically generated arbitrary JSON manifests.

HF semantic coverage now treats the expected semantic classes as the hard
contract and physical operator coverage as a trend. The required test asserts
Attention, RoPE, RMSNorm, MLP, and Linear, permits coverage down to 0.80, and
prints uncovered operator classes for diagnosis. A fixed Windows/Python
3.13/Transformers 4.57.6 job prevents this path from remaining Linux-only. The
matching local CPU environment reported 0.8559 coverage and passed all five HF
semantic tests.

The trusted engine adapter now limits artifact file count and traversal entries
as well as bytes. Process/deadline polling remains at 20 ms while the more
expensive directory scan runs at 250 ms intervals. Scans check their deadline
incrementally, avoid materializing complete directories, and terminate on file
or entry storms with `artifact_file_limit`. Execution records expose observed
artifact bytes, files, and scan entries.

## Verification

- local full suite: `374 passed, 5 skipped`;
- malformed testcase + Agent adversarial subset: `40 passed`;
- Python 3.10 / PyTorch 2.1 relevant subset: `45 passed`;
- Windows Python 3.13 / PyTorch 2.13 CPU / Transformers 4.57.6 semantic suite:
  `5 passed`;
- changed-file Ruff `F`/`I` checks: pass;
- workflow YAML parse and `git diff --check`: pass;
- GitHub Actions: 19/19 jobs, including Core without PyTorch, Agent+MCP,
  Windows/Linux C++, PyTorch floor/current/nightly, generic HF floor, fixed
  Windows HF, fixed Qwen3.5, and latest Transformers.

## Known gaps

- The new Windows HF leg uses the official CPU wheel. The reviewer's exact XPU
  stack and CUDA/device capture remain unverified.
- Artifact monitoring is bounded detection for trusted adapters, not an OS
  sandbox or a filesystem quota; a process may transiently create data between
  scans.
- The JSON fuzz test is deliberately lightweight and deterministic. It is not a
  coverage-guided fuzzer and does not replace future corpus-based fuzzing.
- Semantic coverage is printed in CI logs, not yet persisted as a longitudinal
  metric artifact.
- Formal release tags and versioned 0.1/0.2/0.3 golden compatibility assets are
  still outstanding.

## Suggested next steps

1. Run a blind real-Agent repair benchmark inside the established MCP roots.
2. Add scheduled/manual GPU coverage when suitable runners are available.
3. Persist semantic trend data and add a small malformed-manifest fuzz corpus.
4. Create release tags and cross-version golden artifacts before freezing the
   exchange formats.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests\core\test_testcase_validate.py tests\core\test_agent.py -q
.\.venv-hf457\Scripts\python.exe -m pytest tests\frontend\test_semantic_hf.py -q -s
gh run view 30747593127
```
