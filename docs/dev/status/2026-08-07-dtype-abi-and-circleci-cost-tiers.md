# Contract dtype ABI completeness and CircleCI cost tiers

## Closed issues

- Known-contract boundary validation now requires every input/output role to
  declare a non-empty `dtype` string, instead of only comparing dtypes when
  they happen to be present. A hand-crafted contract testcase with
  `payload: null` and no `dtype` is rejected as an ABI violation, not silently
  accepted as a structurally valid non-reproducible case.
- The extractor rejects duplicate `--contract` values (for example the same ID
  passed twice) instead of silently de-duplicating them, tightening the
  "exactly one contract" CLI semantics.
- CircleCI is restructured into three cost tiers:

  | tier | Windows | Linux |
  | --- | --- | --- |
  | every branch/PR | 3 jobs: core py3.10, frontend py3.13+current+HF fixed, C++ | full matrix |
  | main | same 3 | full matrix + nightly warning |
  | version tag | full 7-job qualification (core 3.10/3.12/3.13, torch-min, torch-current, HF fixed, C++) | full matrix |

  Windows jobs move off the expensive executor for most pushes: the daily set
  drops from 7 to 3 Windows jobs (about 57% fewer Windows job-minutes), while
  tag pipelines restore the full qualification.
- `cpp-windows` no longer installs PyTorch or runs the mini-Llama trace chain.
  It now proves MSVC compilation, the reader self-test, and Python-writer →
  C++-reader wire agreement using numpy-only golden `.irtensor` fixtures
  (`scripts/generate_cpp_golden.py`) with positive, float16, and deliberately
  corrupted negative comparisons.
- Windows Python setup probes the launcher (`py -3.x`) and only falls back to
  Chocolatey when the target version is missing; the CMake step probes
  `cmake --version` before installing. Frontend `trace`/`repro` artifacts are
  stored only `when: on_fail`.
- The PyTorch nightly warning runs on the Monday schedule and main pushes only,
  not on every feature branch.

## Acceptance evidence

- `circleci config validate` passes; `circleci config process` shows the
  intended three-tier job/filter layout.
- Python full suite: `532 passed, 9 skipped` plus new dtype-presence and
  duplicate-contract tests.
- The numpy golden fixtures verified locally against the MSVC C++ comparator:
  positive float32 and float16 PASS, corrupted tensor rejected.

## Decisions

- Linux keeps its full matrix on every branch and tag because open-source
  Linux credits are comparatively abundant; Windows credits are the scarce
  resource being conserved.
- The Windows full qualification is tag-only, not main-only: `v*` tags are the
  release boundary, and re-running it on every main push would consume the
  same 4x credits without a release artifact.

## Known gaps

- Tag pipelines trigger the full Windows matrix on any `v*` tag, including
  pre-release tags; a release-only tag convention (for example `v[0-9]+.[0-9]+.[0-9]+`)
  could tighten that further.
- Orphaned `.bak-*`/`.tmp-*` extraction directories after a hard crash still
  have no startup recovery.

## Reproduce

```powershell
circleci config validate .circleci/config.yml
.\.venv\Scripts\python.exe scripts\generate_cpp_golden.py .scratch\cpp-golden
```
