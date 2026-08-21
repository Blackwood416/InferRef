# v0.9.0 Semantic Validation

## Closed issues

- Comparison Spec v0.1 is implemented with conditional testcase/suite `0.3`
  versioning, `comparison_requires_0_3` reader validation, per-field
  `effective_comparison.sources`, and a two-axis tolerance model.
- Comparator plugin architecture is implemented:
  - `inferref.comparators` entry-point group;
  - entry-point name is the full comparator ID;
  - `validate_config` runs before engine launch;
  - `tensor/numeric/v1` built-in;
  - multi-output comparator fixture pack and object-detection example.
- Agent summary mode (`inferref-agent-summary` v0.1) is implemented for run,
  compare, and scenario operations.
- Region preview (`region list --details`) and recommendation
  (`region recommend`) are implemented.
- Doctor hardware details are extended as best-effort non-gating fields.
- 0.8.1 review preflight items are closed: `Testcase::WriteOutputs` with
  caller-label diagnostics, generated-project slow compile test, stale path
  corrections, and EXTENDING standard-mode wording.

## Acceptance evidence

- Python full suite: `749 passed, 6 skipped, 3 deselected`.
- C++ NMake build succeeds; `irtensor_selftest`, `testcase_selftest`, and
  `bridge_selftest` all pass.
- `scripts/check_agent_workflow_skill.py` passes.
- `git diff --check` is clean.

## Decisions

- Testcase/suite format becomes `0.3` only when `comparison` is present; old
  readers reject `0.3` instead of silently downgrading.
- `--tolerance` is a per-dtype default table replacement; `--atol`/`--rtol`
  are scalar overrides in a separate axis.
- `effective_comparison` records per-field `sources` for audit.
- Comparators are trusted same-process Python; no sandbox or timeout is
  promised in 0.9.

## Known gaps

- Dataset-level evaluation remains external.
- GPU/XPU gates were not rerun for this release; no hardware-specific code
  changed in the comparator work.
- The Codex skill has structural validation but has not been forward-tested in
  a real coding-agent session.

## Reproduce

```powershell
python -m pytest -q
cmd /c "call ...\VsDevCmd.bat -arch=x64 -host_arch=x64 >nul && cmake -S cpp -B cpp/build-090 -G \"NMake Makefiles\" -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build-090 && ctest --test-dir cpp/build-090 --output-on-failure"
python scripts/check_agent_workflow_skill.py
```
