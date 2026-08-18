# v0.7.0 CI retrigger and upstream compatibility fixes

## Closed issues

- The v0.7.0 push events were created while GitHub was unavailable, so
  CircleCI never received the original branch/tag push. A temporary branch and
  `v0.7.0-ci.1` were used to re-trigger CI on the exact release commit; the
  temporary branches were deleted after pipelines started.
- `frontend-linux-qwen35-latest` failed against `transformers==5.15.0`:
  Transformers now records three state `copy_` mutations instead of four
  because the decode-time conv-state copy is no longer emitted. The test now
  asserts the storage-version contract (two state storages, version 0 start,
  consecutive transitions, at least one decode transition) instead of pinning
  the exact upstream copy count.
- `frontend-linux-py3.10-torch-min`, `frontend-linux-generic-min`, and
  `frontend-windows-py3.10-torch-min-tag` failed because Python 3.10 argparse
  renders subcommand descriptions on the following line. The agent workflow
  check harness parsed only the same-line layout, so `agent` subcommands were
  treated as unknown. The parser now accepts both layouts.

## Acceptance evidence

- Full Python suite with `transformers==5.15.0` in an isolated environment:
  `670 passed, 9 skipped, 1 deselected`.
- `tests/frontend/test_hf_hybrid_cache.py` passes with both
  `transformers==5.14.1` and `transformers==5.15.0`.
- `tests/docs/test_agent_workflow_skill.py` passes in the Python 3.10 +
  torch 2.1 environment (`9 passed`) and in the current Python 3.13
  environment (`9 passed`).
- `scripts/check_agent_workflow_skill.py` reports OK under both environments.

## Decisions

- Keep the `qwen35-latest` job instead of pinning it down. The test now
  tolerates Transformers optimizing state writes without weakening the
  truthfulness guarantee.
- CI verification for the release commit is carried by `v0.7.0-ci.1`; the
  original `v0.7.0` tag and GitHub Release remain unchanged.

## Known gaps

- The skill has still not been forward-tested in a real coding-agent session;
  structural validation and install smoke remain the automated coverage.
- The CircleCI statuses for `v0.7.0-ci.1` cover the release commit, while the
  branch pipeline covers the empty retrigger commit on `main`; code content is
  identical.
- GPU/XPU code did not change; no new hardware gate is required for this fix.

## Reproduce

```powershell
uv run --no-project --with torch --with transformers==5.15.0 --with pytest --with numpy python -m pytest tests -q
.\.venv-torch21\Scripts\python.exe -m pytest tests\docs\test_agent_workflow_skill.py -q
python scripts/check_agent_workflow_skill.py
```
