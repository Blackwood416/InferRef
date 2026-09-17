# Contributing to InferRef

Start with [the documentation index](docs/index.md) and [extension guide](docs/EXTENDING.md).
For a bug report, include the InferRef version, OS/Python version, `inferref doctor`
output, a minimal command and fixture, and expected versus actual output. Remove
credentials, private model weights, and proprietary trace payloads before sharing.

## Development

```bash
git clone https://github.com/Blackwood416/InferRef.git
cd InferRef
uv venv
uv pip install -e ".[dev]"
uv run --no-sync python -m pytest tests/core
```

Install `.[torch,dev]` to run `tests/frontend`, or `.[agent,dev]` for MCP tests.
HF model integrations need the `hf` or `qwen35` extra; accelerator tests need
their documented hardware. Do not turn a missing environment into a pass claim.
Slow tests are excluded by default; run them explicitly with
`uv run --no-sync python -m pytest tests/core -m slow`.

Before submitting, run the relevant tests, the full core suite,
`uvx ruff check inferref tests`, and
`uv run --no-sync python scripts/check_agent_workflow_skill.py`.
For packaging changes, follow [distribution checks](docs/DISTRIBUTION.md).

## Pull requests and contracts

- Add a regression that fails before the fix. User-facing CLI changes need a real
  happy-path invocation and JSON/exit-code assertions, not only mocked dispatch.
- Keep the core importable without Torch. Preserve filesystem containment and
  treat adapters/plugins as executable trusted configuration.
- Do not renumber dtype wire codes or silently change report fields. Document
  compatibility changes and bump the affected artifact schema when necessary.
- Add user-visible changes to `CHANGELOG.md` under Unreleased. Keep unrelated
  refactors and generated traces/build products out of the patch.
- Historical status reports are dated evidence, not the current user manual.

## Extension authors

See [detectors and extensions](docs/EXTENDING.md),
[semantic plugins](docs/semantic-plugins.md), and the
[comparator protocol](docs/spec/InferRef_0.9_Semantic_Validation.md).
Comparator distributions register entry points in `inferref.comparators`; the
entry-point name must match the plugin's comparator ID. Test discovery, config
validation, passing/mismatching artifacts, and plugin exceptions in an isolated
environment. `inferref comparator list --json` inspects discovery; custom
comparators operate on testcase output roles, not full trace pairs.

The package ships `py.typed` so type checkers can consume existing inline hints;
this is not a claim that every public API is fully statically typed.
