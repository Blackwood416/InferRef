# Agent workflow guide and Codex skill

## Closed issues

- The 0.7 Agent Workflow milestone artifacts are implemented:
  - `docs/AGENT_WORKFLOW.md` is the human-facing end-to-end guide (install,
    protocol discovery, MCP configuration, standard agent loop, trace
    inspection, testcase extraction, engine run/compare, scenario run,
    envelope interpretation, common traps, and a worked mini-Llama example).
  - `skills/inferref/` is the distributable Codex skill: a lean `SKILL.md`
    (82 lines) with the standard loop and operation decision table, plus one
    level of references (`cli.md`, `adapter.md`, `protocol.md`,
    `workflows.md`) covering the full CLI/MCP inventory, Engine Adapter v0.2,
    envelope semantics, and three complete workflows.
  - `README.md` points to both artifacts with the one-line Codex skill
    install command.
  - `scripts/check_agent_workflow_skill.py` is a stdlib-only validation
    harness: required files, frontmatter, every `inferref <subcommand>` and
    `inferref_<tool>` mention (verified against the current CLI help tree and
    the `agent capabilities` MCP tool list), and author-path/credential
    cleanliness. It exits non-zero with stable actionable messages.
  - `tests/docs/test_agent_workflow_skill.py` wraps the harness with eight
    pytest cases, including failure modes and a `python -S` run that proves
    the harness stays stdlib-only.

## Acceptance evidence

- Full Python suite on current `main` (`683334f`) with the untracked
  milestone files present: `666 passed, 6 skipped, 1 deselected`
  (the +8 count over the v0.6.0 baseline is exactly the new `tests/docs`
  tests).
- `python -m pytest tests/docs/test_agent_workflow_skill.py -q`: 8 passed.
- `python scripts/check_agent_workflow_skill.py`: `OK` (6 files scanned,
  28 command mentions, 6 MCP tool mentions).
- `skills/inferref` passes the `skill-creator` `quick_validate.py`, and an
  install smoke (copy to a fresh directory + validate) succeeds.
- Every command mentioned in the guide, skill, and references exists in the
  current CLI help tree; all six MCP tools (`inferref_capabilities`,
  `inferref_context`, `inferref_extract_testcase`, `inferref_compare_outputs`,
  `inferref_run_engine`, `inferref_run_scenario`) match
  `inferref agent capabilities --json`.
- Manual smoke: `inferref agent capabilities --json` and
  `inferref scenario validate tests/fixtures/scenarios/kv-chain --json` both
  pass.

## Decisions

- The skill follows the `skill-creator` conventions: frontmatter contains only
  `name` and `description`; `agents/openai.yaml` carries display metadata and a
  `default_prompt` referencing `$inferref`; references stay one level deep and
  are linked from `SKILL.md`.
- The validation harness extracts command mentions only from code spans
  (fenced blocks and inline code) so prose does not produce false positives,
  and skips JSON field names such as `"inferref_version"`.
- MCP tool names are verified against `agent capabilities --json` instead of
  importing the optional `mcp` package, keeping the harness and CI torch-free
  and MCP-free.
- `scenario validate` takes the scenario root directory (not
  `scenario.json`); the guide, skill, and harness all use the directory form.
- The harness accepts `--repo` so tests can point at minimal fixture trees
  while command verification still runs against the real repository CLI.

## Known gaps

- The 0.7 spec's manual smoke section still shows
  `inferref scenario validate tests/fixtures/scenarios/kv-chain/scenario.json`,
  which fails because the CLI expects the scenario root directory. The
  milestone artifacts use the correct directory form; the spec line should be
  corrected.
- The milestone files (`docs/AGENT_WORKFLOW.md`, `skills/inferref/`, the
  harness, tests, and this report) are not yet committed at snapshot time;
  review and commit are pending.
- The skill has been structurally validated and install-smoked but not yet
  forward-tested in a real coding-agent session; that remains a manual
  release smoke.
- The harness verifies MCP tool names against the capabilities payload, not a
  live MCP round-trip. MCP remains optional and is not exercised in CI.
- CircleCI full-suite legs (`pytest tests -q`) collect `tests/docs`
  automatically; the core-only legs do not, so a broken harness would not be
  caught by a `tests/core`-only gate until the full leg runs.
- No XPU/GPU code changed in this docs-only milestone; the previous v0.6.0
  A770 gate evidence stands, the XPU runner still must be started manually
  (not a service), and no GPU self-hosted runner is registered.

## Reproduce

```powershell
python -m pytest tests/docs/test_agent_workflow_skill.py -q
python scripts/check_agent_workflow_skill.py
python -m pytest tests -q
python -m inferref.cli.main agent capabilities --json
python -m inferref.cli.main scenario validate tests/fixtures/scenarios/kv-chain --json
```
