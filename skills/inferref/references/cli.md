# InferRef CLI and MCP reference

Every command accepts `--json` for agent and CI consumption, except `export`,
which always emits JSON. `agent run`, `agent compare`, `scenario run`, and
`suite run` stop at the first divergence by default; pass `--all-failures`
(where available) to compare everything. Plain `inferref compare` reports all
differences unless `--first-failure` is passed.

## Environment and runtime

- `inferref doctor [--device <cpu|cuda|xpu>] [--verify-plugins] [--json]` - prove
  the runtime and accelerator state before starting.
- Install: `uv pip install -e ".[torch,dev,agent]"`.

## Trace lifecycle

- `inferref trace <script> -o <trace-dir> [--capture-tensors <mode>] [--semantic-analysis] [--json]` -
  record a reference trace. Script arguments go after `--`.
- `inferref inspect <trace> [--tensor <value-id>] [-v] [--json]` - list operators,
  tensors, modules, and sources.
- `inferref analyze <trace> [--top <n>] [--json]` - summarise operator and region
  coverage.
- `inferref validate <trace> [--json]` - check Trace IR invariants.

## Regions

- `inferref region detect <trace> [--dry-run] [--detector <name>] [--replace] [-v] [--json]` -
  recognise semantic regions such as Linear, RMSNorm, RoPE, and Attention.
- `inferref region list <trace> [--details] [--json]` - list regions, with
  boundary/reproducibility details when requested.
- `inferref region recommend <trace> [--top <n>] [--min-score <n>] [--json]` -
  score and rank regions for engine extraction.
- `inferref region create <trace> --name <name> (--from-op <id> --to-op <id> | --module <path> | --source-function <name>) [--semantic <label>] [--engine-op <kernel>] [--json]`.
- `inferref region delete <trace> <name> [--json]`.

## Testcases

- `inferref testcase extract <trace> (--op <id> | --region <name>) -o <dir> [--name <name>] [--input-names <a,b,c>] [--output-names <x,y>] [--contract <contract-id>] [--force] [--json]`.
- `inferref testcase dedup <trace> [--operator <name>] [--limit <n>] [--json]` -
  group operators into unique signatures.

## Contracts

- `inferref contract list [--json]` - list built-in and discovered contracts.
- `inferref contract show <contract-id> [--json]` - show one resolved contract.
- `inferref contract validate <file> [--json]` - validate a `.contract.json` file.

## Comparators

- `inferref comparator list [--json]` - list built-in and discovered comparison plugins.
- `inferref comparator show <id> [--json]` - show one resolved comparator plugin.

`--comparator <id>` and `--comparison-config <json|file>` are accepted by
`compare`, `agent compare`, `agent run`, `agent run_scenario`, `scenario run`,
and `suite run`. The default comparator is `tensor/numeric/v1`.

## Agent operations

- `inferref agent capabilities [--json]` - formats, operations, and MCP tool names.
- `inferref agent context <artifact> [--json]` - compact trace/testcase summary.
- `inferref agent extract <trace> (--op <id> | --region <name>) -o <dir> [--name <name>] [--input-names <a,b,c>] [--output-names <x,y>] [--contract <contract-id>] [--json]`.
- `inferref agent compare <testcase> <engine-output> [--atol <f>] [--rtol <f>] [--ignore-stride] [--strict-layout] [--all-failures] [--first-failure] [--json]`.
- `inferref agent run <testcase> --adapter <adapter.json> [--runs-dir <dir>] [--atol <f>] [--rtol <f>] [--ignore-stride] [--strict-layout] [--all-failures] [--first-failure] [--json]`.
- `inferref agent run_scenario <scenario-dir> --adapter <adapter.json> [--runs-dir <dir>] [--state-mode reference|engine] [--compare-state] [--atol <f>] [--rtol <f>] [--ignore-stride] [--strict-layout] [--all-failures] [--first-failure] [--json]`.
- `inferref agent evaluate <benchmark> --agents <a,b> --report-dir <dir> [--claude-settings <path>] [--claude-model <model>] [--public-attestation <path>] [--json]` -
  run the blind repair benchmark with external coding Agents.

## Scenario

- `inferref scenario validate <scenario-dir> [--allow-nonreproducible] [--json]`.
- `inferref scenario run <scenario-dir> --adapter <adapter.json> --runs-dir <dir> [--state-mode reference|engine] [--compare-state] [--allow-unsupported] [--fail-fast] [--atol <f>] [--rtol <f>] [--ignore-stride] [--strict-layout] [--all-failures] [--first-failure] [--json]`.

## Suite

- `inferref suite validate <suite.json> [--allow-nonreproducible] [--json]`.
- `inferref suite run <suite.json> --adapter <adapter.json> [--adapter <more.json> ...] --runs-dir <dir> [--allow-unsupported] [--fail-fast] [--json]`.
- `inferref suite report <suite-run.json> --output <report.html> [--json]`.

## MCP tools

Start the server with `inferref-mcp --read-root <workspace> --write-root <runs>`
(repeatable; write roots default to read roots).

- `inferref_capabilities()` - discover formats, operations, and MCP tools.
- `inferref_context(path)` - summarise and validate a trace or testcase.
- `inferref_extract_testcase(trace, output, region|op_id, name, input_names, output_names, contracts)`.
- `inferref_compare_outputs(testcase, engine_output, atol, rtol, ignore_stride, strict_layout, first_failure)`.
- `inferref_run_engine(testcase, adapter, runs_root, atol, rtol, ignore_stride, strict_layout, first_failure)`.
- `inferref_run_scenario(scenario, adapter, runs_root, state_mode, compare_state, atol, rtol, ignore_stride, strict_layout, first_failure)`.

MCP roots are path-policy containment, not a sandbox. Exact root paths are host
policy and are never committed to the repository.
