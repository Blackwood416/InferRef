# v0.9.1 core-linux MCP test fix

## Closed issues

- `tests/core/test_comparison_spec.py` had one test that created the MCP
  server without the optional `agent` extra installed. Core-linux jobs install
  only `[dev]`, so the test failed with `ModuleNotFoundError: mcp`.
- The test now uses `pytest.importorskip("mcp")` and is skipped on core-only
  environments, matching the existing optional-MCP test convention.

## Acceptance evidence

- Linux container (`python:3.12-slim`, `pip install -e ".[dev]"`):
  `504 passed, 2 skipped, 3 deselected`.
- Windows full suite remains green.

## Reproduce

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace python:3.12-slim \
  sh -c "pip install -q -e '.[dev]' && python -m pytest tests/core -q"
```
