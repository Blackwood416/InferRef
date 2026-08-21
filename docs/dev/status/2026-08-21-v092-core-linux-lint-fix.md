# v0.9.2 core-linux lint gate fix

## Closed issues

- The 0.9 core-linux jobs failed on the new `ruff check inferref tests` step.
- Added a project Ruff policy for intentional broad catches and optional
  frontend/runtime best-effort code, fixed unused imports, registry global
  handling, and explicit `subprocess.run(check=...)` calls.
- Core-linux now passes both the regular core suite and the lint gate.

## Acceptance evidence

- `uvx ruff check inferref tests`: `All checks passed!`.
- Python full suite: `749 passed, 6 skipped, 3 deselected`.
- Linux container core suite: `504 passed, 2 skipped, 3 deselected`.

## Reproduce

```bash
uvx ruff check inferref tests
python -m pytest -q
```
