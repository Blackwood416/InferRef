# InferRef 0.6 Docs Housekeeping

> **Project:** InferRef
> **Status:** Draft / 0.6.0 scope
> **Audience:** worker agent

---

# 1. Goal

The README has drifted from the repository's real state:

- `pyproject.toml` is 0.5.6 but README still claims fixed test counts from an
  older release (`277 tests`, `161 tests`);
- README points CI readers at `.github/workflows/ci.yml` while the CPU matrix
  now runs on CircleCI;
- there is no single "where do I plug in my thing" guide, which is the largest
  user-experience gap for external engine developers.

This task fixes the drift and adds `docs/EXTENDING.md`.

---

# 2. Scope

1. Remove all hardcoded test counts from README.
2. Replace them with a CI status badge and dependency-bound commands that stay
   correct.
3. Correct CI pointers: CPU matrix on CircleCI; GitHub Actions hosts the
   GPU/XPU/nightly gates.
4. Add `docs/EXTENDING.md` with four short recipes.

No product code changes. No changes to test behavior.

---

# 3. README Changes

## 3.1 Tests section

Replace:

```text
python -m pytest tests -q          # 277 tests
python -m pytest tests/core -q     # 161 tests, no PyTorch required
```

with:

```text
python -m pytest tests -q
python -m pytest tests/core -q     # no PyTorch required
```

Keep the explanation of the dependency split and the torch import-block
verification. Add a CI status badge at the top of the README:

```markdown
![CI](https://img.shields.io/endpoint?url=<circleci-badge-endpoint>)
```

Use the CircleCI badge endpoint for the project repository. The exact badge
URL is filled in from the CircleCI project settings; do not invent one that
does not resolve.

## 3.2 CI pointer

The paragraph that says:

```text
CI runs the core suite ... see [.github/workflows/ci.yml](.github/workflows/ci.yml).
```

must be updated to say:

- the core/frontend CPU matrix runs on CircleCI (`.circleci/config.yml`);
- GitHub Actions workflows host the GPU, XPU, and nightly gates
  (`.github/workflows/`).

Link both locations.

## 3.3 Other stale references

Scan the README for:

- any remaining fixed test counts;
- any statement that all CI lives in `.github/workflows`;
- version claims that contradict `pyproject.toml`.

Do not rewrite product content beyond these corrections.

---

# 4. `docs/EXTENDING.md`

One page, four recipes. Each recipe is:

1. goal;
2. files to create;
3. minimal example;
4. verification command.

Required content:

## 4.1 Add a semantic detector

- entry point group `inferref.semantic_detectors`;
- implement `SemanticDetector` (name + `detect(package)`);
- verify with `inferref doctor --verify-plugins` and
  `inferref region detect --detector <name>`;
- link to `InferRef_Contract_Schema_v0.1.md` if the detector ships contracts
  too.

## 4.2 Add an executable contract

- entry point group `inferref.contracts`;
- minimal `inferref-contract` schema example (Swiglu-style);
- verify with `inferref contract list`, `inferref contract validate`, then
  `inferref testcase extract --contract <id>`;
- link to the contract schema spec.

## 4.3 Add an engine adapter

- `inferref-engine-adapter` v0.2 JSON shape;
- `{testcase}`, `{output}`, `{adapter_dir}`, `{python}` placeholders;
- capability declaration and per-contract capabilities;
- verify with `inferref agent run ... --json` or `inferref suite run ...`;
- link to `InferRef_Engine_Adapter_v0.2.md`.

## 4.4 Add a corpus case

- standalone testcase directory layout;
- suite manifest entry;
- verify with `inferref suite validate` and `inferref suite run`;
- link to `InferRef_Suite_v0.1.md`.

---

# 5. Acceptance

1. `rg -n "277 tests|161 tests|workflows/ci.yml" README.md` returns no hits.
2. README contains exactly one CI badge line and no other hardcoded test
   count.
3. `docs/EXTENDING.md` exists, is under 200 lines, and every command in it is
   valid against the current CLI.
4. No source or test file changed.

---

# 6. Implementation Task

Files:

- `README.md`
- add `docs/EXTENDING.md`

Single task, no sequencing dependencies. Verify with the commands in section
5 and a final `python -m pytest tests/core -q` smoke run to confirm the doc
change did not affect the environment.

