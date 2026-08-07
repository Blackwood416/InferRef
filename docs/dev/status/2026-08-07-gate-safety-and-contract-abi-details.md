# Gate deletion safety, contract ABI details, JSON grammar strictness

## Closed issues

- The XPU gate script no longer trusts a string-prefix containment check before
  recursive deletion. The evidence directory is now required to be a real
  subdirectory of the repository (rejecting `.`, `..`, rooted paths, and
  sibling checkouts) and to live under one of the two well-known evidence
  roots (`xpu-sycl-evidence` or `.scratch`). A typo like
  `-EvidenceDirectory . -CleanEvidenceDirectory` can no longer delete the
  repository or a sibling directory.
- `rmsnorm/last-dim/v1` epsilon is now exactly one scalar value in both the
  Python profile validator (`_numel(epsilon) != 1` is an error) and the native
  engine (`e_tensor.Numel() != 1` → `invalid_testcase`). Shape `[2]` is
  rejected instead of silently reading only the first element.
- Per-contract capabilities must now be subsets of the adapter-wide
  declaration at load time: `contract_capabilities[C].dtypes ⊆ dtypes`,
  `.features ⊆ features`, `.max_rank ≤ max_rank`. A contradictory adapter file
  is rejected by the parser, not merely at preflight.
- KV contracts now define their dtype relation: `cache.dtype == update.dtype ==
  cache_out.dtype`, enforced by the Python boundary validator and the native
  engine. Mixed-dtype cache/update inputs are an ABI violation, not an implicit
  conversion.
- The `json.hpp` parser now rejects unescaped control characters (`< 0x20`) in
  strings, matching the JSON grammar; the self-test covers the rejection.
- Documentation now states the extraction publish guarantee precisely
  (rename-based atomic publication under normal process execution, with the
  hard-crash window between the two `--force` renames) and the input/output
  ABI asymmetry: contract outputs are an exact role set, while contract inputs
  are required named roles with optional extra non-contract input tensors.

## Acceptance evidence

- Python full suite: `529 passed, 9 skipped`.
- New tests cover: non-scalar epsilon rejection, per-contract subset
  rejection (dtype/rank/features), KV dtype relation, and JSON control
  characters (C++ self-test).
- Gate script path guard verified against `.`, sibling checkouts, unsafe
  roots, and outside paths; all refused before any build or deletion.
- Native A770 gate: 13/13 positive cells and 3/3 injected-error negatives
  PASS with the new engine checks and Level Zero evidence.

## Decisions

- KV dtype policy is scheme A (all three tensors share one dtype), matching the
  traditional KV-cache layout and the engine's `EncodeLike(cache)` output.
- Extractor crash recovery for the `.bak-*`/`.tmp-*` window is explicitly
  out of scope for now; the guarantee is documented instead.

## Known gaps

- CircleCI still runs the full matrix for every branch/tag and saves trace
  artifacts unconditionally; correctness is fine, cost optimization is not
  done yet.
- The extractor has no startup recovery scan for orphaned `.bak-*`/`.tmp-*`
  directories after a hard crash.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest tests\core\test_contracts.py tests\core\test_adapter_capabilities.py -q
ctest --test-dir cpp/build --output-on-failure
```
