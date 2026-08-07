# Contract observable outputs, relational validation, atomic extraction

## Closed issues

- Standalone contract validation now checks the full boundary, not just
  inputs. A known contract enforces an exact observable output set
  (`contract_output_missing` / `contract_unexpected_output` errors), so a
  RoPE testcase that keeps `q_embed` but drops `k_embed` is invalid even with
  legal inputs. A PASS can no longer hide a broken half of a contract.
- Contract profiles gained output and relation validators in addition to
  input shape checks: RMSNorm `y.shape == x.shape` + dtype; RoPE
  `q_embed.shape == query.shape` / `k_embed.shape == key.shape` + dtypes; KV
  append sequence = cache + update with shared rank/width; indexed update
  `cache_out.shape == cache.shape`.
- Extraction is now staging + atomic publish. Payloads and manifest are built
  under a hidden sibling directory, run through the same `validate_testcase`
  used by Suite/engine, and only then promoted. Existing outputs are refused
  unless `--force`, and `--force` replaces via backup-and-swap so a failed
  extraction cannot corrupt a previously valid testcase.
- v0.2 restricts a testcase to exactly one executable contract: the manifest
  `contracts` array must have a single unique ID, extraction rejects repeated
  `--contract` values, and the native engine already rejects multiple
  supported contracts. Adapter `contracts`/`contract_capabilities` remain
  multi-contract, since one engine legitimately supports many operations.

## Acceptance evidence

- Python full suite: `529 passed, 9 skipped`.
- All 13 committed XPU corpus testcases pass the new output-set and relation
  validation unchanged.
- New tests cover missing/unexpected observable outputs, wrong output shapes,
  exact-one contract arrays, extraction refusal without `--force`, atomic
  replace with `--force`, failed-extraction preservation of an existing
  testcase, and absence of leftover staging directories.

## Decisions

- Exact output sets are enforced at the policy layer (validator/extractor);
  shape and relation invariants live in the registry profiles, keeping
  profiles composable for future composite contracts.
- The extractor's final `validate_testcase` gate tolerates `payload_missing`
  errors, because metadata-only extraction legitimately produces a
  non-reproducible testcase with declared missing payloads.

## Known gaps

- Dtype/value-level semantic invariants (rotation convention, numeric ranges)
  remain in the engine and hardware gate; the registry stays a Python-validator
  profile rather than a general constraint DSL, by design.
- The index contract still uses a float32 index tensor; an integer-typed
  contract can be added as a future v2.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest tests\core\test_contracts.py tests\core\test_testcase_validate.py tests\frontend\test_testcase_and_regions.py -q
```
