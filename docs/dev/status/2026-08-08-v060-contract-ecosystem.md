# v0.6.0 contract ecosystem and CI/XPU gate

## Closed issues

- The 0.6 Contract Ecosystem milestone is implemented: executable contracts
  move out of the hard-coded tuple into an `inferref.contracts` registry with
  the lightweight declarative schema (`inferref-contract` v0.1), relation
  expressions, a Python escape hatch, and entry-point discovery for third-party
  contract packs.
- Scenario v0.1 adds the stateful layer between Testcase and Suite: explicit
  state bindings, effective-testcase compilation, `reference-state` /
  `engine-state` replay modes, and `--compare-state` verification. Scenario
  execution reuses the existing adapter ABI and testcase harness.
- Suite v0.2 adds the additive `kind` field, Agent protocol gains
  `run_scenario`, and the CLI exposes contract and scenario commands. Existing
  suite 0.1 readers remain supported.
- Windows slow-test acceptance is fixed by removing `--no-build-isolation`
  from the fixture-pack `pip install`. Python 3.10 and 3.12/3.13 venvs otherwise
  disagree on whether `setuptools`/`wheel` is available.
- CircleCI no longer runs the shared Linux matrix on tag pipelines. Jobs using
  `pipeline.git.branch or pipeline.git.tag` were changed to branch-only
  `branches.only` + `tags.ignore`, so pushing a tag after the branch no longer
  re-runs the same Linux/agent/frontend/C++ commit gate.
- The XPU gate was executed on the local A770. The runner is installed at
  `E:\InferRefRunner\actions-runner` but is not a Windows service; it was
  started with `run.cmd` in a hidden background process.

## Acceptance evidence

- Python full suite on current `main` (`a3c4f6f`): `658 passed, 6 skipped,
  1 deselected`.
- CircleCI branch statuses for `a3c4f6f` completed green before the tag was
  created, including Linux core/agent/frontend/C++ and the Windows daily
  smoke set.
- Tag pipeline `v0.6.0-ci.1` completed the Windows tag qualification green.
- Intel XPU compatibility workflow completed green on the A770 runner.
- `circleci config validate .circleci/config.yml` passes with the new
  branch-only shared-job filters.

## Decisions

- `v0.6.0` marks the feature release. CI-only fixes after a feature tag use
  `v<version>-ci.N` and do not bump `pyproject.toml` or `INFERREF_VERSION`.
- Branch pipelines are the Linux correctness gate. Tag pipelines exist only to
  run the expensive Windows qualification on the exact tagged commit, avoiding
  duplicate CI for the same revision.
- GPU/XPU workflows remain `workflow_dispatch` only because self-hosted
  runners execute repository code with the runner account's privileges.

## Known gaps

- Any `v*` tag still triggers the tag pipeline, including CI-only tags such as
  `v0.6.0-ci.1`. A release-only convention such as `v[0-9]+.[0-9]+.[0-9]+`
  could narrow this further.
- The tag pipeline no longer re-runs the Linux matrix. Release safety relies
  on the main branch pipeline having passed the exact tagged commit.
- The XPU runner is not registered as a Windows service, so it stops on reboot
  or interactive session changes and must be started manually again.
- No GPU self-hosted runner is registered, so the GPU workflow cannot currently
  be dispatched.

## Reproduce

```powershell
python -m pytest -q
circleci config validate .circleci/config.yml
Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','E:\InferRefRunner\actions-runner\run.cmd' -WorkingDirectory 'E:\InferRefRunner\actions-runner' -WindowStyle Hidden
```
