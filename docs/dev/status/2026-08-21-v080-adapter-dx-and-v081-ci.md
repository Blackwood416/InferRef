# v0.8.0 Adapter DX and v0.8.1 CI hardening

## Closed issues

- The 0.8 Adapter DX milestone is implemented:
  - `cpp/include/inferref/testcase.hpp` provides a high-level C++ testcase
    loader, named input/output access, `Finish()` manifest publication, and a
    self-test;
  - `inferref adapter scaffold` generates a compilable C++ adapter project
    from a testcase, including `CMakeLists.txt`, `adapter.json`, `main.cpp`,
    and `README.md`;
  - `cpp/include/inferref/bridge.hpp` and `cpp/examples/runtime_bridge/main.cpp`
    provide a generic runtime bridge with named `DebugInvoke` outputs;
  - bridge-mode scaffolding and the first YOLO/Aila TFCM runbook are
    committed.
- Windows CircleCI jobs failed because Chocolatey could not install Python
  (`504 Gateway Timeout`) and the `py` launcher had no usable runtime. The
  installer now tries `py` first and falls back to a downloaded `uv`, which
  installs Python and creates a seeded venv.
- `cpp-windows` failed under the default Visual Studio generator. The job now
  locates Visual Studio through `vswhere`, initializes the VS Developer
  Command Prompt, and builds with NMake + single-config ctest.
- All CircleCI statuses are green on the 0.8.1 release commit.

## Acceptance evidence

- Python 3.13 full suite: `676 passed, 6 skipped, 1 deselected`.
- Python 3.10 full suite: `580 passed, 12 skipped, 1 deselected`.
- Python 3.12 `tests/core`: `433 passed`.
- C++ NMake build succeeds; `irtensor_selftest`, `testcase_selftest`, and
  `bridge_selftest` all pass.
- CircleCI: 22/22 statuses `success` on `444cb61`.
- `inferref 0.8.1` wheel and sdist are attached to the GitHub Release.

## Decisions

- Windows Python provisioning avoids Chocolatey as the primary path. `uv` is
  the reliable fallback because the project already uses it for local
  environments.
- `cpp-windows` uses `vswhere` + `VsDevCmd.bat` + NMake so MSVC resource
  tools are on PATH and the build remains a single-config Release build.
- `v0.8.0` remains the original feature tag; `v0.8.1` carries the CI fixes and
  is the green public release.

## Known gaps

- Semantic comparison remains deferred to 0.9.
- GPU/XPU gates were not rerun for this CI-only patch; no hardware code
  changed.
- The Codex skill has structural validation but has not been forward-tested in
  a real coding-agent session.

## Reproduce

```powershell
python -m pytest -q
uv run --no-project --python 3.10 --with torch --with transformers==5.14.1 --with pytest --with numpy python -m pytest tests -q
circleci config validate .circleci/config.yml
```
