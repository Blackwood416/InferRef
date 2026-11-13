# Package distribution

InferRef 0.12.2 is published to PyPI. Install it with:

```bash
python -m pip install inferref==0.12.2
```

The immutable `v0.12.2` source tag can also be installed directly without a
manual clone:

```bash
python -m pip install "inferref @ git+https://github.com/Blackwood416/InferRef.git@v0.12.2"
```

## Build and verify locally

Use a fresh output directory, separate from historical `dist/` contents:

```bash
uv build --out-dir .scratch/package-check
uvx twine check .scratch/package-check/*
python scripts/check_distributions.py .scratch/package-check
python scripts/check_distributions.py .scratch/package-check --install
```

The checker requires exactly one matching wheel/sdist pair, matching embedded
versions, the typing marker, and alignment modules. Install the wheel into a
fresh environment and run `inferref --version` from outside the repository to
avoid accidentally importing the source checkout. Repeat installation with the
sdist. Building alone does not publish a package.

## TestPyPI release test

The 0.12.1 candidate was uploaded to TestPyPI for a staging check before
publication. The commands below remain useful for a future pre-release check;
do not use TestPyPI as the production install source.

```powershell
uv build --out-dir .scratch/testpypi-0.12.1
uvx twine check `
  .scratch/testpypi-0.12.1/inferref-0.12.1-py3-none-any.whl `
  .scratch/testpypi-0.12.1/inferref-0.12.1.tar.gz
python scripts/check_distributions.py .scratch/testpypi-0.12.1 --install
```

Set the TestPyPI API token only in the current shell. Do not write it into the
repository, `pyproject.toml`, workflow files, or a committed `.pypirc`. The
test token is `UV_TEST_PUBLISH_TOKEN`; `UV_PUBLISH_TOKEN` is reserved for the
production PyPI upload. `uv publish` falls back to `UV_PUBLISH_TOKEN` when
`--token` is omitted, so always pass the TestPyPI token explicitly and never
put a TestPyPI token in `UV_PUBLISH_TOKEN`:

```powershell
$env:UV_TEST_PUBLISH_TOKEN = '<set in this shell only>'
uv publish --publish-url https://test.pypi.org/legacy/ `
  --token $env:UV_TEST_PUBLISH_TOKEN `
  .scratch/testpypi-0.12.1/inferref-0.12.1-py3-none-any.whl `
  .scratch/testpypi-0.12.1/inferref-0.12.1.tar.gz
```

Run a dry run first when validating credentials:

```powershell
uv publish --publish-url https://test.pypi.org/legacy/ `
  --token $env:UV_TEST_PUBLISH_TOKEN --dry-run `
  .scratch/testpypi-0.12.1/inferref-0.12.1-py3-none-any.whl `
  .scratch/testpypi-0.12.1/inferref-0.12.1.tar.gz
```

TestPyPI versions are immutable. If the candidate needs changes after upload,
do not overwrite it; use a new version or TestPyPI pre-release version and record
why. Verify the uploaded package in a fresh environment outside the checkout:

```powershell
python -m venv .scratch/testpypi-venv
. .scratch/testpypi-venv/Scripts/Activate.ps1
python -m pip install `
  --index-url https://test.pypi.org/simple/ `
  --extra-index-url https://pypi.org/simple `
  inferref==0.12.1
inferref --version
python -c "import inferref; print(inferref.__version__)"
```

TestPyPI may lag behind PyPI and may not mirror every dependency. The
`--extra-index-url` allows normal dependencies such as numpy to resolve from
PyPI while installing InferRef from TestPyPI. The TestPyPI upload is not a
PyPI publication and does not create a GitHub Release.

## GitHub release assets

The package workflow builds/tests distributions on main pushes, pull requests,
manual dispatch, and new published releases. CI artifacts contain the wheel and
sdist. For a published release, a separate write-scoped job checks that the tag
matches the embedded version and the tagged commit's required CircleCI checks
are green before attaching the assets. Existing assets are not overwritten.

A release carries four assets plus checksums:

- `inferref-<version>-py3-none-any.whl` — Python package, CLI and MCP server
- `inferref-<version>.tar.gz` — minimal Python source distribution
- `inferref-cpp-<version>.tar.gz` — header-only C++ SDK, CMake project and
  quantization goldens
- `inferref-skill-<version>.tar.gz` — Agent skill files
- `SHA256SUMS` — SHA-256 for all four artifacts

The PyPI/TestPyPI distribution is intentionally Python-only. C++ headers are
not installed by `pip install inferref`; use the C++ archive or the source
repository. The skill archive extracts as `inferref/` for a skills directory.

The local asset builder is:

```powershell
python scripts/build_release_assets.py --output .scratch/release-assets-0.12.2
```

To include wheel/sdist hashes in `SHA256SUMS`, pass them explicitly:

```powershell
python scripts/build_release_assets.py --output .scratch/release-assets-0.12.2 `
  --extra-artifact .scratch/release-0.12.2/inferref-0.12.2-py3-none-any.whl `
  --extra-artifact .scratch/release-0.12.2/inferref-0.12.2.tar.gz
```

Before releasing: update both source version declarations, move Unreleased
changes to the new changelog entry, run core/frontend/slow and C++ gates, push
the commit, wait for branch CI, push its version tag, wait for tag qualification,
and publish the GitHub Release. Releasing a new version is a separate operation
from merging these QA fixes. The workflow is not retroactively triggered for
the already published v0.10.0 release.

For the published 0.12.2 release, tag `v0.12.2` was created only after branch
CI and the release decision. Qualification for 0.12.2 includes the PyPI README
refresh, package metadata, install diagnostics and the existing 0.12.1
qualification surface. See [release notes](releases/0.12.2.md).

## PyPI publication

Production PyPI uses `UV_PUBLISH_TOKEN`; never use it for TestPyPI. Upload the
exact wheel and sdist built from the release tag, after tag qualification:

```powershell
uv publish --publish-url https://upload.pypi.org/legacy/ `
  --token $env:UV_PUBLISH_TOKEN `
  .scratch/release-0.12.2/inferref-0.12.2-py3-none-any.whl `
  .scratch/release-0.12.2/inferref-0.12.2.tar.gz
```

Verify PyPI project-name ownership, configure credentials or a trusted publisher
and protected release environment, and approve a new version before enabling
publication. This repository does not automatically upload to PyPI. Never store
publishing credentials in source or re-upload a used version.
