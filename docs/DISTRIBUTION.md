# Package distribution

InferRef is not currently published to PyPI (checked 2026-09-17). Until a
maintainer establishes ownership and a publishing policy, install from a tag
without cloning manually:

```bash
python -m pip install "inferref @ git+https://github.com/Blackwood416/InferRef.git@v0.10.0"
```

This requires Git. The v0.10.0 source tag is immutable and does **not** include
later Unreleased fixes. For those fixes, use the reviewed working tree or a
subsequently released version. Do not rebuild changed code as a replacement
v0.10.0 release asset.

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

## GitHub release assets

The package workflow builds/tests distributions on main pushes, pull requests,
manual dispatch, and new published releases. CI artifacts contain the wheel and
sdist. For a published release, a separate write-scoped job checks that the tag
matches the embedded version and the tagged commit's required CircleCI checks
are green before attaching the pair. Existing assets are not overwritten.

Before releasing: update both source version declarations, move Unreleased
changes to the new changelog entry, run core/frontend/slow and C++ gates, push
the commit, wait for branch CI, push its version tag, wait for tag qualification,
and publish the GitHub Release. Releasing a new version is a separate operation
from merging these QA fixes. The workflow is not retroactively triggered for
the already published v0.10.0 release.

## PyPI remains an explicit maintainer step

Verify PyPI project-name ownership, configure credentials or a trusted publisher
and protected release environment, test on TestPyPI, and approve a new version
before enabling publication. This repository does not automatically upload to
PyPI. Never store publishing credentials in source or re-upload a used version.
