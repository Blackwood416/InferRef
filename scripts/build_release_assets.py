"""Build the non-Python GitHub Release assets for InferRef.

The PyPI distribution contains the Python package, CLI and MCP entry points.
The C++ header-only SDK and the Agent skill are separate release assets so
``pip install inferref`` stays small and does not pretend to install C++.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from contextlib import contextmanager
from pathlib import Path

PACKAGE_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
SKIP_DIRECTORY_NAMES = {"__pycache__", ".pytest_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def source_version(root: Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = PACKAGE_VERSION_RE.search(text)
    if not match:
        raise SystemExit("could not read project version from pyproject.toml")
    return match.group(1)


def _add_file(tar: tarfile.TarFile, source: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(source), arcname)
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = 0
    info.mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
    with source.open("rb") as handle:
        tar.addfile(info, handle)


def _add_bytes(
    tar: tarfile.TarFile, data: bytes, arcname: str, mode: int = 0o644
) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(data)
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = 0
    info.mode = mode
    tar.addfile(info, fileobj=io.BytesIO(data))


def _add_tree(tar: tarfile.TarFile, source: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(source), arcname + "/")
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mtime = 0
    info.mode = 0o755
    tar.addfile(info)
    for child in sorted(source.iterdir(), key=lambda item: item.name):
        if child.name in SKIP_DIRECTORY_NAMES or child.suffix in SKIP_SUFFIXES:
            continue
        child_arcname = f"{arcname}/{child.name}"
        if child.is_dir():
            _add_tree(tar, child, child_arcname)
        elif child.is_file():
            _add_file(tar, child, child_arcname)


@contextmanager
def _open_archive(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
    ):
        yield tar


def _build_cpp(root: Path, version: str, output: Path) -> Path:
    asset = output / f"inferref-cpp-{version}.tar.gz"
    readme = (
        f"# InferRef C++ SDK {version}\n\n"
        "Header-only C++17 readers and comparators for InferRef artifacts.\n\n"
        "Build the bundled examples and self-tests with:\n\n"
        "```bash\n"
        f"cmake -S inferref-cpp-{version}/cpp -B build\n"
        "cmake --build build --config Release\n"
        "ctest --test-dir build -C Release --output-on-failure\n"
        "```\n\n"
        "The public headers are under `cpp/include/inferref/`; the CMake target\n"
        "`inferref_headers` exposes them as an INTERFACE library. The committed\n"
        "`tests/fixtures/cpp_quantization_goldens/` files are included so the\n"
        "quantization and bridge self-tests remain runnable from this archive.\n"
    ).encode()
    with _open_archive(asset) as tar:
        tar_root = f"inferref-cpp-{version}"
        _add_bytes(tar, readme, f"{tar_root}/README.md")
        _add_file(tar, root / "LICENSE", f"{tar_root}/LICENSE")
        _add_file(tar, root / "cpp/CMakeLists.txt", f"{tar_root}/cpp/CMakeLists.txt")
        _add_tree(tar, root / "cpp/include", f"{tar_root}/cpp/include")
        _add_tree(tar, root / "cpp/examples", f"{tar_root}/cpp/examples")
        _add_tree(
            tar,
            root / "tests/fixtures/cpp_quantization_goldens",
            f"{tar_root}/tests/fixtures/cpp_quantization_goldens",
        )
    return asset


def _build_skill(root: Path, version: str, output: Path) -> Path:
    asset = output / f"inferref-skill-{version}.tar.gz"
    with _open_archive(asset) as tar:
        _add_tree(tar, root / "skills/inferref", "inferref")
        _add_file(tar, root / "LICENSE", "inferref/LICENSE")
    return asset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(".scratch/release-assets"))
    parser.add_argument("--version")
    parser.add_argument(
        "--extra-artifact",
        action="append",
        type=Path,
        default=[],
        help="additional artifact to include in SHA256SUMS (repeatable or glob-expanded)",
    )
    args = parser.parse_args()

    root = args.source.resolve()
    version = args.version or source_version(root)
    expected = source_version(root)
    if version != expected:
        raise SystemExit(
            f"release asset version {version!r} does not match pyproject {expected!r}"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    assets = [
        _build_cpp(root, version, output),
        _build_skill(root, version, output),
    ]
    checksum_assets = [*assets, *[path.resolve() for path in args.extra_artifact]]
    checksum_lines = []
    for path in checksum_assets:
        if not path.is_file():
            raise SystemExit(f"checksum artifact does not exist: {path}")
        checksum_lines.append(f"{_sha256(path)}  {path.name}")
    checksum_file = output / "SHA256SUMS"
    checksum_file.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "version": version,
                "assets": [str(path.name) for path in assets],
                "checksums": str(checksum_file),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
