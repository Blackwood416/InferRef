"""Content-addressing helpers for Alignment sessions.

The alignment format deliberately hashes *content* rather than filesystem
metadata.  This module is kept independent from the rest of the alignment
implementation so callers such as the Agent and the CLI can use the same
digest rules without importing the session store.

Two details are important here:

* tree hashes include a canonical, sorted manifest and the bytes of every
  regular file; symlinks and special files are rejected instead of silently
  following them;
* adapter hashes are made from the validated public adapter definition.  The
  values of configured environment variables are never part of that
  definition (only their names are retained).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


class HashingError(ValueError):
    """Raised when a path cannot be hashed safely."""


def canonical_json(value: Any) -> bytes:
    """Encode JSON using the alignment canonical representation.

    ``allow_nan=False`` matters: NaN and infinities do not have a portable
    JSON meaning and accepting them would allow two implementations to hash
    the same logical record differently.
    """

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HashingError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 digest of :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value)).hexdigest()


# Friendly aliases used by integrations which predate the explicit
# ``canonical_json`` name.
canonical_sha256 = canonical_json_sha256
hash_json = canonical_json_sha256


def sha256_bytes(data: bytes | bytearray | memoryview) -> str:
    """Hash a byte sequence."""

    return hashlib.sha256(bytes(data)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without following a symlink."""

    candidate = Path(path)
    try:
        stat = candidate.lstat()
    except OSError as exc:
        raise HashingError(f"cannot stat file {candidate}: {exc}") from exc
    if not _is_regular_stat(stat):
        if _is_symlink_stat(stat):
            raise HashingError(f"refusing to hash symlink: {candidate}")
        raise HashingError(f"expected a regular file: {candidate}")

    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HashingError(f"cannot read file {candidate}: {exc}") from exc
    return digest.hexdigest()


def _is_symlink_stat(stat: os.stat_result) -> bool:
    # ``stat.S_ISLNK`` is intentionally avoided here so this module remains
    # tiny; mode's file-type bits are stable on all supported platforms.
    return (stat.st_mode & 0o170000) == 0o120000


def _is_regular_stat(stat: os.stat_result) -> bool:
    return (stat.st_mode & 0o170000) == 0o100000


def _safe_tree_root(path: str | Path) -> Path:
    candidate = Path(path)
    try:
        stat = candidate.lstat()
    except OSError as exc:
        raise HashingError(f"cannot stat tree {candidate}: {exc}") from exc
    if _is_symlink_stat(stat):
        raise HashingError(f"refusing to hash symlink tree root: {candidate}")
    if not (candidate.is_dir() or _is_regular_stat(stat)):
        raise HashingError(f"tree root is not a file or directory: {candidate}")
    return candidate.resolve()


def tree_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Return a deterministic manifest for a file or directory tree.

    Directory entries are included so an empty directory and a missing
    directory cannot accidentally share a digest.  Paths are always POSIX
    relative names; a single file uses its basename as the relative name.
    Symlinks, sockets, devices, and other special files are rejected.  A
    symlink encountered below a directory is rejected even if it points back
    inside the tree.
    """

    root = _safe_tree_root(path)
    try:
        root_stat = root.lstat()
    except OSError as exc:  # pragma: no cover - root was checked above
        raise HashingError(f"cannot stat tree {root}: {exc}") from exc

    if _is_regular_stat(root_stat):
        return [
            {
                "path": root.name,
                "kind": "file",
                "size": root_stat.st_size,
                "sha256": sha256_file(root),
            }
        ]

    entries: list[dict[str, Any]] = []

    def visit(directory: Path, relative: PurePosixPath) -> None:
        try:
            with os.scandir(directory) as scanner:
                children = sorted(scanner, key=lambda item: item.name)
        except OSError as exc:
            raise HashingError(f"cannot enumerate tree {directory}: {exc}") from exc
        for item in children:
                # lstat is essential: Path.is_dir() follows links.
                try:
                    stat = item.stat(follow_symlinks=False)
                except OSError as exc:
                    raise HashingError(f"cannot stat tree entry {item.path}: {exc}") from exc
                item_relative = relative / item.name
                relative_text = item_relative.as_posix()
                if _is_symlink_stat(stat):
                    raise HashingError(f"refusing to hash symlink: {item.path}")
                if (stat.st_mode & 0o170000) == 0o040000:
                    entries.append({"path": relative_text, "kind": "directory"})
                    visit(Path(item.path), item_relative)
                elif _is_regular_stat(stat):
                    entries.append(
                        {
                            "path": relative_text,
                            "kind": "file",
                            "size": stat.st_size,
                            "sha256": sha256_file(Path(item.path)),
                        }
                    )
                else:
                    raise HashingError(f"refusing to hash special file: {item.path}")

    visit(root, PurePosixPath("."))
    # os.scandir traversal is already sorted, but sorting the final list also
    # makes this invariant explicit and protects future traversal changes.
    return sorted(entries, key=lambda entry: (entry["path"], entry["kind"]))


def tree_sha256(path: str | Path) -> str:
    """Return a deterministic content digest for a file or directory tree."""

    return canonical_json_sha256(tree_manifest(path))


def hash_tree(path: str | Path) -> str:
    """Alias for :func:`tree_sha256`."""

    return tree_sha256(path)


def canonical_tree_sha256(path: str | Path) -> str:
    """Explicitly named alias used by the Alignment specification."""

    return tree_sha256(path)


def trace_metadata_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Return the content manifest for Trace IR metadata files.

    Payloads under ``tensors/`` are deliberately excluded.  A trace metadata
    digest describes the graph/region contract; testcase payloads are linked
    separately by their testcase tree digest.  Missing optional metadata files
    are not synthesized, but the two required IR files are required to exist.
    """

    root = _safe_tree_root(path)
    if not root.is_dir():
        raise HashingError(f"trace metadata root is not a directory: {root}")
    names = (
        "manifest.json",
        "graph.json",
        "modules.json",
        "sources.json",
        "regions.json",
        "storages.json",
    )
    entries: list[dict[str, Any]] = []
    for name in names:
        candidate = root / name
        try:
            stat = candidate.lstat()
        except FileNotFoundError:
            if name in {"manifest.json", "graph.json"}:
                raise HashingError(f"missing required trace metadata file: {name}")
            continue
        except OSError as exc:
            raise HashingError(f"cannot stat trace metadata {candidate}: {exc}") from exc
        if _is_symlink_stat(stat):
            raise HashingError(f"refusing to hash symlink trace metadata: {candidate}")
        if not _is_regular_stat(stat):
            raise HashingError(f"trace metadata is not a regular file: {candidate}")
        entries.append(
            {
                "path": name,
                "kind": "file",
                "size": stat.st_size,
                "sha256": sha256_file(candidate),
            }
        )
    return entries


def trace_metadata_sha256(path: str | Path) -> str:
    """Hash the graph/manifest metadata of one Trace IR package."""

    return canonical_json_sha256(trace_metadata_manifest(path))


def _adapter_public_record(adapter: Any) -> Mapping[str, Any]:
    """Get a validated/public adapter mapping without retaining secrets."""

    if isinstance(adapter, (str, Path)):
        source = Path(adapter)
        try:
            from inferref.agent.protocol import EngineAdapter

            return EngineAdapter.load(source).to_dict()
        except Exception:
            try:
                value = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise HashingError(f"cannot read adapter definition {source}: {exc}") from exc
    elif hasattr(adapter, "to_dict") and callable(adapter.to_dict):
        value = adapter.to_dict()
    else:
        value = adapter
    if not isinstance(value, Mapping):
        raise HashingError("adapter definition must be a JSON object")
    return value


def normalize_adapter(adapter: Any) -> dict[str, Any]:
    """Return the redacted, normalized Adapter v0.2 definition.

    The adapter loader's ``to_dict`` already redacts environment values, but
    this function also accepts raw JSON so hashing can happen before the
    adapter object is constructed.  Unknown fields are intentionally ignored:
    they are not part of the executable Adapter ABI and may contain secrets.
    The source filename is likewise excluded because it is a reference, not a
    definition.
    """

    raw = _adapter_public_record(adapter)
    known = (
        "format",
        "format_version",
        "name",
        "command",
        "timeout_seconds",
        "cwd",
        "max_output_chars",
        "max_artifact_bytes",
        "max_artifact_files",
        "target_device",
        "capabilities",
    )
    normalized: dict[str, Any] = {
        key: raw[key]
        for key in known
        if key in raw and key != "source"
    }

    # Prefer keys from the raw environment map.  For an EngineAdapter object
    # only ``environment_keys`` is exposed by ``to_dict``.
    environment = raw.get("environment")
    if isinstance(environment, Mapping):
        if not all(isinstance(key, str) for key in environment):
            raise HashingError("adapter environment keys must be strings")
        environment_keys = sorted(environment)
    else:
        keys = raw.get("environment_keys", [])
        if keys is None:
            keys = []
        if not isinstance(keys, (list, tuple)) or not all(
            isinstance(key, str) for key in keys
        ):
            raise HashingError("adapter environment_keys must be a string array")
        environment_keys = sorted(set(keys))
    normalized["environment_keys"] = environment_keys

    # Normalise harmless JSON shape differences.  Do not coerce command
    # arguments or capability values: the adapter validator owns their exact
    # ABI semantics and a digest must change when they change.
    command = normalized.get("command")
    if isinstance(command, tuple):
        normalized["command"] = list(command)
    elif command is not None and not isinstance(command, list):
        raise HashingError("adapter command must be a string array")
    return normalized


def normalized_adapter_sha256(adapter: Any) -> str:
    """Hash the redacted normalized adapter definition."""

    return canonical_json_sha256(normalize_adapter(adapter))


def adapter_normalized_sha256(adapter: Any) -> str:
    """Alias for :func:`normalized_adapter_sha256`."""

    return normalized_adapter_sha256(adapter)


def hash_adapter(adapter: Any) -> str:
    """Alias used by callers that do not need to spell out normalization."""

    return normalized_adapter_sha256(adapter)


def hash_path(path: str | Path) -> str:
    """Hash a regular file or a directory tree."""

    candidate = Path(path)
    try:
        stat = candidate.lstat()
    except OSError as exc:
        raise HashingError(f"cannot stat path {candidate}: {exc}") from exc
    if _is_symlink_stat(stat):
        raise HashingError(f"refusing to hash symlink: {candidate}")
    if _is_regular_stat(stat):
        return sha256_file(candidate)
    if (stat.st_mode & 0o170000) == 0o040000:
        return tree_sha256(candidate)
    raise HashingError(f"path is not a regular file or directory: {candidate}")


__all__ = [
    "HashingError",
    "adapter_normalized_sha256",
    "canonical_json",
    "canonical_json_sha256",
    "canonical_sha256",
    "canonical_tree_sha256",
    "hash_adapter",
    "hash_json",
    "hash_path",
    "hash_tree",
    "normalize_adapter",
    "normalized_adapter_sha256",
    "sha256_bytes",
    "sha256_file",
    "trace_metadata_manifest",
    "trace_metadata_sha256",
    "tree_manifest",
    "tree_sha256",
]
