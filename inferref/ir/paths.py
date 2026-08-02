"""Cross-platform path containment for every InferRef artifact boundary."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath


class PathBoundaryError(ValueError):
    """Raised when an artifact path escapes its configured root."""


def resolve_contained_path(
    root: str | Path,
    relative: str | Path,
    *,
    kind: str = "artifact path",
) -> Path:
    """Resolve a relative path and require it to remain below ``root``.

    Absolute syntax is rejected for both POSIX and Windows even when validation
    runs on the other platform. ``Path.resolve`` follows existing symlinks and
    junctions, so an in-root link to an external target is rejected too.
    """

    text = str(relative)
    if not text:
        raise PathBoundaryError(f"{kind} must not be empty")
    posix_path = PurePosixPath(text)
    windows_path = PureWindowsPath(text)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
    ):
        raise PathBoundaryError(f"{kind} must be relative: {text}")
    if ".." in posix_path.parts or ".." in windows_path.parts:
        raise PathBoundaryError(f"{kind} escapes root via parent traversal: {text}")

    resolved_root = Path(root).resolve()
    resolved = (resolved_root / Path(text)).resolve()
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise PathBoundaryError(f"{kind} escapes root {resolved_root}: {text}")
    return resolved


def resolve_allowed_path(
    path: str | Path,
    allowed_roots: Iterable[str | Path],
    *,
    kind: str = "path",
) -> Path:
    """Resolve a host path and require it below one configured allowed root."""

    resolved = Path(path).resolve()
    roots = tuple(Path(root).resolve() for root in allowed_roots)
    if not roots:
        raise PathBoundaryError(f"no allowed roots are configured for {kind}")
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        rendered = ", ".join(str(root) for root in roots)
        raise PathBoundaryError(
            f"{kind} is outside configured allowed roots: {resolved} "
            f"(allowed: {rendered})"
        )
    return resolved


def validate_allowed_roots(
    roots: Iterable[str | Path], *, kind: str
) -> tuple[Path, ...]:
    """Resolve server roots once and reject missing/non-directory policy entries."""

    resolved: list[Path] = []
    for root in roots:
        candidate = Path(root).resolve()
        if not candidate.is_dir():
            raise PathBoundaryError(
                f"configured {kind} root is not a directory: {candidate}"
            )
        if candidate not in resolved:
            resolved.append(candidate)
    if not resolved:
        raise PathBoundaryError(f"configure at least one {kind} root")
    return tuple(resolved)
