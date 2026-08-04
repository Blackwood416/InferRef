"""Bounded traversal of adapter-produced artifacts."""

from __future__ import annotations

import stat
import time
from dataclasses import dataclass
from pathlib import Path

_DEADLINE_CHECK_ENTRIES = 64


@dataclass(frozen=True)
class ArtifactScan:
    total_bytes: int = 0
    files: int = 0
    entries: int = 0
    limit: str | None = None


def scan_artifacts(root: Path, *, deadline: float, max_bytes: int, max_files: int) -> ArtifactScan:
    total = files = entries_seen = 0
    max_entries = max(1_024, max_files * 2)
    pending = [root]
    visited: set[tuple[int, int]] = set()
    while pending:
        if time.monotonic() >= deadline:
            return ArtifactScan(total, files, entries_seen, "deadline")
        directory = pending.pop()
        try:
            directory_stat = directory.stat()
        except OSError:
            continue
        identity = (directory_stat.st_dev, directory_stat.st_ino)
        if identity in visited:
            continue
        visited.add(identity)
        try:
            entries = directory.iterdir()
            for entry in entries:
                entries_seen += 1
                if entries_seen > max_entries:
                    return ArtifactScan(total, files, entries_seen, "entries")
                if entries_seen % _DEADLINE_CHECK_ENTRIES == 0 and time.monotonic() >= deadline:
                    return ArtifactScan(total, files, entries_seen, "deadline")
                try:
                    entry_stat = entry.lstat()
                    attrs = getattr(entry_stat, "st_file_attributes", 0)
                    reparse = bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
                    if stat.S_ISLNK(entry_stat.st_mode) or reparse:
                        files += 1
                        total += entry_stat.st_size
                    elif stat.S_ISDIR(entry_stat.st_mode):
                        pending.append(entry)
                    else:
                        files += 1
                        total += entry_stat.st_size
                    if files > max_files:
                        return ArtifactScan(total, files, entries_seen, "files")
                    if total > max_bytes:
                        return ArtifactScan(total, files, entries_seen, "bytes")
                except OSError:
                    continue
        except OSError:
            continue
    return ArtifactScan(total, files, entries_seen)
