"""Host-configured filesystem policy for InferRef's MCP transport."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from inferref.ir.paths import resolve_allowed_path, validate_allowed_roots


@dataclass(frozen=True)
class MCPPathPolicy:
    """Allowed read and write roots applied before an MCP tool touches disk."""

    read_roots: tuple[Path, ...]
    write_roots: tuple[Path, ...]

    @classmethod
    def create(
        cls,
        *,
        read_roots: Iterable[str | Path] | None = None,
        write_roots: Iterable[str | Path] | None = None,
    ) -> MCPPathPolicy:
        default = (Path.cwd(),)
        reads = validate_allowed_roots(read_roots or default, kind="MCP read")
        writes = validate_allowed_roots(write_roots or reads, kind="MCP write")
        return cls(read_roots=reads, write_roots=writes)

    def read(self, path: str | Path, *, kind: str) -> Path:
        return resolve_allowed_path(path, self.read_roots, kind=kind)

    def write(self, path: str | Path, *, kind: str) -> Path:
        return resolve_allowed_path(path, self.write_roots, kind=kind)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "read_roots": [str(path) for path in self.read_roots],
            "write_roots": [str(path) for path in self.write_roots],
        }
