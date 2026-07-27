"""Source mapping records (IR §26, §27; SPEC §16).

Source records are deduplicated: many operators typically share one source
location. Embedding source text is optional and off by default to keep traces
small and avoid embedding model source into artifacts (SPEC §58).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferref.ir._common import Record, drop_none

#: Source path handling modes (IR §27).
PATH_MODES = ("absolute", "relative", "redacted")


@dataclass(frozen=True)
class SourceFrame:
    """One Python stack frame (IR §26)."""

    file: str
    line: int
    function: str
    source_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return drop_none(
            {
                "file": self.file,
                "line": self.line,
                "function": self.function,
                "source_text": self.source_text,
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceFrame":
        return cls(
            file=data.get("file", "<unknown>"),
            line=int(data.get("line", 0)),
            function=data.get("function", "<unknown>"),
            source_text=data.get("source_text"),
        )

    def __str__(self) -> str:
        return f"{self.file}:{self.line} in {self.function}"


@dataclass
class SourceRecord(Record):
    """A deduplicated source mapping (IR §26)."""

    id: int = -1
    #: Innermost non-framework frame — the location a human wants to see.
    primary: SourceFrame | None = None
    #: Full filtered stack, innermost first.
    stack: tuple[SourceFrame, ...] = ()
    source_text: str | None = None

    _KNOWN = ("id", "primary", "stack", "source_text")

    def _encode(self) -> dict[str, Any]:
        return drop_none(
            {
                "id": self.id,
                "primary": self.primary.to_dict() if self.primary else None,
                "stack": [f.to_dict() for f in self.stack],
                "source_text": self.source_text,
            }
        )

    @classmethod
    def _decode(cls, data: dict[str, Any]) -> dict[str, Any]:
        primary = data.get("primary")
        return {
            "id": int(data["id"]),
            "primary": SourceFrame.from_dict(primary) if primary else None,
            "stack": tuple(SourceFrame.from_dict(f) for f in data.get("stack", ())),
            "source_text": data.get("source_text"),
        }

    def key(self) -> tuple[Any, ...]:
        """Dedup key over the full stack."""
        return tuple((f.file, f.line, f.function) for f in self.stack)

    def __str__(self) -> str:
        return str(self.primary) if self.primary else "<no source>"
