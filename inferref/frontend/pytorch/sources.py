"""Source mapping (IR §26, §27; SPEC §16, §58).

Runtime operators are mapped back to the Python source that issued them by
walking the interpreter stack and dropping frames belonging to PyTorch itself
and to InferRef. What remains is model code — for Hugging Face models that is
``modeling_*.py``, which is exactly the location an engine developer wants.

Records are deduplicated by stack signature: a model typically issues thousands
of operators from a few hundred distinct source locations.
"""

from __future__ import annotations

import linecache
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from inferref.ir.source import SourceFrame, SourceRecord

#: Directories whose frames are never interesting as a source mapping.
_FRAMEWORK_ROOTS: tuple[str, ...] = ()


def _framework_roots() -> tuple[str, ...]:
    global _FRAMEWORK_ROOTS
    if _FRAMEWORK_ROOTS:
        return _FRAMEWORK_ROOTS
    roots: list[str] = []
    try:
        import torch

        roots.append(os.path.dirname(os.path.abspath(torch.__file__)))
    except Exception:  # pragma: no cover - torch always present in this frontend
        pass
    import inferref

    roots.append(os.path.dirname(os.path.abspath(inferref.__file__)))
    _FRAMEWORK_ROOTS = tuple(roots)
    return _FRAMEWORK_ROOTS


def _is_framework_frame(filename: str) -> bool:
    if not filename or filename.startswith("<"):
        return True
    absolute = os.path.abspath(filename)
    if any(absolute.startswith(root + os.sep) for root in _framework_roots()):
        return True
    # Import machinery and stdlib runpy noise.
    return "importlib" in absolute and "_bootstrap" in absolute


@dataclass
class SourceMapper:
    """Captures and deduplicates source mappings."""

    path_mode: str = "relative"
    embed_source_text: bool = False
    max_frames: int = 32
    base_dir: Path = field(default_factory=Path.cwd)

    _records: dict[tuple, SourceRecord] = field(default_factory=dict)
    _next_id: int = 1

    def capture(self, skip: int = 0) -> int | None:
        """Capture the current model-level stack; returns a source record id.

        ``skip`` drops that many innermost frames before filtering, letting the
        caller exclude its own plumbing.
        """
        raw = traceback.extract_stack()
        if skip:
            raw = raw[: len(raw) - skip]

        frames: list[SourceFrame] = []
        # Innermost first — the location a human reads is the deepest model frame.
        for summary in reversed(raw):
            if _is_framework_frame(summary.filename):
                continue
            frames.append(
                SourceFrame(
                    file=self._format_path(summary.filename),
                    line=summary.lineno or 0,
                    function=summary.name,
                    source_text=self._source_text(summary.filename, summary.lineno),
                )
            )
            if len(frames) >= self.max_frames:
                break

        if not frames:
            return None

        key = tuple((f.file, f.line, f.function) for f in frames)
        existing = self._records.get(key)
        if existing is not None:
            return existing.id

        record = SourceRecord(id=self._next_id, primary=frames[0], stack=tuple(frames))
        self._next_id += 1
        self._records[key] = record
        return record.id

    def _format_path(self, filename: str) -> str:
        if self.path_mode == "redacted":
            return os.path.basename(filename)
        absolute = os.path.abspath(filename)
        if self.path_mode == "absolute":
            return absolute
        try:
            return os.path.relpath(absolute, self.base_dir).replace(os.sep, "/")
        except ValueError:
            # Different drive on Windows — fall back to the absolute path.
            return absolute.replace(os.sep, "/")

    def _source_text(self, filename: str, lineno: int | None) -> str | None:
        if not self.embed_source_text or not lineno:
            return None
        text = linecache.getline(filename, lineno)
        return text.strip() or None

    def records(self) -> list[SourceRecord]:
        return sorted(self._records.values(), key=lambda r: r.id)


def python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
