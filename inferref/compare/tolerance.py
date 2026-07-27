"""Per-dtype tolerance policy (SPEC §27).

::

    {
      "float16": {"atol": 0.002, "rtol": 0.002},
      "float32": {"atol": 1e-5,  "rtol": 1e-5}
    }

Defaults are scaled to each dtype's machine epsilon rather than being uniform,
because a tolerance that is sane for float32 is meaningless for bfloat16 (which
has 8 mantissa bits).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Default (atol, rtol) per stable dtype name.
DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    "float64": (1e-9, 1e-9),
    "float32": (1e-5, 1e-5),
    "float16": (2e-3, 2e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
    "complex64": (1e-5, 1e-5),
    "complex128": (1e-9, 1e-9),
}

#: Integer and boolean tensors must match exactly.
_EXACT = (0.0, 0.0)


@dataclass
class TolerancePolicy:
    """Resolves the ``(atol, rtol)`` to use for a given dtype."""

    per_dtype: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_TOLERANCES)
    )
    #: Set by ``--atol`` / ``--rtol``; overrides every per-dtype entry.
    override_atol: float | None = None
    override_rtol: float | None = None

    def for_dtype(self, dtype: str) -> tuple[float, float]:
        atol, rtol = self.per_dtype.get(dtype, _EXACT)
        if self.override_atol is not None:
            atol = self.override_atol
        if self.override_rtol is not None:
            rtol = self.override_rtol
        return atol, rtol

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            name: {"atol": atol, "rtol": rtol}
            for name, (atol, rtol) in sorted(self.per_dtype.items())
        }
        if self.override_atol is not None:
            out["_override_atol"] = self.override_atol
        if self.override_rtol is not None:
            out["_override_rtol"] = self.override_rtol
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TolerancePolicy":
        per_dtype = dict(DEFAULT_TOLERANCES)
        for name, entry in data.items():
            if name.startswith("_"):
                continue
            per_dtype[name] = (
                float(entry.get("atol", 0.0)),
                float(entry.get("rtol", 0.0)),
            )
        return cls(per_dtype=per_dtype)

    @classmethod
    def load(cls, path: str | Path) -> "TolerancePolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
