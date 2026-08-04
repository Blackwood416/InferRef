from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_record(output_path: Path, result: dict[str, Any]) -> None:
    (output_path / "inferref-run.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
