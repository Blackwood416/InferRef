from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from inferref.agent.adapter import execute_adapter
from inferref.agent.protocol import EngineAdapter
from inferref.suite.schema import Suite, load_suite


def run_suite(
    suite: str | Path | Suite,
    adapter: str | Path | EngineAdapter,
    runs_dir: str | Path,
    *,
    allow_unsupported: bool = False,
) -> dict[str, Any]:
    loaded = load_suite(suite) if not isinstance(suite, Suite) else suite
    engine = EngineAdapter.load(adapter) if not isinstance(adapter, EngineAdapter) else adapter
    root = Path(runs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    for case in loaded.cases:
        result = execute_adapter(case.testcase, engine, root / case.id)
        accepted = result["status"] == "pass" or (
            allow_unsupported and result["status"] == "unsupported"
        )
        cases.append(
            {
                "id": case.id,
                "tags": list(case.tags),
                "status": result["status"],
                "accepted": accepted,
                "run": result,
            }
        )
    passed = all(case["accepted"] for case in cases)
    report = {
        "format": "inferref-suite-run",
        "format_version": "0.1",
        "status": "pass" if passed else "fail",
        "suite": loaded.to_dict(),
        "adapter": engine.to_dict(),
        "allow_unsupported": allow_unsupported,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "counts": {
            "total": len(cases),
            "pass": sum(case["status"] == "pass" for case in cases),
            "unsupported": sum(case["status"] == "unsupported" for case in cases),
            "failed": sum(not case["accepted"] for case in cases),
        },
        "cases": cases,
    }
    (root / "inferref-suite-run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return report
