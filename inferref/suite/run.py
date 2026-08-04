from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inferref.agent.adapter import execute_adapter
from inferref.agent.protocol import AgentProtocolError, EngineAdapter
from inferref.ir.paths import PathBoundaryError, resolve_contained_path
from inferref.suite.paths import artifact_key
from inferref.suite.schema import Suite, SuiteError, load_suite


AdapterInput = str | Path | EngineAdapter


def _load_adapters(adapter: AdapterInput | Sequence[AdapterInput]) -> tuple[EngineAdapter, ...]:
    raw = adapter if isinstance(adapter, Sequence) and not isinstance(adapter, (str, bytes, Path)) else (adapter,)
    loaded = tuple(item if isinstance(item, EngineAdapter) else EngineAdapter.load(item) for item in raw)
    if not loaded:
        raise ValueError("suite run requires at least one adapter")
    return loaded


def _adapter_ids(adapters: tuple[EngineAdapter, ...]) -> tuple[str, ...]:
    result: list[str] = []
    used: set[str] = set()
    for index, adapter in enumerate(adapters, 1):
        base = re.sub(r"[^a-zA-Z0-9_.-]+", "-", adapter.name).strip("-.") or f"adapter-{index}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}-{suffix}"
            suffix += 1
        used.add(candidate)
        result.append(candidate)
    return tuple(result)


def run_suite(
    suite: str | Path | Suite,
    adapter: AdapterInput | Sequence[AdapterInput],
    runs_dir: str | Path,
    *,
    allow_unsupported: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run each testcase against one or more adapters and write a matrix report."""

    loaded = load_suite(suite) if not isinstance(suite, Suite) else suite
    engines = _load_adapters(adapter)
    adapter_ids = _adapter_ids(engines)
    root = Path(runs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []

    for case in loaded.cases:
        results: list[dict[str, Any]] = []
        for adapter_id, engine in zip(adapter_ids, engines, strict=True):
            try:
                adapter_root = resolve_contained_path(
                    root,
                    artifact_key(adapter_id, fallback="adapter"),
                    kind="suite adapter output",
                )
                case_root = resolve_contained_path(
                    adapter_root,
                    artifact_key(case.id, fallback="case"),
                    kind=f"suite case {case.id!r} output",
                )
            except PathBoundaryError as exc:  # defensive: keys are single components
                raise SuiteError(str(exc)) from exc
            try:
                result = execute_adapter(case.testcase, engine, case_root)
            except (AgentProtocolError, OSError, ValueError) as exc:
                if fail_fast:
                    raise
                result = {
                    "run_id": None,
                    "adapter": engine.to_dict(),
                    "testcase": str(case.testcase),
                    "requirements": None,
                    "capability_status": "error",
                    "status": "infrastructure_error",
                    "execution": None,
                    "comparison": None,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            accepted = result["status"] == "pass" or (
                allow_unsupported and result["status"] == "unsupported"
            )
            cell = {
                "adapter_id": adapter_id,
                "adapter_name": engine.name,
                "status": result["status"],
                "accepted": accepted,
                "run": result,
            }
            results.append(cell)
            cells.append(cell)

        case_accepted = all(item["accepted"] for item in results)
        case_status = _aggregate_status(results, accepted=case_accepted)
        case_record: dict[str, Any] = {
            "id": case.id,
            "tags": list(case.tags),
            "status": case_status,
            "accepted": case_accepted,
            "results": results,
        }
        # Preserve the 0.1 single-adapter convenience shape for API callers.
        if len(results) == 1:
            case_record["status"] = results[0]["status"]
            case_record["run"] = results[0]["run"]
        cases.append(case_record)

    accepted = all(cell["accepted"] for cell in cells)
    status = _aggregate_status(cells, accepted=accepted)
    adapters = [
        {"id": adapter_id, "name": engine.name, "adapter": engine.to_dict()}
        for adapter_id, engine in zip(adapter_ids, engines, strict=True)
    ]
    report: dict[str, Any] = {
        "format": "inferref-suite-run",
        "format_version": "0.2",
        "status": status,
        "accepted": accepted,
        "exit_code_policy_satisfied": accepted,
        "suite": loaded.to_dict(),
        "adapters": adapters,
        "allow_unsupported": allow_unsupported,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "counts": {
            "total": len(cells),
            "pass": sum(cell["status"] == "pass" for cell in cells),
            "unsupported": sum(cell["status"] == "unsupported" for cell in cells),
            "failed": sum(not cell["accepted"] for cell in cells),
        },
        "cases": cases,
    }
    if len(engines) == 1:
        report["adapter"] = engines[0].to_dict()
    (root / "inferref-suite-run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def _aggregate_status(cells: Sequence[dict[str, Any]], *, accepted: bool) -> str:
    if not accepted:
        return "fail"
    statuses = [str(cell["status"]) for cell in cells]
    passed = sum(status == "pass" for status in statuses)
    unsupported = sum(status == "unsupported" for status in statuses)
    if passed == len(statuses):
        return "pass"
    if unsupported == len(statuses):
        return "unsupported"
    if passed + unsupported == len(statuses):
        return "partial"
    return "fail"
