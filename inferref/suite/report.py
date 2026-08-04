from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _load_run(run: str | Path | dict[str, Any]) -> dict[str, Any]:
    data = run if isinstance(run, dict) else json.loads(Path(run).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("format") != "inferref-suite-run":
        raise ValueError("not an InferRef suite run")
    if data.get("format_version") not in {"0.1", "0.2"}:
        raise ValueError(f"unsupported suite run version {data.get('format_version')!r}")
    return data


def _cells(data: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    if data.get("format_version") == "0.1":
        adapter = data.get("adapter", {})
        adapters = [{"id": "adapter", "name": str(adapter.get("name", "adapter"))}]
        rows = []
        for case in data.get("cases", []):
            rows.append({"case": case, "results": [{"adapter_id": "adapter", "status": case.get("status"), "run": case.get("run", {})}]})
        return adapters, rows
    adapters = [{"id": item["id"], "name": item["name"]} for item in data.get("adapters", [])]
    return adapters, [{"case": case, "results": case.get("results", [])} for case in data.get("cases", [])]


def _cell_summary(result: dict[str, Any]) -> dict[str, Any]:
    run = result.get("run") or {}
    comparison = run.get("comparison") or {}
    comparisons = comparison.get("comparisons") or []
    max_error = max(
        (float(item.get("metrics", {}).get("max_abs_error", 0.0)) for item in comparisons),
        default=None,
    )
    first = comparison.get("first_failure") or {}
    mismatch = first.get("metrics", {}).get("first_mismatch")
    execution = run.get("execution") or {}
    return {
        "status": result.get("status", "unknown"),
        "max_abs_error": max_error,
        "first_divergence": mismatch,
        "duration_ms": execution.get("duration_ms"),
        "run_id": run.get("run_id"),
        "unsupported_reasons": run.get("unsupported", []),
    }


def render_suite_report(
    run: str | Path | dict[str, Any], output: str | Path
) -> dict[str, Any]:
    """Write a self-contained HTML matrix and a machine-readable JSON sibling."""

    data = _load_run(run)
    adapters, rows = _cells(data)
    matrix: list[dict[str, Any]] = []
    for row in rows:
        by_adapter = {item.get("adapter_id"): _cell_summary(item) for item in row["results"]}
        matrix.append({
            "id": row["case"].get("id"),
            "tags": row["case"].get("tags", []),
            "engines": by_adapter,
        })
    report = {
        "format": "inferref-suite-report",
        "format_version": "0.1",
        "status": data.get("status"),
        "accepted": data.get("accepted", data.get("status") == "pass"),
        "suite": data.get("suite", {}).get("name"),
        "adapters": adapters,
        "counts": data.get("counts", {}),
        "matrix": matrix,
    }

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    headings = "".join(
        f"<th>{html.escape(item['name'])}"
        + (f"<small>{html.escape(item['id'])}</small>" if item["id"] != item["name"] else "")
        + "</th>"
        for item in adapters
    )
    body: list[str] = []
    for row in matrix:
        cells = []
        for adapter in adapters:
            cell = row["engines"].get(adapter["id"], {"status": "missing"})
            status = str(cell["status"])
            details = []
            if cell.get("max_abs_error") is not None:
                details.append(f"max |error| {cell['max_abs_error']:.6g}")
            if cell.get("duration_ms") is not None:
                details.append(f"{cell['duration_ms']:.1f} ms")
            if cell.get("first_divergence"):
                details.append("first divergence " + html.escape(json.dumps(cell["first_divergence"], separators=(",", ":"))))
            if cell.get("unsupported_reasons"):
                details.extend(html.escape(str(reason)) for reason in cell["unsupported_reasons"])
            cells.append(f'<td class="{html.escape(status)}"><strong>{html.escape(status.upper())}</strong><small>{"<br>".join(details)}</small></td>')
        body.append(f"<tr><th>{html.escape(str(row['id']))}<small>{html.escape(', '.join(row['tags']))}</small></th>{''.join(cells)}</tr>")

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>InferRef suite report — {html.escape(str(report['suite']))}</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;color:#17202a;background:#f7f8fa}}h1{{margin-bottom:.25rem}}p{{color:#566573}}table{{border-collapse:collapse;width:100%;background:white;box-shadow:0 1px 4px #ccd}}th,td{{padding:.75rem;border:1px solid #d5d8dc;text-align:left;vertical-align:top}}thead th{{background:#273746;color:white}}td.pass{{background:#eafaf1}}td.fail,td.mismatch,td.error,td.infrastructure_error{{background:#fdedec}}td.unsupported{{background:#fef9e7}}small{{display:block;color:#566573;margin-top:.3rem;font-weight:400}}</style></head>
<body><h1>{html.escape(str(report['suite']))}</h1><p>Status: <strong>{html.escape(str(report['status']).upper())}</strong> · {report['counts'].get('pass', 0)}/{report['counts'].get('total', 0)} cells passed</p>
<table><thead><tr><th>Case</th>{headings}</tr></thead><tbody>{''.join(body)}</tbody></table></body></html>"""
    output_path.write_text(document, encoding="utf-8")
    report["html"] = str(output_path)
    report["json"] = str(json_path)
    return report
