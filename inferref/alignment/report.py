"""Report generation (Markdown and JSON sidecar) for Alignment Session v0.1 (L6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from inferref.alignment.classification import classify_failure_domain
from inferref.alignment.matrix import build_fixture_matrix
from inferref.alignment.narrowing import compute_narrowing
from inferref.alignment.store import AlignmentStore


def generate_alignment_report(
    store: AlignmentStore,
    *,
    markdown_path: Path | None = None,
    json_sidecar_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate Markdown alignment report and structured JSON sidecar."""
    snapshot = store.snapshot
    matrix = build_fixture_matrix(
        list(store.evidence.values()),
        snapshot,
        known_boundaries=list(snapshot.boundaries.keys()),
        events=store.events,
    )

    evidence_by_b: dict[str, list[Any]] = {}
    for ev in matrix.current_evidence.values():
        if ev.type == "numerical_comparison":
            evidence_by_b.setdefault(ev.subject["boundary"], []).append(ev)

    from inferref.alignment.boundaries import inventory_boundaries
    from inferref.ir.package import TracePackage

    trace_path = Path(snapshot.source["trace"]["path"]).resolve()
    pkg = TracePackage.load(trace_path)
    inventory = inventory_boundaries(pkg)

    narrowing = compute_narrowing(inventory, evidence_by_b)
    candidates: list[Any] = []
    if narrowing.smallest_confirmed_failure:
        notes = [ev for ev in store.evidence.values() if ev.type == "manual_note"]
        candidates, _ = classify_failure_domain(
            narrowing.smallest_confirmed_failure,
            list(matrix.current_evidence.values()),
            notes=notes,
            active_engine_identity=matrix.active_engine_identity,
            events=store.events,
        )

    # 1. Build Markdown report
    tfcm_val = matrix.metrics.get("tfcm_seconds")
    tfcm_str = f"{tfcm_val:.1f}s" if isinstance(tfcm_val, (int, float)) else "N/A"
    ar_val = matrix.metrics.get("adoption_ratio")
    ar_str = f"{ar_val:.1%}" if isinstance(ar_val, (int, float)) else "N/A"
    nat_val = matrix.metrics.get("numerical_alignment_time_seconds")
    nat_str = f"{nat_val:.1f}s" if isinstance(nat_val, (int, float)) else "N/A"
    adhoc_mins = matrix.metrics.get("ad_hoc_debugging_minutes")
    adhoc_str = f"{adhoc_mins:.1f} mins" if isinstance(adhoc_mins, (int, float)) else "N/A"

    lines: list[str] = [
        f"# InferRef Alignment Report: {snapshot.session_id}",
        "",
        f"- **Status**: `{snapshot.status}`",
        f"- **Trace**: `{snapshot.source.get('trace', {}).get('path', 'unknown')}`",
        f"- **Adapter**: `{snapshot.target.get('adapter', {}).get('path', 'unknown')}`",
        f"- **Created**: `{snapshot.created_at}`",
        f"- **Updated**: `{snapshot.updated_at}`",
        "",
        "## Summary Metrics",
        "",
        f"- **Total Boundaries**: {len(matrix.boundary_states)}",
        f"- **Verified**: {len(matrix.verified_boundaries)}",
        f"- **Failing**: {len(matrix.failing_boundaries)}",
        f"- **Untested**: {len(matrix.untested_boundaries)}",
        f"- **Stale**: {len(matrix.stale_boundaries)}",
        f"- **TFCM**: {tfcm_str}",
        f"- **Adoption Ratio**: {ar_str}",
        f"- **Numerical Alignment Time**: {nat_str}",
        f"- **Ad-hoc Debugging Time**: {adhoc_str}",
        "",
        "## Diagnostics & Narrowing",
        "",
    ]

    if narrowing.smallest_confirmed_failure:
        lines.extend([
            f"- **Smallest Confirmed Failure**: `{narrowing.smallest_confirmed_failure}`",
            f"- **Confidence**: `{narrowing.confidence}`",
            f"- **Reason**: {narrowing.reason}",
            f"- **Interaction Gap**: {'Yes' if narrowing.interaction_gap else 'No'}",
            "",
            "### Failure Domain Classification",
            "",
        ])
        if candidates:
            lines.append("| Domain | Confidence | Candidate ID | Summary |")
            lines.append("|---|---|---|---|")
            for c in candidates:
                lines.append(f"| `{c.domain}` | {c.confidence:.2f} | `{c.id}` | {c.summary} |")
        else:
            lines.append("No classification candidates identified.")
        lines.append("")
    else:
        lines.extend([
            "No active failure diagnosed.",
            "",
        ])

    lines.extend([
        "## Boundary Matrix",
        "",
        "| Boundary | Status | Evidence Count |",
        "|---|---|---|",
    ])
    for bid, b_state in sorted(matrix.boundary_states.items()):
        b_info = snapshot.boundaries.get(bid, {})
        ev_count = len(b_info.get("evidence_ids", []))
        lines.append(
            f"| `{bid}` | `{b_state}` | {ev_count} |"
        )
    lines.append("")

    lines.extend([
        "## Recent Events",
        "",
        "| Seq | Event | Time | Data |",
        "|---|---|---|---|",
    ])
    for event in store.events[-15:]:
        lines.append(
            f"| {event.seq} | `{event.event}` | {event.timestamp} | `{json.dumps(event.data, default=str)}` |"
        )
    lines.append("")

    markdown_text = "\n".join(lines)

    # 2. Build JSON sidecar
    sidecar: dict[str, Any] = {
        "format": "inferref-alignment-report",
        "format_version": "0.1",
        "session_id": snapshot.session_id,
        "status": snapshot.status,
        "source": snapshot.source,
        "target": snapshot.target,
        "matrix": matrix.to_dict(),
        "narrowing": narrowing.to_dict(),
        "classification": [c.to_dict() for c in candidates],
        "metrics": matrix.metrics,
    }

    if markdown_path is not None:
        md_p = Path(markdown_path).resolve()
        md_p.parent.mkdir(parents=True, exist_ok=True)
        md_p.write_text(markdown_text, encoding="utf-8")

    if json_sidecar_path is not None:
        js_p = Path(json_sidecar_path).resolve()
        js_p.parent.mkdir(parents=True, exist_ok=True)
        js_p.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    return markdown_text, sidecar
