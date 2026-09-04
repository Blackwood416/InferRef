"""CLI handlers for `inferref align` (Alignment Orchestrator v0.10 L6)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from inferref.agent.protocol import EngineAdapter
from inferref.alignment.boundaries import inventory_boundaries
from inferref.alignment.classification import classify_failure_domain
from inferref.alignment.evidence import build_manual_note_evidence
from inferref.alignment.hashing import (
    normalized_adapter_sha256,
    trace_metadata_sha256,
)
from inferref.alignment.matrix import build_fixture_matrix
from inferref.alignment.narrowing import compute_narrowing
from inferref.alignment.planner import plan_alignment
from inferref.alignment.report import generate_alignment_report
from inferref.alignment.runner import AlignmentRunner
from inferref.alignment.store import AlignmentStore
from inferref.ir.package import TracePackage

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def _resolve_session_dir(raw: str | None) -> Path:
    if raw:
        p = Path(raw).resolve()
        if (p / "alignment.json").is_file():
            return p
        if p.is_file() and p.name == "alignment.json":
            return p.parent
        return p
    # Default to ./alignment
    cwd_align = Path.cwd() / "alignment"
    if (cwd_align / "alignment.json").is_file():
        return cwd_align
    return cwd_align


def _emit(payload: Any, text: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(text)


# -- align init ------------------------------------------------------------


def cmd_align_init(args: argparse.Namespace) -> int:
    trace_path = Path(args.trace).resolve()
    adapter_path = Path(args.adapter).resolve()
    session_dir = Path(args.dir).resolve() if args.dir else Path.cwd() / "alignment"

    if not trace_path.exists():
        print(f"error: trace path does not exist: {trace_path}", file=sys.stderr)
        return EXIT_USAGE
    if not adapter_path.is_file():
        print(f"error: adapter path does not exist: {adapter_path}", file=sys.stderr)
        return EXIT_USAGE

    try:
        pkg = TracePackage.load(trace_path)
        adapter = EngineAdapter.load(adapter_path)
        trace_digest = trace_metadata_sha256(pkg.root)
        adapter_digest = normalized_adapter_sha256(adapter.to_dict())

        import shutil

        if args.force and session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)

        session_id = None
        if args.name:
            session_id = args.name if args.name.startswith("al-") else f"al-{args.name}"

        store = AlignmentStore.initialize(
            session_dir,
            trace_path=trace_path,
            adapter_path=adapter_path,
            trace_metadata_digest=trace_digest,
            adapter_digest=adapter_digest,
            session_id=session_id,
        )
    except Exception as exc:
        print(f"error: failed to initialize alignment session: {exc}", file=sys.stderr)
        return EXIT_USAGE

    payload = store.snapshot.to_dict()
    text = (
        f"Initialized InferRef Alignment Session '{store.snapshot.session_id}'\n"
        f"  Directory: {session_dir}\n"
        f"  Trace:     {trace_path} (digest: {trace_digest[:16]})\n"
        f"  Adapter:   {adapter_path} (digest: {adapter_digest[:16]})\n"
        f"  Status:    {store.snapshot.status}"
    )
    _emit(payload, text, args.json)
    return EXIT_OK


# -- align plan ------------------------------------------------------------


def cmd_align_plan(args: argparse.Namespace) -> int:
    session_dir = _resolve_session_dir(args.session)
    store: AlignmentStore | None = None
    if (session_dir / "alignment.json").is_file():
        store = AlignmentStore.load(session_dir)

    trace_path = (
        Path(args.trace).resolve()
        if args.trace
        else (Path(store.snapshot.source["trace"]["path"]).resolve() if store else None)
    )
    if not trace_path or not trace_path.exists():
        print("error: trace path required or session must contain valid trace", file=sys.stderr)
        return EXIT_USAGE

    adapter_path = (
        Path(args.adapter).resolve()
        if args.adapter
        else (Path(store.snapshot.target["adapter"]["path"]).resolve() if store else None)
    )
    adapter: EngineAdapter | None = None
    if adapter_path and adapter_path.is_file():
        adapter = EngineAdapter.load(adapter_path)

    pkg = TracePackage.load(trace_path)
    plan = plan_alignment(
        pkg,
        session=store.snapshot if store else None,
        adapter=adapter,
    )

    recommended = plan.candidates[0].boundary if plan.candidates else None
    payload = plan.to_dict()
    payload["recommended_boundary"] = recommended
    lines = [
        f"Alignment Plan (candidates: {len(plan.candidates)})",
        f"  Recommended next: {recommended or 'none'}",
        "",
        "Top Candidates:",
    ]
    for idx, cand in enumerate(plan.candidates[:10], start=1):
        lines.append(
            f"  - [{idx:02d}] {cand.boundary:<20} score={cand.score} eligible={cand.runnability.eligible} ({cand.runnability.reason})"
        )
    text = "\n".join(lines)
    _emit(payload, text, args.json)
    return EXIT_OK


# -- align run -------------------------------------------------------------


def cmd_align_run(args: argparse.Namespace) -> int:
    session_dir = _resolve_session_dir(args.session)
    if not (session_dir / "alignment.json").is_file():
        print(f"error: no alignment session found at {session_dir}", file=sys.stderr)
        return EXIT_USAGE

    store = AlignmentStore.load(session_dir)
    trace_path = Path(store.snapshot.source["trace"]["path"]).resolve()
    adapter_path = Path(store.snapshot.target["adapter"]["path"]).resolve()

    pkg = TracePackage.load(trace_path)
    adapter = EngineAdapter.load(adapter_path)
    runner = AlignmentRunner(store, pkg, adapter)

    boundaries = list(args.boundary or [])
    if getattr(args, "subject", None):
        boundaries.append(args.subject)

    external_tc = Path(args.testcase).resolve() if getattr(args, "testcase", None) else None
    if external_tc is not None:
        if not getattr(args, "subject", None) and len(boundaries) != 1:
            print("error: --testcase requires exactly one explicit --subject boundary", file=sys.stderr)
            return EXIT_USAGE
        if len(boundaries) > 1:
            print("error: --testcase cannot be associated with multiple boundaries", file=sys.stderr)
            return EXIT_USAGE
        if not getattr(args, "fixture_kind", None):
            print("error: --testcase requires an explicitly supplied --fixture-kind", file=sys.stderr)
            return EXIT_USAGE

    fixture_kind = getattr(args, "fixture_kind", None) or "reference_derived"
    limit = args.limit or (len(boundaries) if boundaries else 1)

    comparator_overrides: dict[str, Any] = {}
    if getattr(args, "comparator", None):
        comparator_overrides["comparator"] = args.comparator
    if getattr(args, "atol", None) is not None:
        comparator_overrides["atol"] = args.atol
    if getattr(args, "rtol", None) is not None:
        comparator_overrides["rtol"] = args.rtol
    if getattr(args, "ignore_stride", False):
        comparator_overrides["ignore_stride"] = True
    if getattr(args, "strict_layout", False):
        comparator_overrides["strict_layout"] = True

    results = runner.run(
        limit=limit,
        boundary_ids=boundaries,
        continue_on_fail=args.continue_on_fail,
        fixture_kind=fixture_kind,
        fixture_tags=getattr(args, "fixture_tag", None) or (),
        external_testcase=external_tc,
        engine_revision=args.engine_revision,
        comparator_overrides=comparator_overrides or None,
    )

    payload = {
        "results": [
            {
                "boundary": r.boundary,
                "status": r.status,
                "evidence_id": r.evidence.id if r.evidence else None,
                "error_message": r.error_message,
            }
            for r in results
        ],
        "snapshot": store.snapshot.to_dict(),
    }

    lines = [f"Executed {len(results)} boundaries:"]
    for r in results:
        lines.append(f"  - {r.boundary}: {r.status}")
    text = "\n".join(lines)

    _emit(payload, text, args.json)
    has_failure = any(r.status in ("fail", "error") for r in results)
    return EXIT_FAIL if has_failure else EXIT_OK


# -- align status ----------------------------------------------------------


def cmd_align_status(args: argparse.Namespace) -> int:
    session_dir = _resolve_session_dir(args.session)
    if not (session_dir / "alignment.json").is_file():
        print(f"error: no alignment session found at {session_dir}", file=sys.stderr)
        return EXIT_USAGE

    store = AlignmentStore.load(session_dir)
    snap = store.snapshot

    matrix = build_fixture_matrix(
        list(store.evidence.values()),
        snap,
        known_boundaries=list(snap.boundaries.keys()),
        events=store.events,
    )

    evidence_by_b: dict[str, list[Any]] = {}
    for ev in matrix.current_evidence.values():
        if ev.type == "numerical_comparison":
            evidence_by_b.setdefault(ev.subject["boundary"], []).append(ev)

    t_path = Path(store.snapshot.source["trace"]["path"]).resolve()
    pkg = TracePackage.load(t_path)
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

    verified_count = sum(1 for s in matrix.boundary_states.values() if s == "verified")
    failing_count = sum(1 for s in matrix.boundary_states.values() if s == "failing")
    untested_count = sum(1 for s in matrix.boundary_states.values() if s == "untested")
    stale_count = sum(1 for s in matrix.boundary_states.values() if s == "stale")

    active_failure = snap.narrowing.get("current_failure")
    payload = {
        "session_id": snap.session_id,
        "status": snap.status,
        "counts": {
            "verified": verified_count,
            "failing": failing_count,
            "untested": untested_count,
            "stale": stale_count,
            "total": len(snap.boundaries),
        },
        "active_failure": active_failure,
        "smallest_confirmed_failure": narrowing.smallest_confirmed_failure,
        "classification": [c.to_dict() for c in candidates],
    }

    lines = [
        f"Alignment Session: {snap.session_id}",
        f"  Status:       {snap.status}",
        f"  Boundaries:   {verified_count} verified, {failing_count} failing, {untested_count} untested, {stale_count} stale / {len(snap.boundaries)} total",
        f"  Active Fail:  {active_failure or 'none'}",
        f"  Smallest Fail:{narrowing.smallest_confirmed_failure or 'none'}",
    ]
    if candidates:
        lines.append("  Classification Candidates:")
        for c in candidates:
            lines.append(f"    - {c.domain} ({c.confidence:.2f}): {c.summary}")

    text = "\n".join(lines)
    _emit(payload, text, args.json)
    return EXIT_OK


# -- align summary ---------------------------------------------------------


def cmd_align_summary(args: argparse.Namespace) -> int:
    session_dir = _resolve_session_dir(args.session)
    if not (session_dir / "alignment.json").is_file():
        print(f"error: no alignment session found at {session_dir}", file=sys.stderr)
        return EXIT_USAGE

    store = AlignmentStore.load(session_dir)
    matrix = build_fixture_matrix(
        list(store.evidence.values()),
        store.snapshot,
        known_boundaries=list(store.snapshot.boundaries.keys()),
        events=store.events,
    )
    payload = matrix.to_dict()

    m = matrix.metrics
    tfcm_val = m.get("tfcm_seconds")
    tfcm_str = f"{tfcm_val:.1f}s" if isinstance(tfcm_val, (int, float)) else "N/A"
    ar_val = m.get("adoption_ratio")
    ar_str = f"{ar_val:.1%}" if isinstance(ar_val, (int, float)) else "N/A"
    adhoc_mins = m.get("ad_hoc_debugging_minutes")
    adhoc_str = f"{adhoc_mins:.1f} mins" if isinstance(adhoc_mins, (int, float)) else "N/A"
    nat_val = m.get("numerical_alignment_time_seconds")
    nat_str = f"{nat_val:.1f}s" if isinstance(nat_val, (int, float)) else "N/A"

    lines = [
        f"Alignment Summary: {store.snapshot.session_id}",
        f"  Validation Steps: {m.get('inferref_validation_steps', 0)}",
        f"  Engine Builds:    {m.get('engine_builds_count', 0)}",
        f"  Adoption Ratio:   {ar_str}",
        f"  TFCM:             {tfcm_str}",
        f"  Alignment Time:   {nat_str}",
        f"  Ad-hoc Mins:      {adhoc_str}",
    ]
    text = "\n".join(lines)
    _emit(payload, text, args.json)
    return EXIT_OK


# -- align narrow ----------------------------------------------------------


def cmd_align_narrow(args: argparse.Namespace) -> int:
    session_dir = _resolve_session_dir(args.session)
    if not (session_dir / "alignment.json").is_file():
        print(f"error: no alignment session found at {session_dir}", file=sys.stderr)
        return EXIT_USAGE

    store = AlignmentStore.load(session_dir)
    matrix = build_fixture_matrix(
        list(store.evidence.values()),
        store.snapshot,
        known_boundaries=list(store.snapshot.boundaries.keys()),
        events=store.events,
    )

    evidence_by_b: dict[str, list[Any]] = {}
    for ev in matrix.current_evidence.values():
        if ev.type == "numerical_comparison":
            evidence_by_b.setdefault(ev.subject["boundary"], []).append(ev)

    t_path = Path(store.snapshot.source["trace"]["path"]).resolve()
    pkg = TracePackage.load(t_path)
    inventory = inventory_boundaries(pkg)

    narrowing = compute_narrowing(
        inventory,
        evidence_by_b,
        current_failure=args.boundary,
    )
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

    payload = {
        "narrowing": narrowing.to_dict(),
        "classification": [c.to_dict() for c in candidates],
    }

    lines = [
        "Narrowing Analysis:",
        f"  Smallest Confirmed Failure: {narrowing.smallest_confirmed_failure or 'none'}",
        f"  Confidence:                 {narrowing.confidence}",
        f"  Reason:                     {narrowing.reason}",
        f"  Interaction Gap:            {'Yes' if narrowing.interaction_gap else 'No'}",
        f"  Next Candidates:            {', '.join(narrowing.next_candidates) or 'none'}",
    ]
    text = "\n".join(lines)
    _emit(payload, text, args.json)
    return EXIT_OK


# -- align note ------------------------------------------------------------


def cmd_align_note(args: argparse.Namespace) -> int:
    session_dir = _resolve_session_dir(args.session)
    if not (session_dir / "alignment.json").is_file():
        print(f"error: no alignment session found at {session_dir}", file=sys.stderr)
        return EXIT_USAGE

    store = AlignmentStore.load(session_dir)
    t_path = Path(store.snapshot.source["trace"]["path"]).resolve()
    pkg = TracePackage.load(t_path)
    inventory = inventory_boundaries(pkg)

    boundary = inventory.resolve(args.boundary)
    if boundary is None:
        print(f"error: boundary not in inventory: {args.boundary}", file=sys.stderr)
        return EXIT_USAGE

    target = getattr(args, "target", None) or getattr(args, "override_domain", None) or getattr(args, "candidate_id", None)
    duration_minutes = getattr(args, "duration_minutes", None)
    milestone = getattr(args, "milestone", None)

    note_evidence = build_manual_note_evidence(
        subject={"boundary": boundary.id, "name": boundary.name},
        category=args.category,
        reason=args.text,
        duration_minutes=duration_minutes,
        milestone=milestone,
        target=target,
        extensions={"author": args.author},
    )

    with store.lock():
        event_data: dict[str, Any] = {
            "boundary": boundary.id,
            "category": args.category,
            "text": args.text,
        }
        if duration_minutes is not None:
            event_data["duration_minutes"] = duration_minutes
        if milestone is not None:
            event_data["milestone"] = milestone
        if target is not None:
            event_data["target"] = target
        store.commit_evidence(
            note_evidence,
            event_name="note_recorded",
            event_data=event_data,
        )

    payload = note_evidence.to_dict()
    text = f"Added manual note to {boundary.id} [{args.category}]: {note_evidence.id}"
    _emit(payload, text, args.json)
    return EXIT_OK


# -- align report ----------------------------------------------------------


def cmd_align_report(args: argparse.Namespace) -> int:
    session_dir = _resolve_session_dir(args.session)
    if not (session_dir / "alignment.json").is_file():
        print(f"error: no alignment session found at {session_dir}", file=sys.stderr)
        return EXIT_USAGE

    store = AlignmentStore.load(session_dir)
    md_path = Path(args.output).resolve() if args.output else session_dir / "reports" / "alignment-report.md"
    sidecar_path = (
        Path(args.json_sidecar).resolve()
        if args.json_sidecar
        else session_dir / "reports" / "alignment-report.json"
    )

    _md_text, sidecar = generate_alignment_report(
        store,
        markdown_path=md_path,
        json_sidecar_path=sidecar_path,
    )

    if args.json:
        print(json.dumps(sidecar, indent=2, ensure_ascii=False, default=str))
    else:
        print("Generated alignment report:")
        print(f"  Markdown: {md_path}")
        print(f"  JSON:     {sidecar_path}")
    return EXIT_OK


# -- Subparser Registration ------------------------------------------------


def register_align_subparsers(subparsers: argparse._SubParsersAction[Any], name: str = "align") -> None:
    p_align = subparsers.add_parser(
        name,
        help="Alignment Orchestrator: evidence-driven engine alignment (0.10)",
    )
    align_subs = p_align.add_subparsers(dest=f"{name}_command")

    # init
    p_init = align_subs.add_parser("init", help="initialize an alignment session")
    p_init.add_argument("trace", help="path to source trace package")
    p_init.add_argument("adapter", help="path to engine adapter definition")
    p_init.add_argument("--dir", help="session directory (default: ./alignment)")
    p_init.add_argument("--name", help="optional friendly session name")
    p_init.add_argument("--force", action="store_true", help="overwrite existing session")
    p_init.add_argument("--json", action="store_true", help="emit json")
    p_init.set_defaults(func=cmd_align_init)

    # plan
    p_plan = align_subs.add_parser("plan", help="show alignment candidates and recommendation")
    p_plan.add_argument("session", nargs="?", help="session directory (default: ./alignment)")
    p_plan.add_argument("--trace", help="trace path override")
    p_plan.add_argument("--adapter", help="adapter path override")
    p_plan.add_argument("--json", action="store_true", help="emit json")
    p_plan.set_defaults(func=cmd_align_plan)

    # run
    p_run = align_subs.add_parser("run", help="execute boundary alignment runs")
    p_run.add_argument("session", nargs="?", help="session directory (default: ./alignment)")
    p_run.add_argument("--limit", type=int, default=1, help="maximum boundaries to run (default: 1)")
    p_run.add_argument("--boundary", action="append", help="specific boundary ID(s) to run")
    p_run.add_argument("--subject", help="single subject boundary ID alias")
    p_run.add_argument("--testcase", help="explicit external testcase directory to run")
    p_run.add_argument("--fixture-kind", default=None, help="fixture kind (default: reference_derived)")
    p_run.add_argument("--fixture-tag", action="append", default=[], help="fixture tags (repeatable)")
    p_run.add_argument("--comparator", help="override comparator plugin")
    p_run.add_argument("--atol", type=float, help="override absolute tolerance")
    p_run.add_argument("--rtol", type=float, help="override relative tolerance")
    p_run.add_argument("--ignore-stride", action="store_true", help="ignore stride mismatches")
    p_run.add_argument("--strict-layout", action="store_true", help="enforce strict layout matching")
    p_run.add_argument("--continue-on-fail", action="store_true", help="continue running even if a boundary fails")
    p_run.add_argument("--engine-revision", help="explicit engine revision override")
    p_run.add_argument("--json", action="store_true", help="emit json")
    p_run.set_defaults(func=cmd_align_run)

    # status
    p_status = align_subs.add_parser("status", help="show alignment session status")
    p_status.add_argument("session", nargs="?", help="session directory (default: ./alignment)")
    p_status.add_argument("--json", action="store_true", help="emit json")
    p_status.set_defaults(func=cmd_align_status)

    # summary
    p_summary = align_subs.add_parser("summary", help="show alignment metrics and savings")
    p_summary.add_argument("session", nargs="?", help="session directory (default: ./alignment)")
    p_summary.add_argument("--json", action="store_true", help="emit json")
    p_summary.set_defaults(func=cmd_align_summary)

    # narrow
    p_narrow = align_subs.add_parser("narrow", help="show smallest confirmed failure and diagnostics")
    p_narrow.add_argument("session", nargs="?", help="session directory (default: ./alignment)")
    p_narrow.add_argument("--boundary", help="boundary ID to narrow from")
    p_narrow.add_argument("--json", action="store_true", help="emit json")
    p_narrow.set_defaults(func=cmd_align_narrow)

    # note
    p_note = align_subs.add_parser("note", help="add manual note evidence to a boundary")
    p_note.add_argument("session", nargs="?", help="session directory (default: ./alignment)")
    p_note.add_argument("boundary", help="boundary ID")
    p_note.add_argument("category", help="note category (e.g. custom-tool, missing-observation, classification-override)")
    p_note.add_argument("text", help="note markdown content")
    p_note.add_argument("--duration-minutes", type=float, help="ad-hoc debugging minutes spent")
    p_note.add_argument("--milestone", help="milestone tag, e.g. first-correct-model")
    p_note.add_argument("--target", help="target candidate ID or override domain")
    p_note.add_argument("--override-domain", help="classification override domain")
    p_note.add_argument("--candidate-id", help="candidate hypothesis ID")
    p_note.add_argument("--author", default="user", help="author identifier")
    p_note.add_argument("--json", action="store_true", help="emit json")
    p_note.set_defaults(func=cmd_align_note)

    # report
    p_report = align_subs.add_parser("report", help="generate markdown report and json sidecar")
    p_report.add_argument("session", nargs="?", help="session directory (default: ./alignment)")
    p_report.add_argument("--output", help="output markdown file path")
    p_report.add_argument("--json-sidecar", help="output json sidecar file path")
    p_report.add_argument("--json", action="store_true", help="emit json sidecar to stdout")
    p_report.set_defaults(func=cmd_align_report)
