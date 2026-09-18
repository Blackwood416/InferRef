"""Alignment Runner and Execution Orchestration for Alignment Session v0.1 (L4)."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inferref.agent.executor import execute_adapter
from inferref.agent.protocol import EngineAdapter
from inferref.agent.run_record import write_run_record
from inferref.alignment.boundaries import (
    Boundary,
    BoundaryInventory,
    compute_in_scope_frontier,
    inventory_boundaries,
)
from inferref.alignment.classification import classify_failure_domain
from inferref.alignment.evidence import (
    build_comparison_evidence,
)
from inferref.alignment.hashing import (
    canonical_json_sha256,
    normalized_adapter_sha256,
    sha256_file,
    trace_metadata_sha256,
)
from inferref.alignment.narrowing import compute_narrowing
from inferref.alignment.planner import boundary_requirements, plan_alignment
from inferref.alignment.schema import Evidence, utc_now
from inferref.alignment.store import AlignmentStore
from inferref.alignment.testcase_cache import TestcaseCache
from inferref.ir.package import TracePackage


@dataclass
class RunResult:
    """Outcome of one boundary run."""

    status: str  # "pass", "fail", "error", "unsupported"
    boundary: str
    evidence: Evidence | None = None
    execution: dict[str, Any] | None = None
    error_message: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def resolve_engine_identity(
    adapter: EngineAdapter,
    explicit_revision: str | None = None,
) -> dict[str, Any]:
    """Derive a stable engine identity dictionary or require explicit revision."""
    if explicit_revision is not None:
        return {"kind": "manual", "revision": str(explicit_revision), "dirty": False}

    cwd = adapter.working_directory()
    # Try Git resolution in working directory
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if git_head.returncode == 0 and git_head.stdout.strip():
            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            dirty = bool(status_proc.stdout.strip())
            return {
                "kind": "git",
                "revision": git_head.stdout.strip()[:16],
                "dirty": dirty,
            }
    except (OSError, FileNotFoundError):
        pass

    import hashlib

    # Try hashing local executable or script argument if present
    for item in adapter.command:
        expanded = item.replace("{python}", sys.executable)
        candidate_path = Path(expanded)
        if not candidate_path.is_absolute() and cwd.is_dir():
            candidate_path = cwd / expanded
        try:
            # POSIX virtual environments commonly expose their interpreter as
            # a symlink.  The adapter is trusted executable configuration, so
            # bind the identity to the regular file that execution will reach
            # while keeping sha256_file's general no-symlink policy intact.
            candidate_path = candidate_path.resolve(strict=True)
        except OSError:
            continue
        if candidate_path.is_file():
            try:
                h = sha256_file(candidate_path)
                inline_script = ""
                if "-c" in adapter.command:
                    idx = adapter.command.index("-c")
                    if idx + 1 < len(adapter.command):
                        inline_script = adapter.command[idx + 1]
                if inline_script:
                    combined = hashlib.sha256((h + inline_script).encode("utf-8")).hexdigest()
                    return {"kind": "file", "revision": f"sha256:{combined[:16]}", "dirty": False}
                return {"kind": "file", "revision": f"sha256:{h[:16]}", "dirty": False}
            except Exception:
                pass

    raise ValueError(
        "Cannot automatically resolve stable engine identity. "
        "Please provide an explicit revision via --engine-revision."
    )


class AlignmentRunner:
    """Orchestrates testcase extraction, adapter execution, evidence commit, and narrowing."""

    def __init__(
        self,
        store: AlignmentStore,
        package: TracePackage,
        adapter: EngineAdapter,
        *,
        inventory: BoundaryInventory | None = None,
    ) -> None:
        self.store = store
        self.package = package
        self.adapter = adapter
        self.inventory = inventory or inventory_boundaries(package)
        self.cache = TestcaseCache(store.root / "testcases")

        # Verify adapter and trace digests against session definition
        adapter_sha = normalized_adapter_sha256(adapter.to_dict())
        session_adapter_sha = store.snapshot.target.get("adapter", {}).get("normalized_sha256")
        if session_adapter_sha and adapter_sha != session_adapter_sha:
            raise ValueError(
                f"adapter digest mismatch: session expects {session_adapter_sha}, got {adapter_sha}"
            )

        trace_path = getattr(package, "root", None) or package
        trace_sha = trace_metadata_sha256(trace_path)
        session_trace_sha = store.snapshot.source.get("trace", {}).get("metadata_sha256")
        if session_trace_sha and trace_sha != session_trace_sha:
            raise ValueError(
                f"trace metadata digest mismatch: session expects {session_trace_sha}, got {trace_sha}"
            )

    def run_candidate(
        self,
        boundary_or_id: str | Boundary,
        *,
        fixture_kind: str = "reference_derived",
        fixture_tags: Sequence[str] = (),
        external_testcase: Path | None = None,
        engine_revision: str | None = None,
        comparator_overrides: Mapping[str, Any] | None = None,
    ) -> RunResult:
        """Execute one boundary, record immutable Evidence, and update session."""
        # A long-lived API runner must not reuse testcases after source code or
        # scale bytes changed since construction. CLI calls normally recreate
        # the runner, but both entry points share this freshness check.
        current_trace_sha = trace_metadata_sha256(self.package.root)
        expected_trace_sha = self.store.snapshot.source.get("trace", {}).get("metadata_sha256")
        if expected_trace_sha and current_trace_sha != expected_trace_sha:
            raise ValueError("trace metadata digest mismatch: source changed during alignment session")
        boundary = (
            boundary_or_id
            if isinstance(boundary_or_id, Boundary)
            else self.inventory.resolve(boundary_or_id)
        )
        if boundary is None:
            raise ValueError(f"unknown boundary: {boundary_or_id}")

        engine_identity = resolve_engine_identity(self.adapter, engine_revision)

        # 1. Obtain testcase
        if external_testcase is not None:
            tc_path, tc_tree_sha, tc_reused = self.cache.import_external(external_testcase, boundary)
        else:
            tc_path, tc_tree_sha, tc_reused = self.cache.get_or_extract(self.package, boundary)

        event_name = "testcase_reused" if tc_reused else "testcase_extracted"
        try:
            rel_tc_for_event = str(tc_path.relative_to(self.store.root)).replace("\\", "/")
        except ValueError:
            rel_tc_for_event = str(tc_path).replace("\\", "/")
        self.store.commit_event(
            {
                "event": event_name,
                "data": {
                    "boundary": boundary.id,
                    "testcase": rel_tc_for_event,
                    "tree_sha256": tc_tree_sha,
                },
            }
        )

        runs_dir = self.store.root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = utc_now().replace(":", "").replace("-", "")
        import uuid
        fresh_run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
        output_dir = runs_dir / fresh_run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            rel_tc = str(tc_path.relative_to(self.store.root)).replace("\\", "/")
        except ValueError:
            rel_tc = str(tc_path).replace("\\", "/")

        # 2. Capability assessment
        requirements = boundary_requirements(self.package, boundary)
        if self.adapter.capabilities is not None:
            incompatible = list(self.adapter.capabilities.incompatibilities(requirements))
            if incompatible:
                exec_result = {
                    "run_id": fresh_run_id,
                    "adapter": self.adapter.to_dict(),
                    "testcase": str(tc_path),
                    "output": str(output_dir),
                    "status": "unsupported",
                    "execution": None,
                    "comparison": None,
                    "unsupported": incompatible,
                }
                write_run_record(output_dir, exec_result)
                run_record_path = output_dir / "inferref-run.json"
                run_record_sha = sha256_file(run_record_path)
                try:
                    rel_rr = str(run_record_path.relative_to(self.store.root)).replace("\\", "/")
                except ValueError:
                    rel_rr = str(run_record_path).replace("\\", "/")

                evidence = build_comparison_evidence(
                    subject={"boundary": boundary.id, "name": boundary.name},
                    fixture={"kind": fixture_kind, "tags": list(fixture_tags)},
                    provenance={
                        "trace_metadata_sha256": self.store.snapshot.source["trace"]["metadata_sha256"],
                        "testcase": {"path": rel_tc, "tree_sha256": tc_tree_sha},
                        "run_record": {"path": rel_rr, "sha256": run_record_sha},
                        "adapter_normalized_sha256": self.store.snapshot.target["adapter"]["normalized_sha256"],
                        "engine_identity": engine_identity,
                        "effective_comparison_sha256": "0" * 64,
                    },
                    execution_result=exec_result,
                    provided_inputs_valid=True,
                )
                from inferref.alignment.matrix import build_fixture_matrix

                matrix = build_fixture_matrix(
                    list(self.store.evidence.values()),
                    self.store.snapshot,
                    known_boundaries=list(self.store.snapshot.boundaries.keys()),
                    active_engine_identity=engine_identity,
                    events=self.store.events,
                    prospective_evidence=evidence,
                )

                event_data = {
                    "boundary": boundary.id,
                    "name": boundary.name,
                    "reasons": incompatible,
                    "boundary_states": dict(matrix.boundary_states),
                }

                self.store.commit_evidence(
                    evidence,
                    event_name="boundary_unsupported",
                    event_data=event_data,
                )
                self._refresh_diagnostics(active_engine=engine_identity)
                return RunResult(status="unsupported", boundary=boundary.id, evidence=evidence)

        # 3. Execute adapter
        overrides = dict(comparator_overrides or {})
        try:
            exec_result = execute_adapter(
                testcase=tc_path,
                adapter=self.adapter,
                runs_root=runs_dir,
                **overrides,
            )
        except Exception as exc:
            self.store.commit_event(
                {
                    "event": "execution_error",
                    "data": {"boundary": boundary.id, "name": boundary.name, "message": str(exc)},
                }
            )
            return RunResult(
                status="error",
                boundary=boundary.id,
                error_message=str(exc),
            )

        # Write run record
        output_dir = Path(exec_result["output"])
        write_run_record(output_dir, exec_result)
        run_record_path = output_dir / "inferref-run.json"
        run_record_sha = sha256_file(run_record_path)

        try:
            rel_rr = str(run_record_path.relative_to(self.store.root)).replace("\\", "/")
        except ValueError:
            rel_rr = str(run_record_path).replace("\\", "/")

        comp_dict = exec_result.get("effective_comparison") or {}
        comp_sha = canonical_json_sha256(comp_dict)

        evidence = build_comparison_evidence(
            subject={"boundary": boundary.id, "name": boundary.name},
            fixture={"kind": fixture_kind, "tags": list(fixture_tags)},
            provenance={
                "trace_metadata_sha256": self.store.snapshot.source["trace"]["metadata_sha256"],
                "testcase": {"path": rel_tc, "tree_sha256": tc_tree_sha},
                "run_record": {"path": rel_rr, "sha256": run_record_sha},
                "adapter_normalized_sha256": self.store.snapshot.target["adapter"]["normalized_sha256"],
                "engine_identity": engine_identity,
                "effective_comparison_sha256": comp_sha,
            },
            comparison_report=exec_result.get("comparison"),
            execution_result=exec_result,
            provided_inputs_valid=True,
        )

        # Event determination
        if evidence.result == "pass":
            event_name = "boundary_verified"
        elif evidence.result == "fail":
            event_name = "boundary_failed"
        elif evidence.result == "unsupported":
            event_name = "boundary_unsupported"
        else:
            event_name = "execution_error"

        # Build fixture matrix with prospective evidence to compute updated boundary states
        from inferref.alignment.matrix import build_fixture_matrix

        matrix = build_fixture_matrix(
            list(self.store.evidence.values()),
            self.store.snapshot,
            known_boundaries=list(self.store.snapshot.boundaries.keys()),
            active_engine_identity=engine_identity,
            events=self.store.events,
            prospective_evidence=evidence,
        )

        event_data = {
            "boundary": boundary.id,
            "name": boundary.name,
            "boundary_states": dict(matrix.boundary_states),
        }

        self.store.commit_evidence(
            evidence,
            event_name=event_name,
            event_data=event_data,
        )

        # Refresh narrowing and failure domain in snapshot
        self._refresh_diagnostics(active_engine=engine_identity)

        return RunResult(
            status=evidence.result,
            boundary=boundary.id,
            evidence=evidence,
            execution=exec_result.get("execution"),
        )

    def _refresh_diagnostics(self, active_engine: dict[str, Any] | None = None) -> None:
        """Update narrowing and classification in the session after evidence commit."""
        from inferref.alignment.matrix import build_fixture_matrix

        matrix = build_fixture_matrix(
            list(self.store.evidence.values()),
            self.store.snapshot,
            known_boundaries=list(self.store.snapshot.boundaries.keys()),
            active_engine_identity=active_engine,
            events=self.store.events,
        )

        state_changed = False
        for b, state in matrix.boundary_states.items():
            if b in self.store.snapshot.boundaries:
                if self.store.snapshot.boundaries[b].get("state") != state:
                    self.store.snapshot.boundaries[b]["state"] = state
                    state_changed = True
            else:
                self.store.snapshot.boundaries[b] = {
                    "state": state,
                    "last_event_seq": self.store.snapshot.last_event_seq,
                    "evidence_ids": [],
                }
                state_changed = True

        evidence_by_b: dict[str, list[Evidence]] = {}
        for ev in matrix.current_evidence.values():
            if ev.type == "numerical_comparison":
                evidence_by_b.setdefault(ev.subject["boundary"], []).append(ev)

        narrowing_state = compute_narrowing(
            self.inventory,
            evidence_by_b,
            package_or_graph=self.package,
        )
        if narrowing_state.smallest_confirmed_failure:
            self.store.commit_event(
                {
                    "event": "first_divergence_changed",
                    "data": {
                        "narrowing": narrowing_state.to_dict(),
                        "boundary_states": dict(matrix.boundary_states),
                    },
                }
            )

            # Classify failure domain using current evidence and notes
            notes = [ev for ev in self.store.evidence.values() if ev.type == "manual_note"]
            candidates, _ = classify_failure_domain(
                narrowing_state.smallest_confirmed_failure,
                list(matrix.current_evidence.values()),
                notes=notes,
                active_engine_identity=active_engine,
                events=self.store.events,
            )
            self.store.commit_event(
                {
                    "event": "failure_domain_changed",
                    "data": {
                        "classification": {"candidates": [c.to_dict() for c in candidates]},
                        "boundary_states": dict(matrix.boundary_states),
                    },
                }
            )

        # Detect when every required in-scope boundary is verified
        required_frontier = compute_in_scope_frontier(
            self.inventory,
            scope=self.store.snapshot.scope,
            package=self.package,
        )
        all_required_verified = (
            bool(required_frontier)
            and all(
                self.store.snapshot.boundaries.get(bid, {}).get("state") == "verified"
                for bid in required_frontier
            )
        )
        if all_required_verified and self.store.snapshot.status != "verified":
            self.store.commit_event(
                {
                    "event": "session_verified",
                    "data": {"boundary_states": dict(matrix.boundary_states)},
                }
            )
        elif not all_required_verified and self.store.snapshot.status == "verified":
            self.store.commit_event(
                {
                    "event": "plan_selected",
                    "data": {
                        "status": "aligning",
                        "boundary_states": dict(matrix.boundary_states),
                    },
                }
            )
        elif state_changed:
            self.store.persist_snapshot()

    def run(
        self,
        *,
        limit: int = 1,
        boundary_ids: Sequence[str] = (),
        continue_on_fail: bool = False,
        fixture_kind: str = "reference_derived",
        fixture_tags: Sequence[str] = (),
        external_testcase: Path | None = None,
        engine_revision: str | None = None,
        comparator_overrides: Mapping[str, Any] | None = None,
    ) -> list[RunResult]:
        """Run up to limit boundaries."""
        results: list[RunResult] = []

        if boundary_ids:
            selected_ids = list(boundary_ids)[:limit]
        else:
            plan = plan_alignment(self.package, session=self.store.snapshot, adapter=self.adapter)
            selected_ids = [c.boundary for c in plan.candidates if c.runnability.eligible][:limit]

        for bid in selected_ids:
            with self.store.lock():
                res = self.run_candidate(
                    bid,
                    fixture_kind=fixture_kind,
                    fixture_tags=fixture_tags,
                    external_testcase=external_testcase,
                    engine_revision=engine_revision,
                    comparator_overrides=comparator_overrides,
                )
            results.append(res)
            if not continue_on_fail and res.status in ("fail", "error"):
                break

        return results


def hashlib_sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
