"""Alignment Runner and Execution Orchestration for Alignment Session v0.1 (L4)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inferref.agent.executor import execute_adapter, execute_adapter_batch
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
from inferref.compare.compare import compare_testcase
from inferref.compare.tolerance import DEFAULT_TOLERANCES
from inferref.comparison.resolution import (
    EffectiveComparison,
    resolve_comparison_policy,
)
from inferref.ir.package import TracePackage
from inferref.testcase.validate import require_valid_testcase

# ``execute_adapter_batch`` only accepts identifier-like item ids, while a
# boundary id is ``kind:value``. The request index keeps two boundaries apart
# even when their slugs collide (InferRef 0.12 section 7.3).
_BATCH_ITEM_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
# A batch report is only usable when the helper mapped every requested item; any
# other status (unsupported, results_error, execution_error, or a failed process
# such as timeout/crash) leaves no per-item report to finalize from.
_BATCH_TRUSTED_STATUSES = frozenset({"pass", "failed"})


def batch_item_id(index: int, boundary_id: str) -> str:
    """Return the ordered, pattern-safe batch item id for one boundary."""
    return f"{index}-{_BATCH_ITEM_ID_UNSAFE.sub('-', boundary_id)}"


def _resolve_effective_comparison(
    testcase_path: Path, overrides: Mapping[str, Any]
) -> EffectiveComparison:
    """Resolve the comparison policy the sequential executor would apply.

    ``execute_adapter`` owns this resolution for a single launch; the batch
    helper owns transport only, so finalization reuses the same resolution
    function and comparator overrides for each item.
    """
    validation = require_valid_testcase(Path(testcase_path))
    policy = overrides.get("policy")
    cli_atol = (
        policy.override_atol
        if policy and policy.override_atol is not None
        else overrides.get("atol")
    )
    cli_rtol = (
        policy.override_rtol
        if policy and policy.override_rtol is not None
        else overrides.get("rtol")
    )
    cli_tolerance = (
        policy.per_dtype
        if policy and policy.per_dtype != DEFAULT_TOLERANCES
        else overrides.get("tolerance")
    )
    return resolve_comparison_policy(
        testcase_spec=validation.manifest.get("comparison"),
        suite_spec=overrides.get("suite_spec"),
        cli_comparator=overrides.get("comparator"),
        cli_atol=cli_atol,
        cli_rtol=cli_rtol,
        cli_strict_layout=overrides.get("strict_layout") or None,
        cli_ignore_stride=overrides.get("ignore_stride") or None,
        cli_tolerance=cli_tolerance,
        cli_config=overrides.get("comparison_config"),
    )


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


@dataclass(frozen=True)
class PreparedCandidate:
    """Pre-execution context for one boundary run.

    Carries everything the runner resolves before an adapter process starts so
    a caller can finalize the same boundary after a shared batch process
    (InferRef 0.12 section 7.3).
    """

    boundary: Boundary
    fixture_kind: str
    fixture_tags: tuple[str, ...]
    engine_identity: dict[str, Any]
    testcase_path: Path
    testcase_tree_sha256: str
    testcase_rel_path: str
    run_id: str
    output_dir: Path
    unsupported: tuple[str, ...] = ()


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
        prepared = self._prepare_candidate(
            boundary_or_id,
            fixture_kind=fixture_kind,
            fixture_tags=fixture_tags,
            external_testcase=external_testcase,
            engine_revision=engine_revision,
        )
        if isinstance(prepared, RunResult):
            return prepared
        return self._execute_prepared(
            prepared, comparator_overrides=comparator_overrides
        )

    def _execute_prepared(
        self,
        prepared: PreparedCandidate,
        *,
        comparator_overrides: Mapping[str, Any] | None = None,
    ) -> RunResult:
        """Launch one prepared candidate and finalize its output."""
        overrides = dict(comparator_overrides or {})
        try:
            exec_result = execute_adapter(
                testcase=prepared.testcase_path,
                adapter=self.adapter,
                runs_root=prepared.output_dir.parent,
                **overrides,
            )
        except Exception as exc:
            self.store.commit_event(
                {
                    "event": "execution_error",
                    "data": {
                        "boundary": prepared.boundary.id,
                        "name": prepared.boundary.name,
                        "message": str(exc),
                    },
                }
            )
            return RunResult(
                status="error",
                boundary=prepared.boundary.id,
                error_message=str(exc),
            )

        return self.run_candidate_from_output(
            prepared,
            exec_result,
            exec_result["output"],
        )

    def _prepare_candidate(
        self,
        boundary_or_id: str | Boundary,
        *,
        fixture_kind: str = "reference_derived",
        fixture_tags: Sequence[str] = (),
        external_testcase: Path | None = None,
        engine_revision: str | None = None,
    ) -> PreparedCandidate | RunResult:
        """Resolve, cache, and capability-check one boundary before execution.

        Returns a :class:`PreparedCandidate` ready for an adapter launch, or an
        already-finalized ``unsupported`` result when the adapter cannot run
        the testcase at all.
        """
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
        incompatible = list(self._capability_incompatibilities(boundary))

        prepared = PreparedCandidate(
            boundary=boundary,
            fixture_kind=fixture_kind,
            fixture_tags=tuple(fixture_tags),
            engine_identity=engine_identity,
            testcase_path=tc_path,
            testcase_tree_sha256=tc_tree_sha,
            testcase_rel_path=rel_tc,
            run_id=fresh_run_id,
            output_dir=output_dir,
            unsupported=tuple(incompatible),
        )
        if not incompatible:
            return prepared

        # The adapter cannot run this testcase, so there is nothing to execute;
        # record the same unsupported result the sequential path always emitted.
        unsupported_result = {
            "run_id": fresh_run_id,
            "adapter": self.adapter.to_dict(),
            "testcase": str(tc_path),
            "output": str(output_dir),
            "status": "unsupported",
            "execution": None,
            "comparison": None,
            "unsupported": incompatible,
        }
        return self._finalize_candidate(prepared, unsupported_result, output_dir)

    def _capability_incompatibilities(self, boundary: Boundary) -> tuple[str, ...]:
        """Reasons the declared adapter capabilities cannot run this boundary."""
        if self.adapter.capabilities is None:
            return ()
        requirements = boundary_requirements(self.package, boundary)
        return tuple(self.adapter.capabilities.incompatibilities(requirements))

    def _boundary_runnable(self, boundary_id: str) -> bool:
        """Whether the adapter can execute this boundary at all."""
        boundary = self.inventory.resolve(boundary_id)
        return boundary is not None and not self._capability_incompatibilities(boundary)

    def run_candidate_from_output(
        self,
        prepared: PreparedCandidate,
        execution_result: Mapping[str, Any],
        output_dir: str | Path,
    ) -> RunResult:
        """Finalize a prepared candidate whose adapter output already exists.

        This is the batch seam (InferRef 0.12 section 7.3): a caller launches one
        adapter process for several testcases, then finalizes each prepared
        candidate from its own execution result and per-item output directory.
        It never launches an adapter.
        """
        return self._finalize_candidate(prepared, execution_result, output_dir)

    def _finalize_candidate(
        self,
        prepared: PreparedCandidate,
        execution_result: Mapping[str, Any],
        output_dir: str | Path,
    ) -> RunResult:
        """Write the run record and Evidence for one prepared candidate."""
        boundary = prepared.boundary
        output_path = Path(output_dir)
        write_run_record(output_path, dict(execution_result))
        run_record_path = output_path / "inferref-run.json"
        run_record_sha = sha256_file(run_record_path)

        try:
            rel_rr = str(run_record_path.relative_to(self.store.root)).replace("\\", "/")
        except ValueError:
            rel_rr = str(run_record_path).replace("\\", "/")

        if prepared.unsupported:
            comp_sha = "0" * 64
        else:
            comp_dict = execution_result.get("effective_comparison") or {}
            comp_sha = canonical_json_sha256(comp_dict)

        evidence = build_comparison_evidence(
            subject={"boundary": boundary.id, "name": boundary.name},
            fixture={"kind": prepared.fixture_kind, "tags": list(prepared.fixture_tags)},
            provenance={
                "trace_metadata_sha256": self.store.snapshot.source["trace"]["metadata_sha256"],
                "testcase": {
                    "path": prepared.testcase_rel_path,
                    "tree_sha256": prepared.testcase_tree_sha256,
                },
                "run_record": {"path": rel_rr, "sha256": run_record_sha},
                "adapter_normalized_sha256": self.store.snapshot.target["adapter"]["normalized_sha256"],
                "engine_identity": prepared.engine_identity,
                "effective_comparison_sha256": comp_sha,
            },
            comparison_report=execution_result.get("comparison"),
            execution_result=execution_result,
            provided_inputs_valid=True,
        )

        # Build fixture matrix with prospective evidence to compute updated boundary states
        from inferref.alignment.matrix import build_fixture_matrix

        matrix = build_fixture_matrix(
            list(self.store.evidence.values()),
            self.store.snapshot,
            known_boundaries=list(self.store.snapshot.boundaries.keys()),
            active_engine_identity=prepared.engine_identity,
            events=self.store.events,
            prospective_evidence=evidence,
        )

        if prepared.unsupported:
            event_name = "boundary_unsupported"
            event_data = {
                "boundary": boundary.id,
                "name": boundary.name,
                "reasons": list(prepared.unsupported),
                "boundary_states": dict(matrix.boundary_states),
            }
        else:
            # Event determination
            if evidence.result == "pass":
                event_name = "boundary_verified"
            elif evidence.result == "fail":
                event_name = "boundary_failed"
            elif evidence.result == "unsupported":
                event_name = "boundary_unsupported"
            else:
                event_name = "execution_error"
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
        self._refresh_diagnostics(active_engine=prepared.engine_identity)

        return RunResult(
            status=evidence.result,
            boundary=boundary.id,
            evidence=evidence,
            execution=execution_result.get("execution"),
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
        """Run up to limit boundaries.

        When the adapter declares ``batch_execution`` and more than one selected
        boundary is runnable, the boundaries share one adapter process
        (InferRef 0.12 section 7.3). Every other case keeps the sequential
        launch-per-boundary path.
        """
        if boundary_ids:
            selected_ids = list(boundary_ids)[:limit]
        else:
            plan = plan_alignment(self.package, session=self.store.snapshot, adapter=self.adapter)
            selected_ids = [c.boundary for c in plan.candidates if c.runnability.eligible][:limit]

        if limit > 1 and self.adapter.supports_batch_execution and external_testcase is None:
            batched = self._run_selected_batch(
                selected_ids,
                continue_on_fail=continue_on_fail,
                fixture_kind=fixture_kind,
                fixture_tags=fixture_tags,
                engine_revision=engine_revision,
                comparator_overrides=comparator_overrides,
            )
            if batched is not None:
                return batched

        results: list[RunResult] = []
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

    def _run_selected_batch(
        self,
        selected_ids: Sequence[str],
        *,
        continue_on_fail: bool,
        fixture_kind: str,
        fixture_tags: Sequence[str],
        engine_revision: str | None,
        comparator_overrides: Mapping[str, Any] | None,
    ) -> list[RunResult] | None:
        """Run the selected boundaries in one adapter process.

        Returns ``None`` when the batch is not usable, so the caller keeps the
        unchanged sequential path.
        """
        batchable_ids = [bid for bid in selected_ids if self._boundary_runnable(bid)]
        if len(batchable_ids) < 2:
            return None
        if len(batchable_ids) < len(selected_ids) and not continue_on_fail:
            # ``_prepare_candidate`` finalizes an unsupported boundary on the
            # spot, so a failure earlier in the batch could commit evidence the
            # sequential contract stops before. That mix is ambiguous.
            return None

        overrides = dict(comparator_overrides or {})
        with self.store.lock():
            prepared: list[tuple[str, PreparedCandidate | RunResult]] = [
                (
                    bid,
                    self._prepare_candidate(
                        bid,
                        fixture_kind=fixture_kind,
                        fixture_tags=fixture_tags,
                        engine_revision=engine_revision,
                    ),
                )
                for bid in selected_ids
            ]
            batch_candidates = [
                (bid, item)
                for bid, item in prepared
                if isinstance(item, PreparedCandidate)
            ]
            if len(batch_candidates) < 2:
                return self._finalize_prepared_sequence(
                    prepared, overrides=overrides, continue_on_fail=continue_on_fail
                )

            items = [
                (batch_item_id(index, bid), item.testcase_path)
                for index, (bid, item) in enumerate(batch_candidates)
            ]
            batch = execute_adapter_batch(items, self.adapter, self.store.root / "runs")
            if batch["status"] not in _BATCH_TRUSTED_STATUSES:
                self._record_batch_fallback(batch, [bid for bid, _ in batch_candidates])
                return self._finalize_prepared_sequence(
                    prepared, overrides=overrides, continue_on_fail=continue_on_fail
                )

            batch_position = {
                bid: index for index, (bid, _) in enumerate(batch_candidates)
            }
            results: list[RunResult] = []
            for bid, item in prepared:
                if isinstance(item, RunResult):
                    result = item
                else:
                    batch_item = batch["results"][batch_position[bid]]
                    execution_result = self._batch_item_result(
                        item, batch_item, batch, overrides
                    )
                    result = self.run_candidate_from_output(
                        item, execution_result, batch_item["output_dir"]
                    )
                results.append(result)
                if not continue_on_fail and result.status in ("fail", "error"):
                    break
            return results

    def _finalize_prepared_sequence(
        self,
        prepared: Sequence[tuple[str, PreparedCandidate | RunResult]],
        *,
        overrides: Mapping[str, Any],
        continue_on_fail: bool,
    ) -> list[RunResult]:
        """Finish already-prepared boundaries one adapter launch at a time."""
        results: list[RunResult] = []
        for _, item in prepared:
            result = (
                item
                if isinstance(item, RunResult)
                else self._execute_prepared(item, comparator_overrides=overrides)
            )
            results.append(result)
            if not continue_on_fail and result.status in ("fail", "error"):
                break
        return results

    def _record_batch_fallback(
        self, batch: Mapping[str, Any], boundary_ids: Sequence[str]
    ) -> None:
        """Note why a batched launch was abandoned for sequential execution."""
        note = f"batch fallback: {batch.get('status')}"
        detail = str(batch.get("error") or "").strip()
        if detail:
            note += f" ({detail[:300]})"
        self.store.commit_event(
            {
                "event": "note_recorded",
                "data": {
                    "category": "external-blocker",
                    "reason": note,
                    "boundaries": list(boundary_ids),
                },
            }
        )

    def _batch_item_result(
        self,
        prepared: PreparedCandidate,
        item: Mapping[str, Any],
        batch: Mapping[str, Any],
        overrides: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the run-result mapping finalization expects for one batch item."""
        execution: dict[str, Any] = {
            "status": "completed",
            "exit_code": item["exit_code"],
            "batch_manifest": batch["manifest"],
            "batch_output": batch["batch_output"],
            "output_dir": item["output_dir"],
        }
        if item.get("error"):
            execution["error"] = item["error"]
        result: dict[str, Any] = {
            "run_id": prepared.run_id,
            "adapter": self.adapter.to_dict(),
            "testcase": str(prepared.testcase_path),
            "output": item["output_dir"],
            "execution": execution,
            "comparison": None,
            "effective_comparison": None,
        }
        try:
            effective_comparison = _resolve_effective_comparison(
                prepared.testcase_path, overrides
            )
            effective_comparison.validate()
        except Exception as exc:
            # A sequential launch would refuse this candidate before starting and
            # the runner reports an error, so the batch item reports one too.
            result["status"] = "error"
            execution["error"] = str(exc)
            return result
        result["effective_comparison"] = effective_comparison.to_dict()
        if item["status"] != "passed" or item["exit_code"] != 0:
            result["status"] = "execution_error"
            return result
        try:
            report = compare_testcase(
                prepared.testcase_path,
                item["output_dir"],
                effective_comparison=effective_comparison,
                first_failure=True,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result.update(
                status="comparison_error",
                comparison={"status": "error", "message": str(exc)},
            )
            return result
        report_dict = report.to_dict()
        result["comparison"] = report_dict
        if report_dict.get("comparator"):
            result["comparator"] = report_dict["comparator"]
        result["status"] = (
            "pass"
            if report.status == "pass"
            else ("error" if report.status == "error" else "mismatch")
        )
        return result


def hashlib_sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
