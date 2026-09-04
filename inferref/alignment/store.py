"""Crash-consistent persistence for Alignment Session v0.1.

This module owns storage and replay only.  Boundary planning, narrowing, and
classification deliberately remain elsewhere; the reducer below records the
small amount of materialized state needed to make an append-only event log
useful to those workers.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import socket
import sys
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from inferref.alignment.hashing import (
    HashingError,
    adapter_normalized_sha256,
    canonical_json,
    sha256_file,
    trace_metadata_sha256,
    tree_sha256,
)
from inferref.alignment.schema import (
    BOUNDARY_STATES,
    AlignmentEvent,
    AlignmentSchemaError,
    AlignmentSnapshot,
    Evidence,
    make_session_id,
    utc_now,
)
from inferref.ir.paths import PathBoundaryError, resolve_contained_path

ALIGNMENT_MANIFEST = "alignment.json"
ALIGNMENT_FILE = ALIGNMENT_MANIFEST
EVENTS_FILE = "events.jsonl"
EVIDENCE_DIR = "evidence"
TESTCASES_DIR = "testcases"
RUNS_DIR = "runs"
REPORTS_DIR = "reports"
LOCK_FILE = ".alignment.lock"


class AlignmentStoreError(ValueError):
    """Base class for structural and persistence failures."""


class AlignmentIntegrityError(AlignmentStoreError):
    """Raised when an on-disk session cannot be trusted or replayed."""


class AlignmentPathError(AlignmentIntegrityError):
    """Raised when an internal session path is unsafe."""


class AlignmentLockError(AlignmentStoreError):
    """Base class for lock acquisition/recovery errors."""

    code = "alignment_locked"

    def __init__(self, message: str, *, stale: bool = False):
        self.stale = stale
        super().__init__(message)


class AlignmentLockedError(AlignmentLockError):
    """Raised when another live (or unconfirmed) process owns the lock."""


class AlignmentStaleLockError(AlignmentLockError):
    """Raised when a stale lock exists and recovery was not explicitly asked."""

    code = "alignment_stale_lock"


class AlignmentExistsError(FileExistsError):
    """Raised when atomic session initialization targets an existing path."""


class AlignmentVersionError(AlignmentIntegrityError):
    """Raised for an unsupported Alignment/Evidence/Event version."""


class AlignmentSequenceError(AlignmentIntegrityError):
    """Raised for event sequence gaps, duplicates, or regressions."""


class AlignmentConflictError(AlignmentStoreError):
    """Raised when an in-memory store is stale with respect to persistent disk state."""

    code = "session_conflict"


def _json_error(kind: str, path: Path, exc: BaseException) -> AlignmentIntegrityError:
    return AlignmentIntegrityError(f"cannot read {kind} {path}: {exc}")


def _read_json(path: Path, kind: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _json_error(kind, path, exc) from exc


def _fsync_file(path: Path) -> None:
    """Flush one file to stable storage where the host permits it."""

    try:
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    except OSError as exc:
        raise AlignmentStoreError(f"cannot flush {path}: {exc}") from exc


def _fsync_directory(path: Path) -> None:
    # Windows does not expose a directory fsync.  A best-effort flush is still
    # useful on POSIX, while the atomic rename remains the primary invariant.
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON via a same-directory temporary and atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        encoded = canonical_json(dict(payload)) + b"\n"
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise AlignmentStoreError(f"cannot atomically write {path}: {exc}") from exc


def _write_jsonl_line(path: Path, payload: Mapping[str, Any], *, append: bool) -> None:
    encoded = canonical_json(dict(payload)) + b"\n"
    mode = "ab" if append else "xb"
    try:
        with path.open(mode) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise AlignmentStoreError(f"cannot append event log {path}: {exc}") from exc


def _coerce_sequence(values: Any, *, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, (list, tuple)):
        raise AlignmentStoreError(f"{name} must be a string or string sequence")
    result = tuple(value.strip() if isinstance(value, str) else value for value in values)
    if not all(isinstance(value, str) and value for value in result):
        raise AlignmentStoreError(f"{name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise AlignmentStoreError(f"{name} must not contain duplicates")
    return result


def validate_internal_path(
    root: str | Path,
    relative: str | Path,
    *,
    kind: str = "internal path",
    must_exist: bool = False,
) -> Path:
    """Resolve a session-internal POSIX path and enforce containment.

    ``resolve_contained_path`` handles symlinks and junctions.  The additional
    lexical checks here are necessary because Alignment paths are explicitly
    POSIX paths even when a session is being read on Windows.
    """

    text = str(relative)
    if not text or "\x00" in text:
        raise AlignmentPathError(f"{kind} must be a non-empty path without NUL")
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if (
        "\\" in text
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or windows.root
        or any(part in ("", ".", "..") for part in posix.parts)
        or any(part in ("", ".", "..") for part in windows.parts)
    ):
        raise AlignmentPathError(f"{kind} must be a relative POSIX path: {text!r}")
    try:
        resolved = resolve_contained_path(Path(root), text, kind=kind)
    except PathBoundaryError as exc:
        raise AlignmentPathError(str(exc)) from exc
    root_resolved = Path(root).resolve()
    if resolved == root_resolved or not resolved.is_relative_to(root_resolved):
        raise AlignmentPathError(f"{kind} escapes session root: {text!r}")
    if must_exist and not resolved.exists():
        raise AlignmentIntegrityError(f"missing {kind}: {text}")
    return resolved


def _external_path(value: Any, *, kind: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise AlignmentIntegrityError(f"{kind} must be a non-empty path")
    return Path(value).expanduser().resolve()


def _same_or_stale_digest(actual: str, expected: str, *, what: str) -> None:
    if actual != expected:
        raise AlignmentIntegrityError(
            f"{what} digest mismatch: recorded {expected}, actual {actual}"
        )


class _SessionLock:
    """Re-entrant context manager around :class:`AlignmentStore` locking."""

    def __init__(self, store: AlignmentStore, recover_stale_lock: bool):
        self.store = store
        self.recover_stale_lock = recover_stale_lock

    def __enter__(self) -> AlignmentStore:
        self.store.acquire_lock(recover_stale_lock=self.recover_stale_lock)
        return self.store

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.store.release_lock()


class AlignmentStore:
    """A loaded Alignment Session and its append-only audit records."""

    def __init__(
        self,
        root: str | Path,
        *,
        snapshot: AlignmentSnapshot,
        events: list[AlignmentEvent],
        evidence: dict[str, Evidence],
    ) -> None:
        self.root = Path(root).resolve()
        self.snapshot = snapshot
        self.events = list(events)
        self.evidence = dict(evidence)
        self._lock_token: str | None = None
        self._lock_depth = 0

    @property
    def session(self) -> AlignmentSnapshot:
        """Current materialized snapshot (compatibility alias)."""

        return self.snapshot

    @property
    def alignment(self) -> AlignmentSnapshot:
        return self.snapshot

    @property
    def last_event_seq(self) -> int:
        return self.snapshot.last_event_seq

    @property
    def manifest_path(self) -> Path:
        return self.root / ALIGNMENT_MANIFEST

    @property
    def events_path(self) -> Path:
        return self.root / EVENTS_FILE

    @property
    def evidence_path(self) -> Path:
        return self.root / EVIDENCE_DIR

    @property
    def committed_evidence(self) -> dict[str, Evidence]:
        """Evidence records that are referenced by committed events."""
        committed_ids: set[str] = set()
        for ev in self.events:
            committed_ids.update(ev.evidence_ids)
        return {eid: self.evidence[eid] for eid in committed_ids if eid in self.evidence}

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        replay: bool = True,
        validate_artifacts: bool = True,
        refresh_external: bool = True,
    ) -> AlignmentStore:
        """Load, validate, and optionally replay one session.

        Replay is in-memory by default.  It intentionally does not write while
        loading: a caller can inspect a crash-recovered snapshot without
        unexpectedly acquiring a lock.  Call :meth:`rebuild_snapshot` to
        persist the recovered materialized view.
        """

        session_root = Path(root).resolve()
        if not session_root.is_dir():
            raise AlignmentIntegrityError(f"alignment session is not a directory: {session_root}")
        manifest = validate_internal_path(
            session_root, ALIGNMENT_MANIFEST, kind="alignment manifest", must_exist=True
        )
        events_path = validate_internal_path(
            session_root, EVENTS_FILE, kind="alignment event log", must_exist=True
        )
        evidence_dir = validate_internal_path(
            session_root, EVIDENCE_DIR, kind="evidence directory", must_exist=True
        )
        if not evidence_dir.is_dir():
            raise AlignmentIntegrityError("evidence path is not a directory")

        try:
            snapshot = AlignmentSnapshot.from_dict(_read_json(manifest, "alignment snapshot"))
        except AlignmentSchemaError as exc:
            raise AlignmentIntegrityError(str(exc)) from exc

        events = cls._load_events(events_path)
        evidence = cls._load_evidence(session_root, evidence_dir)
        cls._validate_event_stream(snapshot, events, evidence)
        store = cls(session_root, snapshot=snapshot, events=events, evidence=evidence)
        if replay:
            store.snapshot = store._replay_snapshot(full=False)
        if validate_artifacts:
            store._validate_evidence_artifacts()
        if refresh_external:
            store.refresh_external_drift()
        return store

    @staticmethod
    def _load_events(path: Path) -> list[AlignmentEvent]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AlignmentIntegrityError(f"cannot read alignment event log {path}: {exc}") from exc
        if not text:
            raise AlignmentIntegrityError("alignment event log is empty")
        events: list[AlignmentEvent] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                raise AlignmentIntegrityError(f"blank event log line at {line_number}")
            try:
                raw = json.loads(line)
                events.append(AlignmentEvent.from_dict(raw))
            except (ValueError, AlignmentSchemaError) as exc:
                raise AlignmentIntegrityError(
                    f"invalid alignment event at line {line_number}: {exc}"
                ) from exc
        return events

    @staticmethod
    def _load_evidence(root: Path, directory: Path) -> dict[str, Evidence]:
        result: dict[str, Evidence] = {}
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise AlignmentIntegrityError(f"cannot enumerate evidence directory: {exc}") from exc
        for path in entries:
            if path.name.startswith("."):
                continue
            if path.is_dir():
                # Nested evidence paths are not part of the v0.1 layout and
                # could hide records from a naive loader.
                raise AlignmentPathError(f"nested evidence path is not allowed: {path.name}")
            if path.suffix != ".json":
                raise AlignmentIntegrityError(f"unexpected evidence entry: {path.name}")
            validate_internal_path(root, f"{EVIDENCE_DIR}/{path.name}", kind="evidence path")
            try:
                evidence = Evidence.from_dict(_read_json(path, "evidence"))
            except AlignmentSchemaError as exc:
                raise AlignmentIntegrityError(f"invalid evidence {path.name}: {exc}") from exc
            if path.stem != evidence.id:
                raise AlignmentIntegrityError(
                    f"evidence filename {path.name!r} does not match id {evidence.id!r}"
                )
            if evidence.id in result:
                raise AlignmentIntegrityError(f"duplicate evidence id {evidence.id}")
            result[evidence.id] = evidence
        return result

    @staticmethod
    def _validate_event_stream(
        snapshot: AlignmentSnapshot,
        events: list[AlignmentEvent],
        evidence: Mapping[str, Evidence],
    ) -> None:
        if not events:
            raise AlignmentSequenceError("alignment event log has no events")
        if events[0].seq != 1 or events[0].event != "session_initialized":
            raise AlignmentSequenceError(
                "alignment event log must begin with session_initialized seq 1"
            )
        expected = 1
        for record in events:
            if record.seq != expected:
                raise AlignmentSequenceError(
                    f"alignment event sequence gap: expected {expected}, got {record.seq}"
                )
            expected += 1
            missing = [item for item in record.evidence_ids if item not in evidence]
            if missing:
                raise AlignmentIntegrityError(
                    f"event seq {record.seq} references missing evidence: {', '.join(missing)}"
                )
        highest = events[-1].seq
        if snapshot.last_event_seq > highest:
            raise AlignmentSequenceError(
                f"snapshot last_event_seq {snapshot.last_event_seq} exceeds event log {highest}"
            )

    @classmethod
    def initialize(
        cls,
        output: str | Path,
        trace_path: str | Path | None = None,
        adapter_path: str | Path | None = None,
        *,
        source: Mapping[str, Any] | None = None,
        target: Mapping[str, Any] | None = None,
        trace: str | Path | Mapping[str, Any] | None = None,
        adapter: str | Path | Mapping[str, Any] | None = None,
        scope: str | list[str] | tuple[str, ...] = (),
        target_model: str | None = None,
        target_engine: str | None = None,
        model: str | None = None,
        engine: str | None = None,
        required_fixture_kinds: str | list[str] | tuple[str, ...] = (
            "reference_derived",
        ),
        trace_metadata_digest: str | None = None,
        trace_metadata_sha256_value: str | None = None,
        adapter_digest: str | None = None,
        adapter_normalized_sha256_value: str | None = None,
        session_id: str | None = None,
        created_at: str | None = None,
        extensions: Mapping[str, Any] | None = None,
    ) -> AlignmentStore:
        """Create a new session under a staging directory and publish atomically.

        ``trace_path``/``adapter_path`` may be omitted when their already
        computed digests are supplied.  This is useful for remote references,
        while normal local initialization computes and verifies both digests.
        """

        destination = Path(output)
        parent = destination.parent.resolve()
        parent.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
        if destination.exists() or destination.is_symlink():
            raise AlignmentExistsError(f"alignment output already exists: {destination}")

        # Normalize the several public spelling variants before validating the
        # strict snapshot shape.
        if trace_path is None and trace is not None and not isinstance(trace, Mapping):
            trace_path = trace
        if adapter_path is None and adapter is not None and not isinstance(adapter, Mapping):
            adapter_path = adapter
        if source is None:
            if isinstance(trace, Mapping):
                source = {"trace": dict(trace)}
            elif trace_path is not None:
                trace_ref_path = str(Path(trace_path).expanduser().resolve())
                digest = trace_metadata_digest or trace_metadata_sha256_value
                if digest is None:
                    try:
                        digest = trace_metadata_sha256(trace_path)
                    except HashingError as exc:
                        raise AlignmentStoreError(str(exc)) from exc
                source = {
                    "trace": {"path": trace_ref_path, "metadata_sha256": digest}
                }
        if source is None:
            raise AlignmentStoreError("trace_path/source is required")

        if target is None:
            adapter_record: Mapping[str, Any] | None = None
            if isinstance(adapter, Mapping):
                adapter_record = adapter
            elif adapter_path is not None:
                adapter_ref_path = str(Path(adapter_path).expanduser().resolve())
                digest = adapter_digest or adapter_normalized_sha256_value
                if digest is None:
                    try:
                        # Validate the Adapter ABI when a local source is
                        # available.  Hashing its public representation keeps
                        # environment values out of the session.
                        from inferref.agent.protocol import EngineAdapter

                        adapter_object = EngineAdapter.load(adapter_path)
                    except Exception as exc:
                        raise AlignmentStoreError(f"invalid engine adapter: {exc}") from exc
                    digest = adapter_normalized_sha256(adapter_object)
                adapter_record = {
                    "path": adapter_ref_path,
                    "normalized_sha256": digest,
                }
            if adapter_record is None:
                raise AlignmentStoreError("adapter_path/target.adapter is required")
            target = {
                "model": target_model or model or "unknown",
                "engine": target_engine or engine or "unknown",
                "adapter": dict(adapter_record),
            }

        session = AlignmentSnapshot(
            session_id=session_id or make_session_id(),
            created_at=created_at or utc_now(),
            updated_at=created_at or utc_now(),
            last_event_seq=1,
            source=dict(source),
            target=dict(target),
            scope=_coerce_sequence(scope, name="scope"),
            required_fixture_kinds=_coerce_sequence(
                required_fixture_kinds, name="required_fixture_kinds"
            ),
            status="new",
            boundaries={},
            narrowing={},
            classification={},
            metrics={},
            extensions=dict(extensions or {}),
        )
        try:
            session.validate()
        except AlignmentSchemaError as exc:
            raise AlignmentStoreError(str(exc)) from exc

        # The initialization event carries an immutable base snapshot so a
        # complete replay does not depend on a potentially damaged snapshot.
        init_event = AlignmentEvent(
            seq=1,
            timestamp=session.created_at,
            event="session_initialized",
            data={"snapshot": session.to_dict()},
            evidence_ids=(),
        )

        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=parent))
        try:
            (staging / EVIDENCE_DIR).mkdir()
            (staging / TESTCASES_DIR).mkdir()
            (staging / RUNS_DIR).mkdir()
            (staging / REPORTS_DIR).mkdir()
            _atomic_json(staging / ALIGNMENT_MANIFEST, session.to_dict())
            _write_jsonl_line(staging / EVENTS_FILE, init_event.to_dict(), append=False)
            if destination.exists() or destination.is_symlink():
                raise AlignmentExistsError(f"alignment output already exists: {destination}")
            os.replace(staging, destination)
            _fsync_directory(parent)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise
        return cls(
            destination,
            snapshot=session,
            events=[init_event],
            evidence={},
        )

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> AlignmentStore:
        """Alias for :meth:`initialize`."""

        return cls.initialize(*args, **kwargs)

    # -- locking ---------------------------------------------------------

    @property
    def lock_path(self) -> Path:
        return self.root / LOCK_FILE

    def lock(self, *, recover_stale_lock: bool = False) -> _SessionLock:
        """Return a context manager for an exclusive mutation lock."""

        return _SessionLock(self, recover_stale_lock)

    session_lock = lock

    def _read_lock(self) -> dict[str, Any] | None:
        try:
            raw = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            # A truncated lock is stale, but still requires explicit recovery.
            return {"_malformed": True}
        if not isinstance(raw, dict):
            return {"_malformed": True}
        return raw

    @staticmethod
    def _windows_pid_is_stale(pid: int) -> bool:
        """Non-mutating Windows PID probe via stdlib ctypes.

        Uses OpenProcess with PROCESS_QUERY_LIMITED_INFORMATION,
        GetExitCodeProcess, STILL_ACTIVE, and CloseHandle.
        Never terminates, signals, or suspends the probed process.
        """
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_ACCESS_DENIED = 5

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            err = kernel32.GetLastError()
            return err != ERROR_ACCESS_DENIED
        try:
            exit_code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value != STILL_ACTIVE
            err = kernel32.GetLastError()
            return err != ERROR_ACCESS_DENIED
        finally:
            kernel32.CloseHandle(handle)

    @staticmethod
    def _lock_is_stale(info: Mapping[str, Any]) -> bool:
        if info.get("_malformed"):
            return True
        host = info.get("hostname")
        pid = info.get("pid")
        if not isinstance(host, str) or not host:
            return True
        if host != socket.gethostname():
            # A process on another host cannot be proven dead safely.
            return False
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return True
        if sys.platform == "win32":
            return AlignmentStore._windows_pid_is_stale(pid)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        except OSError:
            return True
        return False

    def acquire_lock(self, *, recover_stale_lock: bool = False) -> None:
        """Acquire the lock, requiring an explicit flag for stale recovery."""

        if self._lock_token is not None:
            self._lock_depth += 1
            return
        token = uuid.uuid4().hex
        payload = {
            "format": "inferref-alignment-lock",
            "format_version": "0.1",
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "start_time": utc_now(),
            "token": token,
        }
        for _ in range(2):
            try:
                with self.lock_path.open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump(payload, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                _fsync_directory(self.root)
                self._lock_token = token
                self._lock_depth = 1
                return
            except FileExistsError:
                info = self._read_lock() or {}
                stale = self._lock_is_stale(info)
                if not stale:
                    raise AlignmentLockedError(
                        f"alignment_locked: session lock is held by "
                        f"{info.get('hostname', 'unknown')} pid {info.get('pid', 'unknown')}"
                    )
                if not recover_stale_lock:
                    raise AlignmentStaleLockError(
                        "alignment_locked: stale session lock exists; "
                        "call recover_stale_lock() or pass recover_stale_lock=True",
                        stale=True,
                    )
                try:
                    self.lock_path.unlink()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise AlignmentLockError(f"cannot recover stale alignment lock: {exc}") from exc
        raise AlignmentLockedError("alignment_locked: lock changed while recovering")

    def release_lock(self) -> None:
        """Release only the lock token owned by this store instance."""

        if self._lock_token is None:
            return
        if self._lock_depth > 1:
            self._lock_depth -= 1
            return
        info = self._read_lock()
        if info is not None and info.get("token") == self._lock_token:
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise AlignmentLockError(f"cannot release alignment lock: {exc}") from exc
            _fsync_directory(self.root)
        self._lock_token = None
        self._lock_depth = 0

    def recover_stale_lock(self) -> bool:
        """Explicitly remove a stale lock; live locks are never removed."""

        info = self._read_lock()
        if info is None:
            return False
        if not self._lock_is_stale(info):
            raise AlignmentLockedError(
                f"alignment_locked: lock is live on {info.get('hostname')} "
                f"pid {info.get('pid')}"
            )
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise AlignmentLockError(f"cannot recover stale alignment lock: {exc}") from exc
        _fsync_directory(self.root)
        return True

    # -- artifact validation --------------------------------------------

    def _validate_evidence_artifacts(self) -> None:
        source_digest = self.snapshot.source["trace"]["metadata_sha256"]
        adapter_digest = self.snapshot.target["adapter"]["normalized_sha256"]
        for evidence in self.evidence.values():
            if evidence.type != "numerical_comparison":
                continue
            provenance = evidence.provenance
            if provenance.get("trace_metadata_sha256") != source_digest:
                raise AlignmentIntegrityError(
                    f"evidence {evidence.id} trace metadata digest does not match session"
                )
            if provenance.get("adapter_normalized_sha256") != adapter_digest:
                raise AlignmentIntegrityError(
                    f"evidence {evidence.id} adapter digest does not match session"
                )
            testcase = provenance.get("testcase")
            run_record = provenance.get("run_record")
            if not isinstance(testcase, dict) or not isinstance(run_record, dict):
                raise AlignmentIntegrityError(f"evidence {evidence.id} provenance is incomplete")
            testcase_path = validate_internal_path(
                self.root, testcase["path"], kind=f"evidence {evidence.id} testcase", must_exist=True
            )
            run_path = validate_internal_path(
                self.root, run_record["path"], kind=f"evidence {evidence.id} run record", must_exist=True
            )
            try:
                actual_tree = tree_sha256(testcase_path)
                actual_run = sha256_file(run_path)
            except HashingError as exc:
                raise AlignmentIntegrityError(
                    f"evidence {evidence.id} artifact cannot be hashed: {exc}"
                ) from exc
            _same_or_stale_digest(
                actual_tree,
                testcase["tree_sha256"],
                what=f"evidence {evidence.id} testcase",
            )
            _same_or_stale_digest(
                actual_run,
                run_record["sha256"],
                what=f"evidence {evidence.id} run record",
            )

    def refresh_external_drift(self) -> bool:
        """Mark the session stale when its external trace/adapter changed.

        This is deliberately different from an internal evidence digest
        mismatch: old evidence remains auditable, but verified boundaries are
        no longer current against a changed source definition.
        """

        stale = False
        trace = self.snapshot.source["trace"]
        adapter = self.snapshot.target["adapter"]
        try:
            trace_path = _external_path(trace["path"], kind="trace path")
            if not trace_path.is_dir() or trace_metadata_sha256(trace_path) != trace["metadata_sha256"]:
                stale = True
        except (AlignmentIntegrityError, HashingError, OSError):
            stale = True
        try:
            adapter_path = _external_path(adapter["path"], kind="adapter path")
            if not adapter_path.is_file() or adapter_normalized_sha256(adapter_path) != adapter["normalized_sha256"]:
                stale = True
        except (AlignmentIntegrityError, HashingError, OSError):
            stale = True
        if stale:
            self.snapshot.status = "stale"
            for key, value in list(self.snapshot.boundaries.items()):
                if isinstance(value, dict) and value.get("state") == "verified":
                    updated = dict(value)
                    updated["state"] = "stale"
                    self.snapshot.boundaries[key] = updated
        return stale

    # -- event/evidence commits -----------------------------------------

    def _coerce_event(self, event: AlignmentEvent | Mapping[str, Any], *, next_seq: int) -> AlignmentEvent:
        if isinstance(event, AlignmentEvent):
            return event
        raw = dict(event)
        raw.setdefault("seq", next_seq)
        raw.setdefault("timestamp", utc_now())
        raw.setdefault("format", "inferref-alignment-event")
        raw.setdefault("format_version", "0.1")
        raw.setdefault("data", {})
        raw.setdefault("evidence_ids", [])
        raw.setdefault("extensions", {})
        try:
            return AlignmentEvent.from_dict(raw)
        except AlignmentSchemaError as exc:
            raise AlignmentStoreError(str(exc)) from exc

    def _verify_not_stale_locked(self) -> None:
        """Verify that on-disk events and snapshot match in-memory state before writing."""
        memory_last_seq = self.events[-1].seq if self.events else 0

        # 1. Check events.jsonl on disk
        if self.events_path.exists():
            disk_last_seq = 0
            with open(self.events_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        disk_ev = json.loads(line)
                        disk_last_seq = int(disk_ev.get("seq", 0))
                    except Exception:
                        pass
                    break
            if disk_last_seq != memory_last_seq:
                raise AlignmentConflictError(
                    f"session disk event seq ({disk_last_seq}) does not match in-memory seq ({memory_last_seq}); writer is stale"
                )
        elif memory_last_seq != 0:
            raise AlignmentConflictError(
                f"session disk events.jsonl is missing but in-memory seq is {memory_last_seq}; writer is stale"
            )

        # 2. Check alignment.json manifest on disk
        if self.manifest_path.exists():
            try:
                manifest_data = _read_json(self.manifest_path, "alignment snapshot")
                manifest_last_seq = int(manifest_data.get("last_event_seq", 0))
                if manifest_last_seq != memory_last_seq:
                    raise AlignmentConflictError(
                        f"session disk manifest seq ({manifest_last_seq}) does not match in-memory seq ({memory_last_seq}); writer is stale"
                    )
            except AlignmentConflictError:
                raise
            except Exception as exc:
                raise AlignmentConflictError(f"failed to verify disk manifest revision: {exc}") from exc

    def _commit_event_locked(self, event: AlignmentEvent) -> AlignmentSnapshot:
        expected = self.events[-1].seq + 1 if self.events else 1
        if event.seq != expected:
            raise AlignmentSequenceError(f"event seq must be {expected}, got {event.seq}")
        missing = [item for item in event.evidence_ids if item not in self.evidence]
        if missing:
            raise AlignmentIntegrityError(
                f"event references missing evidence: {', '.join(missing)}"
            )
        _write_jsonl_line(self.events_path, event.to_dict(), append=True)
        candidate = self._apply_event(self.snapshot, event)
        _atomic_json(self.manifest_path, candidate.to_dict())
        self.events.append(event)
        self.snapshot = candidate
        return candidate

    def commit_event(
        self,
        event: AlignmentEvent | Mapping[str, Any],
        *,
        recover_stale_lock: bool = False,
    ) -> AlignmentSnapshot:
        """Append one event and atomically replace the materialized snapshot."""
        with self.lock(recover_stale_lock=recover_stale_lock):
            self._verify_not_stale_locked()
            next_seq = self.events[-1].seq + 1 if self.events else 1
            record = self._coerce_event(event, next_seq=next_seq)
            return self._commit_event_locked(record)

    append_event = commit_event

    def commit_evidence(
        self,
        evidence: Evidence | Mapping[str, Any],
        *,
        event: AlignmentEvent | Mapping[str, Any] | None = None,
        event_name: str | None = None,
        event_data: Mapping[str, Any] | None = None,
        recover_stale_lock: bool = False,
    ) -> AlignmentSnapshot:
        """Write immutable Evidence, then append its event before snapshot commit."""

        if isinstance(evidence, Evidence):
            record = evidence
        else:
            try:
                record = Evidence.from_dict(dict(evidence))
            except AlignmentSchemaError as exc:
                raise AlignmentStoreError(str(exc)) from exc
        record.validate()
        if record.id in self.evidence:
            raise AlignmentIntegrityError(f"evidence id already exists: {record.id}")
        target_path = validate_internal_path(
            self.root, f"{EVIDENCE_DIR}/{record.id}.json", kind="evidence path"
        )
        if target_path.exists():
            raise AlignmentIntegrityError(f"evidence file already exists: {target_path.name}")
        # Validate links before publishing the immutable record.  Manual notes
        # intentionally have no artifact requirements.
        if record.type == "numerical_comparison":
            self._validate_one_evidence_artifacts(record)
        with self.lock(recover_stale_lock=recover_stale_lock):
            self._verify_not_stale_locked()
            target_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(target_path, record.to_dict())
            self.evidence[record.id] = record
            if event is None:
                name = event_name or {
                    "pass": "boundary_verified",
                    "fail": "boundary_failed",
                    "unsupported": "boundary_unsupported",
                    "error": "execution_error",
                    "recorded": "note_recorded",
                }.get(record.result, "note_recorded")
                data = dict(event_data or {})
                data.setdefault("boundary", record.subject["boundary"])
                data.setdefault("name", record.subject["name"])
                event = {
                    "event": name,
                    "data": data,
                    "evidence_ids": [record.id],
                }
            elif isinstance(event, AlignmentEvent):
                if record.id not in event.evidence_ids:
                    raise AlignmentIntegrityError(
                        f"event for evidence {record.id} must reference that evidence id"
                    )
            else:
                event = dict(event)
                ids = list(event.get("evidence_ids", []))
                if record.id not in ids:
                    ids.append(record.id)
                event["evidence_ids"] = ids
            next_seq = self.events[-1].seq + 1 if self.events else 1
            event_record = self._coerce_event(event, next_seq=next_seq)
            # The evidence file may be left unreferenced if a process fails
            # after its atomic publication.  It is retained deliberately:
            # immutability and auditability are safer than destructive
            # rollback, and a later event can reference it.
            return self._commit_event_locked(event_record)

    record_evidence = commit_evidence
    append_evidence = commit_evidence
    write_evidence = commit_evidence

    def _validate_one_evidence_artifacts(self, evidence: Evidence) -> None:
        provenance = evidence.provenance
        source_digest = self.snapshot.source["trace"]["metadata_sha256"]
        adapter_digest = self.snapshot.target["adapter"]["normalized_sha256"]
        if provenance.get("trace_metadata_sha256") != source_digest:
            raise AlignmentIntegrityError(
                f"evidence {evidence.id} trace metadata digest does not match session"
            )
        if provenance.get("adapter_normalized_sha256") != adapter_digest:
            raise AlignmentIntegrityError(
                f"evidence {evidence.id} adapter digest does not match session"
            )
        testcase = provenance["testcase"]
        run_record = provenance["run_record"]
        testcase_path = validate_internal_path(
            self.root, testcase["path"], kind=f"evidence {evidence.id} testcase", must_exist=True
        )
        run_path = validate_internal_path(
            self.root, run_record["path"], kind=f"evidence {evidence.id} run record", must_exist=True
        )
        try:
            actual_tree = tree_sha256(testcase_path)
            actual_run = sha256_file(run_path)
        except HashingError as exc:
            raise AlignmentIntegrityError(str(exc)) from exc
        _same_or_stale_digest(actual_tree, testcase["tree_sha256"], what=f"evidence {evidence.id} testcase")
        _same_or_stale_digest(actual_run, run_record["sha256"], what=f"evidence {evidence.id} run record")

    @staticmethod
    def _apply_event(snapshot: AlignmentSnapshot, event: AlignmentEvent) -> AlignmentSnapshot:
        result = copy.deepcopy(snapshot)
        data = dict(event.data)
        result.last_event_seq = event.seq
        result.updated_at = event.timestamp
        if event.event in {
            "boundary_verified",
            "boundary_failed",
            "boundary_unsupported",
        }:
            boundary = data.get("boundary")
            if not isinstance(boundary, str) or not boundary:
                raise AlignmentIntegrityError(
                    f"{event.event} seq {event.seq} requires data.boundary"
                )
            state = {
                "boundary_verified": "verified",
                "boundary_failed": "failing",
                "boundary_unsupported": "unsupported",
            }[event.event]
            previous = result.boundaries.get(boundary, {})
            if not isinstance(previous, dict):
                raise AlignmentIntegrityError(f"boundary aggregate {boundary!r} is not an object")
            updated = dict(previous)
            updated["state"] = state
            updated["last_event_seq"] = event.seq
            if event.evidence_ids:
                existing = list(updated.get("evidence_ids", []))
                updated["evidence_ids"] = list(dict.fromkeys(existing + list(event.evidence_ids)))
            if isinstance(data.get("name"), str) and data["name"]:
                updated["name"] = data["name"]
            result.boundaries[boundary] = updated
            result.status = str(data.get("session_status", "aligning"))
        elif event.event == "execution_error":
            boundary = data.get("boundary")
            if isinstance(boundary, str) and boundary:
                previous = result.boundaries.get(boundary, {})
                if not isinstance(previous, dict):
                    raise AlignmentIntegrityError(f"boundary aggregate {boundary!r} is not an object")
                updated = dict(previous)
                updated["state"] = "blocked"
                updated["last_event_seq"] = event.seq
                if event.evidence_ids:
                    updated["evidence_ids"] = list(event.evidence_ids)
                result.boundaries[boundary] = updated
            result.status = str(data.get("session_status", "blocked"))
        elif event.event == "session_verified":
            result.status = "verified"
        elif event.event == "failure_domain_changed":
            if isinstance(data.get("classification"), dict):
                result.classification = copy.deepcopy(data["classification"])
        elif event.event == "first_divergence_changed":
            if isinstance(data.get("narrowing"), dict):
                result.narrowing = copy.deepcopy(data["narrowing"])
        elif event.event == "note_recorded":
            # Notes are evidence of workflow activity; they do not imply a
            # numerical status.  Metrics/classification aggregates, when
            # supplied by a higher-level worker, are copied verbatim.
            if isinstance(data.get("metrics"), dict):
                result.metrics = copy.deepcopy(data["metrics"])
        if isinstance(data.get("boundary_states"), Mapping):
            for b_name, b_st in data["boundary_states"].items():
                if isinstance(b_st, str) and b_st in BOUNDARY_STATES:
                    if b_name in result.boundaries and isinstance(result.boundaries[b_name], dict):
                        result.boundaries[b_name]["state"] = b_st
                    elif b_name not in result.boundaries:
                        result.boundaries[b_name] = {
                            "state": b_st,
                            "last_event_seq": event.seq,
                            "evidence_ids": [],
                        }
        explicit_status = data.get("status")
        if explicit_status in {"new", "aligning", "blocked", "verified", "stale", "error"}:
            result.status = explicit_status
        result.validate()
        return result

    def _initial_snapshot_from_event(self) -> AlignmentSnapshot:
        first = self.events[0]
        raw = first.data.get("snapshot")
        if isinstance(raw, dict):
            try:
                initial = AlignmentSnapshot.from_dict(raw)
            except AlignmentSchemaError as exc:
                raise AlignmentIntegrityError(
                    f"session_initialized base snapshot is invalid: {exc}"
                ) from exc
            if initial.session_id != self.snapshot.session_id:
                raise AlignmentIntegrityError(
                    "session_initialized base snapshot session_id does not match alignment.json"
                )
            return initial
        # Older/provisional event writers may only carry a session id.  Keep
        # immutable metadata from alignment.json but reset materialized fields.
        initial = copy.deepcopy(self.snapshot)
        initial.status = "new"
        initial.last_event_seq = 1
        initial.boundaries = {}
        initial.narrowing = {}
        initial.classification = {}
        initial.metrics = {}
        initial.updated_at = first.timestamp
        return initial

    def _replay_snapshot(self, *, full: bool = False) -> AlignmentSnapshot:
        if full:
            result = self._initial_snapshot_from_event()
            start = result.last_event_seq
        else:
            result = copy.deepcopy(self.snapshot)
            start = result.last_event_seq
        for event in self.events:
            if event.seq <= start:
                continue
            result = self._apply_event(result, event)
        if result.last_event_seq != self.events[-1].seq:
            raise AlignmentSequenceError(
                f"replayed snapshot seq {result.last_event_seq} does not reach event log {self.events[-1].seq}"
            )
        return result

    def rebuild_snapshot(
        self,
        *,
        persist: bool = True,
        recover_stale_lock: bool = False,
        full: bool = True,
    ) -> AlignmentSnapshot:
        """Rebuild the materialized view from the append-only event log."""

        if not persist:
            return self._replay_snapshot(full=full)
        with self.lock(recover_stale_lock=recover_stale_lock):
            rebuilt = self._replay_snapshot(full=full)
            _atomic_json(self.manifest_path, rebuilt.to_dict())
            self.snapshot = rebuilt
            return rebuilt

    replay = rebuild_snapshot

    def persist_snapshot(self, *, recover_stale_lock: bool = False) -> None:
        """Atomically commit current in-memory snapshot to disk manifest."""
        with self.lock(recover_stale_lock=recover_stale_lock):
            self._verify_not_stale_locked()
            _atomic_json(self.manifest_path, self.snapshot.to_dict())

    def reload(self, *, replay: bool = True) -> AlignmentStore:
        """Reload this store from disk, preserving no uncommitted memory."""

        return self.load(self.root, replay=replay)

    def validate(self) -> None:
        """Re-run all structural and artifact checks on this loaded store."""

        self.snapshot.validate()
        self._validate_event_stream(self.snapshot, self.events, self.evidence)
        self._validate_evidence_artifacts()


def initialize_session(*args: Any, **kwargs: Any) -> AlignmentStore:
    """Module-level initializer used by CLI/Agent integrations."""

    return AlignmentStore.initialize(*args, **kwargs)


def create_session(*args: Any, **kwargs: Any) -> AlignmentStore:
    """Alias for :func:`initialize_session`."""

    return initialize_session(*args, **kwargs)


def load_session(
    root: str | Path,
    *,
    replay: bool = True,
    validate_artifacts: bool = True,
    refresh_external: bool = True,
) -> AlignmentStore:
    """Module-level session loader."""

    return AlignmentStore.load(
        root,
        replay=replay,
        validate_artifacts=validate_artifacts,
        refresh_external=refresh_external,
    )


open_session = load_session


__all__ = [
    "ALIGNMENT_FILE",
    "ALIGNMENT_MANIFEST",
    "EVENTS_FILE",
    "EVIDENCE_DIR",
    "LOCK_FILE",
    "REPORTS_DIR",
    "RUNS_DIR",
    "TESTCASES_DIR",
    "AlignmentConflictError",
    "AlignmentExistsError",
    "AlignmentIntegrityError",
    "AlignmentLockError",
    "AlignmentLockedError",
    "AlignmentPathError",
    "AlignmentSequenceError",
    "AlignmentStaleLockError",
    "AlignmentStore",
    "AlignmentStoreError",
    "AlignmentVersionError",
    "create_session",
    "initialize_session",
    "load_session",
    "open_session",
    "validate_internal_path",
]
