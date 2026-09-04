from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferref.alignment.hashing import (
    HashingError,
    normalize_adapter,
    normalized_adapter_sha256,
    tree_sha256,
)
from inferref.alignment.schema import (
    AlignmentEvent,
    AlignmentSchemaError,
    Evidence,
    make_evidence_id,
)
from inferref.alignment.store import (
    AlignmentIntegrityError,
    AlignmentLockedError,
    AlignmentSequenceError,
    AlignmentStaleLockError,
    AlignmentStore,
    validate_internal_path,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def _store(tmp_path: Path) -> AlignmentStore:
    return AlignmentStore.initialize(
        tmp_path / "alignment",
        trace_path=tmp_path / "trace.irtrace",
        adapter_path=tmp_path / "adapter.json",
        trace_metadata_digest=DIGEST_A,
        adapter_digest=DIGEST_B,
        target_model="model",
        target_engine="engine",
    )


def _manual_note(evidence_id: str | None = None) -> Evidence:
    return Evidence(
        id=evidence_id or make_evidence_id(),
        created_at="2026-09-03T00:00:00Z",
        type="manual_note",
        subject={"boundary": "region:1", "name": "root"},
        fixture={"kind": "other", "tags": []},
        result="recorded",
        observations={},
        provenance={},
        note={"category": "custom-tool", "reason": "inspect a custom dump"},
    )


def test_initialize_is_atomic_and_loads_layout(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.manifest_path.is_file()
    assert store.events_path.is_file()
    assert (store.root / "evidence").is_dir()
    assert (store.root / "testcases").is_dir()
    assert (store.root / "runs").is_dir()
    assert (store.root / "reports").is_dir()
    loaded = AlignmentStore.load(store.root, validate_artifacts=False)
    assert loaded.snapshot.session_id == store.snapshot.session_id
    assert loaded.snapshot.last_event_seq == 1
    assert loaded.events[0].event == "session_initialized"

    with pytest.raises(FileExistsError):
        AlignmentStore.initialize(
            store.root,
            trace_path=tmp_path / "trace.irtrace",
            adapter_path=tmp_path / "adapter.json",
            trace_metadata_digest=DIGEST_A,
            adapter_digest=DIGEST_B,
        )


def test_manual_evidence_commit_is_immutable_and_replayed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    evidence = _manual_note()
    store.commit_evidence(evidence)
    assert store.snapshot.last_event_seq == 2
    # A note event does not invent a numerical boundary state.
    assert "region:1" not in store.snapshot.boundaries
    with pytest.raises(AlignmentIntegrityError):
        store.commit_evidence(evidence)
    loaded = AlignmentStore.load(store.root, validate_artifacts=False)
    assert evidence.id in loaded.evidence
    assert loaded.snapshot.last_event_seq == 2


def test_replay_recovers_event_committed_before_snapshot(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    evidence = _manual_note()

    # Simulate a process dying after the event append.  The test uses the
    # public evidence API's durable files and leaves alignment.json untouched.
    event = AlignmentEvent(
        seq=2,
        timestamp="2026-09-03T00:01:00Z",
        event="note_recorded",
        data={"category": "custom-tool"},
        evidence_ids=(evidence.id,),
    )
    evidence_path = store.root / "evidence" / f"{evidence.id}.json"
    evidence_path.write_text(json.dumps(evidence.to_dict()), encoding="utf-8")
    store.evidence[evidence.id] = evidence
    with store.events_path.open("ab") as stream:
        stream.write(
            (json.dumps(event.to_dict(), separators=(",", ":")) + "\n").encode()
        )
    recovered = AlignmentStore.load(store.root, validate_artifacts=False)
    assert recovered.snapshot.last_event_seq == 2
    assert recovered.snapshot.updated_at == event.timestamp
    assert recovered.rebuild_snapshot(persist=False).last_event_seq == 2


def test_sequence_gap_and_missing_evidence_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    raw = json.loads(store.events_path.read_text(encoding="utf-8")).copy()
    raw["seq"] = 3
    store.events_path.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    with pytest.raises(AlignmentSequenceError):
        AlignmentStore.load(store.root, validate_artifacts=False)

    store = _store(tmp_path / "second")
    event = {
        "format": "inferref-alignment-event",
        "format_version": "0.1",
        "seq": 2,
        "timestamp": "2026-09-03T00:01:00Z",
        "event": "note_recorded",
        "data": {},
        "evidence_ids": ["ev-missing"],
    }
    with store.events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
    with pytest.raises(AlignmentIntegrityError, match="missing evidence"):
        AlignmentStore.load(store.root, validate_artifacts=False)


def test_internal_paths_and_digest_links_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        validate_internal_path(tmp_path, "../escape")
    with pytest.raises(ValueError):
        validate_internal_path(tmp_path, "C:/escape")
    with pytest.raises(ValueError):
        validate_internal_path(tmp_path, "a\\b")

    evidence = _manual_note()
    # Manual notes have no artifact links, but malformed numerical evidence is
    # rejected by schema before it can become an uncheckable audit record.
    raw = {
        **evidence.to_dict(),
        "id": make_evidence_id(),
        "type": "numerical_comparison",
        "result": "pass",
        "observations": {"provided_inputs_valid": True, "engine_inputs_observed": False},
        "provenance": {
            "trace_metadata_sha256": DIGEST_A,
            "testcase": {"path": "../escape", "tree_sha256": DIGEST_A},
            "run_record": {"path": "runs/r.json", "sha256": DIGEST_A},
            "adapter_normalized_sha256": DIGEST_B,
            "engine_identity": {"kind": "git", "revision": "abc", "dirty": False},
            "effective_comparison_sha256": DIGEST_A,
        },
    }
    with pytest.raises(AlignmentSchemaError):
        Evidence.from_dict(raw)


def test_live_and_stale_lock_recovery_are_explicit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with store.lock():
        other = AlignmentStore.load(store.root, validate_artifacts=False)
        with pytest.raises(AlignmentLockedError, match="alignment_locked"):
            other.acquire_lock()

    store.lock_path.write_text(
        json.dumps(
            {
                "format": "inferref-alignment-lock",
                "format_version": "0.1",
                "pid": 2_147_483_647,
                "hostname": "test-host",
                "start_time": "2020-01-01T00:00:00Z",
                "token": "old",
            }
        ),
        encoding="utf-8",
    )
    # A foreign host is conservatively considered live.  A malformed local
    # lock is a stale lock but cannot be removed without explicit recovery.
    store.lock_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(AlignmentStaleLockError, match="recover_stale_lock"):
        store.acquire_lock()
    assert store.recover_stale_lock() is True
    with store.lock():
        assert store.lock_path.is_file()


def test_tree_and_adapter_hashes_are_canonical_and_redacted(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    (tree / "b").mkdir(parents=True)
    (tree / "a.txt").write_text("a", encoding="utf-8")
    (tree / "b" / "z.txt").write_text("z", encoding="utf-8")
    first = tree_sha256(tree)
    (tree / "a.txt").touch()
    assert tree_sha256(tree) == first

    adapter = {
        "format": "inferref-engine-adapter",
        "format_version": "0.2",
        "name": "engine",
        "command": ["python", "{testcase}", "{output}"],
        "environment": {"TOKEN": "super-secret"},
        "capabilities": {"device_types": ["cpu"]},
    }
    normalized = normalize_adapter(adapter)
    assert "super-secret" not in json.dumps(normalized)
    changed = {**adapter, "environment": {"TOKEN": "another-secret"}}
    assert normalized_adapter_sha256(adapter) == normalized_adapter_sha256(changed)
    assert normalized_adapter_sha256({**adapter, "environment": {"OTHER": "value"}}) != normalized_adapter_sha256(adapter)

    try:
        (tree / "link").symlink_to(tree / "a.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(HashingError):
        tree_sha256(tree)


def test_lock_liveness_probe_does_not_terminate_process() -> None:
    import os
    import socket

    current_pid = os.getpid()
    info = {
        "format": "inferref-alignment-lock",
        "format_version": "0.1",
        "pid": current_pid,
        "hostname": socket.gethostname(),
        "start_time": "2026-09-03T00:00:00Z",
        "token": "probe-test",
    }
    # Current process must be recognized as alive (not stale)
    assert AlignmentStore._lock_is_stale(info) is False
    # And probe must not have terminated or mutated current process
    assert os.getpid() == current_pid

    # Stale PID on local host must be recognized as stale
    stale_info = {
        "format": "inferref-alignment-lock",
        "format_version": "0.1",
        "pid": 2_147_483_647,
        "hostname": socket.gethostname(),
        "start_time": "2020-01-01T00:00:00Z",
        "token": "dead",
    }
    assert AlignmentStore._lock_is_stale(stale_info) is True

    # Foreign host must be conservatively recognized as live (not stale)
    foreign_info = {
        "format": "inferref-alignment-lock",
        "format_version": "0.1",
        "pid": 2_147_483_647,
        "hostname": "other-host",
        "start_time": "2020-01-01T00:00:00Z",
        "token": "other",
    }
    assert AlignmentStore._lock_is_stale(foreign_info) is False


def test_stale_writer_rejected_and_log_preserved(tmp_path: Path) -> None:
    from inferref.alignment.evidence import build_manual_note_evidence
    from inferref.alignment.store import AlignmentConflictError

    # Two separate instances loaded concurrently at seq=1
    store_a = _store(tmp_path)
    session_dir = store_a.root
    store_b = AlignmentStore.load(session_dir)

    assert len(store_a.events) == 1
    assert len(store_b.events) == 1

    # Writer A commits a note evidence -> seq=2
    note_a = build_manual_note_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        category="custom-tool",
        reason="note from writer A",
    )
    store_a.commit_evidence(note_a)
    assert len(store_a.events) == 2
    assert store_a.events[-1].seq == 2

    # Writer B attempts to commit a note evidence while stale
    note_b = build_manual_note_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        category="custom-tool",
        reason="note from writer B",
    )
    with pytest.raises(AlignmentConflictError):
        store_b.commit_evidence(note_b)

    # Writer B attempts to commit an event while stale
    with pytest.raises(AlignmentConflictError):
        store_b.commit_event({"event": "note_recorded", "data": {"test": "val"}})

    # Disk must NOT have duplicate seq=2 or corrupted events.jsonl
    reloaded = AlignmentStore.load(session_dir)
    seqs = [e.seq for e in reloaded.events]
    assert seqs == [1, 2]
