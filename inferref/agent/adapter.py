"""Shell-free engine execution followed by an InferRef comparison."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inferref.agent.protocol import AgentProtocolError, EngineAdapter
from inferref.compare.compare import compare_testcase
from inferref.compare.tolerance import TolerancePolicy
from inferref.testcase.validate import require_valid_testcase

_STREAM_CHUNK_BYTES = 64 * 1024
_RESOURCE_POLL_SECONDS = 0.02
_ARTIFACT_SCAN_SECONDS = 0.25
_DEADLINE_CHECK_ENTRIES = 64


def execute_adapter(
    testcase: str | Path,
    adapter: EngineAdapter,
    runs_root: str | Path,
    *,
    policy: TolerancePolicy | None = None,
    ignore_stride: bool = False,
    strict_layout: bool = False,
    first_failure: bool = True,
) -> dict[str, Any]:
    """Execute one trusted adapter in a fresh output directory and compare it."""

    testcase_path = Path(testcase).resolve()
    validation = require_valid_testcase(testcase_path)
    if not validation.reproducible:
        blockers = ", ".join(issue.code for issue in validation.issues) or "unknown"
        raise AgentProtocolError(
            "testcase is not independently reproducible after validation "
            f"(blockers: {blockers})"
        )

    cwd = adapter.working_directory()
    if not cwd.is_dir():
        raise AgentProtocolError(f"adapter working directory does not exist: {cwd}")

    output_root = Path(runs_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
    output_path = output_root / run_id
    output_path.mkdir()

    command, configured_env = adapter.expand(
        testcase=testcase_path,
        output=output_path,
        # Resolving a POSIX venv executable follows its symlink to the base
        # interpreter and silently loses the venv's site-packages.
        python=Path(sys.executable).absolute(),
    )
    environment = os.environ.copy()
    environment.update(configured_env)
    environment["INFERREF_TESTCASE"] = str(testcase_path)
    environment["INFERREF_OUTPUT"] = str(output_path)

    started = time.perf_counter()
    execution: dict[str, Any] = {
        "command": list(command),
        "cwd": str(cwd),
        "timeout_seconds": adapter.timeout_seconds,
    }
    stdout_capture = _StreamCapture(
        output_path / "inferref-stdout.log", adapter.max_output_chars
    )
    stderr_capture = _StreamCapture(
        output_path / "inferref-stderr.log", adapter.max_output_chars
    )
    process: subprocess.Popen[bytes] | None = None
    windows_job: int | None = None
    try:
        popen_options: dict[str, Any] = {}
        if os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_options["start_new_session"] = True
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            **popen_options,
        )
        if os.name == "nt":
            windows_job = _assign_windows_kill_job(process)
        stdout_capture.start(process.stdout)
        stderr_capture.start(process.stderr)
        process_status, artifact_scan = _wait_with_limits(
            process,
            deadline=time.monotonic() + adapter.timeout_seconds,
            output_path=output_path,
            max_artifact_bytes=adapter.max_artifact_bytes,
            max_artifact_files=adapter.max_artifact_files,
            captures=(stdout_capture, stderr_capture),
            windows_job=windows_job,
        )
        windows_job = None
        stdout_capture.join()
        stderr_capture.join()
        execution.update(
            {
                "status": process_status,
                "exit_code": process.returncode,
                "stdout": stdout_capture.text(),
                "stderr": stderr_capture.text(),
                "stdout_path": stdout_capture.path.name,
                "stderr_path": stderr_capture.path.name,
                "stdout_bytes": stdout_capture.observed_bytes,
                "stderr_bytes": stderr_capture.observed_bytes,
                "max_output_bytes_per_stream": adapter.max_output_chars,
                "artifact_bytes": artifact_scan.total_bytes,
                "artifact_files": artifact_scan.files,
                "artifact_scan_entries": artifact_scan.entries,
                "max_artifact_bytes": adapter.max_artifact_bytes,
                "max_artifact_files": adapter.max_artifact_files,
                "process_tree_strategy": (
                    "windows_job_object" if os.name == "nt" else "posix_process_group"
                ),
            }
        )
    except OSError as exc:
        if windows_job is not None:
            _close_windows_job(windows_job)
            windows_job = None
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        execution.update(
            {
                "status": "error",
                "exit_code": None,
                "stdout": "",
                "stderr": str(exc),
            }
        )
    execution["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)

    result: dict[str, Any] = {
        "run_id": run_id,
        "adapter": adapter.to_dict(),
        "testcase": str(testcase_path),
        "output": str(output_path),
        "execution": execution,
        "comparison": None,
    }
    if execution["status"] != "completed":
        result["status"] = execution["status"]
    elif execution["exit_code"] != 0:
        result["status"] = "execution_error"
    else:
        try:
            report = compare_testcase(
                testcase_path,
                output_path,
                policy=policy or TolerancePolicy(),
                ignore_stride=ignore_stride,
                strict_layout=strict_layout,
                first_failure=first_failure,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result["status"] = "comparison_error"
            result["comparison"] = {
                "status": "error",
                "message": str(exc),
            }
        else:
            result["comparison"] = report.to_dict()
            result["status"] = "pass" if report.status == "pass" else "mismatch"

    (output_path / "inferref-run.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return result


@dataclass
class _StreamCapture:
    path: Path
    limit: int
    exceeded: threading.Event = field(default_factory=threading.Event)
    observed_bytes: int = 0
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self, stream: Any) -> None:
        self._thread = threading.Thread(
            target=self._consume,
            args=(stream,),
            name=f"inferref-capture-{self.path.name}",
            daemon=True,
        )
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5)

    def text(self) -> str:
        if not self.path.is_file():
            return ""
        value = self.path.read_bytes().decode("utf-8", errors="replace")
        if self.exceeded.is_set():
            value += (
                f"\n... stream exceeded hard limit of {self.limit} byte(s); "
                "process tree terminated ...\n"
            )
        return value

    def _consume(self, stream: Any) -> None:
        remaining = self.limit
        with self.path.open("wb") as output:
            while True:
                chunk = stream.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                self.observed_bytes += len(chunk)
                if remaining:
                    accepted = chunk[:remaining]
                    output.write(accepted)
                    output.flush()
                    remaining -= len(accepted)
                if self.observed_bytes > self.limit:
                    self.exceeded.set()
            stream.close()


@dataclass(frozen=True)
class _ArtifactScan:
    total_bytes: int = 0
    files: int = 0
    entries: int = 0
    limit: str | None = None


def _wait_with_limits(
    process: subprocess.Popen[bytes],
    *,
    deadline: float,
    output_path: Path,
    max_artifact_bytes: int,
    max_artifact_files: int,
    captures: tuple[_StreamCapture, _StreamCapture],
    windows_job: int | None,
) -> tuple[str, _ArtifactScan]:
    status = "completed"
    artifact_scan = _ArtifactScan()
    next_artifact_scan = time.monotonic()
    while process.poll() is None:
        if any(capture.exceeded.is_set() for capture in captures):
            status = "output_limit"
            break
        now = time.monotonic()
        if now >= deadline:
            status = "timeout"
            break
        if now >= next_artifact_scan:
            artifact_scan = _scan_artifacts(
                output_path,
                deadline=deadline,
                max_bytes=max_artifact_bytes,
                max_files=max_artifact_files,
            )
            if artifact_scan.limit == "deadline":
                status = "timeout"
                break
            if artifact_scan.limit == "bytes":
                status = "artifact_limit"
                break
            if artifact_scan.limit in ("files", "entries"):
                status = "artifact_file_limit"
                break
            next_artifact_scan = time.monotonic() + _ARTIFACT_SCAN_SECONDS
        time.sleep(_RESOURCE_POLL_SECONDS)

    if windows_job is not None:
        # Closing a kill-on-close Job Object also removes descendants that
        # outlive a normally exiting direct child.
        _close_windows_job(windows_job)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    elif status != "completed":
        _terminate_process_tree(process)
    else:
        process.wait()
        if os.name != "nt":
            # The direct child may exit successfully after spawning a daemon.
            # The adapter contract never permits descendants to outlive a run.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if any(capture.exceeded.is_set() for capture in captures):
            status = "output_limit"
    if status == "completed":
        artifact_scan = _scan_artifacts(
            output_path,
            deadline=deadline,
            max_bytes=max_artifact_bytes,
            max_files=max_artifact_files,
        )
        if artifact_scan.limit == "deadline":
            status = "timeout"
        elif artifact_scan.limit == "bytes":
            status = "artifact_limit"
        elif artifact_scan.limit in ("files", "entries"):
            status = "artifact_file_limit"
    return status, artifact_scan


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        # The direct child may have exited while a descendant ignored SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _assign_windows_kill_job(process: subprocess.Popen[bytes]) -> int:
    """Put the child in a kill-on-close Job Object (Windows only)."""

    import ctypes
    from ctypes import wintypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    info = ExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = 0x00002000
    if not kernel32.SetInformationJobObject(
        handle, 9, ctypes.byref(info), ctypes.sizeof(info)
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(handle)
        raise ctypes.WinError(error)
    if not kernel32.AssignProcessToJobObject(handle, wintypes.HANDLE(process._handle)):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(handle)
        raise ctypes.WinError(error)
    return int(handle)


def _close_windows_job(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(wintypes.HANDLE(handle))


def _scan_artifacts(
    root: Path,
    *,
    deadline: float,
    max_bytes: int,
    max_files: int,
) -> _ArtifactScan:
    total = 0
    files = 0
    entries_seen = 0
    # Directory-only trees are bounded too. This permits ordinary nested
    # layouts while preventing an unbounded walk that contains no file bytes.
    max_entries = max(1_024, max_files * 2)
    pending = [root]
    visited: set[tuple[int, int]] = set()
    while pending:
        if time.monotonic() >= deadline:
            return _ArtifactScan(total, files, entries_seen, "deadline")
        directory = pending.pop()
        try:
            directory_stat = directory.stat()
        except OSError:
            continue
        identity = (directory_stat.st_dev, directory_stat.st_ino)
        if identity in visited:
            continue
        visited.add(identity)
        entries = directory.iterdir()
        while True:
            try:
                entry = next(entries)
            except StopIteration:
                break
            except OSError:
                break
            entries_seen += 1
            if entries_seen > max_entries:
                return _ArtifactScan(total, files, entries_seen, "entries")
            if (
                entries_seen % _DEADLINE_CHECK_ENTRIES == 0
                and time.monotonic() >= deadline
            ):
                return _ArtifactScan(total, files, entries_seen, "deadline")
            try:
                entry_stat = entry.lstat()
                attributes = getattr(entry_stat, "st_file_attributes", 0)
                is_reparse_point = bool(
                    attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                )
                if stat.S_ISLNK(entry_stat.st_mode) or is_reparse_point:
                    files += 1
                    total += entry_stat.st_size
                elif stat.S_ISDIR(entry_stat.st_mode):
                    pending.append(entry)
                else:
                    files += 1
                    total += entry_stat.st_size
                if files > max_files:
                    return _ArtifactScan(total, files, entries_seen, "files")
                if total > max_bytes:
                    return _ArtifactScan(total, files, entries_seen, "bytes")
            except OSError:
                continue
    return _ArtifactScan(total, files, entries_seen)
