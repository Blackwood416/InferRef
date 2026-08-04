"""Cross-platform process-tree, stream, timeout, and artifact limits."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferref.agent.artifact_policy import ArtifactScan, scan_artifacts

_STREAM_CHUNK_BYTES = 64 * 1024
_RESOURCE_POLL_SECONDS = 0.02
_ARTIFACT_SCAN_SECONDS = 0.25


@dataclass
class StreamCapture:
    path: Path
    limit: int
    exceeded: threading.Event = field(default_factory=threading.Event)
    observed_bytes: int = 0
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self, stream: Any) -> None:
        self._thread = threading.Thread(target=self._consume, args=(stream,), daemon=True)
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5)

    def text(self) -> str:
        if not self.path.is_file():
            return ""
        value = self.path.read_bytes().decode("utf-8", errors="replace")
        if self.exceeded.is_set():
            value += f"\n... stream exceeded hard limit of {self.limit} byte(s); process tree terminated ...\n"
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


def wait_with_limits(process: subprocess.Popen[bytes], *, deadline: float, output_path: Path, max_artifact_bytes: int, max_artifact_files: int, captures: tuple[StreamCapture, StreamCapture], windows_job: int | None) -> tuple[str, ArtifactScan]:
    status = "completed"
    scan = ArtifactScan()
    next_scan = time.monotonic()
    while process.poll() is None:
        if any(c.exceeded.is_set() for c in captures):
            status = "output_limit"; break
        now = time.monotonic()
        if now >= deadline:
            status = "timeout"; break
        if now >= next_scan:
            scan = scan_artifacts(output_path, deadline=deadline, max_bytes=max_artifact_bytes, max_files=max_artifact_files)
            if scan.limit == "deadline": status = "timeout"; break
            if scan.limit == "bytes": status = "artifact_limit"; break
            if scan.limit in ("files", "entries"): status = "artifact_file_limit"; break
            next_scan = time.monotonic() + _ARTIFACT_SCAN_SECONDS
        time.sleep(_RESOURCE_POLL_SECONDS)
    if windows_job is not None:
        close_windows_job(windows_job)
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=5)
    elif status != "completed":
        terminate_process_tree(process)
    else:
        process.wait()
        if os.name != "nt":
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        if any(c.exceeded.is_set() for c in captures): status = "output_limit"
    if status == "completed":
        scan = scan_artifacts(output_path, deadline=deadline, max_bytes=max_artifact_bytes, max_files=max_artifact_files)
        if scan.limit == "deadline": status = "timeout"
        elif scan.limit == "bytes": status = "artifact_limit"
        elif scan.limit in ("files", "entries"): status = "artifact_file_limit"
    return status, scan


def terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None: return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False, shell=False, creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired): process.kill()
    else:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=1)
        except subprocess.TimeoutExpired: pass
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait(timeout=5)


def assign_windows_kill_job(process: subprocess.Popen[bytes]) -> int:
    import ctypes
    from ctypes import wintypes
    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]
    class Io(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]
    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Io), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateJobObjectW(None, None)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    info = Extended(); info.BasicLimitInformation.LimitFlags = 0x00002000
    if not kernel32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)) or not kernel32.AssignProcessToJobObject(handle, wintypes.HANDLE(process._handle)):
        error = ctypes.get_last_error(); kernel32.CloseHandle(handle); raise ctypes.WinError(error)
    return int(handle)


def close_windows_job(handle: int) -> None:
    import ctypes
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle(wintypes.HANDLE(handle))
