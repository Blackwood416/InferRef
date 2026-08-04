"""Compatibility facade for the split adapter implementation."""

from inferref.agent.artifact_policy import ArtifactScan, scan_artifacts
from inferref.agent.executor import execute_adapter
from inferref.agent.process_policy import (
    StreamCapture,
    assign_windows_kill_job,
    close_windows_job,
    terminate_process_tree,
    wait_with_limits,
)

# Private aliases are retained for the evaluation harness and older tests.
_ArtifactScan = ArtifactScan
_StreamCapture = StreamCapture
_assign_windows_kill_job = assign_windows_kill_job
_close_windows_job = close_windows_job
_scan_artifacts = scan_artifacts
_terminate_process_tree = terminate_process_tree
_wait_with_limits = wait_with_limits

__all__ = ["execute_adapter"]
