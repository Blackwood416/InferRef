"""Isolated entry point for formal InferRef Agent evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from inferref.agent.evaluation import (
    _canonical_json_sha256,
    _read_regular_file,
)
from inferref.agent.evaluation_host import _evaluate_benchmark_core


def _executable_file_evidence(path: str) -> dict[str, Any]:
    """Hash the worker Python executable without publishing its absolute path."""

    resolved = Path(path)
    try:
        resolved = resolved.resolve(strict=True)
    except OSError:
        pass
    try:
        payload = _read_regular_file(resolved)
    except OSError as exc:
        return {
            "name": resolved.name,
            "size": None,
            "sha256": None,
            "error": str(exc)[:256],
        }
    return {
        "name": resolved.name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _worker_evidence(request_sha256: str) -> dict[str, Any]:
    """Canonical launch-policy evidence bound to the exact request bytes."""

    policy = {
        "flags": ["-I", "-m"],
        "entry_module": "inferref.agent.evaluation_worker",
        "request_transport": "regular-file-json",
        "request_schema_version": "0.1",
        "request_sha256": request_sha256,
    }
    return {
        "python_isolated": bool(sys.flags.isolated),
        "entry_module": policy["entry_module"],
        "launch_policy": policy,
        "launch_policy_sha256": _canonical_json_sha256(policy),
        "python_executable": _executable_file_evidence(sys.executable),
    }


def _optional_string(request: dict[str, Any], key: str) -> str | None:
    value = request.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"formal worker request {key!r} must be a string or null")
    return value


def _required_string(request: dict[str, Any], key: str) -> str:
    value = _optional_string(request, key)
    if value is None:
        raise ValueError(f"formal worker request {key!r} is required")
    return value


def run_request(path: Path) -> dict[str, Any]:
    if not bool(sys.flags.isolated):
        raise RuntimeError("formal evaluation worker requires Python isolated mode")
    request_bytes = _read_regular_file(path)
    request = json.loads(request_bytes.decode("utf-8"))
    if not isinstance(request, dict):
        raise TypeError("formal worker request root must be an object")
    worker_evidence = _worker_evidence(hashlib.sha256(request_bytes).hexdigest())
    allowed = {
        "benchmark",
        "agents",
        "report_dir",
        "public_attestation",
        "claude_settings",
        "claude_model",
    }
    if set(request) != allowed:
        raise ValueError("formal worker request has missing or unexpected fields")
    agents = request["agents"]
    if (
        not isinstance(agents, list)
        or not agents
        or not all(isinstance(item, str) and item for item in agents)
    ):
        raise ValueError("formal worker request agents must be a non-empty string list")
    return _evaluate_benchmark_core(
        _required_string(request, "benchmark"),
        agents=tuple(agents),
        report_dir=_required_string(request, "report_dir"),
        claude_settings=_optional_string(request, "claude_settings"),
        claude_model=_optional_string(request, "claude_model"),
        public_attestation=_required_string(request, "public_attestation"),
        formal_worker_evidence=worker_evidence,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    run_request(Path(args.request).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
