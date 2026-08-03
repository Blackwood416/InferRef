"""Isolated entry point for formal InferRef Agent evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from inferref.agent.evaluation import _read_regular_file
from inferref.agent.evaluation_host import _evaluate_benchmark_core


def _worker_evidence() -> dict[str, Any]:
    launch_digest = hashlib.sha256(
        json.dumps(sys.orig_argv, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "python_isolated": bool(sys.flags.isolated),
        "entry_module": "inferref.agent.evaluation_worker",
        "launch_policy_sha256": launch_digest,
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
    worker_evidence = _worker_evidence()
    if worker_evidence["python_isolated"] is not True:
        raise RuntimeError("formal evaluation worker requires Python isolated mode")
    request = json.loads(_read_regular_file(path))
    if not isinstance(request, dict):
        raise TypeError("formal worker request root must be an object")
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
