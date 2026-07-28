"""Root fixtures shared by both test suites.

Deliberately free of any PyTorch import: ``tests/core/`` must run in an
environment where torch is not installed at all, which is how the dependency
layering of Trace IR §57 criterion 10 is enforced in CI.

The torch-dependent fixtures live in ``tests/frontend/conftest.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make `examples/` importable so the frontend suite can reuse the mini-Llama
# reference model.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    return tmp_path / "trace"
