"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make `examples/` importable so tests can reuse the mini-Llama reference model.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch", reason="tracing tests need the torch extra")


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    return tmp_path / "trace"


@pytest.fixture
def mini_llama():
    """A small deterministic Llama-style block (RMSNorm + RoPE + GQA + SwiGLU)."""
    from examples.mini_llama.model import build_model

    return build_model(seed=0)


@pytest.fixture
def mini_llama_input(mini_llama):
    """Deterministic hidden states for one prefill step."""
    from examples.mini_llama.model import build_inputs

    return build_inputs(mini_llama, batch=1, seq_len=8, seed=1)
