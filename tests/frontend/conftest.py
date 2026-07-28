"""Fixtures for the PyTorch frontend suite.

Everything under ``tests/frontend/`` needs the ``torch`` extra. Collection is
skipped rather than failed when it is absent, so ``pytest tests/`` still works
in a core-only environment.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="frontend tests need the torch extra")


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
