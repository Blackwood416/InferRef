from __future__ import annotations

from pathlib import Path

from inferref.suite import load_suite, validate_suite


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus" / "xpu-v0.1" / "suite.json"


def test_versioned_xpu_corpus_is_valid_and_weight_free() -> None:
    report = validate_suite(CORPUS)
    assert report["cases"] == 13

    suite = load_suite(CORPUS)
    tags = {tag for case in suite.cases for tag in case.tags}
    assert {"rmsnorm", "rope", "kv-cache", "prefill", "decode"} <= tags
    assert {"tiny-llama", "tiny-qwen3.5", "model-derived"} <= tags

    files = [path for path in CORPUS.parent.rglob("*") if path.is_file()]
    assert sum(path.stat().st_size for path in files) < 1_000_000
    assert not any(path.suffix.lower() in {".bin", ".pt", ".pth", ".safetensors"} for path in files)
