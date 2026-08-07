"""Small CI-only assertions shared by CircleCI Linux and Windows jobs.

Keeping Python snippets in a real file avoids CircleCI 2.1 interpreting shell
heredoc ``<<`` tokens as configuration parameter interpolation.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys


def assert_no_torch(message: str) -> None:
    if importlib.util.find_spec("torch") is not None:
        raise SystemExit(message)
    print("confirmed: torch is not installed")


def check_frontend(hf_suite: str) -> None:
    import numpy
    import torch

    print(f"python {sys.version.split()[0]}")
    print(f"torch  {torch.__version__}")
    print(f"numpy  {numpy.__version__}")

    try:
        import transformers
    except ImportError:
        transformers = None
        print("transformers not installed (real-model tests will skip)")
    else:
        print(f"transformers {transformers.__version__}")

    if hf_suite.startswith("generic-"):
        if transformers is None:
            raise SystemExit("generic HF suite requested but transformers is absent")
        from transformers import LlamaConfig, LlamaForCausalLM  # noqa: F401

        print("generic Llama model APIs are available")
    elif hf_suite.startswith("qwen35-"):
        if transformers is None:
            raise SystemExit("Qwen3.5 suite requested but transformers is absent")
        required = (
            "Qwen3_5TextConfig",
            "Qwen3_5ForCausalLM",
            "Qwen3_5ForConditionalGeneration",
        )
        missing = [name for name in required if not hasattr(transformers, name)]
        if missing:
            raise SystemExit("Qwen3.5 capability missing: " + ", ".join(missing))
        print("Qwen3.5 model classes are available")
    elif hf_suite:
        raise SystemExit(f"unknown HF suite: {hf_suite}")

    missing_api: list[str] = []
    try:
        from torch.utils._python_dispatch import TorchDispatchMode  # noqa: F401
    except Exception as exc:  # pragma: no cover - CI diagnostic
        missing_api.append(f"TorchDispatchMode: {exc}")
    try:
        from torch.multiprocessing.reductions import StorageWeakRef  # noqa: F401
    except Exception as exc:  # pragma: no cover - CI diagnostic
        missing_api.append(f"StorageWeakRef: {exc}")

    import torch.nn.modules.module as module_api

    for hook in ("register_module_forward_pre_hook", "register_module_forward_hook"):
        if not hasattr(module_api, hook):
            missing_api.append(f"nn.modules.module.{hook}")

    if missing_api:
        raise SystemExit("private APIs unavailable:\n  " + "\n  ".join(missing_api))
    print("all required PyTorch APIs are present")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    no_torch = sub.add_parser("no-torch")
    no_torch.add_argument("--message", required=True)

    frontend = sub.add_parser("frontend")
    frontend.add_argument("--hf-suite", default="")

    args = parser.parse_args()
    if args.command == "no-torch":
        assert_no_torch(args.message)
    else:
        check_frontend(args.hf_suite)


if __name__ == "__main__":
    main()
