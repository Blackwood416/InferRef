"""Create only the Agent-visible workspace for the RoPE evaluation v0.2."""

from __future__ import annotations

import argparse
from pathlib import Path

from inferref.agent.evaluation import EvaluationBenchmark, prepare_workspace


def setup_workspace(output: str | Path) -> Path:
    benchmark = EvaluationBenchmark.load(Path(__file__).with_name("benchmark.json"))
    return prepare_workspace(benchmark, Path(output))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args(argv)
    root = setup_workspace(args.workspace)
    print(f"Wrote blind Agent evaluation workspace to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
