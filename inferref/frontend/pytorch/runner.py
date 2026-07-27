"""Run a user script under a trace session (SPEC §33).

``inferref trace run_model.py --scope model.layers.0 --output trace/`` executes
an arbitrary script with tracing active. The script needs no InferRef-specific
cooperation: module hierarchy and scope filtering come from PyTorch's global
module hooks (see :mod:`inferref.frontend.pytorch.modules`).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any, Sequence

from inferref.frontend.pytorch.session import TraceSession


def run_script(
    script: str | Path, session: TraceSession, argv: Sequence[str] = ()
) -> dict[str, Any]:
    """Execute ``script`` under ``session`` and return its module globals."""
    script = Path(script)
    if not script.is_file():
        raise FileNotFoundError(f"script not found: {script}")

    saved_argv = list(sys.argv)
    saved_path = list(sys.path)
    sys.argv = [str(script), *argv]
    # Mirror `python script.py`: the script's directory leads the import path.
    sys.path.insert(0, str(script.parent.resolve()))
    try:
        with session:
            return runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = saved_argv
        sys.path[:] = saved_path
