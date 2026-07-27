"""Trace session — the public tracing entry point (SPEC §54).

::

    with inferref.trace(output="trace/", scope="model.layers.0"):
        output = model(**inputs)

The session owns the dispatch mode, module tracker, source mapper, parameter
index and tensor capture, and assembles their output into a
:class:`~inferref.ir.package.TracePackage`.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from inferref.frontend.pytorch.capture import TensorCapture, normalise_policy
from inferref.frontend.pytorch.dispatch import InferRefDispatchMode, TraceRecorder
from inferref.frontend.pytorch.identity import Identity
from inferref.frontend.pytorch.modules import ModuleTracker
from inferref.frontend.pytorch.params import ParameterIndex
from inferref.frontend.pytorch.sources import SourceMapper, python_version
from inferref.ir.graph import Graph, GraphIO
from inferref.ir.manifest import (
    Capture,
    Determinism,
    Environment,
    Execution,
    Manifest,
    ModelInfo,
    NamedVersion,
    SourcePolicy,
)
from inferref.ir.package import TracePackage
from inferref.ir.values import TensorRef
from inferref.ir.version import FORMAT, FORMAT_VERSION, INFERREF_VERSION

FRONTEND_VERSION = "0.1"


def _transformers_version() -> str | None:
    module = sys.modules.get("transformers")
    return getattr(module, "__version__", None) if module else None


@dataclass
class TraceSession:
    """One reference execution capture."""

    output: str | Path
    scope: str | None = None
    exclude: tuple[str, ...] = ()
    capture_tensors: str = "metadata"
    source_map: bool = True
    module_map: bool = True
    embed_source_text: bool = False
    path_mode: str = "relative"
    max_ops: int | None = None
    max_capture_elements: int = 0
    model_name: str = "unknown"
    model_revision: str | None = None
    seed: int | None = None
    device: str = "cpu"

    identity: Identity = field(default_factory=Identity)
    package: TracePackage | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.output = Path(self.output)
        self._policy = normalise_policy(self.capture_tensors)
        self._modules = ModuleTracker(scope=self.scope, exclude=tuple(self.exclude))
        self._modules.on_root = self._on_root_module
        self._sources = SourceMapper(
            path_mode=self.path_mode,
            embed_source_text=self.embed_source_text,
            base_dir=Path.cwd(),
        )
        self._params = ParameterIndex()
        self._capture = TensorCapture(
            root=self.output, policy=self._policy, max_elements=self.max_capture_elements
        )
        self._recorder = TraceRecorder(
            identity=self.identity,
            on_new_value=self._on_new_value,
            module_stack=lambda: self._modules.stack if self.module_map else (),
            capture_source=self._capture_source,
        )
        self._mode = InferRefDispatchMode(
            self._recorder, should_record=self._should_record, max_ops=self.max_ops
        )
        self._roots: list[torch.nn.Module] = []
        self._graph_inputs: list[GraphIO] = []
        self._graph_outputs: list[GraphIO] = []
        self._warnings: list[str] = []
        self._entered = False

    # -- callbacks --------------------------------------------------------

    def _on_root_module(self, module: torch.nn.Module) -> None:
        self._params.index(module)
        if module not in self._roots:
            self._roots.append(module)
            if self.model_name == "unknown":
                self.model_name = type(module).__name__
            if module.training:
                self._warnings.append(
                    "model is in training mode; reference execution may be non-deterministic "
                    "(dropout, batchnorm updates)"
                )

    def _on_new_value(self, tensor: torch.Tensor, value_id: int, is_output: bool) -> None:
        record = self.identity.value(value_id)
        role, qualified_name = self._params.classify(tensor)
        record.role = role
        record.qualified_name = qualified_name
        record.capture = self._capture.capture(
            tensor, record, is_output=is_output, in_scope=self._modules.in_scope()
        )

    def _should_record(self) -> bool:
        if not self.module_map:
            return True
        return self._modules.in_scope()

    def _capture_source(self) -> int | None:
        if not self.source_map:
            return None
        # Drop this frame plus the dispatch-mode plumbing between us and model code.
        return self._sources.capture(skip=1)

    # -- context management ----------------------------------------------

    def __enter__(self) -> "TraceSession":
        if self._entered:
            raise RuntimeError("TraceSession is not reentrant")
        self._entered = True
        self.output.mkdir(parents=True, exist_ok=True)
        if self.seed is not None:
            torch.manual_seed(self.seed)
        if self.module_map:
            self._modules.start()
        self._mode.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._mode.__exit__(exc_type, exc, tb)
        if self.module_map:
            self._modules.stop()
        if exc_type is None:
            self.package = self.build()
            self.package.save(self.output)

    # -- explicit graph boundary -----------------------------------------

    def mark_input(self, name: str, tensor: torch.Tensor) -> None:
        """Declare an external graph input (IR §37)."""
        with self._mode.paused():
            value_id, is_new = self.identity.intern(tensor)
            if is_new:
                self._on_new_value(tensor, value_id, False)
        self.identity.value(value_id).role = "input"
        self._graph_inputs.append(GraphIO(name=name, value=TensorRef(value_id)))

    def mark_output(self, name: str, tensor: torch.Tensor) -> None:
        """Declare an external graph output (IR §37)."""
        with self._mode.paused():
            value_id, is_new = self.identity.intern(tensor)
            if is_new:
                self._on_new_value(tensor, value_id, True)
        self.identity.value(value_id).role = "output"
        self._graph_outputs.append(GraphIO(name=name, value=TensorRef(value_id)))

    # -- assembly ---------------------------------------------------------

    def build(self) -> TracePackage:
        """Assemble the recorded state into a trace package."""
        graph = Graph(
            operators=list(self._recorder.operators),
            values=self.identity.values(),
            inputs=list(self._graph_inputs),
            outputs=list(self._graph_outputs),
        )
        graph.recompute_links()
        if not graph.inputs:
            graph.inputs = self._infer_graph_inputs(graph)
        if not graph.outputs:
            graph.outputs = self._infer_graph_outputs(graph)

        return TracePackage(
            manifest=self._build_manifest(),
            graph=graph,
            modules=self._modules.records() if self.module_map else [],
            sources=self._sources.records() if self.source_map else [],
            regions=[],
            storages=self.identity.storage_records(),
            root=self.output,
        )

    def _infer_graph_inputs(self, graph: Graph) -> list[GraphIO]:
        """Values consumed but never produced are the trace's inputs (IR §37)."""
        out: list[GraphIO] = []
        for value in graph.values:
            if value.producer is None and value.consumers:
                name = value.qualified_name or f"value_{value.id}"
                if value.role == "activation":
                    value.role = "input"
                out.append(GraphIO(name=name, value=TensorRef(value.id)))
        return out

    def _infer_graph_outputs(self, graph: Graph) -> list[GraphIO]:
        """Values produced but never consumed are the trace's outputs (IR §37)."""
        out: list[GraphIO] = []
        for value in graph.values:
            if value.producer is not None and not value.consumers:
                if value.role == "activation":
                    value.role = "output"
                out.append(GraphIO(name=f"value_{value.id}", value=TensorRef(value.id)))
        return out

    def _build_manifest(self) -> Manifest:
        warnings = list(self._warnings)
        if self._mode.skipped_ops:
            warnings.append(
                f"{self._mode.skipped_ops} operators were skipped by the scope filter"
            )
        packages = {"torch": torch.__version__, "numpy": _numpy_version()}
        transformers = _transformers_version()
        if transformers:
            packages["transformers"] = transformers

        return Manifest(
            format=FORMAT,
            format_version=FORMAT_VERSION,
            inferref_version=INFERREF_VERSION,
            frontend=NamedVersion("pytorch", FRONTEND_VERSION),
            reference_framework=NamedVersion("pytorch", torch.__version__),
            model=ModelInfo(name=self.model_name, revision=self.model_revision),
            execution=Execution(mode="inference", device=self.device),
            capture=Capture(
                tensor_policy=self._policy,
                source_mapping=self.source_map,
                module_mapping=self.module_map,
                scope=self.scope,
                exclude=tuple(self.exclude),
                max_ops=self.max_ops,
            ),
            source_policy=SourcePolicy(
                path_mode=self.path_mode, embed_source_text=self.embed_source_text
            ),
            environment=Environment(
                python=python_version(),
                os=platform.system(),
                architecture=platform.machine(),
                device_name=self.device,
                packages=packages,
            ),
            determinism=Determinism(
                seed=self.seed,
                training=any(m.training for m in self._roots),
                grad_enabled=torch.is_grad_enabled(),
                autocast=torch.is_autocast_enabled(),
                warnings=tuple(warnings),
            ),
        )


def _numpy_version() -> str:
    try:
        import numpy

        return numpy.__version__
    except Exception:  # pragma: no cover
        return "unknown"


def trace(
    output: str | Path = "trace/",
    *,
    scope: str | None = None,
    exclude: Iterable[str] = (),
    capture_tensors: str = "metadata",
    source_map: bool = True,
    module_map: bool = True,
    embed_source_text: bool = False,
    path_mode: str = "relative",
    max_ops: int | None = None,
    max_capture_elements: int = 0,
    model_name: str = "unknown",
    seed: int | None = None,
    device: str = "cpu",
) -> TraceSession:
    """Create a :class:`TraceSession` (SPEC §54)."""
    return TraceSession(
        output=output,
        scope=scope,
        exclude=tuple(exclude),
        capture_tensors=capture_tensors,
        source_map=source_map,
        module_map=module_map,
        embed_source_text=embed_source_text,
        path_mode=path_mode,
        max_ops=max_ops,
        max_capture_elements=max_capture_elements,
        model_name=model_name,
        seed=seed,
        device=device,
    )
