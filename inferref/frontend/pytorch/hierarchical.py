"""Hierarchical step trace compilation (SPEC §7.2; 0.12 milestone M3).

``compile_step_trace`` runs exactly one prefill/decode step of a decoder model
under the normal :class:`~inferref.frontend.pytorch.session.TraceSession`
capture and adds nested :class:`~inferref.ir.region.RegionRecord` spans on top of
the captured operator graph::

    WholeModel@prefill
      TransformerBlock@layers.0
        RMSNorm@layers.0.input_layernorm
        Attention@layers.0.self_attn
          Linear@layers.0.self_attn.q_proj
          ...
        SwiGLU@layers.0.mlp
          Linear@layers.0.mlp.down_proj

Operator capture is *not* reimplemented here. Every forward hook only records
the operator execution index at which a module entered and exited; the region
node sets are then read off the single authoritative trace that the session
already recorded, and their boundary inputs/outputs follow the standard IR §34
rule, so each span is directly extractable as a testcase.

Quantization descriptors are attached only from an explicit caller mapping
(:class:`LeafQuantization`). Live ``__torch_dispatch__`` interception does not
see a third-party quantized tensor subclass such as ``bitsandbytes.Params4bit``,
so automatic capture of those is an explicit 0.12 non-goal; callers project the
checkpoint's ``QuantizationSpec`` parameters onto the leaf projection they want
described.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch

from inferref.frontend.pytorch.session import TraceSession
from inferref.ir.operator import Annotation
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import TensorValueRecord
from inferref.ir.validate import validate_package
from inferref.ir.version import QUANTIZED_FORMAT_VERSION
from inferref.quantization.spec import AFFINE_SCHEMES, QuantizationSpec
from inferref.region.boundary import derive_boundary

#: Default selector for decoder blocks: a module path ending in
#: ``layers.<digits>``, optionally below a prefix such as ``model.``.
DECODER_LAYER_PATTERN = r"(?:^|\.)layers\.\d+$"

#: Decoder sub-blocks that get their own span when the layer has them.
SUB_BLOCK_NAMES = (
    "input_layernorm",
    "self_attn",
    "post_attention_layernorm",
    "mlp",
)

#: Leaf projections that get a ``Linear`` span when the model has them.
PROJECTION_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

WHOLE_MODEL = "WholeModel"
TRANSFORMER_BLOCK = "TransformerBlock"
ATTENTION = "Attention"
SWIGLU = "SwiGLU"
MLP = "MLP"
LINEAR = "Linear"

_STEP_ID = re.compile(r"[A-Za-z0-9_.-]+")


class HierarchicalTraceError(ValueError):
    """Raised when a step trace cannot be compiled as requested."""


@dataclass(frozen=True)
class LeafQuantization:
    """Explicit quantization descriptor for one leaf projection.

    The caller supplies the already-projected parameters of the checkpoint: no
    third-party tensor subclass is inspected, and the compiler derives the
    :class:`~inferref.quantization.QuantizationSpec` value IDs from the values it
    interns for these tensors. ``logical_shape`` is taken from the projection's
    observed weight, so a descriptor can never disagree with the trace.
    """

    scheme: str
    scale: torch.Tensor
    zero_point: torch.Tensor | None = None
    codebook: torch.Tensor | None = None
    granularity: str = "per_tensor"
    axis: int | None = None
    group_size: int | None = None


@dataclass(frozen=True)
class _Selection:
    """One module that receives a span."""

    path: str
    kind: str
    module: torch.nn.Module


@dataclass(frozen=True)
class _Span:
    """An observed module execution over operator execution indices."""

    path: str
    kind: str
    first_index: int
    last_index: int


@dataclass
class _SpanCollector:
    """Maps module entry/exit onto operator execution indices."""

    operator_count: Callable[[], int]
    spans: list[_Span] = field(default_factory=list)
    _starts: dict[str, list[int]] = field(default_factory=dict)
    _handles: list[Any] = field(default_factory=list)

    def attach(self, selection: Sequence[_Selection]) -> None:
        for item in selection:
            self._handles.append(
                item.module.register_forward_pre_hook(
                    partial(self._enter, item.path)
                )
            )
            self._handles.append(
                item.module.register_forward_hook(
                    partial(self._exit, item.path, item.kind)
                )
            )

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _enter(self, path: str, module: torch.nn.Module, args: tuple) -> None:
        # A path stack rather than a single index keeps a module that re-enters
        # itself during its own forward from overwriting its outer span.
        self._starts.setdefault(path, []).append(self.operator_count())

    def _exit(
        self,
        path: str,
        kind: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ) -> None:
        starts = self._starts.get(path)
        if not starts:
            return
        first_index = starts.pop()
        last_index = self.operator_count() - 1
        # A module that dispatches nothing (a pure Python pass-through) has no
        # operator span to describe and is left out rather than emitting a
        # region with an empty node set.
        if last_index < first_index:
            return
        self.spans.append(
            _Span(
                path=path,
                kind=kind,
                first_index=first_index,
                last_index=last_index,
            )
        )


class _StepSession(TraceSession):
    """A step capture that declares the quantized trace format when needed.

    Embedded ``quantization`` descriptors are only valid in trace ``1.0``
    (IR §5), and the session would otherwise save the manifest before the
    descriptors attached during the step are visible.
    """

    def build(self) -> TracePackage:
        package = super().build()
        if any(value.quantization is not None for value in package.graph.values):
            package.manifest.format_version = QUANTIZED_FORMAT_VERSION
        return package


def compile_step_trace(
    model: torch.nn.Module,
    example_inputs: Mapping[str, Any] | Sequence[Any] | torch.Tensor,
    step_id: str,
    *,
    output_dir: str | Path,
    output_name: str = "output",
    capture_tensors: str = "all",
    leaf_quantization: Mapping[str, LeafQuantization] | None = None,
    layer_pattern: str = DECODER_LAYER_PATTERN,
    seed: int | None = None,
) -> TracePackage:
    """Run one generation step and compile its hierarchical trace package.

    ``example_inputs`` is either a mapping of keyword arguments (tensors are
    marked as trace inputs under their keyword name) or a positional sequence
    (marked as ``input_0``, ``input_1``, ...). The step output must be a single
    tensor; wrap a model whose step returns a container in a module that returns
    the tensor to trace.

    Region boundaries are only extractable as testcases when their values carry
    payloads, so the default capture policy is ``"all"``. ``leaf_quantization``
    is the explicit projection path described in :class:`LeafQuantization`; it
    is applied after the step and raises rather than guessing when the observed
    storage cannot express the requested scheme.
    """
    if not isinstance(model, torch.nn.Module):
        raise HierarchicalTraceError(
            f"model must be a torch.nn.Module, got {type(model).__name__}"
        )
    if not _STEP_ID.fullmatch(step_id):
        raise HierarchicalTraceError(
            f"step_id {step_id!r} must match {_STEP_ID.pattern!r}"
        )
    selection = _select_modules(model, layer_pattern)

    session = _StepSession(
        output=output_dir,
        capture_tensors=capture_tensors,
        seed=seed,
    )
    # TraceSession does not expose its recorder; the span index is the number of
    # operators already recorded, and recording it independently would duplicate
    # operator capture (SPEC §60: one authoritative physical trace).
    collector = _SpanCollector(lambda: len(session._recorder.operators))

    with session:
        _mark_inputs(session, example_inputs)
        collector.attach(selection)
        try:
            with torch.no_grad():
                result = _invoke(model, example_inputs)
        finally:
            collector.detach()
        _mark_output(session, result, output_name)
        if leaf_quantization:
            _attach_leaf_quantization(session, model, leaf_quantization)

    package = session.package
    if package is None:  # pragma: no cover - raised only if the step failed
        raise HierarchicalTraceError("step capture produced no trace package")
    # The session saves before regions exist, so the region file is rewritten.
    package.regions = _build_regions(package, collector.spans, step_id)
    package.save_regions()

    errors = [
        issue
        for issue in validate_package(package, check_payloads=True)
        if issue.severity == "error"
    ]
    if errors:
        raise HierarchicalTraceError(
            "compiled step trace is invalid: "
            + "; ".join(str(issue) for issue in errors)
        )
    return package


# -- step execution --------------------------------------------------------


def _invoke(
    model: torch.nn.Module,
    example_inputs: Mapping[str, Any] | Sequence[Any] | torch.Tensor,
) -> Any:
    if isinstance(example_inputs, Mapping):
        return model(**example_inputs)
    if isinstance(example_inputs, torch.Tensor):
        return model(example_inputs)
    return model(*example_inputs)


def _mark_inputs(
    session: TraceSession,
    example_inputs: Mapping[str, Any] | Sequence[Any] | torch.Tensor,
) -> None:
    if isinstance(example_inputs, Mapping):
        for name, value in example_inputs.items():
            if isinstance(value, torch.Tensor):
                session.mark_input(name, value)
        return
    tensors = (
        [example_inputs]
        if isinstance(example_inputs, torch.Tensor)
        else list(example_inputs)
    )
    for index, value in enumerate(tensors):
        if isinstance(value, torch.Tensor):
            session.mark_input(f"input_{index}", value)


def _mark_output(session: TraceSession, result: Any, output_name: str) -> None:
    if not isinstance(result, torch.Tensor):
        raise HierarchicalTraceError(
            f"step output must be a tensor, got {type(result).__name__}; wrap the "
            "model in a module that returns the step tensor to trace"
        )
    session.mark_output(output_name, result)


# -- module selection ------------------------------------------------------


def _select_modules(
    model: torch.nn.Module, layer_pattern: str
) -> list[_Selection]:
    """Return every module that gets a region span, parents before children."""
    named = dict(model.named_modules())
    layer_re = re.compile(layer_pattern)
    layer_paths = sorted(
        (path for path in named if layer_re.search(path)),
        key=_layer_order_key,
    )
    if not layer_paths:
        raise HierarchicalTraceError(
            f"no decoder layer modules match {layer_pattern!r}; "
            "pass layer_pattern for a model that does not use 'layers.<i>'"
        )

    selection = [_Selection(path="", kind=WHOLE_MODEL, module=model)]
    selection.extend(
        _Selection(path=path, kind=TRANSFORMER_BLOCK, module=named[path])
        for path in layer_paths
    )

    for layer_path in layer_paths:
        layer = named[layer_path]
        for name in SUB_BLOCK_NAMES:
            sub = getattr(layer, name, None)
            if isinstance(sub, torch.nn.Module):
                selection.append(
                    _Selection(
                        path=f"{layer_path}.{name}",
                        kind=_sub_block_kind(name, sub),
                        module=sub,
                    )
                )

    for name, module in sorted(named.items()):
        if name.rsplit(".", 1)[-1] not in PROJECTION_NAMES:
            continue
        if next(iter(module.children()), None) is not None:
            continue
        if not any(name.startswith(f"{layer_path}.") for layer_path in layer_paths):
            continue
        selection.append(_Selection(path=name, kind=LINEAR, module=module))
    return selection


def _layer_order_key(path: str) -> tuple[int, str]:
    """Order ``layers.2`` before ``layers.10`` (string sort would not)."""
    suffix = path.rsplit(".", 1)[-1]
    return (int(suffix) if suffix.isdigit() else 0, path)


def _sub_block_kind(name: str, module: torch.nn.Module) -> str:
    if name == "self_attn":
        return ATTENTION
    if name == "mlp":
        gated = isinstance(getattr(module, "gate_proj", None), torch.nn.Module)
        return SWIGLU if gated else MLP
    return type(module).__name__


# -- region assembly -------------------------------------------------------


def _build_regions(
    package: TracePackage, spans: Sequence[_Span], step_id: str
) -> list[RegionRecord]:
    """Turn observed spans into nested regions in deterministic order."""
    op_id_by_index = {
        op.execution_index: op.id for op in package.graph.ops_in_execution_order()
    }
    # Parents open at the same execution index as their first child, so a longer
    # span sorts first; ties then break on path so ordering is reproducible.
    ordered = sorted(spans, key=lambda s: (s.first_index, -s.last_index, s.path))

    regions: list[RegionRecord] = []
    for span in ordered:
        node_ids = [
            op_id_by_index[index]
            for index in range(span.first_index, span.last_index + 1)
            if index in op_id_by_index
        ]
        if not node_ids:
            continue
        inputs, outputs = derive_boundary(package.graph, node_ids)
        name = (
            f"{WHOLE_MODEL}@{step_id}"
            if span.path == ""
            else f"{span.kind}@{span.path}"
        )
        regions.append(
            RegionRecord(
                id=len(regions),
                name=name,
                node_ids=tuple(node_ids),
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                semantic=Annotation(
                    type="semantic", name=span.kind, confidence=1.0
                ),
                source_ids=_source_ids(package, node_ids),
                creation_method="module",
            )
        )
    return regions


def _source_ids(package: TracePackage, node_ids: Sequence[int]) -> tuple[int, ...]:
    seen: dict[int, None] = {}
    for node_id in node_ids:
        source_id = package.graph.op(node_id).source_id
        if source_id is not None:
            seen.setdefault(source_id, None)
    return tuple(seen)


# -- explicit leaf quantization -------------------------------------------


def _storage_dtype_for(scheme: str) -> str:
    if scheme in AFFINE_SCHEMES:
        return AFFINE_SCHEMES[scheme]
    if scheme == "nf4/v1":
        return "uint4"
    raise HierarchicalTraceError(
        f"unsupported quantization scheme {scheme!r}; expected one of "
        f"{sorted(AFFINE_SCHEMES)} or 'nf4/v1'"
    )


def _attach_leaf_quantization(
    session: TraceSession,
    model: torch.nn.Module,
    mapping: Mapping[str, LeafQuantization],
) -> None:
    """Attach the caller's explicit descriptors to the leaf weight values."""
    # Interning a parameter must not be recorded as a model operator, exactly as
    # :meth:`TraceSession.mark_input` does for external inputs.
    with session._mode.paused():
        for path in sorted(mapping):
            leaf = mapping[path]
            try:
                module = model.get_submodule(path)
            except AttributeError as exc:
                raise HierarchicalTraceError(
                    f"leaf quantization names unknown module {path!r}"
                ) from exc
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor):
                raise HierarchicalTraceError(
                    f"leaf module {path!r} has no weight tensor to quantize"
                )
            weight_value = _intern_value(session, weight)
            if weight_value.quantization is not None:
                raise HierarchicalTraceError(
                    f"leaf module {path!r} is already quantized in this trace"
                )
            storage_dtype = _storage_dtype_for(leaf.scheme)
            if weight_value.dtype != storage_dtype:
                raise HierarchicalTraceError(
                    f"leaf module {path!r} records dtype {weight_value.dtype!r} but "
                    f"scheme {leaf.scheme!r} stores {storage_dtype!r}; live capture "
                    "observes the tensor torch actually dispatches, so packed low-bit "
                    "storage must be projected by the caller first"
                )
            weight_value.quantization = QuantizationSpec(
                logical_shape=tuple(weight_value.shape),
                scale_value_id=_intern_value(session, leaf.scale).id,
                zero_point_value_id=(
                    _intern_value(session, leaf.zero_point).id
                    if leaf.zero_point is not None
                    else None
                ),
                codebook_value_id=(
                    _intern_value(session, leaf.codebook).id
                    if leaf.codebook is not None
                    else None
                ),
                codebook=(
                    "bitsandbytes-nf4/v1" if leaf.codebook is not None else None
                ),
                scheme=leaf.scheme,
                storage_dtype=storage_dtype,
                granularity=leaf.granularity,
                axis=leaf.axis,
                group_size=leaf.group_size,
            )


def _intern_value(session: TraceSession, tensor: torch.Tensor) -> TensorValueRecord:
    """Intern ``tensor`` as a trace value, classifying it like the session does."""
    value_id, is_new = session.identity.intern(tensor)
    if is_new:
        session._on_new_value(tensor, value_id, False)
    return session.identity.value(value_id)
