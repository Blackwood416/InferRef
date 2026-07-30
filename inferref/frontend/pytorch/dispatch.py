"""Dispatcher-level runtime tracing (SPEC §7.1; IR §7, §22-§25).

``InferRefDispatchMode`` intercepts every dispatched ATen operator and records
it as an :class:`~inferref.ir.operator.OperatorRecord` in real execution order.

Two details drive the shape of this code:

**Mutation is detected from the operator schema, not from ``Tensor._version``.**
The version counter is bumped by the autograd layer, which sits above
``TorchDispatchMode``; from inside ``__torch_dispatch__`` it reads identically
before and after the operator runs. ``schema.arguments[i].alias_info.is_write``
is authoritative and available, so InferRef advances its own storage versions
from it (IR §15, §24).

**Input values are interned before the operator runs, outputs after.** For an
in-place operator that yields two distinct immutable value records for the same
tensor object, differing in ``storage_version`` — exactly what IR §2.4 requires.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from inferref.ir.operator import AliasEffect, Effects, OperatorRecord, StorageMutation
from inferref.ir.values import (
    DictValue,
    ListValue,
    NoneValue,
    OpaqueValue,
    ScalarValue,
    StringValue,
    TensorRef,
    TupleValue,
    Value,
)
from inferref.frontend.pytorch.identity import Identity, torch_dtype_name


def canonical_parts(func: Any) -> tuple[str, str, str]:
    """Split an ``OpOverload`` into ``(namespace, op, overload)`` (IR §23)."""
    schema = getattr(func, "_schema", None)
    if schema is not None:
        # schema.name is "aten::mm"; overload_name is "" for the default overload.
        namespace, _, op = schema.name.partition("::")
        overload = schema.overload_name or "default"
        return namespace or "aten", op or str(func), overload
    text = str(func)
    parts = text.split(".")
    if len(parts) >= 3:
        return parts[0], ".".join(parts[1:-1]), parts[-1]
    return "unknown", text, "default"


def mutated_arg_positions(func: Any) -> list[int]:
    """Schema argument indices the operator declares it writes to (IR §24)."""
    schema = getattr(func, "_schema", None)
    if schema is None:
        return []
    positions: list[int] = []
    for index, argument in enumerate(schema.arguments):
        alias = argument.alias_info
        if alias is not None and alias.is_write:
            positions.append(index)
    return positions


def iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield every tensor nested inside a runtime argument."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)


def iter_written_tensors(
    func: Any, args: tuple, kwargs: dict
) -> Iterator[torch.Tensor]:
    """Yield every tensor the operator schema declares it writes to (IR §24).

    Binding a declared write to the runtime argument that carries it needs more
    than an index into ``args``:

    * **Keyword-only writes.** ``aten.add.out`` declares its write on ``out``,
      which is ``kwarg_only`` and arrives in ``kwargs``; its schema index (3)
      is past the end of ``args``. ``out=`` variants are common, so binding by
      position alone silently loses their mutations.
    * **Container writes.** ``aten._foreach_add_.Scalar`` declares its write on
      ``self``, typed ``List[Tensor]``. The written tensors are nested inside a
      list, not the argument itself.
    * **Positional arguments passed by name.** A non-``kwarg_only`` argument may
      still be supplied as a keyword.

    Arguments that were left at their default are simply absent and skipped.
    """
    schema = getattr(func, "_schema", None)
    if schema is None:
        return
    for index, argument in enumerate(schema.arguments):
        alias = argument.alias_info
        if alias is None or not alias.is_write:
            continue
        if not argument.kwarg_only and index < len(args):
            bound = args[index]
        elif argument.name in kwargs:
            bound = kwargs[argument.name]
        else:
            continue
        yield from iter_tensors(bound)


def _scalar_value(obj: Any) -> Value:
    # bool must precede int: bool is a subclass of int in Python.
    if isinstance(obj, bool):
        return ScalarValue.from_number(obj, "bool")
    if isinstance(obj, int):
        return ScalarValue.from_number(obj, "int64")
    if isinstance(obj, float):
        return ScalarValue.from_number(obj, "float64")
    if isinstance(obj, complex):
        return ScalarValue.from_number(obj, "complex128")
    raise TypeError(type(obj))


@dataclass
class TraceRecorder:
    """Accumulates operator and value records during one trace."""

    identity: Identity = field(default_factory=Identity)
    operators: list[OperatorRecord] = field(default_factory=list)

    #: Hook invoked for every newly interned tensor value, letting the session
    #: apply capture policy and parameter classification.
    on_new_value: Callable[[torch.Tensor, int, bool], None] | None = None
    #: Returns the module id stack for the operator currently executing.
    module_stack: Callable[[], tuple[int, ...]] = lambda: ()
    #: Returns a source record id for the operator currently executing.
    capture_source: Callable[[], int | None] = lambda: None

    _next_op_id: int = 1
    _next_execution_index: int = 0

    def to_value(
        self, obj: Any, *, is_output: bool, memo: dict[tuple, int] | None = None
    ) -> Value:
        """Convert a runtime argument or result into an IR value (IR §9-§11).

        ``memo`` is supplied when converting an operator's *result* so that
        outputs get freshly allocated value records (see :meth:`Identity.intern`).
        """
        if obj is None:
            return NoneValue()
        if isinstance(obj, torch.Tensor):
            value_id, is_new = self.identity.intern(obj, memo)
            if is_new and self.on_new_value is not None:
                self.on_new_value(obj, value_id, is_output)
            return TensorRef(value_id=value_id)
        if isinstance(obj, str):
            return StringValue(value=obj)
        if isinstance(obj, (bool, int, float, complex)):
            return _scalar_value(obj)
        if isinstance(obj, (list,)):
            return ListValue(
                items=tuple(self.to_value(i, is_output=is_output, memo=memo) for i in obj)
            )
        if isinstance(obj, tuple):
            return TupleValue(
                items=tuple(self.to_value(i, is_output=is_output, memo=memo) for i in obj)
            )
        if isinstance(obj, dict):
            return DictValue(
                items=tuple(
                    (
                        self.to_value(k, is_output=is_output, memo=memo),
                        self.to_value(v, is_output=is_output, memo=memo),
                    )
                    for k, v in obj.items()
                )
            )
        if isinstance(obj, torch.dtype):
            return OpaqueValue(type="torch.dtype", repr=torch_dtype_name(obj), portable=True)
        if isinstance(obj, torch.device):
            return OpaqueValue(type="torch.device", repr=str(obj), portable=True)
        if isinstance(obj, torch.layout):
            return OpaqueValue(type="torch.layout", repr=str(obj), portable=True)
        if isinstance(obj, torch.memory_format):
            return OpaqueValue(type="torch.memory_format", repr=str(obj), portable=True)
        # IR §41: anything else is diagnostic-only and blocks independent repro.
        return OpaqueValue(type=type(obj).__name__, repr=repr(obj)[:200], portable=False)

    def next_op_id(self) -> tuple[int, int]:
        op_id = self._next_op_id
        index = self._next_execution_index
        self._next_op_id += 1
        self._next_execution_index += 1
        return op_id, index


class InferRefDispatchMode(TorchDispatchMode):
    """Records every dispatched operator into a :class:`TraceRecorder`."""

    def __init__(
        self,
        recorder: TraceRecorder,
        *,
        should_record: Callable[[], bool] | None = None,
        max_ops: int | None = None,
    ) -> None:
        super().__init__()
        self.recorder = recorder
        self._should_record = should_record or (lambda: True)
        self._max_ops = max_ops
        self._local = threading.local()
        self.skipped_ops = 0

    # -- reentrancy -------------------------------------------------------

    @property
    def _busy(self) -> bool:
        return getattr(self._local, "busy", False)

    @_busy.setter
    def _busy(self, value: bool) -> None:
        self._local.busy = value

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Suspend recording for the duration of the block.

        Needed by callers that touch tensors while the mode is active but
        outside ``__torch_dispatch__`` — for example
        :meth:`~inferref.frontend.pytorch.session.TraceSession.mark_input`,
        whose tensor capture would otherwise have its own ``detach``/``view``
        calls recorded as model operators.
        """
        previous = self._busy
        self._busy = True
        try:
            yield
        finally:
            self._busy = previous

    # -- dispatch ---------------------------------------------------------

    def __torch_dispatch__(
        self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None
    ) -> Any:
        kwargs = kwargs or {}

        # Tensor capture calls .contiguous()/.view()/.cpu(), which re-enter this
        # mode. Pass those through untouched rather than recording them.
        if self._busy:
            return func(*args, **kwargs)

        if self._max_ops is not None and len(self.recorder.operators) >= self._max_ops:
            return func(*args, **kwargs)

        if not self._should_record():
            self.skipped_ops += 1
            return func(*args, **kwargs)

        self._busy = True
        try:
            return self._record(func, args, kwargs)
        finally:
            self._busy = False

    def _record(self, func: Any, args: tuple, kwargs: dict) -> Any:
        recorder = self.recorder
        identity = recorder.identity

        # 1. Snapshot the *pre-call* state of every argument.
        positional = tuple(recorder.to_value(a, is_output=False) for a in args)
        keyword = {k: recorder.to_value(v, is_output=False) for k, v in kwargs.items()}

        # 2. Resolve every tensor this operator declares it will write to and
        #    remember its pre-call storage. We must retain the tensor objects,
        #    not only storage ids: resize_/set_ may rebind the same runtime
        #    object to another allocation during the call.
        written_targets: list[tuple[torch.Tensor, int]] = []
        seen_targets: set[int] = set()
        for target in iter_written_tensors(func, args, kwargs):
            storage_id = identity.storage_id_of(target)
            object_key = id(target)
            if storage_id is not None and object_key not in seen_targets:
                seen_targets.add(object_key)
                written_targets.append((target, storage_id))

        # 3. Execute for real.
        out = func(*args, **kwargs)

        # 4. Advance versions only for allocations that still back a writable
        #    target after execution. If every target moved to another storage,
        #    this was an object/storage rebind, not a mutation of the old
        #    allocation. The new allocation starts at version zero and is
        #    represented by the output value plus a same_object alias effect.
        written_storages: list[int] = []
        seen_storages: set[int] = set()
        for target, storage_before in written_targets:
            storage_after = identity.storage_id_of(target)
            if storage_after == storage_before and storage_before not in seen_storages:
                seen_storages.add(storage_before)
                written_storages.append(storage_before)

        mutations: list[StorageMutation] = []
        for storage_id in written_storages:
            before, after = identity.bump_version(storage_id)
            mutations.append(
                StorageMutation(
                    storage_id=storage_id, version_before=before, version_after=after
                )
            )

        # 5. Intern outputs — now at the bumped version, so an in-place op
        #    yields a distinct immutable value record (IR §2.4). The memo is
        #    per-operator so outputs never collide with their own inputs.
        result = recorder.to_value(out, is_output=True, memo={})

        # 6. Alias effects (IR §25).
        aliases = self._alias_effects(args, kwargs, out, positional, keyword, result, identity)

        op_id, execution_index = recorder.next_op_id()
        namespace, op_name, overload = canonical_parts(func)
        record = OperatorRecord(
            id=op_id,
            execution_index=execution_index,
            namespace=namespace,
            op=op_name,
            overload=overload,
            positional_args=positional,
            keyword_args=keyword,
            result=result,
            source_id=recorder.capture_source(),
            module_stack=recorder.module_stack(),
            effects=Effects(mutated_storages=tuple(mutations), aliases=tuple(aliases)),
        )
        recorder.operators.append(record)
        return out

    def _alias_effects(
        self,
        args: tuple,
        kwargs: dict,
        out: Any,
        positional: tuple[Value, ...],
        keyword: dict[str, Value],
        result: Value,
        identity: Identity,
    ) -> list[AliasEffect]:
        """Detect output/input aliasing by comparing observed identities."""
        inputs: list[tuple[TensorRef, torch.Tensor]] = []
        for value, obj in zip(positional, args):
            if isinstance(value, TensorRef) and isinstance(obj, torch.Tensor):
                inputs.append((value, obj))
        for key, value in keyword.items():
            obj = kwargs.get(key)
            if isinstance(value, TensorRef) and isinstance(obj, torch.Tensor):
                inputs.append((value, obj))
        if not inputs:
            return []

        outputs: list[tuple[TensorRef, torch.Tensor]] = []
        _collect_tensor_outputs(result, out, outputs)

        effects: list[AliasEffect] = []
        for out_ref, out_tensor in outputs:
            out_storage = identity.storage_id_of(out_tensor)
            if out_storage is None:
                continue
            same_object_recorded = False
            storage_alias_recorded = False
            for in_ref, in_tensor in inputs:
                if in_tensor is out_tensor:
                    if not same_object_recorded:
                        effects.append(
                            AliasEffect(out_ref.value_id, in_ref.value_id, "same_object")
                        )
                        same_object_recorded = True
                    continue
                # One canonical storage relationship is enough to establish
                # physical aliasing. Recording every duplicate alias argument
                # creates noisy region boundaries; same_object is retained
                # separately so set_(source) can express both object continuity
                # and its new backing storage.
                if storage_alias_recorded:
                    continue
                if identity.storage_id_of(in_tensor) != out_storage:
                    continue
                same_layout = (
                    tuple(in_tensor.shape) == tuple(out_tensor.shape)
                    and tuple(in_tensor.stride()) == tuple(out_tensor.stride())
                    and in_tensor.storage_offset() == out_tensor.storage_offset()
                )
                relationship = "shared_storage" if same_layout else "view"
                effects.append(AliasEffect(out_ref.value_id, in_ref.value_id, relationship))
                storage_alias_recorded = True
        return effects


def _collect_tensor_outputs(
    value: Value, obj: Any, into: list[tuple[TensorRef, torch.Tensor]]
) -> None:
    """Pair up IR output refs with the runtime tensors they describe."""
    if isinstance(value, TensorRef) and isinstance(obj, torch.Tensor):
        into.append((value, obj))
        return
    if isinstance(value, (ListValue, TupleValue)) and isinstance(obj, (list, tuple)):
        for sub_value, sub_obj in zip(value.items, obj):
            _collect_tensor_outputs(sub_value, sub_obj, into)
