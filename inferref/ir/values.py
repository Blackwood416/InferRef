"""The Trace IR value system (IR §9-§11).

PyTorch operators do not only receive tensors, so the IR models arguments and
results as a small tagged union::

    tensor  scalar  none  string  list  tuple  dict  opaque

Tensors are referenced *by id* (:class:`TensorRef`); their metadata lives in a
separate :class:`~inferref.ir.tensor_value.TensorValueRecord`. This is what makes
the trace a dataflow graph rather than a flat operator log (IR §2.2).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Union

from inferref.ir.dtypes import is_known_dtype

#: JSON-safe encodings for non-finite floats (IR §11).
SPECIAL_NAN = "nan"
SPECIAL_POS_INF = "+inf"
SPECIAL_NEG_INF = "-inf"
_SPECIALS = {SPECIAL_NAN, SPECIAL_POS_INF, SPECIAL_NEG_INF}


@dataclass(frozen=True)
class TensorRef:
    """A reference to a tensor value record (IR §10)."""

    value_id: int

    kind = "tensor"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "tensor", "value_id": self.value_id}


@dataclass(frozen=True)
class ScalarValue:
    """A scalar argument, e.g. ``dim=1`` or ``alpha=0.5`` (IR §10, §11)."""

    dtype: str
    value: int | float | bool | complex | str
    #: ``"special"`` when ``value`` is one of ``nan`` / ``+inf`` / ``-inf`` (IR §11).
    encoding: str | None = None

    kind = "scalar"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": "scalar", "dtype": self.dtype, "value": self.value}
        if self.encoding is not None:
            out["encoding"] = self.encoding
        return out

    @staticmethod
    def from_number(value: Any, dtype: str) -> ScalarValue:
        """Build a scalar, JSON-encoding non-finite floats per IR §11."""
        if isinstance(value, float) and not math.isfinite(value):
            if math.isnan(value):
                special = SPECIAL_NAN
            else:
                special = SPECIAL_POS_INF if value > 0 else SPECIAL_NEG_INF
            return ScalarValue(dtype=dtype, value=special, encoding="special")
        if isinstance(value, complex):
            return ScalarValue(dtype=dtype, value=[value.real, value.imag], encoding="complex")
        return ScalarValue(dtype=dtype, value=value)

    def as_number(self) -> Any:
        """Decode back to a Python number, resolving special encodings."""
        if self.encoding == "special":
            if self.value == SPECIAL_NAN:
                return math.nan
            if self.value == SPECIAL_POS_INF:
                return math.inf
            if self.value == SPECIAL_NEG_INF:
                return -math.inf
            raise ValueError(f"unknown special scalar encoding: {self.value!r}")
        if self.encoding == "complex":
            real, imag = self.value  # type: ignore[misc]
            return complex(real, imag)
        return self.value


@dataclass(frozen=True)
class NoneValue:
    """A ``None`` argument (IR §10)."""

    kind = "none"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "none"}


@dataclass(frozen=True)
class StringValue:
    """A string argument (IR §10)."""

    value: str

    kind = "string"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "string", "value": self.value}


@dataclass(frozen=True)
class ListValue:
    """An ordered list argument (IR §10)."""

    items: tuple[Value, ...] = ()

    kind = "list"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "list", "items": [i.to_dict() for i in self.items]}


@dataclass(frozen=True)
class TupleValue:
    """An ordered tuple argument or multi-output result (IR §10)."""

    items: tuple[Value, ...] = ()

    kind = "tuple"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "tuple", "items": [i.to_dict() for i in self.items]}


@dataclass(frozen=True)
class DictValue:
    """A mapping argument, stored as ordered key/value pairs (IR §10)."""

    items: tuple[tuple[Value, Value], ...] = ()

    kind = "dict"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "dict",
            "items": [{"key": k.to_dict(), "value": v.to_dict()} for k, v in self.items],
        }


@dataclass(frozen=True)
class OpaqueValue:
    """A runtime value the frontend could not portably represent (IR §41).

    Opaque values are diagnostic only. A testcase depending on a non-portable
    opaque value MUST report that it cannot be independently reproduced.
    """

    type: str
    repr: str = "<redacted>"
    portable: bool = False

    kind = "opaque"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "opaque",
            "type": self.type,
            "repr": self.repr,
            "portable": self.portable,
        }


Value = Union[
    TensorRef,
    ScalarValue,
    NoneValue,
    StringValue,
    ListValue,
    TupleValue,
    DictValue,
    OpaqueValue,
]


def value_from_dict(data: dict[str, Any]) -> Value:
    """Parse a :class:`Value` from its JSON form (IR §10)."""
    kind = data.get("kind")
    if kind == "tensor":
        return TensorRef(value_id=int(data["value_id"]))
    if kind == "scalar":
        return ScalarValue(
            dtype=data.get("dtype", "float64"),
            value=data.get("value"),
            encoding=data.get("encoding"),
        )
    if kind == "none":
        return NoneValue()
    if kind == "string":
        return StringValue(value=data.get("value", ""))
    if kind == "list":
        return ListValue(items=tuple(value_from_dict(i) for i in data.get("items", ())))
    if kind == "tuple":
        return TupleValue(items=tuple(value_from_dict(i) for i in data.get("items", ())))
    if kind == "dict":
        return DictValue(
            items=tuple(
                (value_from_dict(entry["key"]), value_from_dict(entry["value"]))
                for entry in data.get("items", ())
            )
        )
    if kind == "opaque":
        return OpaqueValue(
            type=data.get("type", "unknown"),
            repr=data.get("repr", "<redacted>"),
            portable=bool(data.get("portable", False)),
        )
    # IR §46: unknown kinds are degraded to opaque rather than raising, so a
    # newer trace stays partially inspectable by an older reader.
    return OpaqueValue(type=f"unknown:{kind}", repr=repr(data), portable=False)


def walk_tensor_refs(value: Value | None) -> Iterator[TensorRef]:
    """Yield every :class:`TensorRef` nested anywhere inside ``value``."""
    if value is None:
        return
    if isinstance(value, TensorRef):
        yield value
    elif isinstance(value, (ListValue, TupleValue)):
        for item in value.items:
            yield from walk_tensor_refs(item)
    elif isinstance(value, DictValue):
        for key, val in value.items:
            yield from walk_tensor_refs(key)
            yield from walk_tensor_refs(val)


def validate_scalar_dtype(name: str) -> bool:
    """Return whether ``name`` is a stable scalar dtype name (IR §11)."""
    return is_known_dtype(name)
