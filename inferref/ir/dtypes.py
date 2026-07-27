"""Stable dtype names and wire codes (IR §11, §21).

This table is the single source of truth shared by:

* ``graph.json`` (which stores dtype **names**, IR §2.6);
* the ``.irtensor`` binary header (which stores dtype **codes**, IR §21);
* ``cpp/include/inferref/irtensor.hpp`` (the C++ reader's ``DataType`` enum).

The numeric codes are **frozen at format version 0.1**. Never renumber an
existing entry — only append.
"""

from __future__ import annotations

#: dtype name -> (wire code, itemsize in bytes)
_TABLE: dict[str, tuple[int, int]] = {
    "bool": (1, 1),
    "int8": (2, 1),
    "int16": (3, 2),
    "int32": (4, 4),
    "int64": (5, 8),
    "uint8": (6, 1),
    "uint16": (7, 2),
    "uint32": (8, 4),
    "uint64": (9, 8),
    "float16": (10, 2),
    "bfloat16": (11, 2),
    "float32": (12, 4),
    "float64": (13, 8),
    "complex64": (14, 8),
    "complex128": (15, 16),
}

#: dtype name -> wire code
DTYPE_CODES: dict[str, int] = {name: code for name, (code, _) in _TABLE.items()}

#: wire code -> dtype name
DTYPE_NAMES: dict[int, str] = {code: name for name, (code, _) in _TABLE.items()}

#: dtype name -> itemsize in bytes
DTYPE_ITEMSIZES: dict[str, int] = {name: size for name, (_, size) in _TABLE.items()}

#: dtype names whose values are floating point (used by tolerance policies).
FLOAT_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})

#: dtype names whose values are complex.
COMPLEX_DTYPES = frozenset({"complex64", "complex128"})


def is_known_dtype(name: str) -> bool:
    """Return whether ``name`` is a stable InferRef dtype name."""
    return name in _TABLE


def dtype_code(name: str) -> int:
    """Return the ``.irtensor`` wire code for a stable dtype name."""
    try:
        return _TABLE[name][0]
    except KeyError:
        raise ValueError(f"unknown InferRef dtype name: {name!r}") from None


def dtype_itemsize(name: str) -> int:
    """Return the size in bytes of one element of ``name``."""
    try:
        return _TABLE[name][1]
    except KeyError:
        raise ValueError(f"unknown InferRef dtype name: {name!r}") from None


def dtype_name_from_code(code: int) -> str:
    """Return the stable dtype name for a ``.irtensor`` wire code."""
    try:
        return DTYPE_NAMES[code]
    except KeyError:
        raise ValueError(f"unknown InferRef dtype code: {code}") from None
