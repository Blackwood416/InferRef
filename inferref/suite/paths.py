from __future__ import annotations

import hashlib
import re
import unicodedata


_LOGICAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def portable_id_key(value: str) -> str:
    """Return the collision key used by common Windows/macOS filesystems."""

    return unicodedata.normalize("NFC", value).rstrip(" .").casefold()


def validate_case_id(value: object, *, where: str) -> str:
    if not isinstance(value, str) or not _LOGICAL_ID.fullmatch(value):
        raise ValueError(
            f"{where} must match ^[A-Za-z0-9][A-Za-z0-9_.-]{{0,127}}$"
        )
    if value.endswith((".", " ")):
        raise ValueError(f"{where} must not end in a dot or space")
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        raise ValueError(f"{where} uses a Windows reserved name: {value!r}")
    return value


def artifact_key(logical_id: str, *, fallback: str) -> str:
    """Map an untrusted logical identifier to one collision-resistant component."""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", logical_id).strip("-.")
    digest = hashlib.sha256(logical_id.encode("utf-8")).hexdigest()[:12]
    return f"{(slug[:80] or fallback)}-{digest}"
