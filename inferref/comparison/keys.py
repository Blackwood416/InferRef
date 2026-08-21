"""Reserved configuration keys and helper functions for Comparison Spec."""

from __future__ import annotations

from typing import Any

RESERVED_NUMERIC_CONFIG_KEYS: frozenset[str] = frozenset(
    {"per_dtype", "strict_layout", "ignore_stride", "atol", "rtol"}
)


def clean_custom_config(raw_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Strip numeric tolerance and layout defaults before passing config to custom comparator plugins."""
    if not raw_cfg:
        return {}
    return {k: v for k, v in raw_cfg.items() if k not in RESERVED_NUMERIC_CONFIG_KEYS}
