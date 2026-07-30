"""Cache-update detection from source context plus physical trace evidence.

Cache implementations often use a generic method named ``update`` and are
plain Python objects rather than ``nn.Module`` instances. Matching that name
globally would label unrelated optimizers, metrics, dictionaries and model
state as KV-cache work. This detector therefore requires two independent
signals:

* a source frame from a cache-named file; and
* physical cache-like work in that invocation: storage mutation or tensor
  concatenation.

This recognises both static caches (``index_copy_`` / ``copy_`` writes) and
dynamic caches (new tensors produced by ``cat``) without importing the
framework that produced the trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from inferref.ir.package import TracePackage
from inferref.semantic.base import (
    CONFIDENCE_STRONG,
    CONFIDENCE_VERY_STRONG,
    Detection,
)
from inferref.semantic.invocations import split_invocations

DETECTOR_ID = "inferref.semantic.cache_update.v1"

CACHE_FILE_PATTERNS: tuple[str, ...] = (
    r"(?:^|/)(?:cache|cache_utils|kv_cache)\.py$",
)

CACHE_FUNCTION_PATTERNS: tuple[str, ...] = (
    r"^update$",
    r"^update_kv_cache$",
    r"^append_kv_cache$",
    r"^write_kv_cache$",
)

CONCAT_OPS: frozenset[str] = frozenset({"cat", "concat", "concatenate"})


@dataclass
class CacheUpdateDetector:
    """Labels cache update invocations backed by physical operator evidence."""

    name: str = "cache_update"
    file_patterns: tuple[str, ...] = field(default_factory=lambda: CACHE_FILE_PATTERNS)
    function_patterns: tuple[str, ...] = field(
        default_factory=lambda: CACHE_FUNCTION_PATTERNS
    )

    def detect(self, package: TracePackage) -> list[Detection]:
        # (file, function, module scope) -> operators issued under that frame.
        matched: dict[tuple[str, str, str], list[int]] = {}
        for op in package.graph.ops_in_execution_order():
            source = package.source(op.source_id)
            if source is None:
                continue
            for frame in source.stack:
                normalised_file = frame.file.replace("\\", "/")
                if not self._matches(normalised_file, self.file_patterns):
                    continue
                if not self._matches(frame.function, self.function_patterns):
                    continue
                scope = self._scope_for(package, op.module_stack)
                matched.setdefault(
                    (normalised_file, frame.function, scope), []
                ).append(op.id)
                break

        pending: list[tuple[tuple[int, ...], str, str, int, int]] = []
        for (file, function, scope), node_ids in matched.items():
            for run in split_invocations(package.graph, node_ids):
                mutations = sum(
                    len(package.graph.op(node_id).effects.mutated_storages)
                    for node_id in run
                )
                concats = sum(
                    package.graph.op(node_id).op in CONCAT_OPS for node_id in run
                )
                # One write in a file named cache.py is still weak evidence —
                # it may be an LRU counter or unrelated state. KV updates
                # normally handle key and value tensors, so require at least
                # two physical cache-like signals in one contiguous run.
                if mutations + concats < 2:
                    continue
                pending.append((run, scope, f"{file}:{function}", mutations, concats))

        # Prefill and decode repeat at one attention module. Stable ordinals
        # keep their region names distinct without suffixing one-off layers.
        scope_counts: dict[str, int] = {}
        for _run, scope, _source, _mutations, _concats in pending:
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
        scope_ordinals: dict[str, int] = {}

        detections: list[Detection] = []
        for run, base_scope, source, mutations, concats in pending:
            ordinal = scope_ordinals.get(base_scope, 0)
            scope_ordinals[base_scope] = ordinal + 1
            scope = base_scope or "<root>"
            if scope_counts[base_scope] > 1:
                scope = f"{scope}#{ordinal}"
            confidence = CONFIDENCE_VERY_STRONG if mutations else CONFIDENCE_STRONG
            evidence = f"cache source {source}; {mutations} storage mutation(s)"
            if concats:
                evidence += f", {concats} tensor concatenation(s)"
            detections.append(
                Detection(
                    name="KVCacheUpdate",
                    node_ids=run,
                    confidence=confidence,
                    detector=DETECTOR_ID,
                    method="semantic_pattern",
                    scope=scope,
                    evidence=evidence,
                )
            )
        return detections

    @staticmethod
    def _matches(value: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _scope_for(package: TracePackage, module_stack: tuple[int, ...]) -> str:
        if not module_stack:
            return ""
        module = package.module(module_stack[-1])
        return module.path if module else ""
