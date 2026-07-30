"""Cache-update detection from source context plus physical trace evidence.

Cache implementations often use a generic method named ``update`` and are
plain Python objects rather than ``nn.Module`` instances. Matching that name
globally would label unrelated optimizers, metrics, dictionaries and model
state as KV-cache work. Generic KV detection therefore requires two independent
signals:

* a source frame from a cache-named file; and
* physical cache-like work in that invocation: storage mutation or tensor
  concatenation.

Specialised recurrent/conv state helpers are a narrower case. Their explicit
function names plus a real storage mutation are sufficient; single-token
decode may update conv state directly from a model helper rather than through
``cache_utils.py``.

This recognises static KV caches (``index_copy_`` / ``copy_`` writes), dynamic
KV caches (new tensors produced by ``cat``), and recurrent/conv state caches
used by hybrid linear-attention models.  The latter are deliberately labelled
``StateCacheUpdate`` rather than ``KVCacheUpdate``: a DeltaNet or SSM state is
cache state, but it is not a key/value history.
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

STATE_CACHE_FUNCTION_PATTERNS: tuple[str, ...] = (
    r"^update_conv_state$",
    r"^update_recurrent_state$",
    r"^(?:torch_)?causal_conv1d_update$",
)

CONCAT_OPS: frozenset[str] = frozenset({"cat", "concat", "concatenate"})


@dataclass
class CacheUpdateDetector:
    """Labels KV and recurrent-state cache updates with physical evidence."""

    name: str = "cache_update"
    file_patterns: tuple[str, ...] = field(default_factory=lambda: CACHE_FILE_PATTERNS)
    function_patterns: tuple[str, ...] = field(
        default_factory=lambda: CACHE_FUNCTION_PATTERNS
    )
    state_function_patterns: tuple[str, ...] = field(
        default_factory=lambda: STATE_CACHE_FUNCTION_PATTERNS
    )

    def detect(self, package: TracePackage) -> list[Detection]:
        # (semantic, file, function, module scope) -> operators issued under
        # that frame.  Explicit state-update names are strong enough to match
        # outside cache_utils.py: Transformers performs single-token conv-state
        # updates in the model helper itself.
        matched: dict[tuple[str, str, str, str], list[int]] = {}
        for op in package.graph.ops_in_execution_order():
            source = package.source(op.source_id)
            if source is None:
                continue
            for frame in source.stack:
                normalised_file = frame.file.replace("\\", "/")
                cache_file = self._matches(normalised_file, self.file_patterns)
                direct_conv_update = bool(
                    re.search(
                        r"^(?:torch_)?causal_conv1d_update$",
                        frame.function,
                        re.IGNORECASE,
                    )
                )
                semantic_name: str | None = None
                if (
                    self._matches(frame.function, self.state_function_patterns)
                    and (cache_file or direct_conv_update)
                ):
                    semantic_name = "StateCacheUpdate"
                elif cache_file and self._matches(frame.function, self.function_patterns):
                    semantic_name = "KVCacheUpdate"
                if semantic_name is None:
                    continue
                scope = self._scope_for(package, op.module_stack)
                matched.setdefault(
                    (semantic_name, normalised_file, frame.function, scope), []
                ).append(op.id)
                break

        pending: list[tuple[str, tuple[int, ...], str, str, int, int]] = []
        for (semantic_name, file, function, scope), node_ids in matched.items():
            for run in split_invocations(package.graph, node_ids):
                mutations = sum(
                    len(package.graph.op(node_id).effects.mutated_storages)
                    for node_id in run
                )
                concats = sum(
                    package.graph.op(node_id).op in CONCAT_OPS for node_id in run
                )
                if semantic_name == "KVCacheUpdate":
                    # One write in a file named cache.py is still weak evidence
                    # — it may be an LRU counter or unrelated state. KV updates
                    # normally handle key and value tensors, so require at
                    # least two physical cache-like signals in one run.
                    if mutations + concats < 2:
                        continue
                elif mutations < 1:
                    # A specialised state-update function name is strong source
                    # evidence, but it must actually mutate storage.
                    continue
                pending.append(
                    (
                        semantic_name,
                        run,
                        scope,
                        f"{file}:{function}",
                        mutations,
                        concats,
                    )
                )

        # A source function can recur after another cache helper (prefill conv,
        # prefill recurrent, decode conv, decode recurrent).  Assign ordinals
        # by execution order rather than dictionary/function grouping.
        pending.sort(key=lambda item: package.graph.op(item[1][0]).execution_index)

        # Prefill and decode repeat at one attention module. Stable ordinals
        # keep their region names distinct without suffixing one-off layers.
        scope_counts: dict[tuple[str, str], int] = {}
        for semantic_name, _run, scope, _source, _mutations, _concats in pending:
            key = (semantic_name, scope)
            scope_counts[key] = scope_counts.get(key, 0) + 1
        scope_ordinals: dict[tuple[str, str], int] = {}

        detections: list[Detection] = []
        for semantic_name, run, base_scope, source, mutations, concats in pending:
            scope_key = (semantic_name, base_scope)
            ordinal = scope_ordinals.get(scope_key, 0)
            scope_ordinals[scope_key] = ordinal + 1
            scope = base_scope or "<root>"
            if scope_counts[scope_key] > 1:
                scope = f"{scope}#{ordinal}"
            confidence = CONFIDENCE_VERY_STRONG if mutations else CONFIDENCE_STRONG
            evidence = f"cache source {source}; {mutations} storage mutation(s)"
            if concats:
                evidence += f", {concats} tensor concatenation(s)"
            detections.append(
                Detection(
                    name=semantic_name,
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
