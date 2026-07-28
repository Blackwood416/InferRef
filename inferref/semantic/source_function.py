"""Semantic detection from source functions (SPEC §17, §19; IR §31, §32).

Some constructs are plain functions rather than modules — Hugging Face models
implement rotary embedding as ``apply_rotary_pos_emb``, not as an ``nn.Module``
— so module-type detection alone cannot see them.

Matching uses the **whole source stack**, not just the innermost frame. That
detail is what makes the result usable::

    ei=30..34  primary_fn=apply_rotary_pos_emb
    ei=35..38  primary_fn=rotate_half            <-- matched via its caller
    ei=39..47  primary_fn=apply_rotary_pos_emb

``rotate_half`` is called *by* ``apply_rotary_pos_emb``, so its operators carry
the helper as their innermost frame but the target function further up the
stack. Matching on the innermost frame alone yields a node set with holes,
which then picks up spurious region inputs; matching on the stack yields the
contiguous 18-operator run that is the actual RoPE computation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from inferref.ir.package import TracePackage
from inferref.semantic.base import CONFIDENCE_VERY_STRONG, Detection
from inferref.semantic.invocations import split_invocations

DETECTOR_ID = "inferref.semantic.source_function.v1"

#: Function-name patterns -> semantic name, matched case-insensitively against
#: the function names in an operator's source stack. Ordered; first match wins.
FUNCTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^apply_rotary_pos_emb$|^apply_rope$|^apply_rotary_emb$", "RoPE"),
    (r"^repeat_kv$", "RepeatKV"),
    (r"^scaled_dot_product_attention$|^eager_attention_forward$", "Attention"),
    (r"^sdpa_attention_forward$|^flash_attention_forward$", "Attention"),
    (r"^update_causal_mask$|^_prepare_4d_causal_attention_mask", "CausalMask"),
)

#: Helpers that only ever run inside a larger construct. Detecting them
#: separately would bury the useful region under noise.
NESTED_HELPERS: frozenset[str] = frozenset({"rotate_half", "rotate_every_two"})


def semantic_for_function(function: str) -> str | None:
    """Map a source function name to a semantic name, if recognised."""
    if function in NESTED_HELPERS:
        return None
    for pattern, name in FUNCTION_PATTERNS:
        if re.search(pattern, function, re.IGNORECASE):
            return name
    return None


@dataclass
class SourceFunctionDetector:
    """Labels operator runs issued from a recognised source function."""

    name: str = "source_function"
    confidence: float = CONFIDENCE_VERY_STRONG
    patterns: tuple[tuple[str, str], ...] = field(default_factory=lambda: FUNCTION_PATTERNS)

    def detect(self, package: TracePackage) -> list[Detection]:
        # function name -> operator ids whose *stack* contains it
        matched: dict[str, list[int]] = {}
        for op in package.graph.ops_in_execution_order():
            source = package.source(op.source_id)
            if source is None:
                continue
            for frame in source.stack:
                semantic_name = semantic_for_function(frame.function)
                if semantic_name is None:
                    continue
                matched.setdefault(f"{semantic_name}\x00{frame.function}", []).append(op.id)
                break  # innermost recognised frame wins

        detections: list[Detection] = []
        for key, nodes in matched.items():
            semantic_name, function = key.split("\x00", 1)
            runs = split_invocations(package.graph, nodes)
            for index, run in enumerate(runs):
                scope = self._scope_for(package, run)
                detections.append(
                    Detection(
                        name=semantic_name,
                        node_ids=run,
                        confidence=self.confidence,
                        detector=DETECTOR_ID,
                        method="source_function",
                        scope=scope or f"#{index}",
                        evidence=f"operators issued from {function}()",
                    )
                )
        return detections

    @staticmethod
    def _scope_for(package: TracePackage, node_ids: tuple[int, ...]) -> str:
        """Anchor a run to the innermost module that contains all of it."""
        stacks = [
            package.graph.op(n).module_stack
            for n in node_ids
            if package.graph.has_op(n)
        ]
        if not stacks or not all(stacks):
            return ""
        common = stacks[0]
        for stack in stacks[1:]:
            limit = min(len(common), len(stack))
            shared = 0
            while shared < limit and common[shared] == stack[shared]:
                shared += 1
            common = common[:shared]
            if not common:
                return ""
        module = package.module(common[-1])
        return module.path if module else ""
