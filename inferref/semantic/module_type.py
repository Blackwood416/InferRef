"""Semantic detection from module types (SPEC §17; IR §31, §32).

The traced module hierarchy already records each module's class:

    layers.0.input_layernorm     examples.mini_llama.model.RMSNorm
    layers.0.self_attn           examples.mini_llama.model.Attention
    layers.0.self_attn.q_proj    torch.nn.modules.linear.Linear

That is the cheapest reliable semantic signal available, and it needs nothing
beyond what a runtime trace already carries — no static export, no pattern
matching over operator sequences.

Two tiers of evidence, scored per IR §32:

* a **torch built-in class** is a deterministic mapping (confidence 1.0):
  ``torch.nn.Linear`` is a Linear, and cannot be anything else;
* a **custom class name** is very strong but still inference (0.90): a class
  called ``Qwen3RMSNorm`` is almost certainly an RMSNorm, but the name is a
  convention, not a guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from inferref.ir.package import TracePackage
from inferref.region.boundary import nodes_for_module
from inferref.semantic.base import (
    CONFIDENCE_DETERMINISTIC,
    CONFIDENCE_STRONG,
    Detection,
)
from inferref.semantic.invocations import split_invocations

DETECTOR_ID = "inferref.semantic.module_type.v1"

#: Exact ``torch.nn`` class paths -> semantic name. A match here cannot be
#: wrong, so it scores 1.0.
BUILTIN_TYPES: dict[str, str] = {
    "torch.nn.modules.linear.Linear": "Linear",
    "torch.nn.modules.linear.LazyLinear": "Linear",
    "torch.nn.modules.linear.Bilinear": "Bilinear",
    "torch.nn.modules.linear.Identity": "Identity",
    "torch.nn.modules.sparse.Embedding": "Embedding",
    "torch.nn.modules.sparse.EmbeddingBag": "EmbeddingBag",
    "torch.nn.modules.normalization.LayerNorm": "LayerNorm",
    "torch.nn.modules.normalization.RMSNorm": "RMSNorm",
    "torch.nn.modules.normalization.GroupNorm": "GroupNorm",
    "torch.nn.modules.batchnorm.BatchNorm1d": "BatchNorm",
    "torch.nn.modules.batchnorm.BatchNorm2d": "BatchNorm",
    "torch.nn.modules.batchnorm.BatchNorm3d": "BatchNorm",
    "torch.nn.modules.conv.Conv1d": "Conv1d",
    "torch.nn.modules.conv.Conv2d": "Conv2d",
    "torch.nn.modules.conv.Conv3d": "Conv3d",
    "torch.nn.modules.activation.SiLU": "SiLU",
    "torch.nn.modules.activation.GELU": "GELU",
    "torch.nn.modules.activation.ReLU": "ReLU",
    "torch.nn.modules.activation.Softmax": "Softmax",
    "torch.nn.modules.activation.Sigmoid": "Sigmoid",
    "torch.nn.modules.activation.Tanh": "Tanh",
    "torch.nn.modules.activation.MultiheadAttention": "Attention",
    "torch.nn.modules.dropout.Dropout": "Dropout",
}

#: Class-name patterns for custom modules, tried in order — first match wins,
#: so more specific patterns must come first. Matched against the bare class
#: name (``Qwen3RMSNorm``), case-insensitively.
NAME_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"rmsnorm", "RMSNorm"),
    (r"layernorm", "LayerNorm"),
    (r"rotary|rope", "RoPE"),
    (r"swiglu", "SwiGLU"),
    (r"geglu", "GeGLU"),
    (r"grouped.?query.?attention|gqa", "GroupedQueryAttention"),
    (r"attention|attn", "Attention"),
    (r"mlp|feed.?forward|ffn", "MLP"),
    (r"decoder.?layer|encoder.?layer|transformer.?block|^block$", "TransformerBlock"),
    (r"embedding", "Embedding"),
)


def semantic_for_type(type_path: str) -> tuple[str, float] | None:
    """Map a module class path to ``(semantic_name, confidence)``."""
    if not type_path:
        return None
    builtin = BUILTIN_TYPES.get(type_path)
    if builtin is not None:
        return builtin, CONFIDENCE_DETERMINISTIC

    class_name = type_path.rsplit(".", 1)[-1]
    # An unrecognised torch built-in stays unlabelled rather than being guessed
    # at: the exact table above is meant to be exhaustive for what it covers.
    if type_path.startswith("torch.nn."):
        return None
    for pattern, name in NAME_PATTERNS:
        if re.search(pattern, class_name, re.IGNORECASE):
            return name, CONFIDENCE_STRONG
    return None


@dataclass
class ModuleTypeDetector:
    """Labels each module invocation from its class (SPEC §56)."""

    name: str = "module_type"
    #: Skip container modules that merely forward to children and would produce
    #: a region identical to a child's.
    skip_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "torch.nn.modules.container.Sequential",
                "torch.nn.modules.container.ModuleList",
                "torch.nn.modules.container.ModuleDict",
            }
        )
    )

    def detect(self, package: TracePackage) -> list[Detection]:
        detections: list[Detection] = []
        for module in package.modules:
            if module.type in self.skip_types:
                continue
            resolved = semantic_for_type(module.type)
            if resolved is None:
                continue
            semantic_name, confidence = resolved

            # All operators under this module, including nested children, so an
            # Attention region genuinely contains its projections (IR §36
            # permits the resulting nesting).
            nodes = nodes_for_module(package.graph, [module.id])
            if not nodes:
                continue

            runs = split_invocations(package.graph, nodes)
            for index, run in enumerate(runs):
                scope = module.path or "<root>"
                if len(runs) > 1:
                    scope = f"{scope}#{index}"
                detections.append(
                    Detection(
                        name=semantic_name,
                        node_ids=run,
                        confidence=confidence,
                        detector=DETECTOR_ID,
                        method="module",
                        scope=scope,
                        evidence=f"module {module.path or '<root>'} is {module.type}",
                    )
                )
        return detections
