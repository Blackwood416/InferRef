"""Hierarchical step trace compilation (SPEC §7.2; 0.12 milestone M3).

The fixture is a synthetic 28-layer decoder defined in this file: no checkpoint,
no ``transformers``, no accelerator. It dispatches the same ATen operators a
real Llama/Qwen block does, so the compiled regions have the same shape.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from inferref.frontend.pytorch.hierarchical import (
    HierarchicalTraceError,
    LeafQuantization,
    compile_step_trace,
)
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.quantization.spec import QuantizationSpec
from inferref.testcase.extract import extract_region
from inferref.testcase.validate import validate_testcase

NUM_LAYERS = 28
HIDDEN_SIZE = 16
INTERMEDIATE_SIZE = 32
NUM_HEADS = 2
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
SEQ_LEN = 4


class MiniRMSNorm(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + 1e-6))


class MiniAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.k_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.v_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)
        self.o_proj = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, seq_len, NUM_HEADS, HEAD_DIM)
        key = self.k_proj(hidden_states).view(batch, seq_len, NUM_HEADS, HEAD_DIM)
        value = self.v_proj(hidden_states).view(batch, seq_len, NUM_HEADS, HEAD_DIM)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scores = torch.matmul(query, key.transpose(2, 3)) / HEAD_DIM**0.5
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        weights = torch.softmax(scores + mask, dim=-1)
        context = torch.matmul(weights, value).transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(context)


class Int8DownProj(nn.Module):
    """A leaf projection whose weight really is INT8 code storage.

    Used to exercise the explicit quantization attachment: the codes are
    dequantized in float32 exactly as a checkpoint projector would, so the trace
    observes an ``int8`` weight value plus its scale and zero point.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        codes = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
        self.weight = nn.Parameter(codes, requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32), requires_grad=False)
        self.zero_point = nn.Parameter(torch.tensor(0, dtype=torch.int8), requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = (self.weight - self.zero_point).to(torch.float32) * self.scale
        return F.linear(hidden_states, weight)


class MiniSwiGLU(nn.Module):
    def __init__(self, down_proj: nn.Module | None = None) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False)
        self.down_proj = down_proj or nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


class MiniBlock(nn.Module):
    def __init__(self, down_proj: nn.Module | None = None) -> None:
        super().__init__()
        self.input_layernorm = MiniRMSNorm(HIDDEN_SIZE)
        self.self_attn = MiniAttention()
        self.post_attention_layernorm = MiniRMSNorm(HIDDEN_SIZE)
        self.mlp = MiniSwiGLU(down_proj)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class MiniStack(nn.Module):
    def __init__(
        self, num_layers: int = NUM_LAYERS, quantized_first_block: bool = False
    ) -> None:
        super().__init__()
        layers = [MiniBlock() for _ in range(num_layers)]
        if quantized_first_block:
            layers[0] = MiniBlock(Int8DownProj(INTERMEDIATE_SIZE, HIDDEN_SIZE))
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def build_fixture(
    *, num_layers: int = NUM_LAYERS, quantized_first_block: bool = False, seed: int = 0
) -> MiniStack:
    torch.manual_seed(seed)
    model = MiniStack(num_layers, quantized_first_block)
    model.eval()
    return model


def build_inputs(seed: int = 1) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {"hidden_states": torch.randn(1, SEQ_LEN, HIDDEN_SIZE)}


def expected_region_names(
    num_layers: int = NUM_LAYERS, step_id: str = "prefill"
) -> list[str]:
    """The exact region order the compiler must emit for the fixture."""
    names = [f"WholeModel@{step_id}"]
    for index in range(num_layers):
        layer = f"layers.{index}"
        names.extend(
            [
                f"TransformerBlock@{layer}",
                f"MiniRMSNorm@{layer}.input_layernorm",
                f"Attention@{layer}.self_attn",
                f"Linear@{layer}.self_attn.q_proj",
                f"Linear@{layer}.self_attn.k_proj",
                f"Linear@{layer}.self_attn.v_proj",
                f"Linear@{layer}.self_attn.o_proj",
                f"MiniRMSNorm@{layer}.post_attention_layernorm",
                f"SwiGLU@{layer}.mlp",
                f"Linear@{layer}.mlp.gate_proj",
                f"Linear@{layer}.mlp.up_proj",
                f"Linear@{layer}.mlp.down_proj",
            ]
        )
    return names


def test_compile_step_trace_emits_nested_regions(trace_dir) -> None:
    model = build_fixture()
    package = compile_step_trace(
        model, build_inputs(), "prefill", output_dir=trace_dir
    )

    assert validate_package(package, check_payloads=True) == []
    assert package.manifest.execution.device == "cpu"
    assert [region.id for region in package.regions] == list(
        range(len(package.regions))
    )
    assert [region.name for region in package.regions] == expected_region_names()

    whole = package.regions[0]
    operators = package.graph.ops_in_execution_order()
    assert whole.node_ids == tuple(op.id for op in operators)
    # IR §34: parameters are consumed inside the whole-model span but produced
    # nowhere, so they are boundary inputs alongside the single step input.
    roles = [package.graph.value(value_id).role for value_id in whole.inputs]
    assert roles.count("input") == 1
    assert roles.count("parameter") > 0
    assert set(roles) <= {"input", "parameter", "buffer"}
    assert len(whole.outputs) == 1
    assert package.graph.value(whole.outputs[0]).role == "output"

    blocks = {
        region.name: region
        for region in package.regions
        if region.name.startswith("TransformerBlock@")
    }
    assert len(blocks) == NUM_LAYERS
    for index in range(NUM_LAYERS):
        layer = f"layers.{index}"
        block = blocks[f"TransformerBlock@{layer}"]
        assert set(block.node_ids) <= set(whole.node_ids)
        for child in (
            f"MiniRMSNorm@{layer}.input_layernorm",
            f"Attention@{layer}.self_attn",
            f"MiniRMSNorm@{layer}.post_attention_layernorm",
            f"SwiGLU@{layer}.mlp",
        ):
            region = package.region(child)
            assert region is not None
            assert set(region.node_ids) < set(block.node_ids)
            assert region.inputs and region.outputs
        leaves = {
            "self_attn": (
                package.region(f"Attention@{layer}.self_attn"),
                ("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
            "mlp": (
                package.region(f"SwiGLU@{layer}.mlp"),
                ("gate_proj", "up_proj", "down_proj"),
            ),
        }
        for block_name, (parent, projections) in leaves.items():
            assert parent is not None
            for projection in projections:
                leaf = package.region(f"Linear@{layer}.{block_name}.{projection}")
                assert leaf is not None
                assert set(leaf.node_ids) < set(parent.node_ids)
                weights = [
                    value_id
                    for value_id in leaf.inputs
                    if package.graph.value(value_id).role == "parameter"
                ]
                assert weights
                weight = package.graph.value(weights[0])
                assert weight.qualified_name == (
                    f"{layer}.{block_name}.{projection}.weight"
                )


def test_compile_step_trace_is_deterministic(tmp_path) -> None:
    model = build_fixture(num_layers=3)
    inputs = build_inputs()
    first = compile_step_trace(
        model, inputs, "prefill", output_dir=tmp_path / "a", capture_tensors="metadata"
    )
    second = compile_step_trace(
        model, inputs, "prefill", output_dir=tmp_path / "b", capture_tensors="metadata"
    )

    def signature(package: TracePackage) -> list[tuple]:
        return [
            (region.name, region.node_ids, region.inputs, region.outputs)
            for region in package.regions
        ]

    assert signature(first) == signature(second)
    assert first.graph.operators == second.graph.operators


def test_nested_regions_extract_as_reproducible_testcases(tmp_path) -> None:
    model = build_fixture(num_layers=2)
    package = compile_step_trace(
        model, build_inputs(), "prefill", output_dir=tmp_path / "trace"
    )
    for name in (
        "TransformerBlock@layers.0",
        "Attention@layers.1.self_attn",
        "Linear@layers.1.mlp.down_proj",
    ):
        region = package.region(name)
        assert region is not None
        case_dir = tmp_path / name.replace("@", "-").replace(".", "_")
        extracted = extract_region(package, region, case_dir)
        assert extracted.reproducible
        assert validate_testcase(case_dir).valid


def test_leaf_quantization_attachment_is_explicit(trace_dir) -> None:
    model = build_fixture(num_layers=2, quantized_first_block=True)
    down_proj = model.layers[0].mlp.down_proj
    assert isinstance(down_proj, Int8DownProj)
    package = compile_step_trace(
        model,
        build_inputs(),
        "decode-0",
        output_dir=trace_dir,
        leaf_quantization={
            "layers.0.mlp.down_proj": LeafQuantization(
                scheme="affine-int8/v1",
                scale=down_proj.scale,
                zero_point=down_proj.zero_point,
            )
        },
    )

    assert package.manifest.format_version == "1.0"
    quantized_values = [
        value for value in package.graph.values if value.quantization is not None
    ]
    assert len(quantized_values) == 1
    value = quantized_values[0]
    assert value.qualified_name is not None
    assert value.qualified_name.endswith("layers.0.mlp.down_proj.weight")
    assert value.dtype == "int8"
    assert value.shape == (HIDDEN_SIZE, INTERMEDIATE_SIZE)
    spec = value.quantization
    assert isinstance(spec, QuantizationSpec)
    assert QuantizationSpec.from_dict(spec.to_dict()) == spec

    leaf = package.region("Linear@layers.0.mlp.down_proj")
    assert leaf is not None
    assert value.id in leaf.inputs

    reloaded = TracePackage.load(trace_dir)
    assert validate_package(reloaded, check_payloads=True) == []
    assert reloaded.graph.value(value.id).quantization == spec


def test_leaf_quantization_rejects_unprojected_low_bit_storage(trace_dir) -> None:
    model = build_fixture(num_layers=2, quantized_first_block=True)
    down_proj = model.layers[0].mlp.down_proj
    assert isinstance(down_proj, Int8DownProj)
    with pytest.raises(HierarchicalTraceError, match="packed low-bit"):
        compile_step_trace(
            model,
            build_inputs(),
            "prefill",
            output_dir=trace_dir,
            leaf_quantization={
                "layers.0.mlp.down_proj": LeafQuantization(
                    scheme="affine-int4/v1",
                    scale=down_proj.scale,
                    granularity="per_group",
                    axis=1,
                    group_size=4,
                )
            },
        )


def test_compile_step_trace_requires_a_tensor_step_output(trace_dir) -> None:
    class Container(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inner = build_fixture(num_layers=2)

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
            return (self.inner(hidden_states),)

    with pytest.raises(HierarchicalTraceError, match="must be a tensor"):
        compile_step_trace(Container(), build_inputs(), "prefill", output_dir=trace_dir)


def test_compile_step_trace_requires_layer_modules(trace_dir) -> None:
    model = build_fixture()
    with pytest.raises(HierarchicalTraceError, match="no decoder layer modules"):
        compile_step_trace(
            model,
            build_inputs(),
            "prefill",
            output_dir=trace_dir,
            layer_pattern=r"(?:^|\.)blocks\.\d+$",
        )
