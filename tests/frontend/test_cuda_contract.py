"""Manual GPU contract coverage for capture, identity, layout, and mutation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import inferref
from inferref.ir.package import TracePackage
from inferref.tensor import codec

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA contract requires a GPU runner"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_view_mutation_and_payload_capture(tmp_path: Path, dtype: torch.dtype) -> None:
    trace_dir = tmp_path / str(dtype).replace("torch.", "")
    base = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    base = base.to(dtype)

    with inferref.trace(
        output=trace_dir,
        capture_tensors="all",
        source_map=False,
        module_map=False,
    ) as session:
        session.mark_input("base", base)
        view = base[:, 1:]
        view.add_(1)
        session.mark_output("view", view)
    torch.cuda.synchronize()

    package = TracePackage.load(trace_dir)
    base_value = package.graph.value(package.manifest.graph_io.inputs[0].value_id)
    view_value = package.graph.value(package.manifest.graph_io.outputs[0].value_id)
    assert base_value.device.type == view_value.device.type == "cuda"
    assert base_value.storage_id == view_value.storage_id
    assert view_value.storage_version > base_value.storage_version
    payload = codec.read(package.tensor_payload_path(view_value.capture.payload))
    np.testing.assert_allclose(
        payload.as_comparable(), view.detach().float().cpu().numpy(), rtol=0, atol=0
    )


def test_pinned_cpu_and_cuda_storage_ids_do_not_collide(tmp_path: Path) -> None:
    cpu = torch.arange(8, dtype=torch.float32).pin_memory()
    gpu = cpu.to("cuda")
    assert cpu.is_pinned()

    with inferref.trace(
        output=tmp_path / "trace",
        capture_tensors="all",
        source_map=False,
        module_map=False,
    ) as session:
        session.mark_input("cpu", cpu)
        session.mark_input("gpu", gpu)
        session.mark_output("gpu_out", gpu + 1)

    package = TracePackage.load(tmp_path / "trace")
    inputs = [package.graph.value(item.value_id) for item in package.manifest.graph_io.inputs]
    assert {value.device.type for value in inputs} == {"cpu", "cuda"}
    assert len({value.storage_id for value in inputs}) == 2


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="non-default device requires two GPUs"
)
def test_non_default_cuda_device_is_preserved(tmp_path: Path) -> None:
    tensor = torch.ones(4, device="cuda:1")
    with inferref.trace(
        output=tmp_path / "trace",
        capture_tensors="all",
        source_map=False,
        module_map=False,
    ) as session:
        session.mark_output("out", tensor * 2)

    package = TracePackage.load(tmp_path / "trace")
    output = package.graph.value(package.manifest.graph_io.outputs[0].value_id)
    assert output.device.type == "cuda"
    assert output.device.index == 1
