"""Manual accelerator contract for CUDA, ROCm-as-CUDA, and Intel XPU."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import inferref
from inferref.frontend.pytorch.accelerator import accelerator_info
from inferref.frontend.pytorch.capture import logical_bytes
from inferref.ir.package import TracePackage
from inferref.tensor import codec


def _selected_backend() -> str | None:
    required = os.environ.get("INFERREF_ACCELERATOR")
    if required:
        if not accelerator_info(required).available:
            pytest.fail(f"required accelerator {required!r} is unavailable")
        return required
    for candidate in ("cuda", "xpu"):
        if accelerator_info(candidate).available:
            return candidate
    return None


@pytest.fixture(scope="module")
def accelerator() -> str:
    selected = _selected_backend()
    if selected is None:
        pytest.skip("accelerator contract requires a CUDA/ROCm/XPU runner")
    return selected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accelerator_view_mutation_and_payload_capture(
    tmp_path: Path, accelerator: str, dtype: torch.dtype
) -> None:
    trace_dir = tmp_path / str(dtype).replace("torch.", "")
    base = torch.arange(16, device=accelerator, dtype=torch.float32).reshape(4, 4)
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

    package = TracePackage.load(trace_dir)
    base_value = package.graph.value(package.graph.inputs[0].value.value_id)
    view_value = package.graph.value(package.graph.outputs[0].value.value_id)
    assert base_value.device.type == view_value.device.type == accelerator
    assert base_value.storage_id == view_value.storage_id
    assert view_value.storage_version > base_value.storage_version
    assert package.manifest.execution.device == f"{accelerator}:0"
    payload = codec.read(package.tensor_payload_path(view_value.capture.payload))
    expected = torch.frombuffer(
        bytearray(logical_bytes(view.float())), dtype=torch.float32
    ).reshape(view.shape)
    np.testing.assert_allclose(
        payload.as_comparable(), expected.numpy(), rtol=0, atol=0
    )


def test_pinned_cpu_and_accelerator_storage_ids_do_not_collide(
    tmp_path: Path, accelerator: str
) -> None:
    try:
        cpu = torch.arange(8, dtype=torch.float32).pin_memory()
    except RuntimeError as exc:
        pytest.skip(f"backend build has no pinned-memory allocator: {exc}")
    device = cpu.to(accelerator)
    assert cpu.is_pinned()

    with inferref.trace(
        output=tmp_path / "trace",
        capture_tensors="all",
        source_map=False,
        module_map=False,
    ) as session:
        session.mark_input("cpu", cpu)
        session.mark_input("device", device)
        session.mark_output("device_out", device + 1)

    package = TracePackage.load(tmp_path / "trace")
    inputs = [package.graph.value(item.value.value_id) for item in package.graph.inputs]
    assert {value.device.type for value in inputs} == {"cpu", accelerator}
    assert len({value.storage_id for value in inputs}) == 2
    assert package.manifest.execution.device == "mixed"


def test_non_default_accelerator_device_is_preserved(
    tmp_path: Path, accelerator: str
) -> None:
    module = getattr(torch, accelerator)
    if module.device_count() < 2:
        pytest.skip("non-default device requires two accelerator devices")
    tensor = torch.ones(4, device=f"{accelerator}:1")
    with inferref.trace(
        output=tmp_path / "trace",
        capture_tensors="all",
        source_map=False,
        module_map=False,
    ) as session:
        session.mark_output("out", tensor * 2)

    package = TracePackage.load(tmp_path / "trace")
    output = package.graph.value(package.graph.outputs[0].value.value_id)
    assert output.device.type == accelerator
    assert output.device.index == 1
    assert package.manifest.execution.device == f"{accelerator}:1"


def test_capture_materializes_work_from_non_default_stream(
    accelerator: str,
) -> None:
    module = getattr(torch, accelerator)
    stream = module.Stream()
    with module.stream(stream):
        value = torch.arange(4096, device=accelerator, dtype=torch.float32)
        result = value.mul(7).add(3)

    # No caller-side synchronization: logical_bytes owns that contract.
    raw = logical_bytes(result)
    host = torch.frombuffer(bytearray(raw), dtype=torch.float32)
    expected = torch.arange(4096, dtype=torch.float32).mul(7).add(3)
    assert torch.equal(host, expected)
