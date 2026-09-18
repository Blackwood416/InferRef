"""Optional real bnb oracle; no model downloads and no GPU required."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from inferref.compare.compare import compare_testcase
from inferref.quantization import QuantizationError, import_bitsandbytes_quant_state
from inferref.quantization.bitsandbytes import QUANT_STATE_PROFILE
from inferref.tensor import codec

bnb = pytest.importorskip("bitsandbytes")
pytestmark = pytest.mark.skipif(bnb.__version__ != "0.50.1", reason="explicit bnb 0.50.1 profile")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("nested", [False, True])
def test_real_quantize_serialize_import_dequantize(dtype, nested):
    torch.manual_seed(12)
    source = torch.randn(132, 128, dtype=dtype)
    packed, state = bnb.functional.quantize_4bit(source, blocksize=64, quant_type="nf4", compress_statistics=nested)
    # Exercise the actual checkpoint serialization/deserialization path.
    state = bnb.functional.QuantState.from_dict(state.as_dict(packed=True), device=torch.device("cpu"))
    original = copy.deepcopy(state.as_dict(packed=True))
    raw = packed.clone()
    item = import_bitsandbytes_quant_state(packed, state, profile=QUANT_STATE_PROFILE,
                                           scale_value_id=10, codebook_value_id=11)
    actual = item.dequantize_source()
    oracle = bnb.functional.dequantize_4bit(packed, state)
    carrier = oracle.view(torch.uint16).numpy() if dtype == torch.bfloat16 else oracle.numpy()
    np.testing.assert_array_equal(actual.data, carrier)
    assert torch.equal(packed, raw)
    for key, value in state.as_dict(packed=True).items():
        assert torch.equal(value, original[key])
    assert item.source_nested == nested
    if nested:
        scales = bnb.functional.dequantize_blockwise(state.absmax, state.state2) + state.offset
        np.testing.assert_array_equal(item.quantized.parameters[10].data, scales.numpy())


def test_actual_object_gates():
    packed, state = bnb.functional.quantize_4bit(torch.ones(4, 64), quant_type="nf4")
    kwargs = {"profile": QUANT_STATE_PROFILE, "scale_value_id": 10, "codebook_value_id": 11}
    with pytest.raises(QuantizationError, match="profile"):
        import_bitsandbytes_quant_state(packed, state, **{**kwargs, "profile": "auto"})
    with pytest.raises(QuantizationError, match="object"):
        import_bitsandbytes_quant_state(packed, {}, **kwargs)
    with pytest.raises(QuantizationError, match="uint8"):
        import_bitsandbytes_quant_state(packed.float(), state, **kwargs)
    state.quant_type = "fp4"
    with pytest.raises(QuantizationError, match="NF4"):
        import_bitsandbytes_quant_state(packed, state, **kwargs)


def test_safetensors_audit_and_portable_case(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    packed, state = bnb.functional.quantize_4bit(torch.ones(4, 64, dtype=torch.bfloat16),
                                                 quant_type="nf4", compress_statistics=True)
    model = tmp_path / "model"
    model.mkdir()
    name = "layer.weight"
    safetensors.save_file({name: packed, **{name + "." + k: v for k, v in state.as_dict(packed=True).items()}},
                         model / "model.safetensors")
    script = Path(__file__).resolve().parents[2] / "examples/quantization/check_bitsandbytes_model.py"
    output = tmp_path / "audit"
    result = subprocess.run([sys.executable, str(script), str(model), "--output", str(output)],
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "pass" and report["source_unchanged"]
    assert report["checked_elements"] == 256 and report["portable_case_pass"]
    assert (output / "testcase/quantization_parameters/v10.irtensor").is_file()
    engine_output = output / "bitsandbytes-output/y.irtensor"
    corrupted = codec.read(engine_output)
    data = corrupted.data.copy()
    data.flat[0] = 0
    codec.write_array(engine_output, data, dtype=corrupted.dtype)
    assert compare_testcase(output / "testcase", engine_output.parent).failed_count == 1
