# Explicit affine INT8 Linear (unreleased)

This torch-free example validates an engine against the **same quantized codes
and parameters** as the reference. It does not measure quantization loss versus
original floating-point weights or implement W8A8 integer accumulation.

From an installed development checkout, use fresh output directories:

```bash
python examples/quantization/int8_trace.py int8.irtrace
inferref agent context int8.irtrace --json
inferref testcase extract int8.irtrace --op 0 --output int8-case --input-names x,weight --output-names y --contract linear/affine-int8/v1
inferref agent run int8-case --adapter examples/quantization/int8_numpy.adapter.json --runs-dir int8-runs --json
```

Expected: Agent `pass`, output `[[9.5, 10.0]]`. The adapter uses its own elementwise
parameter indexing and `einsum`; it does not import the reference dequantizer
or Linear function. Input roles remain `x,weight`; scales are in the separate
`quantization_parameters` manifest array/directory. Move the testcase anywhere:
it does not need the source trace to run.

Inject a kernel error and compare without changing the reference:

```bash
python examples/quantization/int8_numpy.py int8-case --output int8-bug --inject-bug
inferref compare int8-case int8-bug --first-failure --json
```

Expected: `fail`, first failure at `inferref.linear_affine_int8.default`, with
`comparison_domain: float32_kernel_output`. The first output is increased by 1.

Alignment uses the same adapter and requirements:

```bash
inferref align init int8.irtrace examples/quantization/int8_numpy.adapter.json --dir int8-alignment
inferref align plan int8-alignment
inferref align run int8-alignment --limit 1 --engine-revision int8-example-v1
```

Changing a scale or weight file invalidates the session's source digest; create
a new session for the modified trace. Parameter bytes also participate in the
testcase tree digest and evidence provenance.

## Comparison domains and limits

- Quantized trace outputs default to dequantized float32 comparisons using
  **each trace's own** parameters. Equal codes with different scales may fail.
- `inferref compare ref.irtrace actual.irtrace --quantization-mode raw_codes`
  compares codes and layouts instead; a pass does not assert scale equality or
  numerical equivalence. This flag is trace-only.
- Float32 Linear outputs use ordinary numeric tolerances. A quantized testcase
  reference output is dequantized and compared with a float32 engine output.
  For two quantized outputs with independent parameters, compare two traces.
- Quantized engine-output manifests and custom/per-output comparator policies
  for quantized testcase outputs are rejected until explicitly supported.
- `linear/affine-int8/v1` takes `x: float32[..., in]` and quantized
  `weight: int8[out, in]`, produces `y: float32[..., out]`, and has no bias.
  The standalone reference API separately supports optional float32 bias.
- Trace writers must explicitly select `Manifest(format_version="1.0")`.
  Testcase extraction automatically chooses 0.4 when quantization is present.
  Old Python/C++ readers reject these artifact versions. C++ quantized testcase
  execution and automatic Torch qint8 capture are not implemented.
- The adapter declares versioned features
  `quantization/affine-int8/v1/per_tensor`, `/per_axis`, and `/per_group`.
  Plain int8 support or absent capability declarations are insufficient.

See the [full arithmetic specification](../../docs/spec/InferRef_Quantization.md).

## Packed INT4 / UINT4 W4A16

Use `--packed` for signed INT4, or `--unsigned` for UINT4 with explicit zero
points. For example:

```bash
python examples/quantization/int8_trace.py int4.irtrace --packed
inferref testcase extract int4.irtrace --op 0 --output int4-case --input-names x,weight --output-names y --contract linear/affine-w4a16/v1
inferref agent run int4-case --adapter examples/quantization/int4_numpy.adapter.json --runs-dir int4-runs --json
```

Both variants produce `[[9.5, 10.0]]` in float16. The six weight codes occupy
three payload bytes (tensor format 2). W4A16 accumulates in float32, then rounds
to float16; it does not imply hardware INT4 execution. Use the same injected-bug
and Alignment commands with the INT4 paths/adapter. Existing INT8-only adapters
remain unsupported. The C++ tensor reader can decode packed payloads, but the
C++ testcase runner still rejects quantized testcase version 0.4.

## NF4 with explicit codebook and block scales

```bash
python examples/quantization/nf4_trace.py nf4.irtrace
inferref testcase extract nf4.irtrace --op 0 --output nf4-case --input-names x,weight --output-names y --contract linear/nf4-w4a16/v1
inferref agent run nf4-case --adapter examples/quantization/nf4_numpy.adapter.json --runs-dir nf4-runs --json
```

Expected: `pass`. This example uses 69 NF4 indices in a `[3,23]` weight, two
block scales and an explicit 16-entry codebook. It exercises a block boundary
crossing a row and an odd packed tail. The same engine script supports
`--inject-bug`; an affine UINT4 adapter cannot claim NF4 support merely because
its storage dtype matches.

For existing bitsandbytes data, explicitly project a **trusted, non-nested**
QuantState into host arrays (requires Torch only for this projection):

```python
from inferref.quantization import import_bitsandbytes_nf4
from inferref.quantization.nf4 import BITSANDBYTES_PROFILE

# packed_weights and qs are the uint8 weight buffer and QuantState from the
# pinned bitsandbytes 0.48.2 float16 NF4, blocksize=64 profile.
state = {
    "shape": tuple(qs.shape), "dtype": str(qs.dtype).removeprefix("torch."),
    "quant_type": qs.quant_type, "blocksize": qs.blocksize,
    "absmax": qs.absmax.detach().cpu().numpy(),
    "code": qs.code.detach().cpu().numpy(),
    "nested": qs.nested, "offset": qs.offset, "state2": qs.state2,
}
imported = import_bitsandbytes_nf4(
    packed_weights.detach().cpu().numpy(), state,
    profile=BITSANDBYTES_PROFILE, scale_value_id=10, codebook_value_id=11,
)
```

The importer checks exact fields, dtypes, shapes, codebook and scale contents;
it never guesses the profile from bytes. It converts source high-first nibbles
to canonical low-first nibbles. Compressed statistics (`nested=True`), non-null
offset/state2, different block sizes/dtypes and unknown profiles are rejected.
The standalone importer and runtime reference depend only on numpy, not Torch
or bitsandbytes. See the [pinned profile and limitations](../../docs/spec/InferRef_Quantization.md#pinned-bitsandbytes-importer-profile).

## Real bitsandbytes QuantState and double quantization

For an actual `bitsandbytes==0.50.1` QuantState, use the separate object bridge:

```python
from inferref.quantization import import_bitsandbytes_quant_state
from inferref.quantization.bitsandbytes import QUANT_STATE_PROFILE

imported = import_bitsandbytes_quant_state(
    packed_weights, qs, profile=QUANT_STATE_PROFILE,
    scale_value_id=10, codebook_value_id=11,
)
canonical = imported.quantized  # codes/spec/parameters; float32 NF4 semantics
source_output = imported.dequantize_source()  # TensorView, cast to qs.dtype
```

This reads real Torch/QuantState objects without changing them. Main blocks are
64 elements; optional compressed scales use one float32 blockwise level of 256
elements plus a scalar offset. The importer independently expands that level
with numpy and carries the resulting float32 scales into the testcase. It does
not retain nested compressed storage in the IR. The object bridge requires
optional Torch/bitsandbytes; `import_bitsandbytes_nf4_projection` accepts strict
numpy host fields without either dependency. No automatic version guessing.

Source dtype is float16, bfloat16 or float32, independently of a model's configured
Linear compute dtype. `dequantize_source()` performs that explicit source cast;
it is not the existing W4A16 Linear reference or proof of identical GEMM output.
The original 0.48.2 flat importer continues to reject nested/bfloat16 states.

Audit a local checkpoint without loading the model or executing remote code:

```bash
python -m pip install "bitsandbytes==0.50.1" safetensors
python examples/quantization/check_bitsandbytes_model.py /path/to/model --output real-nf4-audit
inferref agent context real-nf4-audit/trace --json
inferref agent compare real-nf4-audit/testcase real-nf4-audit/bitsandbytes-output --atol 0 --rtol 0 --json
```

The output must be a new directory outside the model. Every `*.safetensors` file
is opened read-only; one weight is checked at a time. `--limit N` selects a small
smoke run (default: all NF4 weights). The script checks both expanded scales and
source-rounded weights against actual bitsandbytes, hashes the source files
before/after, writes `report.json`, and exports the first weight as a standalone
dequantization trace/testcase and bitsandbytes output. It never copies the full
checkpoint into the repository and never runs full-model inference.

Local acceptance, 2026-09-18: the supplied Qwen3-0.6B NF4 checkpoint has 196
double-quantized bfloat16 weight tensors. All **440,401,920** stored weight
elements and all reconstructed scales matched bitsandbytes 0.50.1 on CPU with
zero differences. Source SHA-256 was unchanged; the portable case passed at
zero tolerance. This is CPU weight-decoding evidence, not GPU/kernel/generation
validation. The checkpoint's producer version is not inferred from its filename.

### CPU Linear and portable-workflow qualification

```bash
python examples/quantization/qualify_bitsandbytes_linear.py --model /path/to/model --output linear-audit
# Deterministic, download-free model fixture used by CI:
python examples/quantization/qualify_bitsandbytes_linear.py --output synthetic-linear-audit
```

The real-checkpoint mode selects layer 0 q_proj, k_proj, gate_proj and down_proj
from `model.safetensors`. It checks one-token, four-token and rank-3 inputs
against an independent numpy reference (atol=rtol=0.002), then exercises relocated
testcases through Agent, Suite and Alignment, injected errors, scale changes,
hash changes and stale-session rejection. Original checkpoint bytes remain
read-only. Both commands need the pinned runtime installed as above.

`bitsandbytes_linear.adapter.json` runs actual `bnb.nn.Linear4bit` on CPU using
canonical packing. Optional AVX512 repacking is disabled. This path first rounds
dequantized weights to float16 compute dtype before GEMM; it is intentionally
**not** labelled `linear/nf4-w4a16/v1`, whose reference keeps weights in float32.
The trace explicitly records the weight cast and compute dtype. Neither source
bfloat16 rounding nor full-model generation is silently substituted into this
Linear comparison. No GPU qualification is implied.
