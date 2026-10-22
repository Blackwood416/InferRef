"""Qualify whole-model autoregressive generation and GPU/XPU alignment.

Generates reference prefill and decode steps from a causal language model
(such as Qwen3-0.6B NF4), compiles them into an InferRef Scenario, and
verifies target engine execution, token parity, device evidence and alignment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from inferref.agent.executor import execute_adapter
from inferref.agent.protocol import EngineAdapter
from inferref.alignment.hashing import (
    normalized_adapter_sha256,
    trace_metadata_sha256,
)
from inferref.alignment.planner import plan_alignment
from inferref.alignment.runner import AlignmentRunner
from inferref.alignment.store import AlignmentStore
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import StringValue, TensorRef
from inferref.suite import run_suite
from inferref.tensor import codec
from inferref.testcase.validate import validate_testcase

DEFAULT_PROMPT = "Explain how a small robot sorts red and blue blocks. "


@dataclass
class StepResult:
    step_id: str
    input_tokens: list[int]
    next_token_id: int
    next_token_text: str
    logits: np.ndarray


@dataclass
class WholeModelGenerationReference:
    prompt_text: str
    prompt_tokens: list[int]
    generated_tokens: list[int]
    generated_text: str
    steps: list[StepResult]


def run_reference_generation(
    model_path: Path,
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 4,
    device: str = "cpu",
) -> WholeModelGenerationReference:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(str(model_path), device_map=device).eval()

    inputs = tok(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_tokens = input_ids[0].tolist()

    steps: list[StepResult] = []
    generated_tokens: list[int] = []

    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
        logits_0 = out.logits[:, -1, :].cpu().to(torch.float32).numpy()
        next_id = int(np.argmax(logits_0[0]))
        generated_tokens.append(next_id)
        next_text = tok.decode([next_id])
        steps.append(
            StepResult(
                step_id="prefill",
                input_tokens=prompt_tokens,
                next_token_id=next_id,
                next_token_text=next_text,
                logits=logits_0,
            )
        )
        past_key_values = out.past_key_values

        curr_id = next_id
        current_seq = list(prompt_tokens) + [next_id]
        for i in range(1, max_new_tokens):
            step_id = f"decode-{i - 1}"
            step_input = torch.tensor([[curr_id]], device=device, dtype=torch.long)
            out = model(input_ids=step_input, past_key_values=past_key_values, use_cache=True)
            step_logits = out.logits[:, -1, :].cpu().to(torch.float32).numpy()
            curr_id = int(np.argmax(step_logits[0]))
            generated_tokens.append(curr_id)
            step_text = tok.decode([curr_id])
            steps.append(
                StepResult(
                    step_id=step_id,
                    input_tokens=list(current_seq),
                    next_token_id=curr_id,
                    next_token_text=step_text,
                    logits=step_logits,
                )
            )
            current_seq.append(curr_id)
            past_key_values = out.past_key_values

    full_generated_text = tok.decode(generated_tokens)
    return WholeModelGenerationReference(
        prompt_text=prompt,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        generated_text=full_generated_text,
        steps=steps,
    )


def make_generation_testcase(
    case_dir: Path,
    step: StepResult,
) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = case_dir / "inputs"
    ref_dir = case_dir / "reference"
    inputs_dir.mkdir(exist_ok=True)
    ref_dir.mkdir(exist_ok=True)

    token_arr = np.array([step.input_tokens], dtype=np.int64)
    tokens_path = inputs_dir / "tokens.irtensor"
    codec.write_array(tokens_path, token_arr, dtype="int64")

    logits_path = ref_dir / "logits.irtensor"
    logits_arr = step.logits.astype(np.float32)
    codec.write_array(logits_path, logits_arr, dtype="float32")

    token_shape = list(token_arr.shape)
    token_stride = list(codec.contiguous_stride(token_arr.shape))
    logits_shape = list(logits_arr.shape)
    logits_stride = list(codec.contiguous_stride(logits_arr.shape))

    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.2",
        "name": f"qwen3-generation-{step.step_id}",
        "origin": {
            "generator": "examples/quantization/qualify_whole_model_generation.py",
            "step_id": step.step_id,
        },
        "reproducible": True,
        "inputs": [
            {
                "name": "tokens",
                "value_id": 0,
                "payload": "inputs/tokens.irtensor",
                "dtype": "int64",
                "shape": token_shape,
                "stride": token_stride,
                "storage_offset": 0,
                "numel": int(token_arr.size),
                "nbytes": int(token_arr.nbytes),
            }
        ],
        "outputs": [
            {
                "name": "logits",
                "value_id": 1,
                "payload": "reference/logits.irtensor",
                "dtype": "float32",
                "shape": logits_shape,
                "stride": logits_stride,
                "storage_offset": 0,
                "numel": int(logits_arr.size),
                "nbytes": int(logits_arr.nbytes),
            }
        ],
        "nodes": [],
        "values": [
            {
                "id": 0,
                "dtype": "int64",
                "shape": token_shape,
                "stride": token_stride,
                "storage_offset": 0,
                "numel": int(token_arr.size),
                "nbytes": int(token_arr.nbytes),
            },
            {
                "id": 1,
                "dtype": "float32",
                "shape": logits_shape,
                "stride": logits_stride,
                "storage_offset": 0,
                "numel": int(logits_arr.size),
                "nbytes": int(logits_arr.nbytes),
            },
        ],
    }
    from inferref.testcase.requirements import derive_requirements
    manifest["requirements"] = derive_requirements(manifest)
    (case_dir / "testcase.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validation = validate_testcase(case_dir)
    if not validation.valid:
        raise RuntimeError(f"Generated testcase {case_dir} is invalid: {validation.to_dict()}")
    return case_dir


def create_whole_model_scenario(
    scenario_dir: Path,
    ref: WholeModelGenerationReference,
) -> Path:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = scenario_dir / "cases"
    inputs_dir = scenario_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # Write root scenario inputs
    prompt_arr = np.array([ref.prompt_tokens], dtype=np.int64)
    codec.write_array(inputs_dir / "prompt_tokens.irtensor", prompt_arr, dtype="int64")

    scenario_manifest: dict[str, Any] = {
        "format": "inferref-scenario",
        "format_version": "0.1",
        "id": "qwen3-generation-chain",
        "description": "Qwen3 whole-model autoregressive generation scenario",
        "inputs": {
            "prompt_tokens": {"kind": "tensor"},
        },
        "outputs": {
            "logits": {"kind": "tensor"},
        },
        "steps": [],
    }

    for i, step in enumerate(ref.steps):
        make_generation_testcase(cases_dir / step.step_id, step)
        input_binding_name = "prompt_tokens" if i == 0 else f"decode_tokens_{i-1}"
        if i > 0:
            step_token_arr = np.array([step.input_tokens], dtype=np.int64)
            codec.write_array(inputs_dir / f"{input_binding_name}.irtensor", step_token_arr, dtype="int64")
            scenario_manifest["inputs"][input_binding_name] = {"kind": "tensor"}

        step_entry: dict[str, Any] = {
            "id": step.step_id,
            "testcase": f"cases/{step.step_id}",
            "bindings": {
                "inputs": {
                    "tokens": f"scenario.inputs.{input_binding_name}",
                },
                "outputs": {
                    "logits": "scenario.outputs.logits",
                },
            },
        }
        scenario_manifest["steps"].append(step_entry)

    (scenario_dir / "scenario.json").write_text(
        json.dumps(scenario_manifest, indent=2), encoding="utf-8"
    )
    from inferref.scenario.validate import validate_scenario
    val_report = validate_scenario(scenario_dir)
    if not val_report["schema_valid"]:
        raise RuntimeError(f"Scenario validation failed: {val_report}")
    return scenario_dir


def make_trace_package(root: Path, ref: WholeModelGenerationReference) -> TracePackage:
    root.mkdir(parents=True, exist_ok=True)
    tensors_dir = root / "tensors"
    tensors_dir.mkdir(exist_ok=True)

    token_arr = np.array([ref.prompt_tokens], dtype=np.int64)
    logits_arr = ref.steps[0].logits.astype(np.float32)

    codec.write_array(tensors_dir / "v0.irtensor", token_arr, dtype="int64")
    codec.write_array(tensors_dir / "v1.irtensor", logits_arr, dtype="float32")

    values = [
        TensorValueRecord(
            id=0,
            dtype="int64",
            shape=token_arr.shape,
            stride=codec.contiguous_stride(token_arr.shape),
            consumers=(0,),
            capture=CaptureInfo(mode="full", payload="tensors/v0.irtensor"),
        ),
        TensorValueRecord(
            id=1,
            dtype="float32",
            shape=logits_arr.shape,
            stride=codec.contiguous_stride(logits_arr.shape),
            producer=0,
            capture=CaptureInfo(mode="full", payload="tensors/v1.irtensor"),
        ),
    ]

    graph = Graph(
        values=values,
        operators=[
            OperatorRecord(
                id=0,
                execution_index=0,
                namespace="qwen3",
                op="causal_generation_prefill",
                positional_args=(TensorRef(0),),
                result=TensorRef(1),
                keyword_args={"prompt_length": StringValue(str(len(ref.prompt_tokens)))},
            )
        ],
    )
    package = TracePackage(
        manifest=Manifest(format_version="1.0"),
        graph=graph,
        root=root,
        regions=[
            RegionRecord(
                id=0,
                name="qwen3-prefill",
                node_ids=(0,),
                inputs=(0,),
                outputs=(1,),
            )
        ],
    )
    package.save(root)
    return package


def derive_report_status(
    *,
    adapter_used: bool,
    steps: list[dict[str, Any]],
    workflows: dict[str, Any] | None = None,
) -> str:
    """Return the report status derived from the executed comparisons.

    A hard-coded ``pass`` is unsafe: the 0.11 report kept reporting success
    even when a step comparison or a suite case failed.  With no engine adapter
    the report is explicitly ``reference_only`` rather than a qualification
    pass.
    """

    if not adapter_used:
        return "reference_only"
    if not steps:
        return "fail"
    for step in steps:
        if step.get("adapter_status") != "pass":
            return "fail"
        if step.get("compare_pass") is not True:
            return "fail"
        if step.get("token_match") is False:
            return "fail"

    checks = workflows or {}
    if checks.get("relocation_valid") is False:
        return "fail"
    if "agent" in checks and checks.get("agent") != "pass":
        return "fail"
    if checks.get("injected_error_detected") is False:
        return "fail"
    total = checks.get("suite_total")
    if isinstance(total, (int, float)) and total and checks.get("suite_passed") != total:
        return "fail"
    if checks.get("alignment_status") not in (None, "pass"):
        return "fail"
    if checks.get("plan_run_agree") is False:
        return "fail"
    return "pass"


def qualify_generation(
    output_dir: Path,
    model_path: Path,
    adapter_path: Path | None = None,
    max_new_tokens: int = 4,
    device: str = "cpu",
    workflows: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)

    print(f"Running reference generation on {device}...")
    t0 = time.time()
    ref = run_reference_generation(model_path, max_new_tokens=max_new_tokens, device=device)
    ref_time = time.time() - t0
    print(f"Reference generated {len(ref.generated_tokens)} tokens in {ref_time:.2f}s: {ref.generated_text!r}")

    scenario_dir = output_dir / "scenario"
    create_whole_model_scenario(scenario_dir, ref)
    print(f"Compiled scenario at {scenario_dir}")

    trace_dir = output_dir / "trace"
    package = make_trace_package(trace_dir, ref)

    result: dict[str, Any] = {
        "status": "reference_only",
        "model": str(model_path),
        "reference_device": device,
        "prompt": ref.prompt_text,
        "prompt_tokens": ref.prompt_tokens,
        "generated_tokens": ref.generated_tokens,
        "generated_text": ref.generated_text,
        "steps": [],
    }

    adapter = EngineAdapter.load(adapter_path) if adapter_path else None
    result["engine_device"] = adapter.target_device if adapter else None
    cases_dir = scenario_dir / "cases"

    for step in ref.steps:
        step_info: dict[str, Any] = {
            "step_id": step.step_id,
            "next_token_id": step.next_token_id,
            "next_token_text": step.next_token_text,
        }
        case_path = cases_dir / step.step_id

        if adapter:
            run_dir = output_dir / f"runs-{step.step_id}"
            adapter_res = execute_adapter(case_path, adapter, run_dir, atol=0.25, rtol=0.05)
            step_info["adapter_status"] = adapter_res["status"]

            comp = adapter_res.get("comparison")
            step_info["compare_pass"] = bool(comp and comp.get("status") == "pass")
            if comp and comp.get("first_failure"):
                step_info["compare_failure"] = comp["first_failure"]

            run_id = adapter_res.get("run_id")
            actual_dir = run_dir / run_id if run_id else run_dir
            out_logits_path = actual_dir / "logits.irtensor"
            if out_logits_path.exists():
                eng_logits = codec.read(out_logits_path).data
                eng_top_id = int(np.argmax(eng_logits))
                step_info["engine_top_token_id"] = eng_top_id
                step_info["token_match"] = eng_top_id == step.next_token_id

            dev_path = actual_dir / "inferref-sycl-device.json"
            if dev_path.exists():
                step_info["device_evidence"] = json.loads(dev_path.read_text(encoding="utf-8"))

        result["steps"].append(step_info)

    if workflows and adapter:
        prefill_case = cases_dir / "prefill"
        relocated = output_dir / "relocated_prefill"
        shutil.copytree(prefill_case, relocated)
        result["relocation_valid"] = validate_testcase(relocated).valid

        agent_res = execute_adapter(relocated, adapter, output_dir / "agent-runs", atol=0.25, rtol=0.05)
        result["agent"] = agent_res["status"]

        bug_adapter = replace(adapter, command=(*adapter.command, "--inject-bug"))
        bug_res = execute_adapter(relocated, bug_adapter, output_dir / "negative-runs", atol=0.25, rtol=0.05)
        result["injected_error_detected"] = bug_res["status"] != "pass"

        suite_path = output_dir / "suite.json"
        suite_manifest = {
            "format": "inferref-suite",
            "format_version": "0.1",
            "name": "qwen3-generation-suite",
            "cases": [
                {"id": step.step_id, "testcase": f"scenario/cases/{step.step_id}", "tags": ["generation"]}
                for step in ref.steps
            ],
        }
        suite_path.write_text(json.dumps(suite_manifest, indent=2), encoding="utf-8")
        suite_res = run_suite(suite_path, adapter, output_dir / "suite-runs", atol=0.25, rtol=0.05)
        result["suite_passed"] = suite_res["counts"]["pass"]
        result["suite_total"] = suite_res["counts"]["total"]

        plan = plan_alignment(package, adapter=adapter)
        store = AlignmentStore.initialize(
            output_dir / "alignment",
            trace_path=package.root,
            adapter_path=adapter_path,
            trace_metadata_digest=trace_metadata_sha256(package.root),
            adapter_digest=normalized_adapter_sha256(adapter),
        )
        runner = AlignmentRunner(store, package, adapter)
        runs = runner.run(
            limit=1,
            engine_revision="aila-qwen3-nf4-gpu",
            comparator_overrides={"atol": 0.25, "rtol": 0.05},
        )
        result["alignment_status"] = runs[0].status
        result["plan_run_agree"] = runs[0].boundary == plan.candidates[0].boundary

    report_path = output_dir / "report.json"
    workflow_keys = (
        "relocation_valid",
        "agent",
        "injected_error_detected",
        "suite_passed",
        "suite_total",
        "alignment_status",
        "plan_run_agree",
    )
    result["status"] = derive_report_status(
        adapter_used=adapter is not None,
        steps=result["steps"],
        workflows={key: result[key] for key in workflow_keys if key in result},
    )
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Report written to {report_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("E:/RiderProjects/Aila/models/qwen3-0.6B-bnb-nf4-offline"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-workflows", action="store_true")
    args = parser.parse_args()

    qualify_generation(
        output_dir=args.output,
        model_path=args.model,
        adapter_path=args.adapter,
        max_new_tokens=args.tokens,
        device=args.device,
        workflows=not args.no_workflows,
    )


if __name__ == "__main__":
    main()

