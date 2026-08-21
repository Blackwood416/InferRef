"""Scenario executor: effective testcase compilation and stateful replay.

Implements SPEC §6-§7 on top of the existing adapter ABI: every step is
compiled into a standalone effective testcase and executed with
:func:`inferref.agent.execute_adapter`, so no new adapter placeholders or
process surface are introduced.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from inferref.agent.adapter import execute_adapter
from inferref.agent.protocol import AgentProtocolError, EngineAdapter
from inferref.compare.compare import compare_tensors
from inferref.compare.tolerance import TolerancePolicy
from inferref.ir.paths import PathBoundaryError, resolve_contained_path
from inferref.scenario.schema import (
    Scenario,
    ScenarioError,
    ScenarioStep,
    load_scenario,
    split_reference,
)
from inferref.tensor import codec
from inferref.testcase.validate import require_valid_testcase

SCENARIO_RUN_FORMAT = "inferref-scenario-run"
SCENARIO_RUN_FORMAT_VERSION = "0.1"
STATE_MODES = ("reference", "engine")


def run_scenario(
    scenario: str | Path | Scenario,
    adapter: str | Path | EngineAdapter,
    runs_root: str | Path,
    *,
    state_mode: str = "reference",
    compare_state: bool = False,
    allow_unsupported: bool = False,
    fail_fast: bool = False,
    comparator: str | None = None,
    comparison_config: dict[str, Any] | None = None,
    tolerance: str | Path | dict[str, Any] | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    ignore_stride: bool = False,
    strict_layout: bool = False,
    first_failure: bool = True,
) -> dict[str, Any]:
    """Run an ordered scenario chain and return the v0.1 run report.

    ``state_mode="reference"`` fills state slots from step reference outputs;
    ``"engine"`` feeds the engine's own outputs forward, validating shape and
    dtype (and optionally numeric values) against the reference at each step.
    """

    if state_mode not in STATE_MODES:
        raise ScenarioError(
            [
                {
                    "code": "scenario_invalid_manifest",
                    "message": (
                        f"state_mode must be one of {STATE_MODES}, got {state_mode!r}"
                    ),
                    "where": "state_mode",
                }
            ]
        )
    loaded = load_scenario(scenario) if not isinstance(scenario, Scenario) else scenario
    engine = adapter if isinstance(adapter, EngineAdapter) else EngineAdapter.load(adapter)
    policy = TolerancePolicy()
    policy.override_atol = atol
    policy.override_rtol = rtol

    runs_root_path = Path(runs_root).resolve()
    from inferref.suite.paths import artifact_key

    run_root = runs_root_path / f"scenario-{artifact_key(loaded.id, fallback='scenario')}"
    _reset_run_root(run_root, runs_root_path)
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    _materialize_scenario_inputs(loaded, run_root)
    state_slots = _initialize_state_slots(loaded, run_root)
    step_records: list[dict[str, Any]] = []
    stop = False

    for step in loaded.steps:
        if stop:
            break
        try:
            record = _run_step(
                loaded,
                step,
                engine,
                run_root,
                state_slots,
                state_mode=state_mode,
                compare_state=compare_state,
                comparator=comparator,
                comparison_config=comparison_config,
                tolerance=tolerance,
                policy=policy,
                ignore_stride=ignore_stride,
                strict_layout=strict_layout,
                first_failure=first_failure,
            )
        except ScenarioError as exc:
            if fail_fast:
                raise
            code = exc.issues[0]["code"] if exc.issues else "scenario_invalid_manifest"
            record = _scenario_error_step(step, code, str(exc))
        except (AgentProtocolError, OSError, ValueError) as exc:
            if fail_fast:
                raise
            record = _infrastructure_error_step(step, exc)
        step_records.append(record)
        if (
            record["status"] != "pass"
            and record["state_status"] in _STATE_FAILURE_STATUSES
        ):
            stop = True
        if record["state_status"] == "state_missing":
            stop = True
        if state_mode == "engine" and record["status"] == "unsupported":
            stop = True

    status, accepted = _aggregate_status(
        [record["status"] for record in step_records],
        allow_unsupported=allow_unsupported,
    )
    report: dict[str, Any] = {
        "format": SCENARIO_RUN_FORMAT,
        "format_version": SCENARIO_RUN_FORMAT_VERSION,
        "status": status,
        "accepted": accepted,
        "scenario": {"id": loaded.id, "source": str(loaded.source)},
        "adapter": {"id": engine.name, "name": engine.name},
        "state_mode": state_mode,
        "compare_state": compare_state,
        "steps": step_records,
        "outputs": [item.name for item in loaded.outputs],
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    (run_root / "inferref-scenario-run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return report


_STATE_FAILURE_STATUSES = frozenset(
    {"state_shape_mismatch", "state_dtype_mismatch", "state_mismatch"}
)


def _reset_run_root(run_root: Path, runs_root: Path) -> None:
    """Recreate the deterministic scenario run root, staying contained."""

    if run_root.parent != runs_root or not run_root.name.startswith("scenario-"):
        raise ScenarioError(
            [
                {
                    "code": "scenario_invalid_manifest",
                    "message": f"unsafe scenario run root {run_root}",
                    "where": "runs_root",
                }
            ]
        )
    if run_root.exists():
        shutil.rmtree(run_root)


def _materialize_scenario_inputs(
    scenario: Scenario, run_root: Path
) -> None:
    """Copy committed scenario inputs into the run-local inputs area."""

    inputs_dir = run_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for declared in scenario.inputs:
        source = resolve_contained_path(
            scenario.root,
            f"inputs/{declared.name}.irtensor",
            kind=f"scenario input {declared.name!r}",
        )
        destination = resolve_contained_path(
            inputs_dir,
            f"{declared.name}.irtensor",
            kind=f"materialized scenario input {declared.name!r}",
        )
        if not source.is_file():
            raise ScenarioError(
                [
                    {
                        "code": "scenario_invalid_manifest",
                        "message": (
                            f"scenario input payload is missing: "
                            f"inputs/{declared.name}.irtensor"
                        ),
                        "where": f"inputs.{declared.name}",
                    }
                ]
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _initialize_state_slots(
    scenario: Scenario, run_root: Path
) -> dict[str, Path]:
    """Return the current slot file for each declared state slot."""

    slots: dict[str, Path] = {}
    for slot in scenario.state:
        destination = _state_slot_path(run_root, slot.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if slot.init is not None:
            _, name = split_reference(slot.init)
            source = resolve_contained_path(
                run_root,
                f"inputs/{name}.irtensor",
                kind=f"state init for {slot.name!r}",
            )
            shutil.copyfile(source, destination)
        slots[slot.name] = destination
    return slots


def _state_slot_path(run_root: Path, name: str) -> Path:
    return resolve_contained_path(
        run_root,
        f"state/{name}/{name}.irtensor",
        kind=f"state slot {name!r}",
    )


def _run_step(
    scenario: Scenario,
    step: ScenarioStep,
    engine: EngineAdapter,
    run_root: Path,
    state_slots: dict[str, Path],
    *,
    state_mode: str,
    compare_state: bool,
    comparator: str | None = None,
    comparison_config: dict[str, Any] | None = None,
    tolerance: str | Path | dict[str, Any] | None = None,
    policy: TolerancePolicy,
    ignore_stride: bool,
    strict_layout: bool,
    first_failure: bool,
) -> dict[str, Any]:
    step_root = run_root / "steps" / step.id
    effective = step_root / "testcase"
    shutil.copytree(step.testcase, effective, dirs_exist_ok=False)

    try:
        _patch_bound_inputs(step, effective, run_root, state_slots)
        require_valid_testcase(effective)
    except ScenarioError as exc:
        code = exc.issues[0]["code"] if exc.issues else "scenario_invalid_manifest"
        return _scenario_error_step(step, code, str(exc))
    except (ValueError, OSError) as exc:
        return _validation_error_step(step, exc)

    step_runs_root = step_root
    result = execute_adapter(
        effective,
        engine,
        step_runs_root,
        suite_spec=step.comparison,
        comparator=comparator,
        comparison_config=comparison_config,
        tolerance=tolerance,
        policy=policy,
        ignore_stride=ignore_stride,
        strict_layout=strict_layout,
        first_failure=first_failure,
    )

    status = _adapter_status(result["status"])
    engine_output_dir = Path(result["output"]) if result.get("output") else None
    if engine_output_dir is not None and engine_output_dir.is_dir():
        output_copy = step_root / "output"
        shutil.copytree(engine_output_dir, output_copy)
        source_record = step_root / "inferref-run.json"
        shutil.copyfile(engine_output_dir / "inferref-run.json", source_record)

    state_status = "not_applicable"
    bindings_error: str | None = None
    state_status, bindings_error = _apply_output_bindings(
        step,
        effective,
        engine_output_dir,
        run_root,
        state_slots,
        state_mode=state_mode,
        compare_state=compare_state,
        engine_ran=status not in ("unsupported", "error"),
        policy=policy,
        ignore_stride=ignore_stride,
        strict_layout=strict_layout,
    )
    if bindings_error is not None:
        code = (
            "scenario_state_missing"
            if state_status == "state_missing"
            else "scenario_output_missing"
        )
        result = {
            **result,
            "status": "error",
            "scenario_error": {"code": code, "message": bindings_error},
        }
        status = "error"
        (step_root / "inferref-run.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
    elif state_status in _STATE_FAILURE_STATUSES:
        status = "mismatch"

    return {
        "id": step.id,
        "status": status,
        "state_status": state_status,
        "run": result,
        "inputs": dict(step.input_bindings),
        "outputs": dict(step.output_bindings),
    }


def _patch_bound_inputs(
    step: ScenarioStep,
    effective: Path,
    run_root: Path,
    state_slots: dict[str, Path],
) -> dict[str, Any]:
    """Copy bound input tensors into the effective testcase and patch metadata."""

    manifest_path = effective / "testcase.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest.get("inputs", [])
    by_name = {
        item["name"]: item for item in inputs if isinstance(item, dict)
    }
    values_by_id = {
        item["id"]: item
        for item in manifest.get("values", [])
        if isinstance(item, dict)
    }
    inputs_dir = effective / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for role, reference in step.input_bindings.items():
        prefix, name = split_reference(reference)
        if prefix == "state":
            source = state_slots[name]
        else:
            source = resolve_contained_path(
                run_root,
                f"inputs/{name}.irtensor",
                kind=f"scenario input {name!r}",
            )
        if not source.is_file():
            raise ScenarioError(
                [
                    {
                        "code": "scenario_state_uninitialized",
                        "message": f"state slot {name!r} has no file at run time",
                        "where": f"steps[{step.id}].bindings.inputs.{role}",
                    }
                ]
            )
        destination = resolve_contained_path(
            inputs_dir,
            f"{role}.irtensor",
            kind=f"effective input {role!r}",
        )
        shutil.copyfile(source, destination)
        header = codec.read_header(destination).to_metadata()
        record = by_name.get(role)
        if record is None:
            raise ScenarioError(
                [
                    {
                        "code": "scenario_role_unknown",
                        "message": f"input role {role!r} vanished from patched testcase",
                        "where": f"steps[{step.id}].bindings.inputs.{role}",
                    }
                ]
            )
        record.update({"payload": f"inputs/{role}.irtensor", **header})
        value_id = record.get("value_id")
        if value_id is not None and value_id in values_by_id:
            values_by_id[value_id].update(header)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def _apply_output_bindings(
    step: ScenarioStep,
    effective: Path,
    engine_output_dir: Path | None,
    run_root: Path,
    state_slots: dict[str, Path],
    *,
    state_mode: str,
    compare_state: bool,
    engine_ran: bool,
    policy: TolerancePolicy,
    ignore_stride: bool,
    strict_layout: bool,
) -> tuple[str, str | None]:
    """Propagate step outputs into state slots and scenario outputs."""

    state_status = "not_applicable" if state_mode == "reference" else "not_compared"
    manifest = json.loads((effective / "testcase.json").read_text(encoding="utf-8"))
    output_records = {
        item["name"]: item
        for item in manifest.get("outputs", [])
        if isinstance(item, dict)
    }
    for role, reference in step.output_bindings.items():
        prefix, name = split_reference(reference)
        reference_file = resolve_contained_path(
            effective,
            f"reference/{role}.irtensor",
            kind=f"step reference output {role!r}",
        )
        record = output_records.get(role, {})
        engine_file = (
            _find_engine_output(engine_output_dir, role, record.get("value_id"))
            if engine_output_dir is not None
            else None
        )
        if prefix == "scenario.outputs":
            if not engine_ran:
                continue
            if engine_file is None or not engine_file.is_file():
                return state_status, (
                    f"engine produced no output for scenario output role {role!r}"
                )
            destination = resolve_contained_path(
                run_root,
                f"outputs/{name}.irtensor",
                kind=f"scenario output {name!r}",
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(engine_file, destination)
            continue
        # state.<name>
        slot_path = state_slots[name]
        if state_mode == "reference":
            shutil.copyfile(reference_file, slot_path)
            continue
        if not engine_ran:
            continue
        if engine_file is None or not engine_file.is_file():
            return "state_missing", (
                f"engine produced no output for state role {role!r}"
            )
        shutil.copyfile(engine_file, slot_path)
        try:
            reference_header = codec.read_header(reference_file)
            engine_header = codec.read_header(slot_path)
        except (OSError, ValueError) as exc:
            raise ScenarioError(
                [
                    {
                        "code": "scenario_state_mismatch",
                        "message": f"cannot read state tensor for {name!r}: {exc}",
                        "where": f"steps[{step.id}].bindings.outputs.{role}",
                    }
                ]
            ) from exc
        if list(engine_header.shape) != list(reference_header.shape):
            return "state_shape_mismatch", None
        if engine_header.dtype != reference_header.dtype:
            return "state_dtype_mismatch", None
        if compare_state:
            comparison = compare_tensors(
                f"state.{name}",
                codec.read(reference_file),
                codec.read(slot_path),
                policy=policy,
                ignore_stride=ignore_stride,
                strict_layout=strict_layout,
            )
            if comparison.metrics is not None and comparison.metrics.mismatch_count > 0:
                return "state_mismatch", None
            state_status = "ok"
        else:
            state_status = "not_compared"
    return state_status, None


def _find_engine_output(
    engine_dir: Path, name: str, value_id: Any
) -> Path | None:
    """Locate one engine-produced tensor using the same conventions as comparison."""

    manifest_path = engine_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = None
        if isinstance(manifest, dict):
            for entry in manifest.get("outputs", []):
                if (
                    isinstance(entry, dict)
                    and entry.get("name") == name
                    and isinstance(entry.get("payload"), str)
                ):
                    try:
                        return resolve_contained_path(
                            engine_dir,
                            entry["payload"],
                            kind=f"engine output {name!r}",
                        )
                    except PathBoundaryError:
                        return None
    candidates = [f"{name}.irtensor", f"outputs/{name}.irtensor"]
    if value_id is not None:
        candidates += [
            f"tensor_{value_id}.irtensor",
            f"v{int(value_id):08d}.irtensor",
            f"outputs/tensor_{value_id}.irtensor",
        ]
    for relative in candidates:
        try:
            candidate = resolve_contained_path(
                engine_dir, relative, kind=f"engine output {name!r} candidate"
            )
        except PathBoundaryError:
            continue
        if candidate.is_file():
            return candidate
    return None


def _adapter_status(status: str) -> str:
    if status == "pass":
        return "pass"
    if status == "mismatch":
        return "mismatch"
    if status == "unsupported":
        return "unsupported"
    return "error"


def _validation_error_step(step: ScenarioStep, exc: Exception) -> dict[str, Any]:
    return {
        "id": step.id,
        "status": "error",
        "state_status": "not_applicable",
        "run": {
            "run_id": None,
            "status": "validation_error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        },
        "inputs": dict(step.input_bindings),
        "outputs": dict(step.output_bindings),
    }


def _scenario_error_step(
    step: ScenarioStep, code: str, message: str
) -> dict[str, Any]:
    return {
        "id": step.id,
        "status": "error",
        "state_status": "not_applicable",
        "run": {
            "run_id": None,
            "status": "scenario_error",
            "scenario_error": {"code": code, "message": message},
        },
        "inputs": dict(step.input_bindings),
        "outputs": dict(step.output_bindings),
    }


def _infrastructure_error_step(
    step: ScenarioStep, exc: Exception
) -> dict[str, Any]:
    return {
        "id": step.id,
        "status": "error",
        "state_status": "not_applicable",
        "run": {
            "run_id": None,
            "status": "infrastructure_error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        },
        "inputs": dict(step.input_bindings),
        "outputs": dict(step.output_bindings),
    }


def _aggregate_status(
    step_statuses: list[str], *, allow_unsupported: bool
) -> tuple[str, bool]:
    """Scenario status aggregation per SPEC §6.5."""

    if not step_statuses:
        return "error", False
    if all(status == "pass" for status in step_statuses):
        return "pass", True
    if "error" in step_statuses:
        return "error", False
    if "mismatch" in step_statuses:
        return "fail", False
    if all(status == "unsupported" for status in step_statuses):
        return ("unsupported" if allow_unsupported else "fail"), allow_unsupported
    if all(status in ("pass", "unsupported") for status in step_statuses):
        return ("partial" if allow_unsupported else "fail"), allow_unsupported
    return "fail", False


__all__ = [
    "SCENARIO_RUN_FORMAT",
    "SCENARIO_RUN_FORMAT_VERSION",
    "STATE_MODES",
    "run_scenario",
]
