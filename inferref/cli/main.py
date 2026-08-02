"""InferRef command line interface (SPEC §32-§37, §42).

::

    inferref
    ├── trace
    ├── inspect
    ├── analyze
    ├── compare
    ├── validate
    ├── testcase {extract, dedup}
    ├── region   {list, create, detect, delete}
    ├── agent    {capabilities, context, extract, compare, run, evaluate}
    └── export

Built on :mod:`argparse` so that the reader/comparator side installs with numpy
alone. ``torch`` is imported only inside the ``trace`` handler, keeping Trace IR
v0.1 acceptance criterion #10 (a trace is readable without PyTorch) true of the
CLI itself.

Every command supports ``--json`` for agent and CI consumption (SPEC §42).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inferref.ir.package import TracePackage
from inferref.ir.version import INFERREF_VERSION
from inferref.semantic.base import CONFIDENCE_FLOOR
from inferref.semantic.registry import detector_names

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def _emit(payload: Any, text: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    else:
        print(text)


def _load(path: str | Path) -> TracePackage:
    return TracePackage.load(path)


# -- trace -----------------------------------------------------------------


def cmd_trace(args: argparse.Namespace) -> int:
    try:
        from inferref.frontend.pytorch.runner import run_script
        from inferref.frontend.pytorch.session import TraceSession
    except ImportError as exc:
        print(
            f"error: tracing requires PyTorch ({exc}).\n"
            "Install it with:  pip install 'inferref[torch]'",
            file=sys.stderr,
        )
        return EXIT_USAGE

    session = TraceSession(
        output=args.output,
        scope=args.scope,
        exclude=tuple(args.exclude or ()),
        capture_tensors=args.capture_tensors,
        source_map=not args.no_source_map,
        module_map=True,
        semantic_analysis=args.semantic_analysis,
        embed_source_text=args.embed_source_text,
        path_mode=args.path_mode,
        max_ops=args.max_ops,
        max_capture_elements=args.max_capture_elements,
        model_name=args.model_name,
        seed=args.seed,
        device=args.device,
    )
    run_script(args.script, session, argv=args.script_args or ())

    package = session.package
    if package is None:
        print("error: trace produced no package", file=sys.stderr)
        return EXIT_FAIL

    payload = {
        "output": str(args.output),
        "operators": len(package.graph.operators),
        "values": len(package.graph.values),
        "modules": len(package.modules),
        "sources": len(package.sources),
        "storages": len(package.storages),
        "regions": len(package.regions),
    }
    text = (
        f"Wrote trace to {args.output}\n"
        f"  operators: {payload['operators']}\n"
        f"  values:    {payload['values']}\n"
        f"  modules:   {payload['modules']}\n"
        f"  sources:   {payload['sources']}"
    )
    if package.regions:
        text += f"\n  regions:   {payload['regions']} (semantic analysis)"
    _emit(payload, text, args.json)
    return EXIT_OK


# -- inspect ---------------------------------------------------------------


def cmd_inspect(args: argparse.Namespace) -> int:
    from inferref.inspect.text import render_tensor_detail, render_trace, trace_to_dict

    package = _load(args.trace)
    if args.tensor is not None:
        text = render_tensor_detail(package, args.tensor)
        if args.json:
            graph = package.graph
            payload = (
                graph.value(args.tensor).to_dict()
                if graph.has_value(args.tensor)
                else {"error": f"no such value: {args.tensor}"}
            )
            _emit(payload, text, True)
        else:
            print(text)
        return EXIT_OK

    if args.json:
        _emit(trace_to_dict(package, limit=args.limit), "", True)
    else:
        print(
            render_trace(
                package,
                verbose=args.verbose,
                limit=args.limit,
                module=args.module,
                operator=args.operator,
                show_sources=not args.no_sources,
            )
        )
    return EXIT_OK


# -- analyze ---------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    from inferref.inspect.analyze import analyze, render_analysis

    package = _load(args.trace)
    result = analyze(package)
    _emit(result.to_dict(), render_analysis(result, top=args.top), args.json)
    return EXIT_OK


# -- validate --------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    from inferref.ir.validate import validate_package

    package = _load(args.trace)
    issues = validate_package(package)
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity != "error"]

    payload = {
        "status": "pass" if not errors else "fail",
        "errors": len(errors),
        "warnings": len(warnings),
        "issues": [i.to_dict() for i in issues],
    }
    if issues:
        text = "\n".join(str(i) for i in issues)
        text += f"\n\n{len(errors)} error(s), {len(warnings)} warning(s)"
    else:
        text = "All Trace IR v0.1 invariants hold (IR §48)."
    _emit(payload, text, args.json)
    return EXIT_OK if not errors else EXIT_FAIL


# -- compare ---------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> int:
    from inferref.compare.compare import compare_testcase, compare_traces
    from inferref.compare.report import render_report
    from inferref.compare.tolerance import TolerancePolicy
    from inferref.ir.package import is_trace_package

    policy = TolerancePolicy.load(args.tolerance) if args.tolerance else TolerancePolicy()
    policy.override_atol = args.atol
    policy.override_rtol = args.rtol

    reference = Path(args.reference)
    actual = Path(args.actual)

    if is_trace_package(reference) and is_trace_package(actual):
        report = compare_traces(
            reference,
            actual,
            policy=policy,
            ignore_stride=args.ignore_stride,
            strict_layout=args.strict_layout,
            first_failure=args.first_failure,
        )
    elif (reference / "testcase.json").is_file():
        report = compare_testcase(
            reference,
            actual,
            policy=policy,
            ignore_stride=args.ignore_stride,
            strict_layout=args.strict_layout,
            first_failure=args.first_failure,
        )
    else:
        print(
            f"error: {reference} is neither a trace package nor a testcase directory",
            file=sys.stderr,
        )
        return EXIT_USAGE

    _emit(report.to_dict(), render_report(report, verbose=args.verbose), args.json)
    return EXIT_OK if report.status == "pass" else EXIT_FAIL


# -- testcase --------------------------------------------------------------


def cmd_testcase_extract(args: argparse.Namespace) -> int:
    from inferref.testcase.extract import (
        ExtractionError,
        extract_operator,
        extract_region,
        format_missing_payload,
    )

    package = _load(args.trace)
    input_names = args.input_names.split(",") if args.input_names else None
    output_names = args.output_names.split(",") if args.output_names else None
    try:
        if args.region is not None:
            region = package.region(args.region)
            if region is None:
                print(f"error: no region named {args.region!r}", file=sys.stderr)
                return EXIT_USAGE
            result = extract_region(
                package,
                region,
                args.output,
                name=args.name,
                input_names=input_names,
                output_names=output_names,
            )
        elif args.op is not None:
            result = extract_operator(
                package,
                args.op,
                args.output,
                name=args.name,
                input_names=input_names,
                output_names=output_names,
            )
        else:
            print("error: pass either --op or --region", file=sys.stderr)
            return EXIT_USAGE
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    text = f"Wrote testcase to {result.path}\n"
    text += f"  inputs:  {', '.join(result.inputs) or '(none)'}\n"
    text += f"  outputs: {', '.join(result.outputs) or '(none)'}"
    if not result.reproducible:
        text += "\n  WARNING: not independently reproducible"
        if result.missing_payloads:
            text += (
                f"\n    {len(set(result.missing_payloads))} value(s) have no payload; "
                "details:"
            )
            for detail in result.missing_payload_details:
                text += f"\n      - {format_missing_payload(detail)}"
        if result.non_portable:
            text += f"\n    non-portable values: {', '.join(sorted(set(result.non_portable)))}"
    _emit(result.to_dict(), text, args.json)
    return EXIT_OK if result.reproducible else EXIT_FAIL


def cmd_testcase_dedup(args: argparse.Namespace) -> int:
    from inferref.testcase.dedup import dedup_operators, summarise

    package = _load(args.trace)
    groups = dedup_operators(package, operator=args.operator)
    summary = summarise(groups)

    lines = [
        f"{summary['total_executions']} executions -> "
        f"{summary['total_signatures']} unique signatures",
        "",
    ]
    for group in groups[: args.limit]:
        shapes = " x ".join(str(s) for s in group.input_shapes) or "(no tensor inputs)"
        lines.append(f"  {group.count:5d}x  {group.canonical_name}")
        lines.append(f"          inputs: {shapes}")
        lines.append(f"          representative op id: {group.representative}")
    if len(groups) > args.limit:
        lines.append(f"  ... and {len(groups) - args.limit} more signatures")

    payload = {"summary": summary, "signatures": [g.to_dict() for g in groups]}
    _emit(payload, "\n".join(lines), args.json)
    return EXIT_OK


# -- region ----------------------------------------------------------------


def cmd_region_list(args: argparse.Namespace) -> int:
    package = _load(args.trace)
    if not package.regions:
        _emit({"regions": []}, "No regions defined.", args.json)
        return EXIT_OK

    lines = []
    for region in package.regions:
        lines.append(f"{region.id}: {region.name}")
        lines.append(f"  nodes:   {len(region.node_ids)} ({region.creation_method})")
        lines.append(f"  inputs:  {list(region.inputs)}")
        lines.append(f"  outputs: {list(region.outputs)}")
        if region.engine_op:
            lines.append(f"  engine:  {region.engine_op}")
    _emit({"regions": [r.to_dict() for r in package.regions]}, "\n".join(lines), args.json)
    return EXIT_OK


def cmd_region_create(args: argparse.Namespace) -> int:
    from inferref.region.manager import (
        RegionError,
        create_region_from_module,
        create_region_from_ops,
        create_region_from_source_function,
    )

    package = _load(args.trace)
    try:
        if args.from_op is not None and args.to_op is not None:
            region = create_region_from_ops(
                package,
                args.name,
                args.from_op,
                args.to_op,
                semantic=args.semantic,
                engine_op=args.engine_op,
            )
        elif args.module is not None:
            region = create_region_from_module(
                package,
                args.name,
                args.module,
                semantic=args.semantic,
                engine_op=args.engine_op,
            )
        elif args.source_function is not None:
            region = create_region_from_source_function(
                package,
                args.name,
                args.source_function,
                semantic=args.semantic,
                engine_op=args.engine_op,
            )
        else:
            print(
                "error: pass --from-op/--to-op, --module, or --source-function",
                file=sys.stderr,
            )
            return EXIT_USAGE
    except (RegionError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    package.save_regions()
    text = (
        f"Created region {region.id}: {region.name}\n"
        f"  nodes:   {len(region.node_ids)}\n"
        f"  inputs:  {list(region.inputs)}\n"
        f"  outputs: {list(region.outputs)}"
    )
    _emit(region.to_dict(), text, args.json)
    return EXIT_OK


def cmd_region_delete(args: argparse.Namespace) -> int:
    from inferref.region.manager import RegionError, delete_region

    package = _load(args.trace)
    try:
        region = delete_region(package, args.name)
    except RegionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    package.save_regions()
    _emit(region.to_dict(), f"Deleted region {region.id}: {region.name}", args.json)
    return EXIT_OK


def cmd_region_detect(args: argparse.Namespace) -> int:
    from inferref.semantic import apply_detections, clear_semantic_annotations, detect

    package = _load(args.trace)
    try:
        detections = detect(
            package,
            detector_names=args.detector,
            min_confidence=args.min_confidence,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if not detections:
        _emit(
            {"detections": [], "counts": {"detections": 0}},
            "No semantic regions detected.",
            args.json,
        )
        return EXIT_OK

    if args.dry_run:
        lines = [f"{len(detections)} semantic region(s) would be created:", ""]
        for detection in detections:
            lines.append(
                f"  {detection.confidence:.2f}  {detection.region_name:44s} "
                f"{len(detection.node_ids):4d} ops  [{detection.method}]"
            )
            if args.verbose:
                lines.append(f"        {detection.evidence}")
        lines.append("")
        lines.append("Re-run without --dry-run to write them to regions.json.")
        _emit(
            {"dry_run": True, "detections": [d.to_dict() for d in detections]},
            "\n".join(lines),
            args.json,
        )
        return EXIT_OK

    if args.replace:
        clear_semantic_annotations(package)
        package.regions = [r for r in package.regions if r.semantic is None]

    result = apply_detections(package, detections)
    package.save_regions()
    # Annotations live in graph.json, so the whole package is rewritten.
    package.save(package.root)

    lines = [
        f"Detected {len(result.detections)} semantic region(s), "
        f"created {len(result.regions)}:",
        "",
    ]
    for name, count in result.summary_by_name().items():
        lines.append(f"  {count:4d}x  {name}")
    if result.skipped:
        lines.append("")
        lines.append(f"  {len(result.skipped)} skipped:")
        for detection, reason in result.skipped:
            lines.append(f"    {detection.region_name}: {reason}")
    lines.append("")
    lines.append(f"Annotated {result.annotated_operators} operator(s).")
    _emit(result.to_dict(), "\n".join(lines), args.json)
    return EXIT_OK


# -- agent -----------------------------------------------------------------


def _agent_exit(status: str) -> int:
    if status in {"ok", "pass"}:
        return EXIT_OK
    if status == "fail":
        return EXIT_FAIL
    return EXIT_USAGE


def _agent_emit(response: Any, args: argparse.Namespace) -> int:
    payload = response.to_dict()
    text = f"{response.operation}: {response.status}"
    if response.diagnostics:
        text += "\n" + "\n".join(
            f"  {item.get('severity', 'info')}: {item.get('message', '')}"
            for item in response.diagnostics
        )
    _emit(payload, text, args.json)
    return _agent_exit(response.status)


def cmd_agent_capabilities(args: argparse.Namespace) -> int:
    from inferref.agent import capabilities

    return _agent_emit(capabilities(), args)


def cmd_agent_context(args: argparse.Namespace) -> int:
    from inferref.agent import context

    return _agent_emit(context(args.artifact), args)


def cmd_agent_extract(args: argparse.Namespace) -> int:
    from inferref.agent import extract_testcase

    response = extract_testcase(
        args.trace,
        args.output,
        region=args.region,
        op_id=args.op,
        name=args.name,
        input_names=args.input_names.split(",") if args.input_names else None,
        output_names=args.output_names.split(",") if args.output_names else None,
    )
    return _agent_emit(response, args)


def cmd_agent_compare(args: argparse.Namespace) -> int:
    from inferref.agent import compare_outputs

    response = compare_outputs(
        args.testcase,
        args.engine_output,
        atol=args.atol,
        rtol=args.rtol,
        ignore_stride=args.ignore_stride,
        strict_layout=args.strict_layout,
        first_failure=args.first_failure,
    )
    return _agent_emit(response, args)


def cmd_agent_run(args: argparse.Namespace) -> int:
    from inferref.agent import run_engine

    response = run_engine(
        args.testcase,
        args.adapter,
        args.runs_dir,
        atol=args.atol,
        rtol=args.rtol,
        ignore_stride=args.ignore_stride,
        strict_layout=args.strict_layout,
        first_failure=args.first_failure,
    )
    return _agent_emit(response, args)


def cmd_agent_evaluate(args: argparse.Namespace) -> int:
    from inferref.agent.evaluation_host import evaluate_benchmark

    agents = tuple(item.strip() for item in args.agents.split(",") if item.strip())
    report = evaluate_benchmark(
        args.benchmark,
        agents=agents,
        report_dir=args.report_dir,
        claude_settings=args.claude_settings,
        claude_model=args.claude_model,
    )
    summary = (
        f"agent evaluation: {report['status']} "
        f"({report['acceptance']['passed']}/{report['acceptance']['required_passes']})"
    )
    _emit(report, summary, args.json)
    return EXIT_OK if report["status"] == "pass" else EXIT_FAIL


# -- export ----------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    package = _load(args.trace)
    payload = {
        "manifest": package.manifest.to_dict(),
        "graph": package.graph.to_dict(),
        "modules": [m.to_dict() for m in package.modules],
        "sources": [s.to_dict() for s in package.sources],
        "regions": [r.to_dict() for r in package.regions],
        "storages": [s.to_dict() for s in package.storages],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return EXIT_OK


# -- parser ----------------------------------------------------------------


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON (SPEC §42)"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferref",
        description=(
            "Reference execution tracing, testcase extraction and numerical "
            "comparison for inference engine development."
        ),
    )
    parser.add_argument("--version", action="version", version=f"inferref {INFERREF_VERSION}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # trace
    p = sub.add_parser(
        "trace",
        help="run a script and record a reference trace",
        epilog=(
            "Arguments for the traced script go after a '--' separator, e.g. "
            "inferref trace run.py -o trace/ -- --batch 4"
        ),
    )
    p.add_argument("script", help="Python script to execute under tracing")
    # nargs="*" rather than REMAINDER: REMAINDER would swallow every option
    # after `script`, so `-o` would never reach this parser. With "*", argparse's
    # built-in `--` separator passes trailing arguments through to the script.
    p.add_argument(
        "script_args",
        nargs="*",
        help="arguments passed to the script (place them after '--')",
    )
    p.add_argument("-o", "--output", default="trace/", help="output trace directory")
    p.add_argument("--scope", help="only trace operators under this module path")
    p.add_argument(
        "--exclude", action="append", help="skip operators under this module path (repeatable)"
    )
    p.add_argument(
        "--capture-tensors",
        default="metadata",
        choices=[
            "none",
            "metadata",
            "metadata-only",
            "hash",
            "hash-only",
            "outputs",
            "selected",
            "all",
            "full",
        ],
        help="tensor capture policy (SPEC §14)",
    )
    p.add_argument("--device", default="cpu", help="device label recorded in the manifest")
    p.add_argument("--max-ops", type=int, help="stop recording after this many operators")
    p.add_argument(
        "--max-capture-elements",
        type=int,
        default=0,
        help="skip payloads larger than this many elements (0 = no limit)",
    )
    p.add_argument("--no-source-map", action="store_true", help="disable source mapping")
    p.add_argument(
        "--semantic-analysis",
        action="store_true",
        help=(
            "detect semantic regions after tracing (SPEC §17); off by default so "
            "physical tracing stays usable without it"
        ),
    )
    p.add_argument(
        "--embed-source-text",
        action="store_true",
        help="embed source lines in the trace (off by default; SPEC §58)",
    )
    p.add_argument(
        "--path-mode", default="relative", choices=["absolute", "relative", "redacted"]
    )
    p.add_argument("--model-name", default="unknown", help="model name for the manifest")
    p.add_argument("--seed", type=int, help="seed torch RNG and record it")
    _add_json(p)
    p.set_defaults(func=cmd_trace)

    # inspect
    p = sub.add_parser("inspect", help="print a trace's operators, tensors and sources")
    p.add_argument("trace", help="trace package directory")
    p.add_argument("-v", "--verbose", action="store_true", help="show layout and alias detail")
    p.add_argument("--limit", type=int, help="show at most this many operators")
    p.add_argument("--module", help="only show operators under this module path")
    p.add_argument("--operator", help="only show this canonical operator name")
    p.add_argument("--tensor", type=int, help="show detail for one tensor value id")
    p.add_argument("--no-sources", action="store_true", help="hide source locations")
    _add_json(p)
    p.set_defaults(func=cmd_inspect)

    # analyze
    p = sub.add_parser("analyze", help="summarise operator and region coverage")
    p.add_argument("trace", help="trace package directory")
    p.add_argument("--top", type=int, default=15, help="how many operators to list")
    _add_json(p)
    p.set_defaults(func=cmd_analyze)

    # validate
    p = sub.add_parser("validate", help="check Trace IR invariants (IR §48)")
    p.add_argument("trace", help="trace package directory")
    _add_json(p)
    p.set_defaults(func=cmd_validate)

    # compare
    p = sub.add_parser("compare", help="compare reference and engine outputs")
    p.add_argument("reference", help="reference trace or testcase directory")
    p.add_argument("actual", help="engine trace or engine output directory")
    p.add_argument(
        "--first-failure",
        action="store_true",
        help="stop at the earliest divergence (SPEC §30)",
    )
    p.add_argument("--atol", type=float, help="absolute tolerance override")
    p.add_argument("--rtol", type=float, help="relative tolerance override")
    p.add_argument("--tolerance", help="JSON file with per-dtype tolerances (SPEC §27)")
    p.add_argument(
        "--ignore-stride",
        action="store_true",
        help="do not report stride/storage-offset differences at all",
    )
    p.add_argument(
        "--strict-layout",
        action="store_true",
        help=(
            "treat stride/storage-offset differences as failures; off by default "
            "because a fused engine need not reproduce the reference layout (SPEC §20)"
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true", help="list every compared tensor")
    _add_json(p)
    p.set_defaults(func=cmd_compare)

    # testcase
    p = sub.add_parser("testcase", help="extract and deduplicate testcases")
    tsub = p.add_subparsers(dest="testcase_command", metavar="<subcommand>")

    q = tsub.add_parser("extract", help="extract a standalone testcase")
    q.add_argument("trace", help="trace package directory")
    q.add_argument("--op", type=int, help="operator id to extract")
    q.add_argument("--region", help="region name or id to extract")
    q.add_argument("-o", "--output", required=True, help="output testcase directory")
    q.add_argument("--name", help="testcase name (defaults to the operator/region name)")
    q.add_argument(
        "--input-names",
        help="comma-separated names for the boundary inputs, e.g. query,key,cos,sin",
    )
    q.add_argument(
        "--output-names",
        help="comma-separated names for the boundary outputs, e.g. q_embed,k_embed",
    )
    _add_json(q)
    q.set_defaults(func=cmd_testcase_extract)

    q = tsub.add_parser("dedup", help="group operators into unique signatures (SPEC §24)")
    q.add_argument("trace", help="trace package directory")
    q.add_argument("--operator", help="restrict to one canonical operator name")
    q.add_argument("--limit", type=int, default=20, help="how many signatures to show")
    _add_json(q)
    q.set_defaults(func=cmd_testcase_dedup)

    # region
    p = sub.add_parser("region", help="create and list reference regions")
    rsub = p.add_subparsers(dest="region_command", metavar="<subcommand>")

    q = rsub.add_parser("list", help="list regions")
    q.add_argument("trace", help="trace package directory")
    _add_json(q)
    q.set_defaults(func=cmd_region_list)

    q = rsub.add_parser("create", help="create a region")
    q.add_argument("trace", help="trace package directory")
    q.add_argument("--name", required=True, help="region name")
    q.add_argument("--from-op", type=int, help="first operator id (inclusive)")
    q.add_argument("--to-op", type=int, help="last operator id (inclusive)")
    q.add_argument("--module", help="create from a module boundary")
    q.add_argument("--source-function", help="create from a source function boundary")
    q.add_argument("--semantic", help="semantic label, e.g. RoPE")
    q.add_argument("--engine-op", help="engine kernel this region maps to (SPEC §20)")
    _add_json(q)
    q.set_defaults(func=cmd_region_create)

    q = rsub.add_parser("delete", help="delete a region")
    q.add_argument("trace", help="trace package directory")
    q.add_argument("name", help="region name or id")
    _add_json(q)
    q.set_defaults(func=cmd_region_delete)

    q = rsub.add_parser(
        "detect",
        help="detect semantic regions automatically (SPEC §17)",
        description=(
            "Recognise Linear / RMSNorm / RoPE / Attention and friends from the "
            "module hierarchy and source functions already recorded in the trace, "
            "and turn each invocation into a region."
        ),
    )
    q.add_argument("trace", help="trace package directory")
    q.add_argument(
        "--dry-run", action="store_true", help="list what would be created, write nothing"
    )
    q.add_argument(
        "--min-confidence",
        type=float,
        default=CONFIDENCE_FLOOR,
        help=(
            "drop detections below this confidence (IR §32: 1.0 deterministic, "
            "0.90-0.99 very strong, 0.70-0.89 likely)"
        ),
    )
    q.add_argument(
        "--detector",
        action="append",
        help=f"only run this detector (repeatable); available: {', '.join(detector_names())}",
    )
    q.add_argument(
        "--replace",
        action="store_true",
        help="discard existing semantic regions and annotations first",
    )
    q.add_argument("-v", "--verbose", action="store_true", help="show why each was detected")
    _add_json(q)
    q.set_defaults(func=cmd_region_detect)

    # agent
    p = sub.add_parser(
        "agent",
        help="stable JSON workflow for coding agents and MCP hosts",
    )
    asub = p.add_subparsers(dest="agent_command", metavar="<subcommand>")

    q = asub.add_parser("capabilities", help="discover the Agent protocol and operations")
    _add_json(q)
    q.set_defaults(func=cmd_agent_capabilities)

    q = asub.add_parser("context", help="summarise a trace or testcase for an agent")
    q.add_argument("artifact", help="trace or testcase directory")
    _add_json(q)
    q.set_defaults(func=cmd_agent_context)

    q = asub.add_parser("extract", help="extract a portable operator/region testcase")
    q.add_argument("trace", help="trace package directory")
    selection = q.add_mutually_exclusive_group(required=True)
    selection.add_argument("--op", type=int, help="operator id to extract")
    selection.add_argument("--region", help="region name or id to extract")
    q.add_argument("-o", "--output", required=True, help="output testcase directory")
    q.add_argument("--name", help="testcase name")
    q.add_argument("--input-names", help="comma-separated boundary input names")
    q.add_argument("--output-names", help="comma-separated boundary output names")
    _add_json(q)
    q.set_defaults(func=cmd_agent_extract)

    q = asub.add_parser("compare", help="compare engine outputs with a testcase")
    q.add_argument("testcase", help="standalone testcase directory")
    q.add_argument("engine_output", help="engine output directory")
    _add_agent_compare_options(q)
    _add_json(q)
    q.set_defaults(func=cmd_agent_compare)

    q = asub.add_parser(
        "run",
        help="execute a trusted engine adapter and compare its output",
    )
    q.add_argument("testcase", help="standalone testcase directory")
    q.add_argument("--adapter", required=True, help="trusted adapter JSON file")
    q.add_argument(
        "--runs-dir",
        default="inferref-runs",
        help="parent directory for fresh per-run outputs",
    )
    _add_agent_compare_options(q)
    _add_json(q)
    q.set_defaults(func=cmd_agent_run)

    q = asub.add_parser(
        "evaluate",
        help="run a manual blind repair benchmark with external coding Agents",
    )
    q.add_argument("benchmark", help="InferRef Agent evaluation v0.2 manifest")
    q.add_argument(
        "--agents",
        default="codex,claude",
        help="comma-separated configured Agent drivers (default: codex,claude)",
    )
    q.add_argument(
        "--report-dir",
        required=True,
        help="fresh host-side report directory",
    )
    q.add_argument(
        "--claude-settings",
        help="local Claude Code settings override (path and contents are not reported)",
    )
    q.add_argument(
        "--claude-model",
        help="Claude Code model override, useful with a custom provider settings file",
    )
    _add_json(q)
    q.set_defaults(func=cmd_agent_evaluate)

    # export
    p = sub.add_parser("export", help="export a trace as a single JSON document")
    p.add_argument("trace", help="trace package directory")
    p.add_argument("-o", "--output", help="write to this file instead of stdout")
    p.set_defaults(func=cmd_export)

    return parser


def _add_agent_compare_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--atol", type=float, help="absolute tolerance override")
    parser.add_argument("--rtol", type=float, help="relative tolerance override")
    parser.add_argument(
        "--ignore-stride",
        action="store_true",
        help="do not report stride/storage-offset differences",
    )
    parser.add_argument(
        "--strict-layout",
        action="store_true",
        help="treat stride/storage-offset differences as failures",
    )
    parser.add_argument(
        "--all-failures",
        dest="first_failure",
        action="store_false",
        help="compare all outputs instead of stopping at the first failure",
    )
    parser.set_defaults(first_failure=True)


def main(argv: Sequence[str] | None = None) -> int:
    # Spec references (§) and model paths may be non-ASCII; the Windows console
    # defaults to a codepage that cannot encode them.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass

    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        # A bare group like `inferref testcase` — show the top-level help.
        parser.print_help()
        return EXIT_USAGE

    try:
        return args.func(args)
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except BrokenPipeError:  # pragma: no cover - piping into head
        return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
