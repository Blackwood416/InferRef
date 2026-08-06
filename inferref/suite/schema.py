from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inferref.ir.paths import PathBoundaryError, resolve_contained_path
from inferref.suite.paths import portable_id_key, validate_case_id
from inferref.testcase.validate import validate_testcase

SUITE_FORMAT = "inferref-suite"
SUITE_FORMAT_VERSION = "0.1"


class SuiteError(ValueError):
    pass


@dataclass(frozen=True)
class SuiteCase:
    id: str
    testcase: Path
    tags: tuple[str, ...] = ()

    def to_dict(self, root: Path) -> dict[str, Any]:
        return {
            "id": self.id,
            "testcase": self.testcase.relative_to(root).as_posix(),
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class Suite:
    name: str
    source: Path
    cases: tuple[SuiteCase, ...]

    @property
    def root(self) -> Path:
        return self.source.parent

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SUITE_FORMAT,
            "format_version": SUITE_FORMAT_VERSION,
            "name": self.name,
            "source": str(self.source),
            "cases": [case.to_dict(self.root) for case in self.cases],
        }


def load_suite(path: str | Path, *, validate_cases: bool = True) -> Suite:
    source = Path(path).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SuiteError("suite root must be an object")
    if data.get("format") != SUITE_FORMAT:
        raise SuiteError(f"not an InferRef suite: {data.get('format')!r}")
    if data.get("format_version") != SUITE_FORMAT_VERSION:
        raise SuiteError(
            f"unsupported suite format_version {data.get('format_version')!r}"
        )
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise SuiteError("suite name must be a non-empty string")
    records = data.get("cases")
    if not isinstance(records, list) or not records:
        raise SuiteError("suite cases must be a non-empty array")
    cases: list[SuiteCase] = []
    ids: set[str] = set()
    portable_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SuiteError(f"cases[{index}] must be an object")
        try:
            case_id = validate_case_id(record.get("id"), where=f"cases[{index}].id")
        except ValueError as exc:
            raise SuiteError(str(exc)) from exc
        portable_key = portable_id_key(case_id)
        if case_id in ids or portable_key in portable_ids:
            raise SuiteError(
                f"cases[{index}].id collides on a portable filesystem: {case_id!r}"
            )
        testcase = record.get("testcase")
        if not isinstance(testcase, str):
            raise SuiteError(f"cases[{index}].testcase must be a relative string")
        try:
            testcase_path = resolve_contained_path(
                source.parent, testcase, kind=f"suite case {case_id!r} testcase"
            )
        except PathBoundaryError as exc:
            raise SuiteError(str(exc)) from exc
        tags = record.get("tags", [])
        if not isinstance(tags, list) or not all(
            isinstance(tag, str) and tag for tag in tags
        ):
            raise SuiteError(f"cases[{index}].tags must be a string array")
        if validate_cases:
            validation = validate_testcase(testcase_path)
            if not validation.valid:
                raise SuiteError(
                    f"case {case_id!r} is invalid: {validation.issues[0].message}"
                )
        ids.add(case_id)
        portable_ids.add(portable_key)
        cases.append(SuiteCase(case_id, testcase_path, tuple(tags)))
    return Suite(name, source, tuple(cases))


def validate_suite(
    path: str | Path, *, allow_nonreproducible: bool = False
) -> dict[str, Any]:
    """Validate a suite, separating structural validity from runnability.

    ``schema_valid`` means every referenced testcase passes structural
    validation; ``runnable`` additionally requires every testcase to be
    reproducible. ``allow_nonreproducible`` keeps schema-valid suites accepted
    for corpus inventory while still reporting ``runnable: false``.
    """

    try:
        suite = load_suite(path)
    except (SuiteError, OSError, json.JSONDecodeError) as exc:
        return {
            "format": SUITE_FORMAT,
            "format_version": SUITE_FORMAT_VERSION,
            "status": "fail",
            "schema_valid": False,
            "runnable": False,
            "error": str(exc),
            "name": None,
            "cases": 0,
            "case_ids": [],
            "non_runnable_cases": [],
        }
    non_runnable: list[str] = []
    for case in suite.cases:
        validation = validate_testcase(case.testcase)
        if not validation.reproducible:
            non_runnable.append(case.id)
    runnable = not non_runnable
    return {
        "format": SUITE_FORMAT,
        "format_version": SUITE_FORMAT_VERSION,
        "status": "pass" if runnable else "fail",
        "schema_valid": True,
        "runnable": runnable,
        "non_runnable_cases": non_runnable,
        "allow_nonreproducible": allow_nonreproducible,
        "name": suite.name,
        "cases": len(suite.cases),
        "case_ids": [case.id for case in suite.cases],
    }
