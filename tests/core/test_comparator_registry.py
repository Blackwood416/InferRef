"""Comparator protocol, registry, numeric comparator, CLI and doctor integration tests.

Tests SPEC §7 and §15 requirements without requiring PyTorch.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inferref.cli.main import EXIT_FAIL, EXIT_OK, main
from inferref.comparators import (
    NUMERIC_COMPARATOR_ID,
    Artifact,
    ArtifactSet,
    ComparatorPlugin,
    ComparatorResult,
    NumericComparator,
    builtin_comparators,
    comparator_list,
    comparator_plugin_statuses,
    get_comparator,
    run_comparator,
    verify_comparators,
)
from inferref.comparators import registry as comparator_registry
from inferref.doctor import run_doctor
from inferref.tensor import codec


class MockDistribution:
    def __init__(self, name: str = "inferref-vision", version: str = "0.1.0") -> None:
        self.name = name
        self.version = version


class MockEntry:
    def __init__(
        self,
        name: str,
        value: str,
        factory: object,
        dist: MockDistribution | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self._factory = factory
        self.dist = dist or MockDistribution()

    def load(self) -> object:
        if callable(self._factory) and not isinstance(self._factory, ComparatorPlugin):
            return self._factory()
        return self._factory


class MockEntries(list):
    def select(self, *, group: str) -> MockEntries:
        if group == comparator_registry.ENTRY_POINT_GROUP:
            return MockEntries(self)
        return MockEntries()


def _install_entry_points(monkeypatch: pytest.MonkeyPatch, entries: list[MockEntry]) -> None:
    monkeypatch.setattr(comparator_registry.metadata, "entry_points", lambda: MockEntries(entries))


@pytest.fixture(autouse=True)
def _clean_comparator_registry() -> None:
    comparator_registry._reset_registry()
    yield
    comparator_registry._reset_registry()


def _write_irtensor(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    codec.write_array(path, array)
    return path


# -- 1. Protocol & Dataclasses ----------------------------------------------


def test_artifact_and_comparator_result_dataclasses() -> None:
    art = Artifact("test_role", Path("/tmp/test.irtensor"))
    assert art.name == "test_role"
    assert art.path == Path("/tmp/test.irtensor")
    art_dict = art.to_dict()
    assert art_dict["name"] == "test_role"
    assert "test.irtensor" in art_dict["path"]

    res_pass = ComparatorResult(
        status="pass",
        comparator="test/comparator/v1",
        metrics={"score": 0.99},
        diagnostics=[],
    )
    assert res_pass.passed is True
    assert res_pass.to_dict()["status"] == "pass"

    res_fail = ComparatorResult(
        status="fail",
        comparator="test/comparator/v1",
        metrics={"score": 0.5},
        diagnostics=[{"code": "mismatch", "message": "out of range"}],
        first_failure={"output": "x", "message": "out of range"},
    )
    assert res_fail.passed is False
    res_dict = res_fail.to_dict()
    assert res_dict["status"] == "fail"
    assert res_dict["first_failure"]["output"] == "x"


# -- 2. Built-in Comparator Registry ----------------------------------------


def test_builtin_comparator_registered_by_default() -> None:
    builtins = builtin_comparators()
    assert NUMERIC_COMPARATOR_ID in builtins
    assert isinstance(builtins[NUMERIC_COMPARATOR_ID], NumericComparator)
    assert builtins[NUMERIC_COMPARATOR_ID].id == "tensor/numeric/v1"

    comp = get_comparator(NUMERIC_COMPARATOR_ID)
    assert comp is not None
    assert comp.id == "tensor/numeric/v1"


def test_get_comparator_returns_none_for_unknown() -> None:
    assert get_comparator("unknown/comparator/v1") is None


def test_comparator_list_contains_builtin() -> None:
    entries = comparator_list()
    assert any(e.id == NUMERIC_COMPARATOR_ID and e.source == "builtin" and e.status == "loaded" for e in entries)


# -- 3. Entry Point Discovery & Lazy Loading --------------------------------


def test_entry_point_discovery_does_not_import_on_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = []

    class TrackedPlugin:
        id = "vision/tracker/v1"

        def validate_config(self, config: dict[str, Any] | None = None) -> None:
            pass

        def compare(self, ref: ArtifactSet, act: ArtifactSet, config: dict[str, Any] | None = None) -> ComparatorResult:
            return ComparatorResult(status="pass", comparator=self.id)

    def factory() -> TrackedPlugin:
        loaded.append(True)
        return TrackedPlugin()

    entry = MockEntry("vision/tracker/v1", "pkg:factory", factory)
    _install_entry_points(monkeypatch, [entry])

    # Discovery only: must not call factory / load
    statuses = comparator_plugin_statuses(load=False)
    assert len(statuses) == 1
    assert statuses[0].entry_point == "vision/tracker/v1"
    assert statuses[0].status == "discovered"
    assert len(loaded) == 0

    # Verification / load: calls factory
    verified = verify_comparators()
    assert len(verified) == 1
    assert verified[0].status == "loaded"
    assert verified[0].comparator_id == "vision/tracker/v1"
    assert len(loaded) == 1


def test_entry_point_duplicate_reported_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyPlugin:
        id = "vision/dup/v1"

        def validate_config(self, config: dict[str, Any] | None = None) -> None:
            pass

        def compare(self, ref: ArtifactSet, act: ArtifactSet, config: dict[str, Any] | None = None) -> ComparatorResult:
            return ComparatorResult(status="pass", comparator=self.id)

    entries = [
        MockEntry("vision/dup/v1", "pkg1:DummyPlugin", DummyPlugin, MockDistribution("pack-a")),
        MockEntry("vision/dup/v1", "pkg2:DummyPlugin", DummyPlugin, MockDistribution("pack-b")),
    ]
    _install_entry_points(monkeypatch, entries)

    statuses = comparator_plugin_statuses()
    assert all(s.status == "error" for s in statuses)
    assert all("duplicate comparator entry-point name" in (s.error or "") for s in statuses)

    with pytest.raises(ValueError, match="duplicate comparator entry-point name"):
        get_comparator("vision/dup/v1")


def test_entry_point_shadowing_builtin_reported_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class ShadowPlugin:
        id = "tensor/numeric/v1"

        def validate_config(self, config: dict[str, Any] | None = None) -> None:
            pass

        def compare(self, ref: ArtifactSet, act: ArtifactSet, config: dict[str, Any] | None = None) -> ComparatorResult:
            return ComparatorResult(status="pass", comparator=self.id)

    entries = [
        MockEntry("tensor/numeric/v1", "pkg:ShadowPlugin", ShadowPlugin, MockDistribution("evil-pack")),
    ]
    _install_entry_points(monkeypatch, entries)

    statuses = comparator_plugin_statuses()
    assert len(statuses) == 1
    assert statuses[0].status == "error"
    assert "plugin shadows a built-in comparator" in (statuses[0].error or "")

    # get_comparator should still return the genuine built-in
    comp = get_comparator("tensor/numeric/v1")
    assert isinstance(comp, NumericComparator)


def test_entry_point_id_mismatch_reported_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class MismatchPlugin:
        id = "vision/wrong-id/v1"

        def validate_config(self, config: dict[str, Any] | None = None) -> None:
            pass

        def compare(self, ref: ArtifactSet, act: ArtifactSet, config: dict[str, Any] | None = None) -> ComparatorResult:
            return ComparatorResult(status="pass", comparator=self.id)

    entries = [
        MockEntry("vision/declared-id/v1", "pkg:MismatchPlugin", MismatchPlugin),
    ]
    _install_entry_points(monkeypatch, entries)

    verified = verify_comparators()
    assert len(verified) == 1
    assert verified[0].status == "error"
    assert "comparator entry point name 'vision/declared-id/v1' != plugin.id 'vision/wrong-id/v1'" in (verified[0].error or "")


def test_entry_point_invalid_protocol_reported_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class IncompletePlugin:
        id = "vision/incomplete/v1"
        # missing validate_config and compare

    entries = [
        MockEntry("vision/incomplete/v1", "pkg:IncompletePlugin", IncompletePlugin),
    ]
    _install_entry_points(monkeypatch, entries)

    verified = verify_comparators()
    assert len(verified) == 1
    assert verified[0].status == "error"
    assert "ComparatorPlugin" in (verified[0].error or "")


# -- 4. Built-in NumericComparator Tests -------------------------------------


def test_numeric_comparator_config_validation() -> None:
    comp = NumericComparator()
    # Valid configs
    comp.validate_config(None)
    comp.validate_config({})
    comp.validate_config({
        "atol": 1e-4,
        "rtol": 1e-4,
        "strict_layout": True,
        "ignore_stride": False,
        "per_dtype": {
            "float32": {"atol": 1e-5, "rtol": 1e-5},
            "float16": {"atol": 0.001},
        },
    })

    # Invalid non-dict
    with pytest.raises(ValueError, match="must be a dictionary"):
        comp.validate_config(["invalid"])

    # Unknown key
    with pytest.raises(ValueError, match="unknown numeric comparator config key"):
        comp.validate_config({"unsupported_key": 123})

    # Negative atol / rtol
    with pytest.raises(ValueError, match="atol must be a non-negative float"):
        comp.validate_config({"atol": -0.01})
    with pytest.raises(ValueError, match="rtol must be a non-negative float"):
        comp.validate_config({"rtol": -0.01})

    # Non-bool strict_layout
    with pytest.raises(ValueError, match="strict_layout must be a boolean"):
        comp.validate_config({"strict_layout": "yes"})

    # Malformed per_dtype
    with pytest.raises(ValueError, match="per_dtype must be a dictionary"):
        comp.validate_config({"per_dtype": "invalid"})
    with pytest.raises(ValueError, match="must be a dictionary"):
        comp.validate_config({"per_dtype": {"float32": 0.01}})
    with pytest.raises(ValueError, match="unknown key"):
        comp.validate_config({"per_dtype": {"float32": {"foo": 1.0}}})
    with pytest.raises(ValueError, match="must be a non-negative float"):
        comp.validate_config({"per_dtype": {"float32": {"atol": -1.0}}})


def test_numeric_comparator_compare_pass(tmp_path: Path) -> None:
    comp = NumericComparator()
    arr_ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr_act = np.array([[1.000001, 2.0], [3.0, 3.999999]], dtype=np.float32)

    ref_file = _write_irtensor(tmp_path / "ref" / "y.irtensor", arr_ref)
    act_file = _write_irtensor(tmp_path / "act" / "y.irtensor", arr_act)

    ref_set: ArtifactSet = {"y": Artifact("y", ref_file)}
    act_set: ArtifactSet = {"y": Artifact("y", act_file)}

    result = comp.compare(ref_set, act_set, config={"atol": 1e-4, "rtol": 1e-4})
    assert result.passed is True
    assert result.status == "pass"
    assert result.comparator == "tensor/numeric/v1"
    assert result.metrics["mismatch_count"] == 0
    assert result.first_failure is None


def test_numeric_comparator_compare_fail(tmp_path: Path) -> None:
    comp = NumericComparator()
    arr_ref = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    arr_act = np.array([[1.5, 2.0], [3.0, 4.0]], dtype=np.float32)

    ref_file = _write_irtensor(tmp_path / "ref" / "y.irtensor", arr_ref)
    act_file = _write_irtensor(tmp_path / "act" / "y.irtensor", arr_act)

    ref_set: ArtifactSet = {"y": Artifact("y", ref_file)}
    act_set: ArtifactSet = {"y": Artifact("y", act_file)}

    result = comp.compare(ref_set, act_set, config={"atol": 1e-4, "rtol": 1e-4})
    assert result.passed is False
    assert result.status == "fail"
    assert result.metrics["mismatch_count"] > 0
    assert result.first_failure is not None
    assert result.first_failure["output"] == "y"


def test_numeric_comparator_multi_output(tmp_path: Path) -> None:
    comp = NumericComparator()
    ref_y = _write_irtensor(tmp_path / "ref" / "y.irtensor", np.array([1.0, 2.0], dtype=np.float32))
    ref_z = _write_irtensor(tmp_path / "ref" / "z.irtensor", np.array([10.0, 20.0], dtype=np.float32))
    act_y = _write_irtensor(tmp_path / "act" / "y.irtensor", np.array([1.0, 2.0], dtype=np.float32))
    act_z = _write_irtensor(tmp_path / "act" / "z.irtensor", np.array([10.5, 20.0], dtype=np.float32))

    ref_set: ArtifactSet = {"y": Artifact("y", ref_y), "z": Artifact("z", ref_z)}
    act_set: ArtifactSet = {"y": Artifact("y", act_y), "z": Artifact("z", act_z)}

    result = comp.compare(ref_set, act_set, config={"atol": 1e-3, "rtol": 1e-3})
    assert result.passed is False
    assert result.status == "fail"
    assert result.first_failure["output"] == "z"
    assert "y" in result.metrics
    assert "z" in result.metrics
    assert "summary" in result.metrics
    assert result.metrics["summary"]["max_abs_error"] >= 0.5


# -- 5. Core Short-Circuiting & Exception Isolation -------------------------


def test_run_comparator_short_circuits_missing_roles(tmp_path: Path) -> None:
    called = []

    class DummyPlugin:
        id = "test/dummy/v1"

        def validate_config(self, config: dict[str, Any] | None = None) -> None:
            pass

        def compare(self, ref: ArtifactSet, act: ArtifactSet, config: dict[str, Any] | None = None) -> ComparatorResult:
            called.append(True)
            return ComparatorResult(status="pass", comparator=self.id)

    ref_file = _write_irtensor(tmp_path / "ref" / "y.irtensor", np.array([1.0]))
    ref_set: ArtifactSet = {
        "y": Artifact("y", ref_file),
        "z": Artifact("z", tmp_path / "ref" / "z.irtensor"),
    }
    act_set: ArtifactSet = {
        "y": Artifact("y", tmp_path / "act" / "y.irtensor"),
        # z is missing
    }

    result = run_comparator(
        DummyPlugin(),
        ref_set,
        act_set,
        required_roles=["y", "z"],
    )
    # Must NOT call plugin.compare
    assert len(called) == 0
    assert result.passed is False
    assert result.status == "fail"
    assert result.first_failure["output"] == "z"
    assert "engine produced no output for role 'z'" in result.first_failure["message"]


def test_run_comparator_isolates_plugin_exceptions(tmp_path: Path) -> None:
    class ExplodingPlugin:
        id = "test/exploding/v1"

        def validate_config(self, config: dict[str, Any] | None = None) -> None:
            pass

        def compare(self, ref: ArtifactSet, act: ArtifactSet, config: dict[str, Any] | None = None) -> ComparatorResult:
            raise ZeroDivisionError("division by zero in custom comparator logic")

    ref_file = _write_irtensor(tmp_path / "ref" / "y.irtensor", np.array([1.0]))
    act_file = _write_irtensor(tmp_path / "act" / "y.irtensor", np.array([1.0]))
    ref_set: ArtifactSet = {"y": Artifact("y", ref_file)}
    act_set: ArtifactSet = {"y": Artifact("y", act_file)}

    result = run_comparator(ExplodingPlugin(), ref_set, act_set)
    assert result.passed is False
    assert result.status == "error"
    assert result.comparator == "test/exploding/v1"
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0]["code"] == "comparator_exception"
    assert "ZeroDivisionError" in result.diagnostics[0]["error_type"]


def test_run_comparator_by_id(tmp_path: Path) -> None:
    ref_file = _write_irtensor(tmp_path / "ref" / "y.irtensor", np.array([1.0, 2.0]))
    act_file = _write_irtensor(tmp_path / "act" / "y.irtensor", np.array([1.0, 2.0]))
    ref_set: ArtifactSet = {"y": Artifact("y", ref_file)}
    act_set: ArtifactSet = {"y": Artifact("y", act_file)}

    result = run_comparator(NUMERIC_COMPARATOR_ID, ref_set, act_set)
    assert result.passed is True
    assert result.status == "pass"

    with pytest.raises(ValueError, match="unknown comparator"):
        run_comparator("nonexistent/comparator/v1", ref_set, act_set)


# -- 6. CLI Commands (inferref comparator list / show) -----------------------


def test_cli_comparator_list_text(capsys) -> None:
    assert main(["comparator", "list"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Comparator" in out
    assert "tensor/numeric/v1" in out
    assert "builtin" in out


def test_cli_comparator_list_json(capsys) -> None:
    assert main(["comparator", "list", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "inferref-comparator-list"
    assert any(c["id"] == NUMERIC_COMPARATOR_ID for c in payload["comparators"])


def test_cli_comparator_show_builtin(capsys) -> None:
    assert main(["comparator", "show", "tensor/numeric/v1"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "tensor/numeric/v1" in out
    assert "NumericComparator" in out


def test_cli_comparator_show_json(capsys) -> None:
    assert main(["comparator", "show", "tensor/numeric/v1", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "inferref-comparator-show"
    assert payload["status"] == "ok"
    assert payload["comparator"]["id"] == "tensor/numeric/v1"


def test_cli_comparator_show_unknown(capsys) -> None:
    assert main(["comparator", "show", "unknown/comparator/v1", "--json"]) == EXIT_FAIL
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "inferref-comparator-show"
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "comparator_unknown"


# -- 7. Doctor Integration --------------------------------------------------


def test_doctor_reports_comparator_registry_and_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    class MockPlugin:
        id = "vision/doctor-test/v1"

        def validate_config(self, config: dict[str, Any] | None = None) -> None:
            pass

        def compare(self, ref: ArtifactSet, act: ArtifactSet, config: dict[str, Any] | None = None) -> ComparatorResult:
            return ComparatorResult(status="pass", comparator=self.id)

    entry = MockEntry("vision/doctor-test/v1", "pkg:MockPlugin", MockPlugin, MockDistribution("inferref-vision", "1.0.0"))
    _install_entry_points(monkeypatch, [entry])

    # Unverified doctor
    rep_unverified = run_doctor()
    assert any(c["id"] == "comparator.registry" and c["status"] == "pass" for c in rep_unverified["checks"])
    plugin_check = next(c for c in rep_unverified["checks"] if c["id"] == "comparator.plugin.vision/doctor-test/v1")
    assert plugin_check["status"] == "warn"  # discovered is warn until verified
    assert "discovered" in plugin_check["message"]

    # Verified doctor
    rep_verified = run_doctor(verify_plugins=True)
    plugin_check_v = next(c for c in rep_verified["checks"] if c["id"] == "comparator.plugin.vision/doctor-test/v1")
    assert plugin_check_v["status"] == "pass"  # loaded is pass
    assert "loaded" in plugin_check_v["message"]


# -- 8. Real Distribution Discovery from Fixtures ---------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_fixture_distribution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    source = REPO_ROOT / "tests" / "fixtures" / "comparator_pack" / "inferref_vision"
    target = tmp_path / "site"
    shutil.copytree(source, target / "inferref_vision")
    dist_info = target / "inferref_vision-0.1.0.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: inferref-vision\nVersion: 0.1.0\n",
        encoding="utf-8",
    )
    (dist_info / "entry_points.txt").write_text(
        "[inferref.comparators]\n"
        "vision/object-detection/v1 = inferref_vision.detection:DetectionComparator\n"
        "broken/comparator/v1 = inferref_vision.detection:BrokenComparator\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(target))
    return target


def test_real_distribution_discovery_via_importlib_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fixture_distribution(monkeypatch, tmp_path)
    plugin_ids = [
        entry.id for entry in comparator_list() if entry.source == "plugin"
    ]
    assert "vision/object-detection/v1" in plugin_ids
    assert "broken/comparator/v1" in plugin_ids

    statuses = {status.entry_point: status for status in verify_comparators()}
    assert statuses["vision/object-detection/v1"].status == "loaded"
    assert statuses["broken/comparator/v1"].status == "loaded"

    # Get comparator loads plugin and validates
    comp = get_comparator("vision/object-detection/v1")
    assert comp is not None
    assert comp.id == "vision/object-detection/v1"


def test_comparator_list_cli_with_real_distribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fixture_distribution(monkeypatch, tmp_path)
    assert main(["comparator", "list", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    by_id = {entry["id"]: entry for entry in payload["comparators"]}
    assert by_id["vision/object-detection/v1"]["status"] == "loaded"
    assert by_id["vision/object-detection/v1"]["distribution"] == "inferref-vision"
    assert by_id["broken/comparator/v1"]["status"] == "loaded"

    assert main(["comparator", "show", "vision/object-detection/v1", "--json"]) == EXIT_OK
    show_payload = json.loads(capsys.readouterr().out)
    assert show_payload["status"] == "ok"
    assert show_payload["comparator"]["id"] == "vision/object-detection/v1"
