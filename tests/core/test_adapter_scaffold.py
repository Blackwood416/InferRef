"""Adapter scaffolder tests (InferRef 0.8 Adapter DX, section 6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inferref.adapter import ScaffoldError, scaffold_adapter
from inferref.agent.protocol import EngineAdapter
from inferref.cli.main import EXIT_OK, main
from inferref.tensor import codec
from inferref.testcase.requirements import derive_requirements


def _case(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    x = codec.write_array(
        root / "inputs/x.irtensor", np.zeros((2, 3), dtype=np.float32)
    )
    gate = codec.write_array(
        root / "inputs/gate.irtensor", np.zeros((2, 3), dtype=np.float16)
    )
    y = codec.write_array(
        root / "reference/y.irtensor", np.zeros((2, 3), dtype=np.float32)
    )
    z = codec.write_array(
        root / "reference/z.irtensor", np.zeros((2, 3), dtype=np.float32)
    )
    entries: list[dict[str, object]] = []
    values: list[dict[str, object]] = []
    for value_id, (name, payload) in enumerate(
        [("x", x), ("gate", gate), ("y", y), ("z", z)], start=1
    ):
        metadata = codec.read(payload).to_metadata()
        entries.append(
            {
                "name": name,
                "value_id": value_id,
                "payload": str(payload.relative_to(root)).replace("\\", "/"),
                **metadata,
            }
        )
        values.append({"id": value_id, **metadata})
    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.2",
        "name": "scaffold-fixture",
        "reproducible": True,
        "contracts": ["swiglu/fused/v1"],
        "inputs": entries[:2],
        "outputs": entries[2:],
        "nodes": [],
        "values": values,
    }
    manifest["requirements"] = derive_requirements(manifest)
    (root / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_scaffold_generates_four_files_and_a_valid_adapter(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path / "case")
    out = tmp_path / "adapter"
    result = scaffold_adapter(case, out)
    assert result.files == ("CMakeLists.txt", "adapter.json", "main.cpp", "README.md")
    for name in result.files:
        assert (out / name).is_file()

    adapter = EngineAdapter.load(out / "adapter.json")
    assert adapter.name == "scaffolded-adapter"
    assert adapter.target_device == "cpu"
    assert adapter.capabilities is not None
    assert adapter.capabilities.device_types == ("cpu",)
    assert adapter.capabilities.dtypes == ("float16", "float32")
    assert adapter.capabilities.max_rank == 2
    assert adapter.capabilities.features == ("multiple_outputs",)
    assert adapter.capabilities.contracts == ("swiglu/fused/v1",)
    assert adapter.command == (
        "{adapter_dir}/inferref_adapter",
        "--testcase",
        "{testcase}",
        "--output",
        "{output}",
    )

    main_cpp = (out / "main.cpp").read_text(encoding="utf-8")
    assert "RunYourEngine" in main_cpp
    assert "outputs.at(name)" in main_cpp
    assert "testcase.Finish()" in main_cpp
    cmake = (out / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "INFERREF_CPP_INCLUDE" in cmake
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "scaffold-fixture" in readme
    assert "RunYourEngine" in readme


def test_scaffold_capabilities_match_derive_requirements(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    manifest = json.loads((case / "testcase.json").read_text(encoding="utf-8"))
    requirements = derive_requirements(manifest)
    result = scaffold_adapter(case, tmp_path / "adapter")
    assert result.capabilities["dtypes"] == requirements["dtypes"]
    assert result.capabilities["max_rank"] == requirements["max_rank"]
    assert result.capabilities["features"] == requirements["features"]
    assert result.capabilities["contracts"] == manifest["contracts"]


def test_scaffold_cli_json(tmp_path: Path, capsys) -> None:
    case = _case(tmp_path / "case")
    out = tmp_path / "adapter"
    assert main(["adapter", "scaffold", str(case), "-o", str(out), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "inferref-adapter-scaffold"
    assert payload["status"] == "pass"
    assert payload["files"] == ["CMakeLists.txt", "adapter.json", "main.cpp", "README.md"]
    assert payload["capabilities"]["contracts"] == ["swiglu/fused/v1"]
    assert (out / "adapter.json").is_file()


def test_scaffold_refuses_existing_output(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    out = tmp_path / "adapter"
    out.mkdir()
    (out / "stale.txt").write_text("keep", encoding="utf-8")
    try:
        scaffold_adapter(case, out)
    except ScaffoldError as exc:
        assert "already exists" in str(exc)
    else:  # pragma: no cover - failure path must raise
        raise AssertionError("expected ScaffoldError")
    assert (out / "stale.txt").is_file()


def test_scaffold_refuses_missing_and_invalid_testcases(tmp_path: Path) -> None:
    try:
        scaffold_adapter(tmp_path / "missing", tmp_path / "adapter")
    except ScaffoldError as exc:
        assert "does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ScaffoldError")

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "testcase.json").write_text("{not json", encoding="utf-8")
    try:
        scaffold_adapter(bad, tmp_path / "adapter2")
    except (ScaffoldError, ValueError) as exc:
        assert "invalid testcase" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected validation failure")


def test_scaffold_unsupported_language(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    try:
        scaffold_adapter(case, tmp_path / "adapter", language="rust")
    except ScaffoldError as exc:
        assert "unsupported scaffold language" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ScaffoldError")


def test_scaffold_cli_rejects_unsupported_language(tmp_path: Path) -> None:
    case = _case(tmp_path / "case")
    with pytest.raises(SystemExit):
        main(["adapter", "scaffold", str(case), "--language", "rust"])
