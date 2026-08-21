"""Generate a compilable adapter project from one standalone testcase.

The scaffolder reads a validated testcase, derives the engine capability
declaration with ``derive_requirements()``, and writes the four-file project
from section 6: ``CMakeLists.txt``, ``adapter.json``, ``main.cpp``, and
``README.md``. The only user edit is the body of ``RunYourEngine``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inferref.agent.protocol import AgentProtocolError, EngineAdapter
from inferref.testcase.requirements import derive_requirements, is_contract_id
from inferref.testcase.validate import require_valid_testcase

#: Feature vocabulary the generated adapter may declare (section 6.3).
FEATURE_WHITELIST = frozenset(
    {"multiple_outputs", "strided_inputs", "alias_effects", "mutation_effects"}
)

SUPPORTED_LANGUAGES = ("cpp",)

GENERATED_FILES = ("CMakeLists.txt", "adapter.json", "main.cpp", "README.md")


class ScaffoldError(ValueError):
    """The adapter project cannot be generated from the given inputs."""


@dataclass(frozen=True)
class ScaffoldResult:
    """One generated adapter project."""

    path: Path
    files: tuple[str, ...]
    capabilities: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "files": list(self.files),
            "capabilities": self.capabilities,
        }


_MAIN_CPP = """\
#include <inferref/testcase.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

// The only user edit: implement engine dispatch for this region.
// Inputs and outputs are keyed by role name, matching the testcase manifest.
std::map<std::string, inferref::IRTensor> RunYourEngine(
    const std::string &region_name,
    const std::map<std::string, inferref::IRTensor> &inputs)
{
    // TODO: implement engine dispatch.
    (void)region_name;
    (void)inputs;
    throw std::runtime_error("RunYourEngine is not implemented");
}

int main(int argc, char **argv)
{
    std::string testcase_dir;
    std::string output_dir;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--testcase" && i + 1 < argc)
            testcase_dir = argv[++i];
        else if (arg == "--output" && i + 1 < argc)
            output_dir = argv[++i];
    }
    if (testcase_dir.empty() || output_dir.empty())
    {
        std::cerr << "usage: inferref_adapter --testcase DIR --output DIR\\n";
        return 2;
    }

    try
    {
        auto testcase = inferref::Testcase::Load(testcase_dir);
        testcase.SetOutputDir(output_dir);
        auto inputs = testcase.Inputs();
        auto outputs = RunYourEngine(testcase.RegionName(), inputs);

        testcase.WriteOutputs(outputs, "RunYourEngine");
        testcase.Finish();
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "inferref_adapter: " << error.what() << "\\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "inferref_adapter: unknown non-standard exception\\n";
        return 1;
    }
}
"""

_CMAKE = """\
cmake_minimum_required(VERSION 3.16)
project(inferref_adapter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra)
endif()

# Point INFERREF_CPP_INCLUDE at the InferRef checkout (cpp/include), or copy
# cpp/include/inferref into this tree and set it to the parent directory.
set(INFERREF_CPP_INCLUDE "" CACHE PATH "Path to InferRef cpp/include")

add_executable(inferref_adapter main.cpp)
target_include_directories(inferref_adapter PRIVATE ${INFERREF_CPP_INCLUDE})
"""

_MAIN_CPP_BRIDGE = """\
#include <inferref/bridge.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

int main(int argc, char **argv)
{
    std::string testcase_dir;
    std::string output_dir;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--testcase" && i + 1 < argc)
            testcase_dir = argv[++i];
        else if (arg == "--output" && i + 1 < argc)
            output_dir = argv[++i];
    }
    if (testcase_dir.empty() || output_dir.empty())
    {
        std::cerr << "usage: inferref_bridge --testcase DIR --output DIR\\n";
        return inferref::kBridgeUsage;
    }

    // The only user edit: implement engine dispatch in this callback.
    // The bridge resolves the region key and handles output writing.
    inferref::DebugInvoke invoke =
        [](const std::string &region_name,
           const std::map<std::string, inferref::IRTensor> &inputs)
            -> std::map<std::string, inferref::IRTensor>
    {
        // TODO: implement engine dispatch.
        (void)region_name;
        (void)inputs;
        throw std::runtime_error("DebugInvoke is not implemented");
    };

    return inferref::RunBridge(testcase_dir, output_dir, invoke,
                               "inferref_bridge");
}
"""

_CMAKE_BRIDGE = """\
cmake_minimum_required(VERSION 3.16)
project(inferref_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra)
endif()

# Point INFERREF_CPP_INCLUDE at the InferRef checkout (cpp/include), or copy
# cpp/include/inferref into this tree and set it to the parent directory.
set(INFERREF_CPP_INCLUDE "" CACHE PATH "Path to InferRef cpp/include")

add_executable(inferref_bridge main.cpp)
target_include_directories(inferref_bridge PRIVATE ${INFERREF_CPP_INCLUDE})
"""


def _generate_readme(testcase_name: str, *, bridge_mode: bool) -> str:
    if bridge_mode:
        return f"""\
# Scaffolded InferRef runtime bridge

Generated from the standalone testcase `{testcase_name}` in bridge mode. The
output handling is already wired through `inferref::RunBridge`: it loads the
testcase, resolves the region dispatch key, calls your `DebugInvoke` callback
with named inputs, writes every declared output by role name, and publishes
`manifest.json`.

## What to edit

1. `main.cpp` - implement engine dispatch inside the `DebugInvoke` callback.
   It receives the region key (`origin.region`, `origin.contract`, or
   `contracts[0]`) and a role-name -> tensor map, and must return a
   role-name -> tensor map with every declared output.  The bridge handles
   loading, output writing, and manifest publication.
2. `adapter.json` - the generated file is a valid Engine Adapter v0.2
   declaration. Edit `name`, `target_device`, `capabilities.device_types`, and
   the command executable as needed; the rest (dtypes, max_rank, features,
   contracts) was derived from the testcase.

## Build

Point `INFERREF_CPP_INCLUDE` at the InferRef checkout and build:

```bash
cmake -S . -B build -DINFERREF_CPP_INCLUDE=/path/to/inferref/cpp/include
cmake --build build
```

Or copy `cpp/include/inferref` from the InferRef checkout into this tree and
set `INFERREF_CPP_INCLUDE` to the directory that contains the `inferref`
subdirectory. The generated code needs only those header-only files.

## Run

```bash
./build/inferref_bridge --testcase /path/to/testcase --output /path/to/output
```

Then compare with `inferref agent compare /path/to/testcase /path/to/output`.
"""
    return f"""\
# Scaffolded InferRef adapter

Generated from the standalone testcase `{testcase_name}`. The engine loop is
already wired: `main.cpp` loads the testcase, reads the named inputs, calls
`RunYourEngine`, writes every declared output by role name, and publishes
`manifest.json`.

## What to edit

1. `main.cpp` - replace the `RunYourEngine` stub with your engine dispatch.
   It receives the region dispatch key (`origin.region`, `origin.contract`, or
   `contracts[0]`) and a role-name -> tensor map, and must return a
   role-name -> tensor map with every declared output.
2. `adapter.json` - the generated file is a valid Engine Adapter v0.2
   declaration. Edit `name`, `target_device`, `capabilities.device_types`, and
   the command executable as needed; the rest (dtypes, max_rank, features,
   contracts) was derived from the testcase.

## Build

Point `INFERREF_CPP_INCLUDE` at the InferRef checkout and build:

```bash
cmake -S . -B build -DINFERREF_CPP_INCLUDE=/path/to/inferref/cpp/include
cmake --build build
```

Or copy `cpp/include/inferref` from the InferRef checkout into this tree and
set `INFERREF_CPP_INCLUDE` to the directory that contains the `inferref`
subdirectory. The generated code needs only those header-only files.

## Run

```bash
./build/inferref_adapter --testcase /path/to/testcase --output /path/to/output
```

Then compare with `inferref agent compare /path/to/testcase /path/to/output`.
"""


def _adapter_payload(
    manifest: dict[str, Any], *, bridge_mode: bool
) -> dict[str, Any]:
    requirements = derive_requirements(manifest)
    dtypes = requirements.get("dtypes", [])
    if not dtypes:
        raise ScaffoldError(
            "testcase has no dtype metadata; cannot derive adapter capabilities"
        )
    max_rank = requirements.get("max_rank", 0)
    features = sorted(set(requirements.get("features", [])) & FEATURE_WHITELIST)
    contracts = list(manifest.get("contracts") or ())
    if any(not is_contract_id(contract) for contract in contracts):
        raise ScaffoldError(
            "testcase declares invalid executable contract(s): "
            + ", ".join(contract for contract in contracts if not is_contract_id(contract))
        )
    capabilities: dict[str, Any] = {
        "device_types": ["cpu"],
        "dtypes": dtypes,
        "max_rank": max_rank,
        "features": features,
    }
    if contracts:
        capabilities["contracts"] = contracts
    return {
        "format": "inferref-engine-adapter",
        "format_version": "0.2",
        "name": "scaffolded-adapter",
        "target_device": "cpu",
        "capabilities": capabilities,
        "command": [
            "{adapter_dir}/inferref_bridge"
            if bridge_mode
            else "{adapter_dir}/inferref_adapter",
            "--testcase",
            "{testcase}",
            "--output",
            "{output}",
        ],
        "timeout_seconds": 60,
        "max_output_chars": 65536,
    }


def scaffold_adapter(
    testcase: str | Path,
    output: str | Path,
    *,
    language: str = "cpp",
    runtime_bridge: bool = False,
) -> ScaffoldResult:
    """Generate a compilable adapter project from one standalone testcase.

    With ``runtime_bridge=True`` the generated project is a bridge that
    delegates to a ``DebugInvoke`` callback (section 7.4) instead of a
    ``RunYourEngine`` function.
    """

    if language not in SUPPORTED_LANGUAGES:
        raise ScaffoldError(
            f"unsupported scaffold language {language!r}; supported: "
            + ", ".join(SUPPORTED_LANGUAGES)
        )
    testcase_path = Path(testcase)
    if not testcase_path.is_dir():
        raise ScaffoldError(f"testcase directory does not exist: {testcase_path}")
    validation = require_valid_testcase(testcase_path)
    manifest = validation.manifest

    adapter = _adapter_payload(manifest, bridge_mode=runtime_bridge)
    output_path = Path(output)
    if output_path.exists():
        raise ScaffoldError(
            f"output directory already exists: {output_path}; choose a fresh path"
        )
    output_path.mkdir(parents=True, exist_ok=False)
    files = {
        "CMakeLists.txt": _CMAKE_BRIDGE if runtime_bridge else _CMAKE,
        "adapter.json": json.dumps(adapter, indent=2, ensure_ascii=False) + "\n",
        "main.cpp": _MAIN_CPP_BRIDGE if runtime_bridge else _MAIN_CPP,
        "README.md": _generate_readme(
            str(manifest.get("name") or testcase_path.name),
            bridge_mode=runtime_bridge,
        ),
    }
    for name, content in files.items():
        (output_path / name).write_text(content, encoding="utf-8")

    # The generated declaration must be a valid adapter, not just well-formed
    # JSON (acceptance 11.2).
    try:
        EngineAdapter.load(output_path / "adapter.json")
    except (AgentProtocolError, OSError, ValueError) as exc:
        raise ScaffoldError(f"generated adapter failed validation: {exc}") from exc

    return ScaffoldResult(
        path=output_path,
        files=tuple(files),
        capabilities=adapter["capabilities"],
    )
