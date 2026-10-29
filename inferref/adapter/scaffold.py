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

from inferref.agent.protocol import ADAPTER_FEATURES, AgentProtocolError, EngineAdapter
from inferref.testcase.requirements import derive_requirements, is_contract_id
from inferref.testcase.validate import require_valid_testcase

#: Feature vocabulary the generated adapter may declare (section 6.3): the
#: structural features plus the quantization scheme/granularity tokens that
#: ``derive_requirements()`` emits for testcase 0.4.
FEATURE_WHITELIST = ADAPTER_FEATURES

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
#include <vector>

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
        // __INFERREF_QUANTIZATION_QUERY__
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
#include <vector>

int main(int argc, char **argv)
{
    std::string testcase_dir;
    std::string output_dir;
    std::string batch_manifest;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--testcase" && i + 1 < argc)
            testcase_dir = argv[++i];
        else if (arg == "--output" && i + 1 < argc)
            output_dir = argv[++i];
        else if (arg == "--batch-manifest" && i + 1 < argc)
            batch_manifest = argv[++i];
    }
    if (!batch_manifest.empty())
    {
        if (!testcase_dir.empty() || !output_dir.empty())
        {
            std::cerr << "inferref_bridge: --batch-manifest cannot be combined "
                         "with --testcase/--output\\n";
            return inferref::kBridgeUsage;
        }
    }
    else if (testcase_dir.empty() || output_dir.empty())
    {
        std::cerr << "usage: inferref_bridge --testcase DIR --output DIR\\n"
                     "       inferref_bridge --batch-manifest FILE\\n";
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

    if (!batch_manifest.empty())
        return inferref::RunBridgeBatch(batch_manifest, invoke,
                                        "inferref_bridge");
    return inferref::RunBridge(testcase_dir, output_dir, invoke,
                               "inferref_bridge");
}
"""

_MAIN_CPP_BRIDGE_QUANTIZED = """\
#include <inferref/bridge.hpp>

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    std::string testcase_dir;
    std::string output_dir;
    std::string batch_manifest;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--testcase" && i + 1 < argc)
            testcase_dir = argv[++i];
        else if (arg == "--output" && i + 1 < argc)
            output_dir = argv[++i];
        else if (arg == "--batch-manifest" && i + 1 < argc)
            batch_manifest = argv[++i];
    }
    if (!batch_manifest.empty())
    {
        if (!testcase_dir.empty() || !output_dir.empty())
        {
            std::cerr << "inferref_bridge: --batch-manifest cannot be combined "
                         "with --testcase/--output\\n";
            return inferref::kBridgeUsage;
        }
    }
    else if (testcase_dir.empty() || output_dir.empty())
    {
        std::cerr << "usage: inferref_bridge --testcase DIR --output DIR\\n"
                     "       inferref_bridge --batch-manifest FILE\\n";
        return inferref::kBridgeUsage;
    }

    // The only user edit: implement engine dispatch in this callback.  A 0.4
    // testcase carries quantization descriptors and parameter tensors, so this
    // project registers the Testcase-aware DebugInvokeEx callback.
    inferref::DebugInvokeEx invoke =
        [](const inferref::Testcase &testcase,
           const std::string &region_name,
           const std::map<std::string, inferref::IRTensor> &inputs)
            -> std::map<std::string, inferref::IRTensor>
    {
        // __INFERREF_QUANTIZATION_QUERY__
        // TODO: implement engine dispatch.
        (void)region_name;
        (void)inputs;
        throw std::runtime_error("DebugInvokeEx is not implemented");
    };

    if (!batch_manifest.empty())
        return inferref::RunBridgeBatch(batch_manifest, invoke,
                                        "inferref_bridge");
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


def _quantized_input_roles(manifest: dict[str, Any]) -> list[str]:
    """Input role names that carry a testcase 0.4 quantization descriptor."""

    roles: list[str] = []
    for item in manifest.get("inputs", []):
        if not isinstance(item, dict) or not isinstance(item.get("quantization"), dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            roles.append(name)
    return roles


_QUANTIZATION_QUERY_MARKER = "        // __INFERREF_QUANTIZATION_QUERY__"


def _quantization_query_block(roles: list[str]) -> str:
    """Compilable testcase 0.4 quantization queries for each quantized input."""

    blocks = []
    for role in roles:
        blocks.append(
            f'        // Testcase 0.4: input role "{role}" carries an explicit\n'
            "        // quantization descriptor and immutable parameter tensors.\n"
            f'        if (testcase.HasInputQuantization("{role}"))\n'
            "        {\n"
            "            const inferref::QuantizationSpec &spec =\n"
            f'                testcase.InputQuantization("{role}");\n'
            "            const auto parameters =\n"
            "                testcase.QuantizationParameters();\n"
            "            const inferref::IRTensor scale =\n"
            "                testcase.QuantizationParameter(spec.scale_value_id);\n"
            "            const std::vector<float> logical =\n"
            f'                testcase.DequantizeInputToFloat32("{role}");\n'
            "            (void)parameters;\n"
            "            (void)scale;\n"
            "            (void)logical;\n"
            "        }"
        )
    return "\n".join(blocks)


def _render_main_cpp(*, bridge_mode: bool, quantized_roles: list[str]) -> str:
    """Render main.cpp with quantization queries when the testcase is 0.4."""

    if bridge_mode:
        template = _MAIN_CPP_BRIDGE_QUANTIZED if quantized_roles else _MAIN_CPP_BRIDGE
    else:
        template = _MAIN_CPP
    return template.replace(
        _QUANTIZATION_QUERY_MARKER, _quantization_query_block(quantized_roles)
    )


def _generate_readme(
    testcase_name: str, *, bridge_mode: bool, quantized: bool
) -> str:
    callback = "DebugInvokeEx" if quantized else "DebugInvoke"
    quantization_note = (
        "\nThis testcase is format 0.4 with quantized inputs. The generated\n"
        "`main.cpp` queries `testcase.HasInputQuantization(...)`,\n"
        "`InputQuantization(...)`, `QuantizationParameters()`,\n"
        "`QuantizationParameter(...)`, and `DequantizeInputToFloat32(...)` as a\n"
        "starting point; `adapter.json` declares the matching\n"
        "`quantization/...` capability tokens.\n"
        if quantized
        else ""
    )
    if bridge_mode:
        return f"""\
# Scaffolded InferRef runtime bridge

Generated from the standalone testcase `{testcase_name}` in bridge mode. The
output handling is already wired through `inferref::RunBridge`: it loads the
testcase, resolves the region dispatch key, calls your `{callback}` callback
with named inputs, writes every declared output by role name, and publishes
`manifest.json`.
{quantization_note}
## What to edit

1. `main.cpp` - implement engine dispatch inside the `{callback}` callback.
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
{quantization_note}
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
    quantized_roles = _quantized_input_roles(manifest)
    output_path = Path(output)
    if output_path.exists():
        raise ScaffoldError(
            f"output directory already exists: {output_path}; choose a fresh path"
        )
    output_path.mkdir(parents=True, exist_ok=False)
    files = {
        "CMakeLists.txt": _CMAKE_BRIDGE if runtime_bridge else _CMAKE,
        "adapter.json": json.dumps(adapter, indent=2, ensure_ascii=False) + "\n",
        "main.cpp": _render_main_cpp(
            bridge_mode=runtime_bridge, quantized_roles=quantized_roles
        ),
        "README.md": _generate_readme(
            str(manifest.get("name") or testcase_path.name),
            bridge_mode=runtime_bridge,
            quantized=bool(quantized_roles),
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
