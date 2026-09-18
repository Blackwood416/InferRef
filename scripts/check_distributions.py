"""Validate one wheel/sdist pair; optionally install both outside the checkout."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def source_version(data: bytes) -> str:
    for node in ast.parse(data.decode("utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "INFERREF_VERSION"
            for target in node.targets
        ):
            return str(ast.literal_eval(node.value))
    raise ValueError("INFERREF_VERSION missing from distribution")


def check(root: Path, expected_version: str | None = None) -> tuple[Path, Path, str]:
    wheels, sdists = sorted(root.glob("*.whl")), sorted(root.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one wheel and one sdist in a fresh output directory")
    wheel, sdist = wheels[0], sdists[0]
    required = ("inferref/py.typed", "inferref/alignment/runner.py", "inferref/cli/main.py")
    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("wheel must contain exactly one metadata record")
        metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
        if metadata["Name"] != "inferref":
            raise ValueError("unexpected wheel project name")
        version = str(metadata["Version"])
        if source_version(archive.read("inferref/ir/version.py")) != version:
            raise ValueError("wheel source and metadata versions disagree")
        for name in required:
            archive.read(name)
    with tarfile.open(sdist) as archive:
        prefix = f"inferref-{version}/"
        for name in (*required, "pyproject.toml", "README.md", "PKG-INFO", "inferref/ir/version.py"):
            member = archive.getmember(prefix + name)
            if not member.isfile():
                raise ValueError(f"expected regular distribution member: {name}")
        metadata_file = archive.extractfile(prefix + "PKG-INFO")
        version_file = archive.extractfile(prefix + "inferref/ir/version.py")
        assert metadata_file is not None and version_file is not None
        metadata = BytesParser().parsebytes(metadata_file.read())
        if metadata["Name"] != "inferref" or metadata["Version"] != version:
            raise ValueError("wheel and sdist metadata disagree")
        if source_version(version_file.read()) != version:
            raise ValueError("sdist source and metadata versions disagree")
    if expected_version is not None and version != expected_version:
        raise ValueError(f"expected version {expected_version}, found {version}")
    return wheel, sdist, version


def install_check(artifacts: tuple[Path, Path], version: str) -> None:
    for artifact in artifacts:
        with tempfile.TemporaryDirectory(prefix="inferref-package-smoke-") as temporary:
            root = Path(temporary)
            env = root / "venv"
            subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
            python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            subprocess.run([str(python), "-m", "pip", "install", str(artifact.resolve())], check=True, cwd=root)
            code = (
                "import inferref,sys; from pathlib import Path; "
                "from inferref.alignment.runner import AlignmentRunner; "
                "from inferref.compare.compare import compare_traces; "
                "import numpy as np; "
                "from inferref.quantization import QuantizationSpec, dequantize_affine; "
                "from inferref.quantization.dependencies import quantization_dependencies; "
                "from inferref.tensor import TensorView; "
                "spec = QuantizationSpec(logical_shape=(1,), scale_value_id=7); "
                "codes = TensorView('int8', (1,), (1,), 0, np.array([2], dtype=np.int8)); "
                "scale = TensorView('float32', (), (), 0, np.array(.5, dtype=np.float32)); "
                "assert dequantize_affine(codes, spec, values={7: scale}).tolist() == [1.0]; "
                "assert quantization_dependencies([1], specs={1: spec}, "
                "available_ids=[1, 7]).parameter_ids == (7,); "
                "from inferref.quantization.nf4 import BITSANDBYTES_PROFILE,NF4_CODEBOOK,import_bitsandbytes_nf4,dequantize_nf4; "
                "state = dict(shape=(2,),dtype='float16',quant_type='nf4',blocksize=64,"
                "absmax=np.ones(1,dtype=np.float32),code=np.array(NF4_CODEBOOK,dtype=np.float32),"
                "nested=False,offset=None,state2=None); "
                "nf4 = import_bitsandbytes_nf4(np.array([15],dtype=np.uint8),state,"
                "profile=BITSANDBYTES_PROFILE,scale_value_id=10,codebook_value_id=11); "
                "assert nf4.codes.tensor_format_version == 2; "
                "assert nf4.codes.payload_bytes() == bytes([240]); "
                "assert dequantize_nf4(nf4.codes,nf4.spec,values=nf4.parameters).tolist() == [-1.0,1.0]; "
                "from inferref.quantization.bitsandbytes import QUANT_STATE_PROFILE,import_bitsandbytes_nf4_projection; "
                "projected = import_bitsandbytes_nf4_projection(np.array([15],dtype=np.uint8),state,"
                "profile=QUANT_STATE_PROFILE,scale_value_id=10,codebook_value_id=11); "
                "assert projected.dequantize_source().data.tolist() == [-1.0,1.0]; "
                f"assert inferref.__version__ == {version!r}; "
                "assert Path(inferref.__file__).is_relative_to(Path(sys.prefix)); "
                "assert Path(inferref.__file__).with_name('py.typed').is_file(); "
                "assert 'torch' not in sys.modules"
            )
            subprocess.run([str(python), "-I", "-c", code], check=True, cwd=root)
            cli = python.parent / ("inferref.exe" if sys.platform == "win32" else "inferref")
            output = subprocess.check_output([str(cli), "--version"], text=True, cwd=root)
            if output.strip() != f"inferref {version}":
                raise ValueError(f"unexpected installed CLI version: {output!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-version")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    wheel, sdist, version = check(args.directory, args.expected_version)
    if args.install:
        install_check((wheel, sdist), version)
    print(json.dumps({"version": version, "wheel": str(wheel), "sdist": str(sdist),
                      "installed": args.install}))


if __name__ == "__main__":
    main()
