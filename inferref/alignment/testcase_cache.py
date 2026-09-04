"""Content-addressed testcase cache and extractor for Alignment Session v0.1 (L4)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

from inferref.alignment.boundaries import Boundary
from inferref.alignment.hashing import tree_sha256
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.testcase.extract import extract_operator, extract_region
from inferref.testcase.validate import require_valid_testcase


class TestcaseCache:
    """Manages content-addressed testcase artifacts under the session."""

    def __init__(self, testcases_root: Path) -> None:
        self.root = Path(testcases_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def get_or_extract(
        self,
        package: TracePackage,
        boundary: Boundary,
    ) -> tuple[Path, str, bool]:
        """Obtain a validated testcase for a boundary.

        Returns (testcase_path, tree_sha256, reused: bool).
        """
        clean_key = boundary.id.replace(":", "_")

        # Check existing testcases matching prefix
        for candidate_dir in self.root.glob(f"{clean_key}-*"):
            if candidate_dir.is_dir() and (candidate_dir / "testcase.json").is_file():
                try:
                    validation = require_valid_testcase(candidate_dir)
                    if validation.reproducible:
                        tree_sha = tree_sha256(candidate_dir)
                        if candidate_dir.name.endswith(tree_sha[:16]):
                            return candidate_dir, tree_sha, True
                except Exception as exc:
                    logger.debug("Failed validating candidate testcase cache %s: %s", candidate_dir, exc)
                    continue

        # Extract to staging directory
        staging_dir = self.root / f".tmp-{clean_key}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

        staging_dir.mkdir(parents=True, exist_ok=True)

        try:
            if boundary.kind == "region":
                # Find or construct RegionRecord
                region = next((r for r in package.regions if r.id == boundary.metadata.get("region_id")), None)
                if region is None or not region.outputs:
                    region = RegionRecord(
                        id=int(boundary.metadata.get("region_id", 0)),
                        name=boundary.name,
                        node_ids=boundary.node_ids,
                        inputs=boundary.inputs,
                        outputs=boundary.outputs,
                    )
                extract_region(package, region, staging_dir, force=True)
            elif boundary.kind == "op":
                op_id = int(boundary.metadata.get("operator_id", boundary.node_ids[0]))
                extract_operator(package, op_id, staging_dir, force=True)
            else:
                # Module or derived boundary
                region = RegionRecord(
                    id=abs(hash(boundary.id)) % 2_000_000_000,
                    name=boundary.name,
                    node_ids=boundary.node_ids,
                    inputs=boundary.inputs,
                    outputs=boundary.outputs,
                )
                extract_region(package, region, staging_dir, force=True)

            validation = require_valid_testcase(staging_dir)
            if not validation.reproducible:
                raise ValueError(f"extracted testcase for {boundary.id} is not reproducible")

            tree_sha = tree_sha256(staging_dir)
            final_dir = self.root / f"{clean_key}-{tree_sha[:16]}"
            if final_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
                return final_dir, tree_sha, True

            staging_dir.rename(final_dir)
            return final_dir, tree_sha, False
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
    def import_external(
        self,
        external_path: Path,
        boundary: Boundary,
    ) -> tuple[Path, str, bool]:
        """Import an external testcase into the content-addressed session cache.

        Returns (imported_path, tree_sha256, reused: bool).
        """
        ext_p = Path(external_path).resolve()
        validation = require_valid_testcase(ext_p)
        if not validation.reproducible:
            raise ValueError(f"external testcase {ext_p} is not reproducible")

        tree_sha = tree_sha256(ext_p)
        clean_key = boundary.id.replace(":", "_")
        final_dir = self.root / f"{clean_key}-{tree_sha[:16]}"
        if final_dir.is_dir() and (final_dir / "testcase.json").is_file():
            try:
                cached_val = require_valid_testcase(final_dir)
                if cached_val.reproducible and tree_sha256(final_dir) == tree_sha:
                    return final_dir, tree_sha, True
            except Exception as exc:
                logger.debug(
                    "Cached external testcase %s is invalid or modified, re-importing: %s",
                    final_dir,
                    exc,
                )
            shutil.rmtree(final_dir, ignore_errors=True)

        staging_dir = self.root / f".tmp-ext-{clean_key}"
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

        try:
            shutil.copytree(ext_p, staging_dir)
            staged_val = require_valid_testcase(staging_dir)
            if not staged_val.reproducible:
                raise ValueError(f"imported external testcase in {staging_dir} is not reproducible")
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            staging_dir.rename(final_dir)
            return final_dir, tree_sha, False
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
