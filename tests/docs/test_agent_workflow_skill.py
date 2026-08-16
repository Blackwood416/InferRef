"""A4 validation harness for the agent workflow guide and Codex skill.

Runs in tests/core style: no PyTorch and no MCP package are imported by the
checked paths. The harness itself is stdlib-only.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_agent_workflow_skill.py"

MINIMAL_SKILL = """---
name: inferref
description: "validate inference engines"
---
# InferRef

Use `inferref doctor --json`.
"""

MINIMAL_GUIDE = """# Workflow guide

```powershell
inferref doctor --json
```
"""


def _run(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _tree(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "skills" / "inferref" / "references").mkdir(parents=True)
    return repo


def _write(repo: Path, relative: str, content: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_check_passes_on_current_repo() -> None:
    result = _run(REPO_ROOT)
    assert result.returncode == 0, result.stderr
    assert "OK: agent workflow checks passed" in result.stdout


def test_check_fails_when_guide_is_missing(tmp_path: Path) -> None:
    repo = _tree(tmp_path)
    _write(repo, "skills/inferref/SKILL.md", MINIMAL_SKILL)
    result = _run(repo)
    assert result.returncode == 1
    assert "docs/AGENT_WORKFLOW.md" in result.stderr


def test_check_fails_on_unknown_command(tmp_path: Path) -> None:
    repo = _tree(tmp_path)
    _write(repo, "docs/AGENT_WORKFLOW.md", MINIMAL_GUIDE)
    _write(repo, "skills/inferref/SKILL.md", MINIMAL_SKILL)
    _write(
        repo,
        "docs/AGENT_WORKFLOW.md",
        "```powershell\ninferref frobnicate --json\n```\n",
    )
    result = _run(repo)
    assert result.returncode == 1
    assert "unknown CLI command: inferref frobnicate" in result.stderr


def test_check_fails_on_bad_frontmatter(tmp_path: Path) -> None:
    repo = _tree(tmp_path)
    _write(repo, "docs/AGENT_WORKFLOW.md", MINIMAL_GUIDE)
    _write(
        repo,
        "skills/inferref/SKILL.md",
        "---\nname: wrong-name\ndescription: \"x\"\n---\n# InferRef\n",
    )
    result = _run(repo)
    assert result.returncode == 1
    assert "name must be 'inferref'" in result.stderr


def test_check_fails_on_missing_reference_file(tmp_path: Path) -> None:
    repo = _tree(tmp_path)
    _write(repo, "docs/AGENT_WORKFLOW.md", MINIMAL_GUIDE)
    _write(
        repo,
        "skills/inferref/SKILL.md",
        MINIMAL_SKILL
        + "\n| [references/cli.md](references/cli.md) | exact CLI commands |\n",
    )
    result = _run(repo)
    assert result.returncode == 1
    assert "referenced file does not exist: references/cli.md" in result.stderr


def test_check_fails_on_author_machine_path(tmp_path: Path) -> None:
    repo = _tree(tmp_path)
    _write(repo, "skills/inferref/SKILL.md", MINIMAL_SKILL)
    _write(
        repo,
        "docs/AGENT_WORKFLOW.md",
        MINIMAL_GUIDE + "\n```text\nE:\\RiderProjects\\private\n```\n",
    )
    result = _run(repo)
    assert result.returncode == 1
    assert "author machine path" in result.stderr


def test_check_fails_on_credential_like_value(tmp_path: Path) -> None:
    repo = _tree(tmp_path)
    _write(repo, "skills/inferref/SKILL.md", MINIMAL_SKILL)
    _write(
        repo,
        "docs/AGENT_WORKFLOW.md",
        MINIMAL_GUIDE + "\n```text\napi_key = sk-abcdefghijklmnopqrstuvwx\n```\n",
    )
    result = _run(repo)
    assert result.returncode == 1
    assert "credential-like value" in result.stderr


def test_script_runs_without_site_packages() -> None:
    """The harness itself must stay stdlib-only (-S disables site-packages)."""

    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--repo", str(REPO_ROOT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK: agent workflow checks passed" in result.stdout
