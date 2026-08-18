#!/usr/bin/env python3
"""Validate the InferRef agent workflow guide and Codex skill package.

Stdlib-only. Run from the repository root:

    python scripts/check_agent_workflow_skill.py [--repo <root>]

Checks:

1. ``docs/AGENT_WORKFLOW.md``, ``skills/inferref/SKILL.md``, and every file
   linked from the skill's references table exist.
2. ``SKILL.md`` frontmatter has ``name: inferref`` and a non-empty
   ``description``.
3. Every ``inferref <subcommand>`` and ``inferref_<tool>`` name mentioned in
   code spans of the guide, skill, and references exists in the current CLI
   (verified against ``python -m inferref.cli.main`` help) and in the Agent
   capabilities ``mcp_tools`` list. Flags used with each command are checked
   against that command's own help output.
4. No scanned file contains author-machine absolute paths or credential-like
   values.

Exit code is 0 when every check passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GUIDE_REL = Path("docs") / "AGENT_WORKFLOW.md"
SKILL_REL = Path("skills") / "inferref" / "SKILL.md"
OPENAI_YAML_REL = Path("skills") / "inferref" / "agents" / "openai.yaml"
SKILL_DIR_REL = Path("skills") / "inferref"
COMMAND_GROUPS = ("agent", "scenario", "suite", "contract", "region", "testcase")

_PATH_PATTERNS = (
    (re.compile(r"E:\\RiderProjects", re.IGNORECASE), "author machine path"),
    (re.compile(r"C:\\Users", re.IGNORECASE), "absolute Windows user path"),
    (re.compile(r"C:/Users"), "absolute Windows user path"),
    (re.compile(r"/Users/"), "absolute macOS user path"),
    (re.compile(r"/home/"), "absolute Linux user path"),
    (re.compile(r"\b[A-Za-z]:\\"), "absolute drive path"),
)

_CREDENTIAL_PATTERNS = (
    re.compile(
        r"\b(?:api[_-]?key|password|passwd|secret)\b\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bbearer\s+[A-Za-z0-9._~+/-]+=*", re.IGNORECASE),
)

_COMMAND_RE = re.compile(
    r"\binferref\b\s+([a-z][a-z0-9_]*)(?:\s+([a-z][a-z0-9_]*))?"
)
_FLAG_RE = re.compile(r"(?<![A-Za-z0-9])(--[a-z][a-z0-9-]*|-[a-z])(?![A-Za-z0-9])")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=str(REPO_ROOT),
        help="repository root whose guide/skill files are checked "
        "(default: the repository containing this script)",
    )
    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()
    errors: list[str] = []
    mentions: tuple[set[tuple[str, str]], set[str], dict[tuple[str, str], set[str]]] = (
        set(),
        set(),
        {},
    )

    _check_required_files(repo, errors)
    skill_path = repo / SKILL_REL
    if skill_path.is_file():
        skill_text = skill_path.read_text(encoding="utf-8")
        _check_frontmatter(skill_text, errors)
        _check_reference_links(skill_path, skill_text, errors)

    files = _scanned_files(repo)
    if files:
        _check_cleanliness(files, errors)
        mentions = _extract_mentions(files)
        _check_commands(mentions, errors)

    if errors:
        for message in errors:
            print(f"ERROR: {message}", file=sys.stderr)
        print(
            f"FAIL: {len(errors)} agent workflow check(s) failed",
            file=sys.stderr,
        )
        return 1
    print(
        "OK: agent workflow checks passed "
        f"({len(files)} file(s), {len(mentions[0])} command mention(s), "
        f"{sum(len(flags) for flags in mentions[2].values())} flag mention(s), "
        f"{len(mentions[1])} MCP tool mention(s))"
    )
    return 0


def _check_required_files(repo: Path, errors: list[str]) -> None:
    required = (
        (GUIDE_REL, "guide is missing; Task A1 must create it"),
        (SKILL_REL, "skill SKILL.md is missing; Task A2 must create it"),
        (OPENAI_YAML_REL, "skill agents/openai.yaml is missing"),
    )
    for relative, reason in required:
        if not (repo / relative).is_file():
            errors.append(f"{relative.as_posix()}: {reason}")


def _check_frontmatter(skill_text: str, errors: list[str]) -> None:
    frontmatter = _parse_frontmatter(skill_text)
    if not frontmatter:
        errors.append(f"{SKILL_REL.as_posix()}: frontmatter is missing or malformed")
        return
    if frontmatter.get("name") != "inferref":
        errors.append(
            f"{SKILL_REL.as_posix()}: frontmatter name must be 'inferref', "
            f"got {frontmatter.get('name')!r}"
        )
    if not frontmatter.get("description"):
        errors.append(
            f"{SKILL_REL.as_posix()}: frontmatter description must be non-empty"
        )


def _check_reference_links(
    skill_path: Path, skill_text: str, errors: list[str]
) -> None:
    skill_dir = skill_path.parent
    for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", skill_text):
        if link.startswith(("http://", "https://", "#", "mailto:")):
            continue
        target = (skill_dir / link).resolve()
        if not target.is_file():
            errors.append(
                f"{SKILL_REL.as_posix()}: referenced file does not exist: {link}"
            )


def _scanned_files(repo: Path) -> list[Path]:
    files = []
    for relative in (GUIDE_REL, SKILL_REL):
        path = repo / relative
        if path.is_file():
            files.append(path)
    references = repo / SKILL_DIR_REL / "references"
    if references.is_dir():
        files.extend(sorted(references.glob("*.md")))
    return files


def _check_cleanliness(files: list[Path], errors: list[str]) -> None:
    for path in files:
        text = path.read_text(encoding="utf-8")
        for pattern, label in _PATH_PATTERNS:
            match = pattern.search(text)
            if match is not None:
                errors.append(
                    f"{_display(path)}: contains {label}: "
                    f"{match.group(0)!r}"
                )
        for pattern in _CREDENTIAL_PATTERNS:
            match = pattern.search(text)
            if match is not None:
                errors.append(
                    f"{_display(path)}: contains credential-like value: "
                    f"{match.group(0)!r}"
                )


def _extract_mentions(
    files: list[Path],
) -> tuple[
    set[tuple[str, str]], set[str], dict[tuple[str, str], set[str]]
]:
    commands: set[tuple[str, str]] = set()
    tools: set[str] = set()
    flags: dict[tuple[str, str], set[str]] = {}
    for path in files:
        text = path.read_text(encoding="utf-8")
        for kind, span in _code_spans(text):
            for command, line_flags in _span_commands((kind, span)):
                commands.add(command)
                flags.setdefault(command, set()).update(line_flags)
            for tool_match in re.finditer(r"\binferref_[a-z_]+", span):
                # Skip JSON field names such as "inferref_version".
                if (
                    span[tool_match.start() - 1 : tool_match.start()] == '"'
                    and span[tool_match.end() : tool_match.end() + 1] == '"'
                ):
                    continue
                tools.add(tool_match.group(0))
    return commands, tools, flags


def _check_commands(
    mentions: tuple[
        set[tuple[str, str]], set[str], dict[tuple[str, str], set[str]]
    ],
    errors: list[str],
) -> None:
    commands, tools, flags = mentions
    top_commands = _cli_choices(REPO_ROOT)
    group_commands = {
        group: _cli_choices(REPO_ROOT, group) for group in COMMAND_GROUPS
    }
    mcp_tools = _mcp_tools(REPO_ROOT)
    flag_cache: dict[tuple[str, ...], set[str]] = {}

    for top, sub in sorted(commands, key=lambda item: (item[0], item[1] or "")):
        if top not in top_commands:
            errors.append(
                f"unknown CLI command: inferref {top}"
                f"{(' ' + sub) if sub else ''}"
            )
        elif sub and top in group_commands and sub not in group_commands[top]:
            errors.append(
                f"unknown CLI subcommand: inferref {top} {sub}"
            )
        help_args = (top, sub) if sub else (top,)
        known_flags = flag_cache.get(help_args)
        if known_flags is None:
            known_flags = _cli_flags(REPO_ROOT, *help_args)
            flag_cache[help_args] = known_flags
        for flag in sorted(flags.get((top, sub), ())):
            if flag not in known_flags:
                rendered = f"inferref {top}" + (f" {sub}" if sub else "")
                errors.append(f"unknown flag {flag!r} for {rendered}")
    for tool in sorted(tools):
        if tool not in mcp_tools:
            errors.append(f"unknown MCP tool: {tool}")


def _code_spans(text: str) -> list[tuple[str, str]]:
    spans: list[tuple[str, str]] = []
    for match in re.finditer(r"```.*?```", text, flags=re.DOTALL):
        spans.append(("block", match.group(0)[3:-3]))
    for match in re.finditer(r"`([^`\n]+)`", text):
        spans.append(("inline", match.group(1)))
    return spans


def _span_commands(
    span: tuple[str, str],
) -> list[tuple[tuple[str, str], set[str]]]:
    kind, content = span
    result: list[tuple[tuple[str, str], set[str]]] = []
    if kind == "inline":
        match = _COMMAND_RE.search(content)
        if match is not None:
            result.append(
                ((match.group(1), match.group(2)), _line_flags(content))
            )
        return result
    current: tuple[str, str] | None = None
    for line in content.splitlines():
        match = _COMMAND_RE.search(line)
        if match is not None:
            current = (match.group(1), match.group(2))
            result.append((current, _line_flags(line)))
            continue
        if not line.strip():
            current = None
            continue
        if current is not None:
            result.append((current, _line_flags(line)))
    return result


def _line_flags(line: str) -> set[str]:
    return set(_FLAG_RE.findall(line))


def _cli_choices(repo: Path, *args: str) -> set[str]:
    result = subprocess.run(
        [sys.executable, "-m", "inferref.cli.main", *args, "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    choices: set[str] = set()
    for line in result.stdout.splitlines():
        match = re.match(r"^ {4}([a-z_]+)(?: {2,}.*)?$", line)
        if match is not None:
            choices.add(match.group(1))
    return choices


def _cli_flags(repo: Path, *args: str) -> set[str]:
    result = subprocess.run(
        [sys.executable, "-m", "inferref.cli.main", *args, "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return set(_FLAG_RE.findall(result.stdout))


def _mcp_tools(repo: Path) -> set[str]:
    result = subprocess.run(
        [sys.executable, "-m", "inferref.cli.main", "agent", "capabilities", "--json"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return set()
    return set(payload.get("data", {}).get("mcp_tools", []))


def _parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    end = next(
        (index for index in range(1, len(lines)) if lines[index].strip() == "---"),
        None,
    )
    if end is None:
        return {}
    frontmatter: dict[str, str] = {}
    for line in lines[1:end]:
        match = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if match is not None:
            value = match.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            frontmatter[match.group(1)] = value
    return frontmatter


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
