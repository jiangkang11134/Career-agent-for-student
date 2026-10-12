from __future__ import annotations

"""Safe Git workflow tool for Code Agent."""

import re
import subprocess
from pathlib import Path

from smart_agent.tooling import ToolDefinition, ToolResult

_ACTIONS = ("status", "diff", "log", "add", "restore", "commit", "branch", "review")
_BRANCH_OPS = ("list", "create", "switch")
_SAFE_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")


def _validate(input_data: dict) -> dict:
    action = input_data.get("action")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("action is required")
    action = action.strip().lower()
    if action not in _ACTIONS:
        raise ValueError(f"action must be one of: {', '.join(_ACTIONS)}")

    message = str(input_data.get("message") or "").strip()
    if action == "commit" and not message:
        raise ValueError("message is required for commit action")

    paths = input_data.get("paths")
    if paths is None and input_data.get("path") is not None:
        paths = [input_data.get("path")]
    if paths is None:
        paths = []
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list):
        raise ValueError("paths must be a list of relative paths")

    branch_op = str(input_data.get("branch_op") or "list").strip().lower()
    if branch_op not in _BRANCH_OPS:
        raise ValueError(f"branch_op must be one of: {', '.join(_BRANCH_OPS)}")

    branch_name = str(input_data.get("branch") or input_data.get("branch_name") or "").strip()
    start_point = str(input_data.get("start_point") or "").strip()
    restore_mode = str(input_data.get("restore_mode") or "staged").strip().lower()
    if restore_mode not in {"staged", "worktree"}:
        raise ValueError("restore_mode must be staged or worktree")

    try:
        max_lines = int(input_data.get("max_lines", 80))
    except (TypeError, ValueError):
        max_lines = 80
    max_lines = min(max(max_lines, 1), 500)

    return {
        "action": action,
        "message": message,
        "paths": [str(path).strip() for path in paths if str(path).strip()],
        "branch_op": branch_op,
        "branch_name": branch_name,
        "start_point": start_point,
        "restore_mode": restore_mode,
        "max_lines": max_lines,
        "staged": bool(input_data.get("staged", False)),
    }


def _run(input_data: dict, context) -> ToolResult:
    cwd = str(Path(context.cwd).resolve())
    action = input_data["action"]
    try:
        if action == "status":
            return _run_status(cwd)
        if action == "diff":
            return _run_diff(cwd, input_data["max_lines"], input_data["paths"], input_data["staged"])
        if action == "log":
            return _run_log(cwd, input_data["max_lines"])
        if action == "add":
            return _run_add(cwd, input_data["paths"], context)
        if action == "restore":
            return _run_restore(cwd, input_data["paths"], input_data["restore_mode"], context)
        if action == "commit":
            return _run_commit(cwd, input_data["message"], context)
        if action == "branch":
            return _run_branch(cwd, input_data["branch_op"], input_data["branch_name"], input_data["start_point"], context)
        if action == "review":
            return _run_review(cwd, input_data["max_lines"])
    except FileNotFoundError:
        return ToolResult(ok=False, output="Git is not installed or not in PATH.")
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, output="Git command timed out.")
    except Exception as exc:
        return ToolResult(ok=False, output=f"Git error: {exc}")

    return ToolResult(ok=False, output=f"Unknown action: {action}")


def _run_git(args: list[str], cwd: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _ensure_git_allowed(context, args: list[str]) -> None:
    permissions = getattr(context, "permissions", None)
    if permissions is not None:
        permissions.ensure_command("git", args, str(Path(context.cwd).resolve()))


def _normalize_paths(cwd: str, paths: list[str], *, required: bool) -> list[str]:
    if required and not paths:
        raise ValueError("paths is required for this action")
    root = Path(cwd).resolve()
    normalized: list[str] = []
    for raw in paths:
        if "\x00" in raw:
            raise ValueError("path contains invalid null byte")
        if raw.startswith(":"):
            raise ValueError("git pathspec magic is not allowed")
        if raw in {".", "./"}:
            normalized.append(".")
            continue
        path = Path(raw)
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        try:
            rel = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path must stay inside repository: {raw}") from exc
        normalized.append("." if str(rel) == "." else rel.as_posix())
    return normalized


def _format_lines(title: str, stdout: str, max_lines: int) -> ToolResult:
    if not stdout:
        return ToolResult(ok=True, output=f"{title}: no output.")
    lines = stdout.splitlines()
    body = lines[:max_lines]
    if len(lines) > max_lines:
        body.append(f"... and {len(lines) - max_lines} more lines")
    return ToolResult(ok=True, output="\n".join([title, "", *body]))


def _branch_name_is_safe(branch: str) -> bool:
    return bool(branch and _SAFE_BRANCH_RE.match(branch) and ".." not in branch and not branch.endswith(("/", ".", ".lock")))


def _working_tree_dirty(cwd: str) -> bool:
    rc, stdout, _ = _run_git(["status", "--porcelain"], cwd)
    return rc == 0 and bool(stdout)


def _run_status(cwd: str) -> ToolResult:
    rc, stdout, stderr = _run_git(["status", "--short", "--branch"], cwd)
    if rc != 0:
        return ToolResult(ok=False, output=f"Git status failed: {stderr}")
    if not stdout:
        return ToolResult(ok=True, output="Working tree clean. Nothing to commit.")
    staged = sum(1 for line in stdout.splitlines() if line and not line.startswith((" ", "??", "##")))
    unstaged = sum(1 for line in stdout.splitlines() if line.startswith((" ", "??")))
    lines = ["Git Status:", f"  Staged changes: {staged}", f"  Unstaged/untracked changes: {unstaged}", "", "Files:"]
    lines.extend(f"  {line}" for line in stdout.splitlines()[:40])
    if stdout.count("\n") >= 40:
        lines.append(f"... and {stdout.count(chr(10)) - 39} more files")
    return ToolResult(ok=True, output="\n".join(lines))


def _run_diff(cwd: str, max_lines: int, paths: list[str], staged: bool) -> ToolResult:
    safe_paths = _normalize_paths(cwd, paths, required=False)
    args = ["diff", "--cached" if staged else "--stat"]
    if staged:
        args.append("--stat")
    if safe_paths:
        args.extend(["--", *safe_paths])
    rc, stdout, stderr = _run_git(args, cwd)
    if rc != 0:
        return ToolResult(ok=False, output=f"Git diff failed: {stderr}")
    title = "Staged Changes" if staged else "Unstaged Changes"
    if not stdout:
        return ToolResult(ok=True, output=f"No {title.lower()}.")
    return _format_lines(title, stdout, max_lines)


def _run_log(cwd: str, max_lines: int) -> ToolResult:
    rc, stdout, stderr = _run_git(["log", "--oneline", f"-{max_lines}"], cwd)
    if rc != 0:
        return ToolResult(ok=False, output=f"Git log failed: {stderr}")
    return _format_lines("Recent Commits", stdout or "No commits found.", max_lines)


def _run_add(cwd: str, paths: list[str], context) -> ToolResult:
    safe_paths = _normalize_paths(cwd, paths, required=True)
    args = ["add", "--", *safe_paths]
    _ensure_git_allowed(context, args)
    rc, _, stderr = _run_git(args, cwd)
    if rc != 0:
        return ToolResult(ok=False, output=f"Git add failed: {stderr}")
    rc, staged, stderr = _run_git(["diff", "--cached", "--stat"], cwd)
    if rc != 0:
        return ToolResult(ok=False, output=f"Git add verification failed: {stderr}")
    return ToolResult(ok=True, output=staged or "No staged changes after git add.")


def _run_restore(cwd: str, paths: list[str], restore_mode: str, context) -> ToolResult:
    if restore_mode == "worktree":
        return ToolResult(ok=False, output="Blocked: restoring worktree files can discard local changes. Only restore_mode=staged is allowed.")
    safe_paths = _normalize_paths(cwd, paths, required=True)
    head_rc, _, _ = _run_git(["rev-parse", "--verify", "HEAD"], cwd)
    args = (
        ["restore", "--staged", "--", *safe_paths]
        if head_rc == 0
        else ["rm", "--cached", "-r", "--", *safe_paths]
    )
    _ensure_git_allowed(context, args)
    rc, _, stderr = _run_git(args, cwd)
    if rc != 0:
        return ToolResult(ok=False, output=f"Git restore failed: {stderr}")
    return ToolResult(ok=True, output="Paths unstaged successfully.")


def _run_commit(cwd: str, message: str, context) -> ToolResult:
    rc, staged, stderr = _run_git(["diff", "--cached", "--stat"], cwd)
    if rc != 0:
        return ToolResult(ok=False, output=f"Git commit precheck failed: {stderr}")
    if not staged:
        return ToolResult(ok=False, output="No staged changes. Use git action=add with explicit paths first.")
    args = ["commit", "-m", message]
    _ensure_git_allowed(context, args)
    rc, stdout, stderr = _run_git(args, cwd)
    if rc != 0:
        return ToolResult(ok=False, output=f"Commit failed: {stderr}")
    return ToolResult(ok=True, output="\n".join([f"Committed: {message}", "", stdout, "", "Changes:", staged]).strip())


def _run_branch(cwd: str, branch_op: str, branch_name: str, start_point: str, context) -> ToolResult:
    if branch_op == "list":
        rc, stdout, stderr = _run_git(["branch", "--list"], cwd)
        if rc != 0:
            return ToolResult(ok=False, output=f"Git branch failed: {stderr}")
        return _format_lines("Branches", stdout or "(no branches)", 200)

    if not _branch_name_is_safe(branch_name):
        return ToolResult(ok=False, output="Invalid or unsafe branch name.")
    if _working_tree_dirty(cwd):
        return ToolResult(ok=False, output="Working tree has local changes. Commit or unstage before switching branches.")

    if branch_op == "create":
        args = ["switch", "-c", branch_name]
        if start_point:
            args.append(start_point)
    else:
        args = ["switch", branch_name]
    _ensure_git_allowed(context, args)
    rc, stdout, stderr = _run_git(args, cwd)
    if rc != 0:
        return ToolResult(ok=False, output=f"Git branch {branch_op} failed: {stderr}")
    return ToolResult(ok=True, output=stdout or f"Branch {branch_op} completed: {branch_name}")


def _run_review(cwd: str, max_lines: int) -> ToolResult:
    rc, log, stderr = _run_git(["log", "--oneline", "-5"], cwd)
    if rc != 0:
        return ToolResult(ok=False, output=f"Git review failed: {stderr}")
    _, status, _ = _run_git(["status", "--short"], cwd)
    _, unstaged, _ = _run_git(["diff", "--stat"], cwd)
    _, staged, _ = _run_git(["diff", "--cached", "--stat"], cwd)
    lines = [
        "Git Review",
        "=" * 40,
        "",
        "Recent commits:",
        log or "(no commits)",
        "",
        "Staged:",
        staged or "(none)",
        "",
        "Unstaged:",
        unstaged or "(none)",
        "",
        "Status:",
        status or "Working tree clean",
    ]
    return _format_lines("Git Review", "\n".join(lines), max_lines)


git_tool = ToolDefinition(
    name="git",
    description=(
        "Safe Git workflow tool. Actions: status, diff, log, add, restore, commit, branch, review. "
        "Supports explicit staging, staged diff, safe unstage, clean-tree branch create/switch, and commit."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(_ACTIONS), "description": "Git action to perform"},
            "paths": {"type": "array", "items": {"type": "string"}, "description": "Explicit relative paths for add/diff/restore"},
            "path": {"type": "string", "description": "Single relative path alias for paths"},
            "staged": {"type": "boolean", "description": "For diff: show staged diff stat"},
            "message": {"type": "string", "description": "Commit message for commit action"},
            "branch_op": {"type": "string", "enum": list(_BRANCH_OPS), "description": "For branch: list, create, or switch"},
            "branch": {"type": "string", "description": "Branch name for create/switch"},
            "branch_name": {"type": "string", "description": "Branch name alias"},
            "start_point": {"type": "string", "description": "Optional start point for branch create"},
            "restore_mode": {"type": "string", "enum": ["staged", "worktree"], "description": "Only staged is allowed by default"},
            "max_lines": {"type": "number", "description": "Maximum output lines, 1-500"},
        },
        "required": ["action"],
    },
    validator=_validate,
    run=_run,
)
