import shutil
import subprocess
import os

import pytest

from smart_agent.tooling import ToolContext
from smart_agent.tools.git import git_tool
from smart_agent.tools.run_command import run_command_tool


def _git(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _call(tool, payload, repo):
    return tool.run(tool.validator(payload), ToolContext(cwd=str(repo)))


@pytest.fixture()
def git_repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "agent@example.com")
    _git(repo, "config", "user.name", "Agent Test")
    return repo


def test_git_tool_stage_unstage_commit_and_branch(git_repo):
    tracked = git_repo / "agent.py"
    tracked.write_text("print('v1')\n", encoding="utf-8")

    result = _call(git_tool, {"action": "add", "paths": ["agent.py"]}, git_repo)
    assert result.ok
    assert "agent.py" in result.output

    result = _call(git_tool, {"action": "diff", "staged": True}, git_repo)
    assert result.ok
    assert "agent.py" in result.output

    result = _call(git_tool, {"action": "restore", "paths": ["agent.py"]}, git_repo)
    assert result.ok

    result = _call(git_tool, {"action": "status"}, git_repo)
    assert result.ok
    assert "?? agent.py" in result.output

    assert _call(git_tool, {"action": "add", "paths": ["agent.py"]}, git_repo).ok
    result = _call(git_tool, {"action": "commit", "message": "add agent file"}, git_repo)
    assert result.ok
    assert "Committed: add agent file" in result.output

    result = _call(git_tool, {"action": "branch", "branch_op": "create", "branch": "agent/test"}, git_repo)
    assert result.ok

    result = _call(git_tool, {"action": "branch"}, git_repo)
    assert result.ok
    assert "agent/test" in result.output


def test_git_tool_rejects_unsafe_paths_and_destructive_restore(git_repo, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')\n", encoding="utf-8")

    result = _call(git_tool, {"action": "add", "paths": [str(outside)]}, git_repo)
    assert not result.ok
    assert "inside repository" in result.output

    result = _call(git_tool, {"action": "restore", "paths": ["agent.py"], "restore_mode": "worktree"}, git_repo)
    assert not result.ok
    assert "Blocked" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["reset", "--hard"],
        ["clean", "-fd"],
        ["checkout", "--", "agent.py"],
        ["push", "--force"],
        ["restore", "--source=HEAD~1", "agent.py"],
    ],
)
def test_run_command_blocks_high_risk_git_commands(git_repo, args):
    result = _call(run_command_tool, {"command": "git", "args": args}, git_repo)
    assert not result.ok
    assert "blocked by default" in result.output


def test_run_command_windows_shell_snippet_uses_real_shell(git_repo):
    if os.name != "nt":
        pytest.skip("Windows shell selection only applies on Windows")

    result = _call(run_command_tool, {"command": "echo smart-agent-shell | head -1"}, git_repo)

    assert result.ok
    assert "smart-agent-shell" in result.output
