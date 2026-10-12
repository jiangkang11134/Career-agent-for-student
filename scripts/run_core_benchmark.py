"""Run deterministic checks for Smart Agent's core safety contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from pathlib import Path

# Allow both ``python scripts/run_core_benchmark.py`` and ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smart_agent.agents.contracts import AgentArtifact
from smart_agent.context_manager import ContextManager
from smart_agent.domains.runtime import require_explicit_domain
from smart_agent.file_review import load_existing_file
from smart_agent.tooling import ToolContext
from smart_agent.tools.career_task import _validate as validate_career_task
from smart_agent.tools.edit_file import edit_file_tool
from smart_agent.tools.read_file import read_file_tool
from smart_agent.tools.task import _validate as validate_code_task


def _check_path_and_lock(root: Path) -> dict[str, object]:
    target = root / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    context = ToolContext(
        cwd=str(root),
        _runtime={"optimisticLockRequired": True},
    )

    read_input = read_file_tool.validator({"path": "sample.py"})
    read = read_file_tool.run(read_input, context)
    digest = hashlib.sha256(load_existing_file(target).encode("utf-8")).hexdigest()
    target.write_text("value = 2\n", encoding="utf-8")
    conflict_input = edit_file_tool.validator({
            "path": "sample.py",
            "search": "value = 2",
            "replace": "value = 3",
            "expected_hash": digest,
        })
    conflict = edit_file_tool.run(conflict_input, context)
    try:
        read_file_tool.run(read_file_tool.validator({"path": "..\\outside.txt"}), context)
    except (PermissionError, OSError):
        outside_rejected = True
    else:
        outside_rejected = False
    current_digest = hashlib.sha256(load_existing_file(target).encode("utf-8")).hexdigest()
    applied = edit_file_tool.run(edit_file_tool.validator({
            "path": "sample.py",
            "search": "value = 2",
            "replace": "value = 3",
            "expected_hash": current_digest,
        }), context)
    return {
        "read_hash_present": digest in read.output,
        "optimistic_conflict_rejected": not conflict.ok and "conflict" in conflict.output.casefold(),
        "valid_edit_applied": applied.ok and target.read_text(encoding="utf-8") == "value = 3\n",
        "path_escape_rejected": outside_rejected,
    }


def _check_context() -> dict[str, object]:
    manager = ContextManager(model="offline", context_window=200)
    for _ in range(10):
        manager.add_message({"role": "user", "content": "long context " + "x" * 80})
    before = manager.get_stats().total_tokens
    manager.compact_messages(force=True)
    after = manager.get_stats().total_tokens
    return {
        "compaction_triggered": len(manager.compaction_history) > 0,
        "tokens_before": before,
        "tokens_after": after,
        "token_reduction_rate": round(1 - after / max(1, before), 6),
    }


def _check_artifact() -> dict[str, object]:
    artifact = AgentArtifact.create(
        "benchmark",
        domain=require_explicit_domain("code").value,
        session_id="benchmark-session",
        created_by="benchmark",
        content={"ok": True},
    )
    artifact.verify_integrity()
    rejected = False
    try:
        # Bypass constructor validation to simulate an in-memory mutation.
        object.__setattr__(artifact, "content", {"ok": False})
        artifact.verify_integrity()
    except ValueError:
        rejected = True
    return {"valid_artifact": True, "tamper_rejected": rejected}


def _check_agent_boundaries() -> dict[str, object]:
    missing_project_rejected = False
    try:
        validate_code_task({"description": "inspect", "prompt": "inspect", "agent_type": "general"})
    except ValueError:
        missing_project_rejected = True
    missing_approval_rejected = False
    try:
        validate_career_task(
            {
                "agent_type": "project_modify",
                "description": "change code",
                "artifacts": [{"artifact_type": "project_modification_request", "content": {"project_id": "demo", "change_goal": "fix"}}],
            }
        )
    except ValueError:
        missing_approval_rejected = True
    return {
        "unregistered_project_rejected": missing_project_rejected,
        "unapproved_project_modify_rejected": missing_approval_rejected,
    }


def run(output: Path) -> dict[str, object]:
    checks: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory(prefix="smart-agent-benchmark-") as name:
        checks["path_and_lock"] = _check_path_and_lock(Path(name))
    checks["context"] = _check_context()
    checks["artifact"] = _check_artifact()
    checks["agent_boundaries"] = _check_agent_boundaries()
    assertions = [value for group in checks.values() for value in group.values() if isinstance(value, bool)]
    passed = sum(assertions)
    report = {
        "protocol_version": "core-contracts-v1",
        "checks": checks,
        "passed": passed,
        "failed": len(assertions) - passed,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic core contract checks")
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/core-contracts.json"))
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
