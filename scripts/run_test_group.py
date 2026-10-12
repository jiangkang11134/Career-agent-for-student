from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

GROUPS = (
    "core",
    "code",
    "career",
    "web",
    "integration",
    "optional_dependency",
)

CODE_TESTS = {
    "test_agent_flow.py",
    "test_agent_loop.py",
    "test_cli_commands.py",
    "test_hooks.py",
    "test_review_system.py",
    "test_run_command_encoding.py",
    "test_turn_kernel.py",
}
WEB_TESTS = {"test_web_app.py"}
INTEGRATION_TESTS = {
    "test_headless.py",
    "test_integration.py",
    "test_memory_benchmark.py",
    "test_memory_e2e.py",
    "test_memory_integration.py",
    "test_product_surfaces.py",
    "test_subagents.py",
}
OPTIONAL_TESTS = {"test_career_live_api.py"}


def classify(path: Path) -> str:
    name = path.name
    if name in OPTIONAL_TESTS:
        return "optional_dependency"
    if name.startswith("test_career_"):
        return "career"
    if name in WEB_TESTS:
        return "web"
    if name in INTEGRATION_TESTS:
        return "integration"
    if name in CODE_TESTS:
        return "code"
    return "core"


def grouped_tests(tests_root: Path) -> dict[str, list[Path]]:
    grouped = {group: [] for group in GROUPS}
    for path in sorted(tests_root.glob("test_*.py")):
        grouped[classify(path)].append(path)
    return grouped


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in GROUPS:
        print(f"usage: {Path(sys.argv[0]).name} <{'|'.join(GROUPS)}>", file=sys.stderr)
        return 2

    group = sys.argv[1]
    selected = grouped_tests(Path("tests"))[group]
    if not selected:
        print(f"No tests assigned to group: {group}; skipped")
        return 0

    print(f"[{group}] {len(selected)} files")
    command = [sys.executable, "-m", "pytest", *map(str, selected), "-q"]
    with tempfile.TemporaryDirectory(prefix=f"smart-agent-{group}-") as temp_home:
        env = os.environ.copy()
        env["HOME"] = temp_home
        env["USERPROFILE"] = temp_home
        return subprocess.run(command, check=False, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
