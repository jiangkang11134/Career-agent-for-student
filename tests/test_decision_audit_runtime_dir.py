import os
import subprocess
import sys
from pathlib import Path


def test_decision_audit_uses_configured_runtime_dir(tmp_path):
    env = os.environ.copy()
    env["SMART_AGENT_RUNTIME_DIR"] = str(tmp_path / "runtime")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from smart_agent.decision_audit import get_auditor; print(get_auditor().log_dir)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert Path(proc.stdout.strip()) == (tmp_path / "runtime" / "audit").resolve()
