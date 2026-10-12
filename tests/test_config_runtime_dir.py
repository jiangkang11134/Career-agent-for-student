import json
import os
import subprocess
import sys
from pathlib import Path


def _read_config_paths(env):
    code = (
        "import json; "
        "from smart_agent import config; "
        "print(json.dumps({'runtime_dir': str(config.SMART_AGENT_DIR), 'project_root': str(config.PROJECT_ROOT)}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(proc.stdout)


def test_runtime_dir_defaults_to_project_runtime():
    env = os.environ.copy()
    env.pop("SMART_AGENT_RUNTIME_DIR", None)

    data = _read_config_paths(env)

    assert Path(data["runtime_dir"]) == Path(data["project_root"]) / ".runtime"


def test_runtime_dir_can_be_overridden(tmp_path):
    env = os.environ.copy()
    env["SMART_AGENT_RUNTIME_DIR"] = str(tmp_path / "runtime")

    data = _read_config_paths(env)

    assert Path(data["runtime_dir"]) == (tmp_path / "runtime").resolve()
