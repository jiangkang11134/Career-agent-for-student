import json

import smart_agent.config as config
from smart_agent.web.app import _validate_main_agent_connection


def test_main_agent_updates_use_runtime_environment_keys():
    runtime = {
        "model": "deepseek-chat",
        "providerOverride": "anthropic",
        "baseUrl": "https://old.example",
        "authToken": "old-token",
        "apiKey": "",
    }

    updates = _validate_main_agent_connection(
        {"base_url": "https://new.example/v1", "api_key": "new-key"},
        runtime,
    )

    assert updates == {
        "ANTHROPIC_BASE_URL": "https://new.example/v1",
        "ANTHROPIC_API_KEY": "new-key",
        "ANTHROPIC_AUTH_TOKEN": "",
    }


def test_saved_main_agent_connection_overrides_global_settings(monkeypatch, tmp_path):
    settings_path = tmp_path / "settings.json"
    claude_path = tmp_path / "claude.json"
    settings_path.write_text(
        json.dumps({"baseUrl": "https://saved.example/v1", "apiKey": "saved-key"}),
        encoding="utf-8",
    )
    claude_path.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_BASE_URL": "https://global.example/anthropic",
                    "ANTHROPIC_AUTH_TOKEN": "global-token",
                    "ANTHROPIC_MODEL": "deepseek-chat",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "SMART_AGENT_SETTINGS_PATH", settings_path)
    monkeypatch.setattr(config, "CLAUDE_SETTINGS_PATH", claude_path)
    for name in (
        "SMART_AGENT_MODEL",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)

    runtime = config.load_runtime_config(tmp_path)

    assert runtime["baseUrl"] == "https://saved.example/v1"
    assert runtime["apiKey"] == "saved-key"
    assert runtime["authToken"] is None
