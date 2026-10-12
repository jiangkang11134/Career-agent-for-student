from smart_agent.model_registry import Provider, build_provider_config, detect_provider
from smart_agent.web.app import (
    _apply_frontend_model_override,
    _is_unsupported_model_error,
    _run_agent_turn_with_model_fallback,
    _sanitize_user_facing_text,
    _user_visible_progress_message,
)


def test_frontend_model_override_wins_over_backend_anthropic_config(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "old-backend-key")
    monkeypatch.delenv("CUSTOM_API_BASE_URL", raising=False)
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)

    runtime = _apply_frontend_model_override(
        {"model": "claude-sonnet-4-20250514", "baseUrl": "https://api.anthropic.com", "apiKey": "old-runtime-key"},
        model="claude-sonnet-4-20250514",
        base_url="https://proxy.example.com/v1/",
        api_key="frontend-key",
    )

    assert runtime["providerOverride"] == "custom"
    assert runtime["customBaseUrl"] == "https://proxy.example.com/v1"
    assert runtime["customApiKey"] == "frontend-key"
    assert detect_provider(runtime["model"], runtime) is Provider.CUSTOM

    config = build_provider_config(runtime["model"], runtime)
    assert config.provider is Provider.CUSTOM
    assert config.base_url == "https://proxy.example.com/v1"
    assert config.api_key == "frontend-key"


def test_frontend_model_override_uses_anthropic_when_anthropic_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "old-backend-key")

    runtime = _apply_frontend_model_override(
        {"model": "claude-sonnet-4-20250514", "apiKey": "old-runtime-key"},
        base_url="https://api.anthropic.com",
        api_key="frontend-key",
    )

    assert runtime["providerOverride"] == "anthropic"
    assert detect_provider(runtime["model"], runtime) is Provider.ANTHROPIC

    config = build_provider_config(runtime["model"], runtime)
    assert config.provider is Provider.ANTHROPIC
    assert config.base_url == "https://api.anthropic.com"
    assert config.api_key == "frontend-key"


def test_frontend_model_override_api_key_only_wins_over_backend_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "old-backend-key")

    runtime = _apply_frontend_model_override(
        {"model": "claude-sonnet-4-20250514", "baseUrl": "https://api.anthropic.com"},
        api_key="frontend-key",
    )

    assert runtime["providerOverride"] == "anthropic"
    config = build_provider_config(runtime["model"], runtime)
    assert config.provider is Provider.ANTHROPIC
    assert config.api_key == "frontend-key"


def test_frontend_model_override_api_key_only_keeps_existing_custom_endpoint(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CUSTOM_API_BASE_URL", raising=False)
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)

    runtime = _apply_frontend_model_override(
        {
            "model": "gpt-5.6-sol",
            "customBaseUrl": "https://backend.example.com/v1",
            "customApiKey": "old-backend-key",
        },
        api_key="frontend-key",
    )

    assert runtime["providerOverride"] == "custom"
    config = build_provider_config(runtime["model"], runtime)
    assert config.provider is Provider.CUSTOM
    assert config.base_url == "https://backend.example.com/v1"
    assert config.api_key == "frontend-key"


def test_model_fallback_retries_only_for_unsupported_model(monkeypatch):
    calls = []

    class DummyModel:
        def __init__(self, name):
            self.name = name

    def fake_turn(*, model, tools, runtime, **kwargs):
        calls.append(model.name)
        if model.name == "deepseek-chat":
            raise RuntimeError('Model "deepseek-chat" is not supported by any configured account in this group')
        return [{"role": "assistant", "content": "ok"}]

    monkeypatch.setattr("smart_agent.web.app.run_agent_turn", fake_turn)
    monkeypatch.setattr(
        "smart_agent.web.app.create_model_adapter",
        lambda *, model, tools, runtime: DummyModel(model),
    )
    runtime = {"model": "deepseek-chat"}
    result = _run_agent_turn_with_model_fallback(
        model=DummyModel("deepseek-chat"),
        tools=object(),
        fallback_model_name="gpt-5.6-sol",
        current_model_name="deepseek-chat",
        runtime=runtime,
        messages=[],
        cwd=".",
        max_steps=1,
    )

    assert result == [{"role": "assistant", "content": "ok"}]
    assert calls == ["deepseek-chat", "gpt-5.6-sol"]
    assert runtime["model"] == "gpt-5.6-sol"
    assert _is_unsupported_model_error(RuntimeError("temporary network timeout")) is False


def test_internal_runtime_messages_are_not_user_visible():
    raw = "Runtime phase: explore. inspect, decompose, and anchor the task before committing (verification=light, remaining_steps=39)."
    assert _sanitize_user_facing_text(raw) == ""
    assert _user_visible_progress_message(raw) == "正在分析任务结构"
    assert _user_visible_progress_message("Verification guard: final answer withheld") == "正在补充验证并整理结果"
