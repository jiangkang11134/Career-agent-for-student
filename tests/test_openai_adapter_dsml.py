from smart_agent.openai_adapter import _parse_textual_tool_calls


def test_dsml_bash_maps_to_run_command_and_is_removed():
    content = (
        "<|DSML|><|invoke name=\"bash\">"
        "<|parameter name=\"command\" string=\"true\">"
        "find . -maxdepth 2 -iname \"*.md\" | head -50"
        "<|/parameter|><|/invoke|><|/DSML|>"
    )
    text, calls = _parse_textual_tool_calls(content, {"run_command"})
    assert text == ""
    assert calls == [{
        "id": "textual-1",
        "toolName": "run_command",
        "input": {"command": 'find . -maxdepth 2 -iname "*.md" | head -50'},
    }]


def test_dsml_preserves_surrounding_text():
    content = "先检查目录。<|invoke name='bash'><|parameter name='command'>pwd<|/parameter|><|/invoke|>完成。"
    text, calls = _parse_textual_tool_calls(content, {"run_command"})
    assert text == "先检查目录。完成。"
    assert calls[0]["input"] == {"command": "pwd"}


def test_dsml_unknown_tool_is_not_converted():
    content = '<|invoke name="delete_everything"><|parameter name="path">/tmp<|/parameter|><|/invoke|>'
    text, calls = _parse_textual_tool_calls(content, {"run_command"})
    assert calls == []
    assert text == ""


def test_dsml_alias_requires_registered_target_tool():
    content = '<|invoke name="shell"><|parameter name="command" string="true">ls<|/parameter|><|/invoke|>'
    text, calls = _parse_textual_tool_calls(content, set())
    assert text == ""
    assert calls == []


def test_plain_text_is_unchanged():
    content = "普通回复，不包含工具协议。"
    assert _parse_textual_tool_calls(content, {"run_command"}) == (content, [])
