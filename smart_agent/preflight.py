"""Low-latency request classification before model/tool initialization."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum

class PreflightIntent(str, Enum):
    GENERAL = "general"
    PROJECT_INFO = "project_info"
    TASK = "task"


@dataclass(frozen=True, slots=True)
class PreflightDecision:
    intent: PreflightIntent
    complexity: str
    needs_tools: bool
    confidence: float
    reason: str


_WORK_PATTERN = re.compile(
    r"(?i)(?:write|create|implement|modify|edit|fix|debug|refactor|run|test|review|"
    r"upload|save|delete|rename|move|commit|代码|文件|项目|仓库|接口|测试|运行|修改|"
    r"修复|调试|重构|部署|分析|简历|面试|岗位|职位|求职)"
)
_ARITHMETIC_PATTERN = re.compile(r"^[\s\d()+\-*/%.=？?等于多少几加减乘除]+$")
_GREETING_PATTERN = re.compile(r"^(你好|您好|嗨|hello|hi|谢谢|好的|收到|在吗)[!！。.?？\s]*$", re.I)
_GENERAL_INFO_PATTERN = re.compile(
    r"(?i)(?:介绍(?:一下|下)?|什么是|是做什么的|做什么|用途|功能|能力|如何使用|怎么用|"
    r"explain|describe|introduce|what is|tell me about|who are you)"
)
_PROJECT_INFO_PATTERN = re.compile(
    r"(?i)(?:smart\s*agent|本项目|这个项目|当前项目|项目架构|项目结构|项目功能|代码是怎么|源码|仓库)"
)


def classify_input(text: str) -> PreflightDecision:
    """Classify only high-confidence cheap cases; remain conservative otherwise."""
    value = " ".join(str(text or "").strip().split())
    if not value:
        return PreflightDecision(PreflightIntent.GENERAL, "simple", False, 1.0, "empty")
    if (
        len(value) <= 160
        and _PROJECT_INFO_PATTERN.search(value)
        and re.search(r"介绍|说明|概述|架构|结构|功能|能力|怎么工作|如何工作|是什么|explain|describe|what is", value, re.I)
        and not _WORK_PATTERN.search(value.replace("Smart Agent", "").replace("smart agent", ""))
    ):
        return PreflightDecision(PreflightIntent.PROJECT_INFO, "simple", True, 0.97, "project_read_only_inspection")
    if len(value) <= 80 and _ARITHMETIC_PATTERN.fullmatch(value) and re.search(r"\d", value) and re.search(r"[+\-*/%=]", value):
        return PreflightDecision(PreflightIntent.GENERAL, "simple", False, 0.99, "arithmetic")
    if len(value) <= 32 and _GREETING_PATTERN.fullmatch(value):
        return PreflightDecision(PreflightIntent.GENERAL, "simple", False, 0.98, "small_talk")
    if len(value) <= 120 and not _WORK_PATTERN.search(value) and _GENERAL_INFO_PATTERN.search(value):
        return PreflightDecision(PreflightIntent.GENERAL, "simple", False, 0.96, "general_information")
    if len(value) <= 48 and not _WORK_PATTERN.search(value) and ("?" in value or "？" in value):
        return PreflightDecision(PreflightIntent.GENERAL, "simple", False, 0.88, "short_question")
    return PreflightDecision(PreflightIntent.TASK, "moderate", True, 0.55, "requires_contextual_routing")


def classify_with_optional_model(
    text: str,
    *,
    runtime: dict | None = None,
) -> PreflightDecision:
    """Use a configured small model only for locally ambiguous requests.

    The classifier receives no tools and has a strict output budget. Any
    timeout, provider error, or malformed response falls back to the local
    conservative decision, which never grants tools based on model text.
    """
    local = classify_input(text)
    if local.confidence >= 0.85:
        return local
    config = dict(runtime or {})
    model_name = str(
        config.get("preflightModel")
        or os.environ.get("SMART_AGENT_PREFLIGHT_MODEL")
        or ""
    ).strip()
    if not model_name:
        return local
    tools = None
    try:
        from smart_agent.model_registry import create_model_adapter
        from smart_agent.tooling import ToolRegistry

        tools = ToolRegistry([])
        classifier_runtime = {
            **config,
            "model": model_name,
            "maxOutputTokens": 80,
            "preflight": True,
        }
        adapter = create_model_adapter(model=model_name, tools=tools, runtime=classifier_runtime)
        step = adapter.next([
            {
                "role": "system",
                "content": (
                    "Classify the user's request. Return JSON only: "
                    '{"intent":"general|task|project_info","complexity":"simple|moderate|complex",'
                    '"needs_tools":true|false,"confidence":0..1}. '
                    "Classify the request for routing only; personal facts remain in the normal agent conversation."
                ),
            },
            {"role": "user", "content": str(text or "")[:2000]},
        ])
        raw = str(getattr(step, "content", "") or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(match.group(0) if match else raw)
        intent = PreflightIntent(str(payload.get("intent") or "").strip().lower())
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
        needs_tools = bool(payload.get("needs_tools", intent is PreflightIntent.TASK))
        if intent is not PreflightIntent.TASK:
            needs_tools = False
        if intent is PreflightIntent.PROJECT_INFO:
            needs_tools = True
        return PreflightDecision(
            intent=intent,
            complexity=str(payload.get("complexity") or "moderate"),
            needs_tools=needs_tools,
            confidence=confidence,
            reason="small_model",
        )
    except Exception:
        return local
    finally:
        try:
            if tools is not None:
                tools.dispose()
        except Exception:
            pass


__all__ = ["PreflightDecision", "PreflightIntent", "classify_input", "classify_with_optional_model"]
