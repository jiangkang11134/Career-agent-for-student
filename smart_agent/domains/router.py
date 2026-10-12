"""Deterministic first-stage router for Smart Agent domain selection."""

from __future__ import annotations

import os
import re

from smart_agent.domains.registry import AgentDomain

_CAREER_TERMS = (
    "jd", "job description", "resume", "cv", "interview", "job application",
    "求职", "招聘", "岗位", "职位", "简历", "面试", "投递", "网申", "职业规划",
)
_CODE_TERMS = (
    "code", "repository", "function", "class", "api", "bug", "debug", "refactor", "test",
    "代码", "仓库", "函数", "接口", "报错", "调试", "重构", "测试", "实现", "开发",
)


def route_domain(user_input: str, configured: str | None = None) -> AgentDomain:
    """Select a domain, honoring an explicit configuration before auto routing."""
    requested = (configured or os.environ.get("SMART_AGENT_DOMAIN", "auto")).strip().lower()
    if requested in {AgentDomain.CODE.value, AgentDomain.CAREER.value}:
        return AgentDomain(requested)

    text = re.sub(r"\s+", " ", (user_input or "").lower())
    career_score = sum(1 for term in _CAREER_TERMS if term in text)
    code_score = sum(1 for term in _CODE_TERMS if term in text)
    return AgentDomain.CAREER if career_score > code_score else AgentDomain.CODE
