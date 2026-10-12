"""Named plugin entry for Code Review."""

from .contracts import AgentPluginSpec, lazy_factory

SPEC = AgentPluginSpec("code_review", lazy_factory("smart_agent.agents.code_agent", "CodeAgent"))
