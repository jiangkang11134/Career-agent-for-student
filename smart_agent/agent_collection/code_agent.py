"""Named plugin entry for the unified Code Agent."""

from .contracts import AgentPluginSpec, lazy_factory

SPEC = AgentPluginSpec("code", lazy_factory("smart_agent.agents.code_agent", "CodeAgent"))
