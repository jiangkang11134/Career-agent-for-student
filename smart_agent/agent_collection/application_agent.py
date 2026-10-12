"""Named plugin entry for Career Application."""

from .contracts import AgentPluginSpec, lazy_factory

SPEC = AgentPluginSpec("application", lazy_factory("smart_agent.agents.career", "ApplicationAgent"))
