"""Named plugin entry for the Career domain agent."""

from .contracts import AgentPluginSpec, lazy_factory

SPEC = AgentPluginSpec("career", lazy_factory("smart_agent.career.agent", "CareerAgent"))
