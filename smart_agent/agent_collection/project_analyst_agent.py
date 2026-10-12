"""Compatibility plugin entry for Code Agent project-analysis mode."""

from .contracts import AgentPluginSpec, lazy_factory

SPEC = AgentPluginSpec(
    "project_analyst",
    lazy_factory("smart_agent.agents.code_agent", "CodeAgent"),
)
