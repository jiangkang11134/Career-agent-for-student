"""Named plugin entry for the root orchestrator."""

from .contracts import AgentPluginSpec, lazy_factory

SPEC = AgentPluginSpec(
    "root_orchestrator",
    lazy_factory("smart_agent.agents.orchestrator", "RootOrchestrator"),
)
