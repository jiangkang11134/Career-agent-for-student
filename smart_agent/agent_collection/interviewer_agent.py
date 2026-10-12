"""Named plugin entry for Interviewer."""

from .contracts import AgentPluginSpec, lazy_factory

SPEC = AgentPluginSpec("interviewer", lazy_factory("smart_agent.agents.career", "InterviewerAgent"))
