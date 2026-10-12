"""Named plugin entry for resume verification."""

from .contracts import AgentPluginSpec, lazy_factory

SPEC = AgentPluginSpec(
    "resume_verification",
    lazy_factory("smart_agent.career.verification", "ResumeVerificationAgent"),
)
