"""Registry and default plugin set for all domain agents."""

from collections.abc import Iterable
from .application_agent import SPEC as APPLICATION_SPEC
from .career_agent import SPEC as CAREER_SPEC
from .code_agent import SPEC as CODE_AGENT_SPEC
from .contracts import lazy_factory
from .contracts import AgentPlugin, AgentPluginSpec
from .interviewer_agent import SPEC as INTERVIEWER_SPEC
from .resume_verification_agent import SPEC as RESUME_VERIFICATION_SPEC
from .root_orchestrator_agent import SPEC as ROOT_ORCHESTRATOR_SPEC


class AgentRegistry:
    def __init__(self, specs: Iterable[AgentPluginSpec] = ()) -> None:
        self._specs: dict[str, AgentPluginSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: AgentPluginSpec, *, replace: bool = False) -> None:
        if spec.agent_id in self._specs and not replace:
            raise ValueError(f"agent already registered: {spec.agent_id}")
        self._specs[spec.agent_id] = spec

    def unregister(self, agent_id: str) -> None:
        self._specs.pop(agent_id, None)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def create(self, agent_id: str, *args, **kwargs) -> AgentPlugin:
        try:
            return self._specs[agent_id].factory(*args, **kwargs)
        except KeyError as exc:
            raise KeyError(f"unknown agent: {agent_id}") from exc


def build_default_registry() -> AgentRegistry:
    """Return the legacy agent set without importing implementations eagerly."""
    registry = AgentRegistry(
        (
            CODE_AGENT_SPEC,
            INTERVIEWER_SPEC,
            APPLICATION_SPEC,
            CAREER_SPEC,
            RESUME_VERIFICATION_SPEC,
            ROOT_ORCHESTRATOR_SPEC,
        )
    )
    # Compatibility IDs resolve to the unified Code Agent; callers should use
    # ``code`` with an explicit mode for new integrations.
    code_factory = lazy_factory("smart_agent.agents.code_agent", "CodeAgent")
    registry.register(AgentPluginSpec("code_review", code_factory))
    registry.register(AgentPluginSpec("project_analysis", code_factory))
    registry.register(AgentPluginSpec("project_analyst", code_factory))
    return registry
