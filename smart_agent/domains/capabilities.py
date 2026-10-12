"""Domain and workflow-stage capability assembly."""

from __future__ import annotations

from dataclasses import dataclass

from smart_agent.domains.registry import AgentDomain, get_domain_module
from smart_agent.domains.runtime import require_explicit_domain

CORE_TOOL_NAMES = frozenset({
    "ask_user",
    "todo_write",
    "complete_task",
    "web_fetch",
    "web_search",
    "load_skill",
})


@dataclass(frozen=True)
class CapabilitySet:
    """Immutable capabilities visible to one domain at one workflow stage."""

    domain: AgentDomain
    stage: str
    system_prompt: str
    memory_namespace: str
    skill_agent_type: str
    tool_names: frozenset[str]


class CapabilityAssembler:
    """Combine core, domain, and stage capabilities deterministically."""

    def assemble(self, domain: AgentDomain | str, stage: str = "default") -> CapabilitySet:
        module = get_domain_module(require_explicit_domain(domain))
        normalized_stage = str(stage or "default").strip().lower()
        if normalized_stage not in module.stage_tool_names:
            raise ValueError(
                f"Unknown stage '{normalized_stage}' for domain '{module.name.value}'."
            )
        return CapabilitySet(
            domain=module.name,
            stage=normalized_stage,
            system_prompt=module.system_prompt,
            memory_namespace=module.memory_namespace,
            skill_agent_type=module.skill_agent_type,
            tool_names=frozenset(
                CORE_TOOL_NAMES
                | module.tool_names
                | module.stage_tool_names[normalized_stage]
            ),
        )


def assemble_capabilities(
    domain: AgentDomain | str,
    stage: str = "default",
) -> CapabilitySet:
    return CapabilityAssembler().assemble(domain, stage)
