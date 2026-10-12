"""Domain modules exposed by the Smart Agent runtime."""

from smart_agent.domains.capabilities import (
    CapabilityAssembler,
    CapabilitySet,
    assemble_capabilities,
)
from smart_agent.domains.registry import (
    AgentDomain,
    DomainModule,
    get_domain_module,
    list_domain_modules,
)
from smart_agent.domains.router import route_domain
from smart_agent.domains.runtime import DomainSelectionError, DomainSession, require_explicit_domain

__all__ = [
    "AgentDomain",
    "CapabilityAssembler",
    "CapabilitySet",
    "DomainModule",
    "DomainSelectionError",
    "DomainSession",
    "assemble_capabilities",
    "get_domain_module",
    "list_domain_modules",
    "require_explicit_domain",
    "route_domain",
]
