"""Pluggable agent collection.

The collection is intentionally lazy: importing the registry does not import
optional model or UI dependencies. Existing ``smart_agent.agents`` exports
remain the compatibility surface.
"""

from .contracts import AgentPlugin, AgentPluginSpec
from .registry import AgentRegistry, build_default_registry

__all__ = ["AgentPlugin", "AgentPluginSpec", "AgentRegistry", "build_default_registry"]
