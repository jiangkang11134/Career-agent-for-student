"""Small, stable contract for replaceable agents."""

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class AgentPlugin(Protocol):
    """Marker for an agent implementation created by the registry."""


@dataclass(frozen=True)
class AgentPluginSpec:
    agent_id: str
    factory: Callable[..., Any]
    version: str = "1"


def lazy_factory(module: str, attribute: str) -> Callable[..., Any]:
    def create(*args: Any, **kwargs: Any) -> Any:
        import importlib

        cls = getattr(importlib.import_module(module), attribute)
        return cls(*args, **kwargs)

    return create
