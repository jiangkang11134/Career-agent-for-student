"""Fixed domain binding for a Smart Agent session."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from smart_agent.domains.registry import AgentDomain


class DomainSelectionError(ValueError):
    """Raised when a domain is missing, automatic, or conflicts with a session."""


def require_explicit_domain(value: AgentDomain | str | None) -> AgentDomain:
    """Resolve a user-selected domain and reject automatic or missing values."""
    if isinstance(value, AgentDomain):
        domain = value
    else:
        normalized = str(value or "").strip().lower()
        try:
            domain = AgentDomain(normalized)
        except ValueError as exc:
            raise DomainSelectionError("Domain must be explicitly selected as 'code' or 'career'.") from exc
    if domain is AgentDomain.AUTO:
        raise DomainSelectionError("Automatic domain routing is disabled; select 'code' or 'career'.")
    return domain


@dataclass(frozen=True)
class DomainSession:
    """The immutable domain binding and current capability stage for one session."""

    session_id: str
    domain: AgentDomain
    stage: str = "default"
    created_at: float = 0.0

    @classmethod
    def create(
        cls,
        domain: AgentDomain | str,
        *,
        session_id: str | None = None,
        stage: str = "default",
        created_at: float | None = None,
    ) -> "DomainSession":
        return cls(
            session_id=session_id or uuid.uuid4().hex[:12],
            domain=require_explicit_domain(domain),
            stage=str(stage or "default").strip().lower(),
            created_at=time.time() if created_at is None else float(created_at),
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        session_id: str,
        created_at: float,
    ) -> "DomainSession":
        payload = data if isinstance(data, dict) else {}
        # Sessions created before domain binding existed were Code sessions.
        domain = payload.get("domain") or AgentDomain.CODE.value
        return cls.create(
            domain,
            session_id=session_id,
            stage=str(payload.get("stage") or "default"),
            created_at=float(payload.get("created_at") or created_at),
        )

    def assert_domain(self, requested: AgentDomain | str | None) -> None:
        domain = require_explicit_domain(requested)
        if domain is not self.domain:
            raise DomainSelectionError(
                f"Session '{self.session_id}' is bound to '{self.domain.value}', "
                f"not '{domain.value}'. Create a new session to switch domains."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "domain": self.domain.value,
            "stage": self.stage,
            "created_at": self.created_at,
        }
