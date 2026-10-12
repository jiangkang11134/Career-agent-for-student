"""Persisted gate decisions and controlled project deep-dive execution."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from smart_agent.career.models import (
    EvidenceMatch,
    JDRequirement,
    ProjectDescriptionDocument,
    ProjectDescriptionStatus,
)
from smart_agent.career.project_gate import ProjectSearchGate
from smart_agent.career.project_library import ProjectLibraryStore

DEEP_DIVE_SCHEMA_VERSION = "1.0"


def _execution_policy(
    *,
    candidate_project_ids: tuple[str, ...],
    project_versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "agent": "project_analyst",
        "mode": "readonly_subagent",
        "allowed_actions": (
            "list_project_files",
            "read_project_document",
            "analyze_project_structure",
            "draft_project_description",
        ),
        "denied_actions": (
            "edit_project_files",
            "write_personal_evidence",
            "confirm_project_description",
            "spawn_subagent",
            "run_unscoped_shell",
        ),
        "source_scope": {
            "project_ids": candidate_project_ids,
            "project_versions": project_versions,
        },
        "output_contract": {
            "allowed_record_status": "needs_user_confirmation",
            "requires_source_refs": True,
            "requires_user_confirmation_before_personal_evidence": True,
        },
    }


def _validate_result_contract(result: dict[str, Any]) -> None:
    description = result.get("project_description")
    if isinstance(description, dict) and description.get("status") == "user_confirmed":
        raise ValueError("Deep-dive analyzer cannot confirm project evidence; user confirmation is required")
    if result.get("personal_evidence") or result.get("confirmed_evidence"):
        raise ValueError("Deep-dive analyzer cannot write confirmed personal evidence")


def _description_preference(project: ProjectDescriptionDocument) -> tuple[int, int, int, float]:
    status_rank = {
        ProjectDescriptionStatus.USER_CONFIRMED: 3,
        ProjectDescriptionStatus.NEEDS_USER_CONFIRMATION: 2,
        ProjectDescriptionStatus.DRAFT: 1,
        ProjectDescriptionStatus.INVALIDATED: 0,
    }.get(project.status, 0)
    populated_fields = sum(
        bool(getattr(project, field_name))
        for field_name in (
            "goals",
            "role",
            "responsibilities",
            "core_flow",
            "architecture",
            "technology_choices",
            "core_modules",
            "technologies",
            "technical_tags",
            "implemented_features",
            "outcomes",
            "challenges",
        )
    )
    return (status_rank, int(bool(project.evidence_id)), populated_fields, project.updated_at)


@dataclass(frozen=True, slots=True)
class ProjectDeepDiveDecision:
    decision_id: str
    requirement_id: str
    requirement_text: str
    action: str
    confidence: float
    threshold: float
    reason: str
    candidate_project_ids: tuple[str, ...]
    candidate_projects: tuple[dict[str, Any], ...]
    project_versions: dict[str, str]
    intent: str
    status: str
    budget_cost: int
    scorer: str = ""
    execution_policy: dict[str, Any] | None = None
    degraded: bool = False
    degradation_reason: str = ""
    result_run_id: str = ""
    error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    schema_version: str = DEEP_DIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"planned", "skipped", "running", "completed", "failed"}:
            raise ValueError(f"Unsupported deep-dive status: {self.status}")
        if self.budget_cost not in {0, 1}:
            raise ValueError("budget_cost must be zero or one")
        if self.schema_version != DEEP_DIVE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported deep-dive schema: {self.schema_version}")
        object.__setattr__(
            self,
            "execution_policy",
            dict(
                self.execution_policy
                or _execution_policy(
                    candidate_project_ids=self.candidate_project_ids,
                    project_versions=self.project_versions,
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProjectDeepDiveDecision":
        payload = dict(value)
        payload["candidate_project_ids"] = tuple(payload.get("candidate_project_ids") or ())
        payload["candidate_projects"] = tuple(dict(item) for item in payload.get("candidate_projects") or ())
        payload["project_versions"] = dict(payload.get("project_versions") or {})
        payload["execution_policy"] = dict(payload.get("execution_policy") or {})
        return cls(**payload)


class ProjectDeepDiveStore:
    _ROOT_LOCKS: dict[str, threading.RLock] = {}
    _ROOT_LOCKS_GUARD = threading.Lock()

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        key = str(self.root).casefold()
        with self._ROOT_LOCKS_GUARD:
            self._lock = self._ROOT_LOCKS.setdefault(key, threading.RLock())

    def put(self, decision: ProjectDeepDiveDecision) -> ProjectDeepDiveDecision:
        path = self.root / f"{decision.decision_id}.json"
        with self._lock:
            if path.exists():
                raise ValueError(f"Deep-dive decision already exists: {decision.decision_id}")
            self._write(path, decision)
        return decision

    def get(self, decision_id: str) -> ProjectDeepDiveDecision:
        path = self.root / f"{str(decision_id or '').strip()}.json"
        with self._lock:
            if not path.is_file():
                raise KeyError(f"Unknown deep-dive decision: {decision_id}")
            return ProjectDeepDiveDecision.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def update(self, decision: ProjectDeepDiveDecision, *, expected_status: str) -> ProjectDeepDiveDecision:
        path = self.root / f"{decision.decision_id}.json"
        with self._lock:
            current = self.get(decision.decision_id)
            if current.status != expected_status:
                raise ValueError(
                    f"Deep-dive decision expected state {expected_status}, found {current.status}"
                )
            self._write(path, decision)
        return decision

    def list(self) -> tuple[ProjectDeepDiveDecision, ...]:
        with self._lock:
            values = [
                ProjectDeepDiveDecision.from_dict(json.loads(path.read_text(encoding="utf-8")))
                for path in self.root.glob("*.json")
            ]
        return tuple(sorted(values, key=lambda item: (item.created_at, item.decision_id)))

    def purge(self, project_id: str) -> int:
        selected = str(project_id or "").strip()
        if not selected:
            raise ValueError("project_id is required")
        removed = 0
        with self._lock:
            for path in tuple(self.root.glob("*.json")):
                try:
                    decision = ProjectDeepDiveDecision.from_dict(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if (
                    selected not in decision.candidate_project_ids
                    and selected not in decision.project_versions
                ):
                    continue
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                removed += 1
        return removed

    def find_reusable(
        self,
        *,
        requirement_id: str,
        requirement_text: str,
        candidate_project_ids: tuple[str, ...],
        project_versions: dict[str, str],
    ) -> ProjectDeepDiveDecision | None:
        candidates = tuple(candidate_project_ids)
        versions = dict(project_versions)
        for decision in reversed(self.list()):
            if decision.status not in {"planned", "running", "completed"}:
                continue
            if decision.action != "deep_scan_recommended":
                continue
            if (
                decision.requirement_id == requirement_id
                and decision.requirement_text == requirement_text
                and decision.candidate_project_ids == candidates
                and decision.project_versions == versions
            ):
                return decision
        return None

    @staticmethod
    def _write(path: Path, decision: ProjectDeepDiveDecision) -> None:
        fd, temporary = tempfile.mkstemp(prefix="deep-dive-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(decision.to_dict(), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                Path(temporary).unlink()
            except OSError:
                pass


class ProjectDeepDiveService:
    def __init__(
        self,
        *,
        project_store: ProjectLibraryStore,
        decision_store: ProjectDeepDiveStore,
        gate: ProjectSearchGate | None = None,
    ) -> None:
        self.project_store = project_store
        self.decision_store = decision_store
        self.gate = gate or ProjectSearchGate()

    def plan(
        self,
        requirements: tuple[JDRequirement, ...] | list[JDRequirement],
        *,
        matches: tuple[EvidenceMatch, ...] | list[EvidenceMatch],
        projects: tuple[ProjectDescriptionDocument, ...] | list[ProjectDescriptionDocument],
        budget: int,
    ) -> tuple[ProjectDeepDiveDecision, ...]:
        if budget < 0:
            raise ValueError("deep-dive budget cannot be negative")
        normalized_requirements = tuple(requirements)
        # Project descriptions are durable Career records and can outlive a
        # deleted library source. Never let those stale records become deep
        # scan candidates or trigger a lookup for a project that no longer
        # exists.
        active_projects = {
            source.project_id.casefold(): source
            for source in self.project_store.list()
        }
        normalized_by_project: dict[str, ProjectDescriptionDocument] = {}
        for project in projects:
            source = active_projects.get(project.project_id.casefold())
            if source is None:
                continue
            # Old Career records may preserve the pre-normalized casing of a
            # project ID. Always pass the library's canonical ID to the gate
            # and subsequent source-version lookup.
            normalized = (
                project if project.project_id == source.project_id
                else replace(project, project_id=source.project_id, content_hash="")
            )
            previous = normalized_by_project.get(source.project_id)
            if previous is None or _description_preference(normalized) > _description_preference(previous):
                normalized_by_project[source.project_id] = normalized
        projects = tuple(normalized_by_project.values())
        raw_decisions = self.gate.decide_batch(
            normalized_requirements,
            matches=matches,
            projects=projects,
            deep_scan_budget=len(normalized_requirements),
        )
        remaining = budget
        planned_projects: set[str] = set()
        results: list[ProjectDeepDiveDecision] = []
        now = time.time()
        for requirement, raw in zip(normalized_requirements, raw_decisions, strict=True):
            action = raw.action
            candidates = raw.candidate_project_ids
            candidate_details_by_id = {
                str(item.get("project_id") or ""): dict(item)
                for item in raw.candidate_projects
                if isinstance(item, dict)
            }
            new_candidates = tuple(item for item in candidates if item not in planned_projects)
            budget_cost = 0
            if action == "deep_scan_recommended" and new_candidates:
                candidates = new_candidates
            elif action == "deep_scan_recommended":
                action = "skip_already_planned"
                candidates = ()
            versions = {
                project_id: self.project_store.get(project_id, refresh=True).project_version
                for project_id in candidates
            }
            candidate_projects = (
                tuple(
                    candidate_details_by_id.get(project_id, {"project_id": project_id})
                    for project_id in candidates
                )
                if candidates
                else tuple(dict(item) for item in raw.candidate_projects)
            )
            if action == "deep_scan_recommended":
                reusable = self.decision_store.find_reusable(
                    requirement_id=requirement.requirement_id,
                    requirement_text=requirement.text,
                    candidate_project_ids=candidates,
                    project_versions=versions,
                )
                if reusable is not None:
                    planned_projects.update(candidates)
                    results.append(reusable)
                    continue
                if remaining:
                    budget_cost = 1
                    remaining -= 1
                    planned_projects.update(candidates)
                else:
                    action = "skip_budget_exhausted"
                    candidates = ()
                    versions = {}
                    candidate_projects = tuple(dict(item) for item in raw.candidate_projects)
            decision = ProjectDeepDiveDecision(
                decision_id=f"deep-dive-{uuid.uuid4().hex[:16]}",
                requirement_id=requirement.requirement_id,
                requirement_text=requirement.text,
                action=action,
                confidence=raw.confidence,
                threshold=raw.threshold,
                reason=raw.reason,
                candidate_project_ids=candidates,
                candidate_projects=candidate_projects,
                project_versions=versions,
                intent=f"JD requirement: {requirement.text}",
                status="planned" if action == "deep_scan_recommended" else "skipped",
                budget_cost=budget_cost,
                scorer=raw.scorer,
                execution_policy=_execution_policy(
                    candidate_project_ids=candidates,
                    project_versions=versions,
                ),
                degraded=raw.degraded,
                degradation_reason=raw.degradation_reason,
                created_at=now,
                updated_at=now,
            )
            self.decision_store.put(decision)
            results.append(decision)
        return tuple(results)

    def execute(
        self,
        decision_id: str,
        *,
        project_id: str,
        analyzer: Callable[[str, str], dict[str, Any]],
    ) -> tuple[ProjectDeepDiveDecision, dict[str, Any]]:
        running, selected = self._begin(decision_id, project_id)
        try:
            result = analyzer(selected, running.intent)
            return self._complete(running, result), result
        except Exception as exc:
            self._fail(running, exc)
            raise

    async def execute_async(
        self,
        decision_id: str,
        *,
        project_id: str,
        analyzer: Callable[[str, str], Awaitable[dict[str, Any]]],
    ) -> tuple[ProjectDeepDiveDecision, dict[str, Any]]:
        running, selected = self._begin(decision_id, project_id)
        try:
            result = await analyzer(selected, running.intent)
            return self._complete(running, result), result
        except Exception as exc:
            self._fail(running, exc)
            raise

    def _begin(self, decision_id: str, project_id: str) -> tuple[ProjectDeepDiveDecision, str]:
        decision = self.decision_store.get(decision_id)
        if decision.status != "planned" or decision.action != "deep_scan_recommended":
            raise ValueError("Deep-dive decision is not executable")
        selected = str(project_id or "").strip()
        if selected not in decision.candidate_project_ids:
            raise ValueError("Project is not a candidate for this deep-dive decision")
        source = self.project_store.get(selected, refresh=True)
        if source.project_version != decision.project_versions.get(selected):
            raise ValueError("Project version changed after the deep-dive decision")
        running = replace(decision, status="running", updated_at=time.time())
        self.decision_store.update(running, expected_status="planned")
        return running, selected

    def _complete(
        self,
        running: ProjectDeepDiveDecision,
        result: dict[str, Any],
    ) -> ProjectDeepDiveDecision:
        _validate_result_contract(result)
        run_id = str(
            result.get("analysis_run", {}).get("run_id")
            or result.get("analysis", {}).get("analysis_run_id")
            or ""
        )
        completed = replace(running, status="completed", result_run_id=run_id, updated_at=time.time())
        return self.decision_store.update(completed, expected_status="running")

    def _fail(self, running: ProjectDeepDiveDecision, error: Exception) -> ProjectDeepDiveDecision:
        failed = replace(running, status="failed", error=str(error), updated_at=time.time())
        return self.decision_store.update(failed, expected_status="running")


__all__ = [
    "DEEP_DIVE_SCHEMA_VERSION",
    "ProjectDeepDiveDecision",
    "ProjectDeepDiveService",
    "ProjectDeepDiveStore",
]
