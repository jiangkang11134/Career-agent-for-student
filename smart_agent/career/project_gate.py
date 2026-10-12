"""Cost-aware gate for deciding whether a JD requirement merits project reading."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Protocol

from smart_agent.career.models import (
    EvidenceMatch,
    JDRequirement,
    ProjectDescriptionDocument,
    ProjectDescriptionStatus,
    RequirementCategory,
    SupportLevel,
)

_CHINESE_STOP_BIGRAMS = frozenset(
    {
        "具备",
        "优先",
        "熟悉",
        "掌握",
        "了解",
        "能力",
        "基础",
        "经验",
        "相关",
        "负责",
        "岗位",
        "要求",
    }
)
_CHINESE_STOP_PHRASES = (
    "具备",
    "优先",
    "熟悉",
    "掌握",
    "了解",
    "能力",
    "基础",
    "经验",
    "相关",
    "岗位",
    "要求",
    "者",
)
_ENGLISH_STOP_WORDS = frozenset(
    {
        "and",
        "or",
        "with",
        "preferred",
        "required",
        "experience",
        "knowledge",
        "skill",
        "skills",
        "basic",
        "familiar",
        "proficient",
        "understanding",
        "candidate",
        "plus",
        "application",
        "app",
        "development",
    }
)

_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "backend_service": (
        "backend",
        "server",
        "service",
        "api",
        "fastapi",
        "flask",
        "django",
        "后端",
        "服务端",
        "接口",
    ),
    "frontend": (
        "frontend",
        "react",
        "vue",
        "tsx",
        "web ui",
        "前端",
        "页面",
        "交互",
    ),
    "llm_application": (
        "llm",
        "large language model",
        "大模型",
        "生成式ai",
        "生成式 ai",
        "rag",
        "retrieval augmented generation",
        "检索增强",
    ),
    "retrieval": (
        "retrieval",
        "search",
        "bm25",
        "embedding",
        "vector",
        "rerank",
        "rag",
        "检索",
        "向量",
        "召回",
        "重排",
    ),
    "deep_learning": (
        "deep learning",
        "pytorch",
        "tensorflow",
        "torch",
        "cnn",
        "transformer",
        "深度学习",
        "神经网络",
        "模型推理",
    ),
    "data_pipeline": (
        "pipeline",
        "etl",
        "spark",
        "airflow",
        "数据处理",
        "数据清洗",
        "数据管道",
    ),
    "database": (
        "database",
        "sql",
        "mysql",
        "postgres",
        "redis",
        "mongodb",
        "数据库",
        "缓存",
    ),
    "deployment": (
        "docker",
        "kubernetes",
        "k8s",
        "ci/cd",
        "deploy",
        "部署",
        "容器",
        "上线",
    ),
    "testing_quality": (
        "test",
        "pytest",
        "unit test",
        "coverage",
        "测试",
        "质量",
        "验证",
    ),
    "performance": (
        "performance",
        "latency",
        "throughput",
        "optimization",
        "性能",
        "延迟",
        "吞吐",
        "优化",
    ),
}

_WEAK_CONCEPTS = frozenset({"llm_application", "testing_quality", "performance"})

_FIELD_WEIGHTS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("technologies", 0.23, ("technologies", "technical_tags")),
    ("technical_choices", 0.17, ("technology_choices", "architecture")),
    ("implementation", 0.20, ("implemented_features", "core_modules", "core_flow")),
    ("ownership", 0.13, ("role", "responsibilities")),
    ("outcomes", 0.12, ("outcomes", "challenges")),
    ("intent", 0.15, ("title", "background", "goals", "problem_statement")),
)


def _concepts(text: str) -> set[str]:
    lowered = str(text or "").casefold()
    matched: set[str] = set()
    for concept, aliases in _CONCEPT_ALIASES.items():
        if any(alias.casefold() in lowered for alias in aliases):
            matched.add(concept)
    return matched


def _tokens(text: str) -> set[str]:
    lowered = str(text or "").casefold()
    values = {
        token
        for token in re.findall(r"[a-z][a-z0-9+#._/-]{1,}|\d+(?:\.\d+)?", lowered)
        if token not in _ENGLISH_STOP_WORDS
    }
    for run in re.findall(r"[\u4e00-\u9fff]+", lowered):
        cleaned = run
        for phrase in _CHINESE_STOP_PHRASES:
            cleaned = cleaned.replace(phrase, " ")
        for segment in cleaned.split():
            values.update(
                token
                for token in (segment[index : index + 2] for index in range(len(segment) - 1))
                if token not in _CHINESE_STOP_BIGRAMS
            )
    return values | _concepts(text)


def _field_text(document: ProjectDescriptionDocument, field_names: tuple[str, ...]) -> str:
    values: list[str] = []
    for field_name in field_names:
        value = getattr(document, field_name)
        if isinstance(value, str):
            values.append(value)
        else:
            values.extend(str(item or "") for item in value)
    return " ".join(values)


def _project_text(document: ProjectDescriptionDocument) -> str:
    return " ".join(_field_text(document, field_names) for _label, _weight, field_names in _FIELD_WEIGHTS)


def _overlap_score(query: set[str], document: set[str]) -> float:
    if not query or not document:
        return 0.0
    overlap = query & document
    if not overlap:
        return 0.0
    coverage = len(overlap) / len(query)
    precision = len(overlap) / min(len(document), max(4, len(query) * 4))
    return 0.78 * coverage + 0.22 * min(1.0, precision)


def _ranking_payload(ranking: "ProjectCandidateRanking") -> dict[str, Any]:
    return {
        "project_id": ranking.project_id,
        "score": ranking.score,
        "reason": ranking.reason,
        "matched_terms": list(ranking.matched_terms),
        "matched_fields": list(ranking.matched_fields),
    }


class ProjectRelevanceScorer(Protocol):
    name: str

    def rank(self, requirement: JDRequirement, project: ProjectDescriptionDocument) -> "ProjectCandidateRanking": ...

    def score(self, requirement: JDRequirement, project: ProjectDescriptionDocument) -> float: ...


@dataclass(frozen=True, slots=True)
class ProjectCandidateRanking:
    project_id: str
    score: float
    reason: str
    matched_terms: tuple[str, ...] = ()
    matched_fields: tuple[str, ...] = ()


class ProjectCandidateSelector(Protocol):
    name: str

    def select(
        self,
        requirements: tuple[JDRequirement, ...],
        projects: tuple[ProjectDescriptionDocument, ...],
    ) -> dict[str, tuple[ProjectCandidateRanking, ...]]: ...


class LLMProjectCandidateSelector:
    """Use one structured model call to choose project cards worth investigating."""

    def __init__(self, client: Any) -> None:
        if not callable(getattr(client, "complete_json", None)):
            raise ValueError("Project matcher client must expose complete_json(...)")
        self.client = client
        self.name = f"project-catalog-agent:{getattr(client, 'name', 'custom')}"

    def select(
        self,
        requirements: tuple[JDRequirement, ...],
        projects: tuple[ProjectDescriptionDocument, ...],
    ) -> dict[str, tuple[ProjectCandidateRanking, ...]]:
        active_projects = tuple(
            project for project in projects if project.status is not ProjectDescriptionStatus.INVALIDATED
        )
        if not requirements or not active_projects:
            return {requirement.requirement_id: () for requirement in requirements}
        response = self.client.complete_json(
            system_prompt=(
                "You are the Career Agent's project investigation planner. Read JD requirements and compact personal "
                "project cards, then choose which real projects are worth opening with a separate read-only Project "
                "Analyst. Reason semantically across Chinese and English; exact keyword overlap is not required. "
                "First rank projects against job_intent_text as one whole role signal; then choose candidates for "
                "individual requirements from that overall fit unless one requirement has an unambiguous different "
                "project fit. A RAG project should win RAG-focused roles; an Agent orchestration project should win "
                "roles about planning, context construction, tool orchestration, verification, stability, or reliability. "
                "A project card is only a search lead, not proof of personal ownership. Do not infer skills absent "
                "from every card. Return JSON with a requirements array. Include every requirement_id exactly once. "
                "Each item contains requirement_id and candidates. Each candidate contains project_id, score from 0 "
                "to 1 used only for ranking, reason, matched_terms, and matched_fields. Return an empty candidates "
                "array when opening a project is unlikely to change resume or interview evidence."
            ),
            user_payload={
                "job_intent_text": "\n".join(requirement.text for requirement in requirements),
                "requirements": [
                    {
                        "requirement_id": requirement.requirement_id,
                        "text": requirement.text,
                        "required": requirement.required,
                        "category": requirement.category.value,
                    }
                    for requirement in requirements
                ],
                "project_cards": [self._project_card(project) for project in active_projects],
            },
            max_output_tokens=4096,
        )
        return self._validate_response(
            response.value,
            requirement_ids={item.requirement_id for item in requirements},
            project_ids={item.project_id for item in active_projects},
        )

    @staticmethod
    def _project_card(project: ProjectDescriptionDocument) -> dict[str, Any]:
        return {
            "project_id": project.project_id,
            "title": project.title,
            "status": project.status.value,
            "goals": list(project.goals),
            "role": project.role,
            "responsibilities": list(project.responsibilities),
            "core_flow": list(project.core_flow),
            "architecture": project.architecture,
            "technology_choices": list(project.technology_choices),
            "technologies": list(project.technologies),
            "technical_tags": list(project.technical_tags),
            "implemented_features": list(project.implemented_features),
            "outcomes": list(project.outcomes),
            "unknowns": list(project.unknowns),
        }

    @staticmethod
    def _validate_response(
        value: dict[str, Any],
        *,
        requirement_ids: set[str],
        project_ids: set[str],
    ) -> dict[str, tuple[ProjectCandidateRanking, ...]]:
        rows = value.get("requirements")
        if not isinstance(rows, list):
            raise ValueError("Project matcher response requires a requirements array")
        selected: dict[str, tuple[ProjectCandidateRanking, ...]] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Project matcher requirement result must be an object")
            requirement_id = str(row.get("requirement_id") or "").strip()
            if requirement_id not in requirement_ids or requirement_id in selected:
                raise ValueError("Project matcher returned an unknown or duplicate requirement_id")
            raw_candidates = row.get("candidates")
            if not isinstance(raw_candidates, list):
                raise ValueError("Project matcher candidates must be an array")
            candidates: list[ProjectCandidateRanking] = []
            seen_projects: set[str] = set()
            for candidate in raw_candidates:
                if not isinstance(candidate, dict):
                    raise ValueError("Project matcher candidate must be an object")
                project_id = str(candidate.get("project_id") or "").strip()
                score = float(candidate.get("score", -1))
                reason = str(candidate.get("reason") or "").strip()
                if project_id not in project_ids or project_id in seen_projects:
                    raise ValueError("Project matcher returned an unknown or duplicate project_id")
                if not 0 <= score <= 1 or not reason:
                    raise ValueError("Project matcher candidate requires score in [0, 1] and reason")
                matched_terms = tuple(
                    str(item).strip()
                    for item in candidate.get("matched_terms", ())
                    if str(item or "").strip()
                )
                matched_fields = tuple(
                    str(item).strip()
                    for item in candidate.get("matched_fields", ())
                    if str(item or "").strip()
                )
                seen_projects.add(project_id)
                candidates.append(
                    ProjectCandidateRanking(
                        project_id,
                        round(score, 6),
                        reason,
                        tuple(dict.fromkeys(matched_terms)),
                        tuple(dict.fromkeys(matched_fields)),
                    )
                )
            selected[requirement_id] = tuple(sorted(candidates, key=lambda item: item.score, reverse=True))
        if set(selected) != requirement_ids:
            raise ValueError("Project matcher must return every requested requirement_id")
        return selected


class LexicalProjectRelevanceScorer:
    """Field-weighted offline scorer over already-produced project catalog cards."""

    name = "project-catalog-field-weighted-v3"

    def rank(self, requirement: JDRequirement, project: ProjectDescriptionDocument) -> ProjectCandidateRanking:
        query = _tokens(requirement.text)
        weighted_score = 0.0
        max_field_score = 0.0
        matched_terms: set[str] = set()
        matched_fields: list[str] = []
        for label, weight, field_names in _FIELD_WEIGHTS:
            text = _field_text(project, field_names)
            document = _tokens(text)
            field_score = _overlap_score(query, document)
            max_field_score = max(max_field_score, field_score)
            if field_score:
                matched_fields.append(label)
                matched_terms.update(sorted(query & document))
            weighted_score += weight * field_score

        whole_document = _tokens(_project_text(project))
        broad_score = _overlap_score(query, whole_document)
        query_concepts = _concepts(requirement.text)
        project_concepts = _concepts(_project_text(project))
        concept_overlap = query_concepts & project_concepts
        matched_terms.update(concept_overlap)
        concept_boost = min(0.08, 0.03 * len(concept_overlap - _WEAK_CONCEPTS) + 0.015 * len(concept_overlap & _WEAK_CONCEPTS))
        score = min(1.0, 0.46 * max_field_score + 0.28 * weighted_score + 0.16 * broad_score + concept_boost)
        technology_terms = _tokens(" ".join(project.technologies + project.technical_tags))
        if query & technology_terms:
            score = min(1.0, score + 0.06)
        score = round(max(0.0, score), 6)
        terms = tuple(sorted(matched_terms))
        fields = tuple(dict.fromkeys(matched_fields))
        reason = (
            f"Field-weighted project card match; fields={','.join(fields) or 'none'}; "
            f"terms={','.join(terms[:8]) or 'none'}."
        )
        return ProjectCandidateRanking(project.project_id, score, reason, terms, fields)

    def score(self, requirement: JDRequirement, project: ProjectDescriptionDocument) -> float:
        return self.rank(requirement, project).score


@dataclass(frozen=True, slots=True)
class ProjectSearchDecision:
    requirement_id: str
    action: str
    confidence: float
    threshold: float
    reason: str
    current_support: str
    candidate_project_ids: tuple[str, ...]
    scorer: str
    degraded: bool = False
    degradation_reason: str = ""
    candidate_projects: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict:
        return {
            "requirement_id": self.requirement_id,
            "action": self.action,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "reason": self.reason,
            "current_support": self.current_support,
            "candidate_project_ids": list(self.candidate_project_ids),
            "candidate_projects": [dict(item) for item in self.candidate_projects],
            "scorer": self.scorer,
            "degraded": self.degraded,
            "degradation_reason": self.degradation_reason,
        }


class ProjectSearchGate:
    def __init__(
        self,
        *,
        scorer: ProjectRelevanceScorer | None = None,
        selector: ProjectCandidateSelector | None = None,
        high_threshold: float = 0.65,
        medium_threshold: float = 0.40,
        sufficient_evidence_score: float = 0.72,
        max_candidate_projects: int = 3,
    ) -> None:
        if not 0 <= medium_threshold <= high_threshold <= 1:
            raise ValueError("Invalid project search thresholds")
        if not 0 <= sufficient_evidence_score <= 1 or max_candidate_projects < 1:
            raise ValueError("Invalid project search gate configuration")
        self.scorer = scorer or LexicalProjectRelevanceScorer()
        self.selector = selector
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.sufficient_evidence_score = sufficient_evidence_score
        self.max_candidate_projects = max_candidate_projects
        self._active_ranker_name = self.scorer.name
        self._selector_error = ""

    def decide(
        self,
        requirement: JDRequirement,
        *,
        matches: tuple[EvidenceMatch, ...] | list[EvidenceMatch],
        projects: tuple[ProjectDescriptionDocument, ...] | list[ProjectDescriptionDocument],
        budget_available: bool = True,
    ) -> ProjectSearchDecision:
        rankings = self._rank((requirement,), tuple(projects))
        return self._decide(
            requirement,
            matches=matches,
            ranked=rankings.get(requirement.requirement_id, ()),
            budget_available=budget_available,
        )

    def _decide(
        self,
        requirement: JDRequirement,
        *,
        matches: tuple[EvidenceMatch, ...] | list[EvidenceMatch],
        ranked: tuple[ProjectCandidateRanking, ...],
        budget_available: bool,
    ) -> ProjectSearchDecision:
        relevant_matches = tuple(
            match for match in matches if match.requirement_ref.record_id == requirement.requirement_id
        )
        supported = tuple(
            match for match in relevant_matches if match.support_level is not SupportLevel.UNSUPPORTED
        )
        current_support = (
            max(supported, key=lambda item: item.score).support_level.value if supported else "not_supported"
        )
        candidate_details = tuple(_ranking_payload(candidate) for candidate in ranked[: self.max_candidate_projects])
        if any(
            match.support_level is SupportLevel.DIRECT and match.score >= self.sufficient_evidence_score
            for match in supported
        ):
            return self._decision(
                requirement,
                action="skip_sufficient_personal_evidence",
                confidence=0.0,
                threshold=1.0,
                reason="Confirmed personal evidence is already strong enough; project deep scan is unnecessary.",
                current_support=current_support,
            )
        if requirement.category is RequirementCategory.EDUCATION:
            return self._decision(
                requirement,
                action="skip_not_project_verifiable",
                confidence=0.0,
                threshold=1.0,
                reason="Education requirements cannot be verified by reading project repositories.",
                current_support=current_support,
            )
        if not budget_available:
            return self._decision(
                requirement,
                action="skip_budget_exhausted",
                confidence=0.0,
                threshold=1.0,
                reason="Project deep-scan budget is exhausted for this run.",
                current_support=current_support,
                candidate_projects=candidate_details,
            )

        confidence = ranked[0].score if ranked else 0.0
        is_preferred = not requirement.required
        threshold = self.high_threshold if is_preferred else self.medium_threshold
        candidates = tuple(
            candidate.project_id
            for candidate in ranked
            if candidate.score >= threshold
        )[: self.max_candidate_projects]
        if confidence < threshold or not candidates:
            return self._decision(
                requirement,
                action="skip_low_confidence",
                confidence=confidence,
                threshold=threshold,
                reason=(
                    ranked[0].reason
                    if ranked
                    else "No project card has enough signal to justify opening the project repository."
                ),
                current_support=current_support,
                candidate_projects=candidate_details,
            )
        return self._decision(
            requirement,
            action="deep_scan_recommended",
            confidence=confidence,
            threshold=threshold,
            reason=ranked[0].reason,
            current_support=current_support,
            candidate_project_ids=candidates,
            candidate_projects=tuple(
                _ranking_payload(candidate)
                for candidate in ranked
                if candidate.project_id in set(candidates)
            ),
        )

    def decide_batch(
        self,
        requirements: tuple[JDRequirement, ...] | list[JDRequirement],
        *,
        matches: tuple[EvidenceMatch, ...] | list[EvidenceMatch],
        projects: tuple[ProjectDescriptionDocument, ...] | list[ProjectDescriptionDocument],
        deep_scan_budget: int = 3,
    ) -> tuple[ProjectSearchDecision, ...]:
        if deep_scan_budget < 0:
            raise ValueError("deep_scan_budget cannot be negative")
        normalized_requirements = tuple(requirements)
        normalized_projects = tuple(projects)
        rankings = self._rank(normalized_requirements, normalized_projects)
        remaining = deep_scan_budget
        decisions: list[ProjectSearchDecision] = []
        for requirement in normalized_requirements:
            decision = self._decide(
                requirement,
                matches=matches,
                ranked=rankings.get(requirement.requirement_id, ()),
                budget_available=True,
            )
            if decision.action == "deep_scan_recommended":
                if remaining:
                    remaining -= 1
                else:
                    decision = self._decide(
                        requirement,
                        matches=matches,
                        ranked=rankings.get(requirement.requirement_id, ()),
                        budget_available=False,
                    )
            decisions.append(decision)
        return tuple(decisions)

    def _rank(
        self,
        requirements: tuple[JDRequirement, ...],
        projects: tuple[ProjectDescriptionDocument, ...],
    ) -> dict[str, tuple[ProjectCandidateRanking, ...]]:
        active_projects = tuple(
            project for project in projects if project.status is not ProjectDescriptionStatus.INVALIDATED
        )
        offline = self._offline_rank(requirements, active_projects)
        global_ranked = self._global_rank(requirements, active_projects)
        guarded = self._apply_global_job_context(offline, global_ranked)
        if self.selector is None:
            self._active_ranker_name = self.scorer.name
            self._selector_error = ""
            return guarded
        try:
            selected = self.selector.select(requirements, active_projects)
        except Exception as exc:  # noqa: BLE001 - degraded fallback is surfaced in the decision payload.
            self._active_ranker_name = self.scorer.name
            self._selector_error = f"{type(exc).__name__}: {str(exc)[:180]}"
            return guarded
        self._active_ranker_name = f"{self.selector.name}+{self.scorer.name}"
        self._selector_error = ""
        return self._merge_selector_and_offline(selected, guarded)

    def _rank_one(self, requirement: JDRequirement, project: ProjectDescriptionDocument) -> ProjectCandidateRanking:
        if callable(getattr(self.scorer, "rank", None)):
            return self.scorer.rank(requirement, project)
        return ProjectCandidateRanking(
            project.project_id,
            self.scorer.score(requirement, project),
            "Offline project card match.",
        )

    def _offline_rank(
        self,
        requirements: tuple[JDRequirement, ...],
        projects: tuple[ProjectDescriptionDocument, ...],
    ) -> dict[str, tuple[ProjectCandidateRanking, ...]]:
        return {
            requirement.requirement_id: tuple(
                sorted(
                    (self._rank_one(requirement, project) for project in projects),
                    key=lambda item: item.score,
                    reverse=True,
                )
            )
            for requirement in requirements
        }

    def _global_rank(
        self,
        requirements: tuple[JDRequirement, ...],
        projects: tuple[ProjectDescriptionDocument, ...],
    ) -> tuple[ProjectCandidateRanking, ...]:
        if not requirements or not projects:
            return ()
        combined_text = "\n".join(requirement.text for requirement in requirements if requirement.text)
        if not combined_text.strip():
            return ()
        combined = replace(
            requirements[0],
            requirement_id="__job_intent__",
            text=combined_text,
            original_text=combined_text,
            required=True,
            content_hash="",
        )
        return tuple(
            sorted(
                (self._rank_one(combined, project) for project in projects),
                key=lambda item: item.score,
                reverse=True,
            )
        )

    def _apply_global_job_context(
        self,
        rankings: dict[str, tuple[ProjectCandidateRanking, ...]],
        global_ranked: tuple[ProjectCandidateRanking, ...],
    ) -> dict[str, tuple[ProjectCandidateRanking, ...]]:
        if not global_ranked:
            return rankings
        global_by_project = {candidate.project_id: candidate for candidate in global_ranked}
        guarded: dict[str, tuple[ProjectCandidateRanking, ...]] = {}
        for requirement_id, candidates in rankings.items():
            adjusted: list[ProjectCandidateRanking] = []
            for candidate in candidates:
                global_candidate = global_by_project.get(candidate.project_id)
                if global_candidate is None:
                    adjusted.append(candidate)
                    continue
                score = round(0.68 * candidate.score + 0.32 * global_candidate.score, 6)
                adjusted.append(
                    ProjectCandidateRanking(
                        candidate.project_id,
                        score,
                        f"{candidate.reason} Whole-JD fit={global_candidate.score:.3f}.",
                        tuple(dict.fromkeys((*candidate.matched_terms, *global_candidate.matched_terms))),
                        tuple(dict.fromkeys((*candidate.matched_fields, *global_candidate.matched_fields))),
                    )
                )
            guarded[requirement_id] = tuple(sorted(adjusted, key=lambda item: item.score, reverse=True))
        return guarded

    @staticmethod
    def _merge_selector_and_offline(
        selected: dict[str, tuple[ProjectCandidateRanking, ...]],
        offline: dict[str, tuple[ProjectCandidateRanking, ...]],
    ) -> dict[str, tuple[ProjectCandidateRanking, ...]]:
        merged: dict[str, tuple[ProjectCandidateRanking, ...]] = {}
        for requirement_id, offline_rows in offline.items():
            if requirement_id not in selected:
                merged[requirement_id] = offline_rows
                continue
            offline_by_project = {candidate.project_id: candidate for candidate in offline_rows}
            selected_rows = selected.get(requirement_id, ())
            rows: list[ProjectCandidateRanking] = []
            for candidate in selected_rows:
                base = offline_by_project.get(candidate.project_id)
                if base is None:
                    rows.append(candidate)
                    continue
                score = round(0.55 * candidate.score + 0.45 * base.score, 6)
                rows.append(
                    ProjectCandidateRanking(
                        candidate.project_id,
                        score,
                        f"{candidate.reason} Offline fit={base.score:.3f}.",
                        tuple(dict.fromkeys((*candidate.matched_terms, *base.matched_terms))),
                        tuple(dict.fromkeys((*candidate.matched_fields, *base.matched_fields))),
                    )
                )
            merged[requirement_id] = tuple(sorted(rows, key=lambda item: item.score, reverse=True))
        return merged

    def _decision(
        self,
        requirement: JDRequirement,
        *,
        action: str,
        confidence: float,
        threshold: float,
        reason: str,
        current_support: str,
        candidate_project_ids: tuple[str, ...] = (),
        candidate_projects: tuple[dict[str, Any], ...] = (),
    ) -> ProjectSearchDecision:
        return ProjectSearchDecision(
            requirement_id=requirement.requirement_id,
            action=action,
            confidence=round(float(confidence), 6),
            threshold=float(threshold),
            reason=reason,
            current_support=current_support,
            candidate_project_ids=candidate_project_ids,
            scorer=self._active_ranker_name,
            degraded=self.selector is None or bool(self._selector_error),
            degradation_reason=(
                "Project matcher model is not configured; used field-weighted offline fallback."
                if self.selector is None
                else f"Project matcher failed; used field-weighted offline fallback: {self._selector_error}"
                if self._selector_error
                else ""
            ),
            candidate_projects=candidate_projects,
        )
