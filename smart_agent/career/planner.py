"""Constraint-driven resume content planning with a CP-SAT backend."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from smart_agent.career.models import (
    RESUME_SECTION_ORDER,
    EvidenceMatch,
    JDDocument,
    JDRequirement,
    PersonalEvidence,
    ResumeItem,
    ResumePlan,
    ResumeSource,
    SourceReference,
    SupportLevel,
    VersionRef,
    normalize_resume_section,
    record_to_dict,
    version_ref,
)
from smart_agent.career.store import CareerStore, RecordNotFoundError, ReferenceIntegrityError

REQUIRED_ITEM_FEATURES = (
    "jd_relevance",
    "evidence_strength",
    "title_and_brand_value",
    "result_strength",
    "narrative_value",
)


@dataclass(frozen=True, slots=True)
class PlannerWeights:
    coverage: float = 4.0
    jd_relevance: float = 2.0
    evidence_strength: float = 2.0
    title_and_brand_value: float = 1.2
    result_strength: float = 1.0
    narrative_value: float = 0.8
    length_cost: float = 0.3

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("Planner weights cannot be negative.")
        if not any(value > 0 for value in asdict(self).values()):
            raise ValueError("At least one planner weight must be positive.")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "PlannerWeights":
        if not value:
            return cls()
        unknown = set(value) - set(asdict(cls()))
        if unknown:
            raise ValueError(f"Unknown planner weights: {sorted(unknown)}")
        return cls(**{key: float(item) for key, item in value.items()})


@dataclass(frozen=True, slots=True)
class PlanningMetrics:
    selected_count: int
    total_length: int
    length_violation: int
    jd_coverage_rate: float
    duplicate_rate: float
    unsupported_content_rate: float
    objective_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlanningComparison:
    constrained: PlanningMetrics
    similarity_baseline: PlanningMetrics
    llm_baseline: PlanningMetrics | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "constrained": self.constrained.to_dict(),
            "similarity_baseline": self.similarity_baseline.to_dict(),
            "llm_baseline": self.llm_baseline.to_dict() if self.llm_baseline else None,
        }


@dataclass(frozen=True, slots=True)
class PlanResult:
    plan: ResumePlan
    selected_items: tuple[ResumeItem, ...]
    excluded_items: tuple[ResumeItem, ...]
    solver_backend: str
    comparison: PlanningComparison

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver_backend": self.solver_backend,
            "plan": record_to_dict(self.plan),
            "selected_items": [record_to_dict(item) for item in self.selected_items],
            "excluded_items": [record_to_dict(item) for item in self.excluded_items],
            "comparison": self.comparison.to_dict(),
        }

    def persist(self, store: CareerStore) -> "PlanResult":
        try:
            plan = store.get(self.plan.plan_id, expected_type=ResumePlan)
        except RecordNotFoundError:
            store.put(self.plan)
            plan = self.plan
        return PlanResult(
            plan=plan,
            selected_items=self.selected_items,
            excluded_items=self.excluded_items,
            solver_backend=self.solver_backend,
            comparison=self.comparison,
        )


class ResumeCandidateBuilder:
    """Convert model-proposed candidate units into verified ResumeItem records."""

    def __init__(self, store: CareerStore) -> None:
        self.store = store

    def build(self, drafts: list[dict[str, Any]]) -> tuple[ResumeItem, ...]:
        if not isinstance(drafts, list) or not drafts:
            raise ValueError("candidate_items must be a non-empty array.")
        supported_matches = tuple(
            match
            for match in self.store.list_latest(EvidenceMatch)
            if match.support_level is not SupportLevel.UNSUPPORTED
        )
        items: list[ResumeItem] = []
        for draft in drafts:
            item = self._build_one(draft, supported_matches)
            try:
                current = self.store.get(item.item_id, expected_type=ResumeItem)
            except RecordNotFoundError:
                self.store.put(item)
                current = item
            items.append(current)
        if len({item.item_id for item in items}) != len(items):
            raise ValueError("candidate_items cannot contain semantic duplicates.")
        return tuple(items)

    def _build_one(
        self,
        draft: dict[str, Any],
        matches: tuple[EvidenceMatch, ...],
    ) -> ResumeItem:
        if not isinstance(draft, dict):
            raise ValueError("Each resume candidate must be an object.")
        raw_section = str(draft.get("section") or "").strip()
        proposed_text = str(draft.get("proposed_text") or "").strip()
        if not raw_section or not proposed_text:
            raise ValueError("Resume candidate section and proposed_text are required.")
        section = normalize_resume_section(raw_section)
        evidence_ids = self._ids(draft.get("evidence_ids"), "evidence_ids")
        fact_ids = self._ids(draft.get("evidence_fact_ids"), "evidence_fact_ids")
        requirement_ids = self._ids(
            draft.get("requirement_ids"),
            "requirement_ids",
            allow_empty=True,
        )
        evidence = tuple(
            self.store.get(identity, expected_type=PersonalEvidence)
            for identity in evidence_ids
        )
        requirements = tuple(
            self.store.get(identity, expected_type=JDRequirement)
            for identity in requirement_ids
        )
        available_fact_ids = {fact.fact_id for record in evidence for fact in record.facts}
        if not set(fact_ids).issubset(available_fact_ids):
            raise ReferenceIntegrityError("Candidate cites facts outside its evidence records.")
        requirement_fact_ids = self._ids(
            draft.get("requirement_evidence_fact_ids", list(fact_ids)),
            "requirement_evidence_fact_ids",
            allow_empty=not requirements,
        )
        if not set(requirement_fact_ids).issubset(set(fact_ids)):
            raise ReferenceIntegrityError(
                "Candidate requirement facts must be a subset of its cited evidence facts."
            )
        evidence_refs = tuple(version_ref(record) for record in evidence)
        requirement_refs = tuple(version_ref(record) for record in requirements)
        self._validate_match_support(evidence_refs, requirement_fact_ids, requirement_refs, matches)

        raw_features = draft.get("features")
        if not isinstance(raw_features, dict):
            raise ValueError("Resume candidate features must be an object.")
        missing_features = set(REQUIRED_ITEM_FEATURES) - set(raw_features)
        if missing_features:
            raise ValueError(f"Resume candidate is missing features: {sorted(missing_features)}")
        features = {str(key): float(value) for key, value in raw_features.items()}
        if any(not 0 <= value <= 1 for value in features.values()):
            raise ValueError("Resume candidate features must be between 0 and 1.")
        length_cost = int(draft.get("length_cost", 0))
        if length_cost < 1:
            raise ValueError("Resume candidate length_cost must be at least 1.")
        redundancy_group = str(draft.get("redundancy_group") or "").strip()
        identity_payload = {
            "section": section,
            "proposed_text": proposed_text,
            "evidence_refs": [(ref.record_id, ref.revision) for ref in evidence_refs],
            "fact_ids": fact_ids,
            "requirement_refs": [(ref.record_id, ref.revision) for ref in requirement_refs],
            "features": features,
            "length_cost": length_cost,
            "redundancy_group": redundancy_group,
        }
        digest = hashlib.sha256(
            json.dumps(identity_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return ResumeItem(
            item_id=f"item-{digest[:16]}",
            section=section,
            proposed_text=proposed_text,
            evidence_refs=evidence_refs,
            evidence_fact_ids=fact_ids,
            requirement_refs=requirement_refs,
            redundancy_group=redundancy_group,
            features=features,
            length_cost=length_cost,
            source_refs=self._sources(evidence, requirements),
        )

    @staticmethod
    def _ids(value: Any, field_name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
        if not isinstance(value, list) or (not value and not allow_empty):
            raise ValueError(f"{field_name} must be an array with at least one item.")
        normalized = tuple(str(item).strip() for item in value if str(item).strip())
        if (not normalized and not allow_empty) or len(normalized) != len(set(normalized)):
            raise ValueError(f"{field_name} must contain unique non-empty IDs.")
        return normalized

    @staticmethod
    def _validate_match_support(
        evidence_refs: tuple[VersionRef, ...],
        fact_ids: tuple[str, ...],
        requirement_refs: tuple[VersionRef, ...],
        matches: tuple[EvidenceMatch, ...],
    ) -> None:
        if not requirement_refs:
            return
        cited_facts = set(fact_ids)
        supported_facts: set[str] = set()
        for requirement_ref in requirement_refs:
            requirement_matches = [
                match
                for match in matches
                if match.requirement_ref == requirement_ref
                and match.evidence_ref in evidence_refs
                and cited_facts.intersection(match.evidence_fact_ids)
            ]
            if not requirement_matches:
                raise ReferenceIntegrityError(
                    f"Candidate requirement lacks a matching verified fact: {requirement_ref.record_id}"
                )
            for match in requirement_matches:
                supported_facts.update(cited_facts.intersection(match.evidence_fact_ids))
        if supported_facts != cited_facts:
            raise ReferenceIntegrityError(
                f"Candidate cites facts not supported by its requirements: {sorted(cited_facts - supported_facts)}"
            )

    @staticmethod
    def _sources(
        evidence: tuple[PersonalEvidence, ...],
        requirements: tuple[JDRequirement, ...],
    ) -> tuple[SourceReference, ...]:
        unique: dict[tuple[str, str, str], SourceReference] = {}
        for source in (
            *(source for record in evidence for source in record.source_refs),
            *(source for record in requirements for source in record.source_refs),
        ):
            key = (source.source_id, source.content_hash, json.dumps(source.locator, sort_keys=True))
            unique[key] = source
        return tuple(unique.values())


class ConstraintResumePlanner:
    """Select candidate content under evidence, coverage, duplicate, and length constraints."""

    def __init__(self, *, weights: PlannerWeights | None = None) -> None:
        self.weights = weights or PlannerWeights()

    def plan(
        self,
        *,
        jd: JDDocument,
        requirements: tuple[JDRequirement, ...],
        items: tuple[ResumeItem, ...],
        content_budget: int,
        section_order: tuple[str, ...] | list[str],
        resume_source: ResumeSource | None = None,
        llm_selected_item_ids: tuple[str, ...] | list[str] | None = None,
        required_item_ids: tuple[str, ...] | list[str] | None = None,
    ) -> PlanResult:
        if content_budget < 1:
            raise ValueError("content_budget must be at least 1.")
        if not items:
            raise ValueError("At least one ResumeItem is required.")
        requirements = tuple(requirement for requirement in requirements if requirement.active)
        if not requirements:
            raise ValueError("At least one active JDRequirement is required.")
        for item in items:
            missing = set(REQUIRED_ITEM_FEATURES) - set(item.features)
            if missing:
                raise ValueError(f"ResumeItem is missing planner features: {sorted(missing)}")
            if item.length_cost < 1:
                raise ValueError("ResumeItem length_cost must be at least 1.")
        if any(requirement.jd_ref.record_id != jd.jd_id for requirement in requirements):
            raise ReferenceIntegrityError("All planner requirements must belong to the selected JD.")
        known_requirements = {requirement.requirement_id for requirement in requirements}
        if any(not set(item.requirement_ids).issubset(known_requirements) for item in items):
            raise ReferenceIntegrityError("ResumeItem cites a requirement outside the planner JD set.")

        required_ids = tuple(dict.fromkeys(str(item).strip() for item in (required_item_ids or ()) if str(item).strip()))
        known_item_ids = {item.item_id for item in items}
        unknown_required = set(required_ids) - known_item_ids
        if unknown_required:
            raise ReferenceIntegrityError(f"required_item_ids contains unknown items: {sorted(unknown_required)}")
        required_indexes = tuple(
            index for index, item in enumerate(items) if item.item_id in set(required_ids)
        )
        conflicting_required = [
            tuple(items[index].item_id for index in indexes if index in required_indexes)
            for indexes in self._duplicate_groups(items)
            if len(set(indexes).intersection(required_indexes)) > 1
        ]
        if conflicting_required:
            raise ReferenceIntegrityError(
                "required_item_ids contain duplicate evidence blocks: "
                + "; ".join(", ".join(group) for group in conflicting_required)
            )
        required_length = sum(items[index].length_cost for index in required_indexes)
        effective_budget = max(content_budget, required_length)
        selected_indexes, backend = self._solve(items, requirements, effective_budget, required_ids=required_ids)
        selected_set = set(selected_indexes)
        normalized_sections = self._section_order(section_order, items, selected_indexes)
        selected_items = tuple(
            item
            for section in normalized_sections
            for index, item in enumerate(items)
            if index in selected_set and normalize_resume_section(item.section) == section
        )
        excluded_items = tuple(item for index, item in enumerate(items) if index not in selected_set)
        excluded_reasons = {
            item.item_id: self._exclusion_reason(index, selected_indexes, items, effective_budget)
            for index, item in enumerate(items)
            if index not in selected_set
        }
        comparison = self._comparison(
            items,
            requirements,
            selected_indexes,
            effective_budget,
            llm_selected_item_ids,
        )
        constraints = {
            "content_budget": effective_budget,
            "requested_content_budget": content_budget,
            "required_items_length": required_length,
            "budget_adjusted_for_required_items": effective_budget != content_budget,
            "solver_backend": backend,
            "weights": asdict(self.weights),
            "selected_total_length": sum(item.length_cost for item in selected_items),
            "covered_requirement_ids": sorted(
                {identity for item in selected_items for identity in item.requirement_ids}
            ),
            "hard_constraints": [
                "verified_evidence_only",
                "content_budget",
                "unique_fact",
                "unique_redundancy_group",
                "required_items",
            ],
        }
        identity_payload = {
            "jd": [jd.jd_id, jd.revision],
            "items": [(item.item_id, item.revision) for item in items],
            "selected": [item.item_id for item in selected_items],
            "budget": effective_budget,
            "sections": normalized_sections,
            "weights": asdict(self.weights),
            "backend": backend,
            "resume_source": (
                [resume_source.resume_source_id, resume_source.revision]
                if resume_source is not None
                else None
            ),
        }
        digest = hashlib.sha256(
            json.dumps(identity_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        plan = ResumePlan(
            plan_id=f"plan-{digest[:16]}",
            jd_ref=version_ref(jd),
            item_refs=tuple(version_ref(item) for item in items),
            selected_item_ids=tuple(item.item_id for item in selected_items),
            excluded_reasons=excluded_reasons,
            section_order=normalized_sections,
            constraints=constraints,
            resume_source_ref=version_ref(resume_source) if resume_source is not None else None,
            source_refs=self._plan_sources(jd, items, resume_source),
        )
        return PlanResult(plan, selected_items, excluded_items, backend, comparison)

    def _solve(
        self,
        items: tuple[ResumeItem, ...],
        requirements: tuple[JDRequirement, ...],
        budget: int,
        *,
        required_ids: tuple[str, ...] = (),
    ) -> tuple[tuple[int, ...], str]:
        try:
            return self._solve_cp_sat(items, requirements, budget, required_ids=required_ids), "cp_sat"
        except ImportError:
            if len(items) <= 20:
                return self._solve_exact(items, requirements, budget, required_ids=required_ids), "exact_fallback"
            return self._solve_greedy(items, requirements, budget, required_ids=required_ids), "greedy_fallback"

    def _solve_cp_sat(
        self,
        items: tuple[ResumeItem, ...],
        requirements: tuple[JDRequirement, ...],
        budget: int,
        *,
        required_ids: tuple[str, ...] = (),
    ) -> tuple[int, ...]:
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        selected = [model.NewBoolVar(f"item_{index}") for index in range(len(items))]
        model.Add(sum(item.length_cost * selected[index] for index, item in enumerate(items)) <= budget)
        required = set(required_ids)
        for index, item in enumerate(items):
            if item.item_id in required:
                model.Add(selected[index] == 1)
        for indexes in self._duplicate_groups(items):
            if len(indexes) > 1:
                model.Add(sum(selected[index] for index in indexes) <= 1)

        scale = 10_000
        objective_terms = [
            round(self._item_value(item, budget) * scale) * selected[index]
            for index, item in enumerate(items)
        ]
        for requirement in requirements:
            indexes = [
                index
                for index, item in enumerate(items)
                if requirement.requirement_id in item.requirement_ids
            ]
            if not indexes:
                continue
            covered = model.NewBoolVar(f"covered_{requirement.requirement_id}")
            model.AddMaxEquality(covered, [selected[index] for index in indexes])
            benefit = self.weights.coverage * (1.0 if requirement.required else 0.5)
            objective_terms.append(round(benefit * scale) * covered)
        model.Maximize(sum(objective_terms))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError("CP-SAT could not find a feasible ResumePlan.")
        return tuple(index for index, variable in enumerate(selected) if solver.Value(variable))

    def _solve_exact(
        self,
        items: tuple[ResumeItem, ...],
        requirements: tuple[JDRequirement, ...],
        budget: int,
        *,
        required_ids: tuple[str, ...] = (),
    ) -> tuple[int, ...]:
        best: tuple[int, ...] = ()
        best_value = 0.0
        for mask in range(1 << len(items)):
            indexes = tuple(index for index in range(len(items)) if mask & (1 << index))
            if not set(required_ids).issubset({items[index].item_id for index in indexes}):
                continue
            if not self._is_feasible(indexes, items, budget):
                continue
            value = self._objective(indexes, items, requirements, budget)
            if value > best_value + 1e-9 or (
                abs(value - best_value) <= 1e-9
                and self._tie_break(indexes, items) < self._tie_break(best, items)
            ):
                best = indexes
                best_value = value
        return best

    def _solve_greedy(
        self,
        items: tuple[ResumeItem, ...],
        requirements: tuple[JDRequirement, ...],
        budget: int,
        *,
        required_ids: tuple[str, ...] = (),
    ) -> tuple[int, ...]:
        required = set(required_ids)
        selected = [index for index, item in enumerate(items) if item.item_id in required]
        if not self._is_feasible(tuple(selected), items, budget):
            raise RuntimeError("Required resume items do not fit within the content budget.")
        remaining = set(range(len(items)))
        remaining.difference_update(selected)
        while remaining:
            current_value = self._objective(tuple(selected), items, requirements, budget)
            candidates = []
            for index in remaining:
                proposed = tuple(sorted((*selected, index)))
                if self._is_feasible(proposed, items, budget):
                    gain = self._objective(proposed, items, requirements, budget) - current_value
                    candidates.append((gain / items[index].length_cost, gain, -index, index))
            if not candidates:
                break
            _, gain, _, chosen = max(candidates)
            if gain <= 0:
                break
            selected.append(chosen)
            remaining.remove(chosen)
        return tuple(sorted(selected))

    def _objective(
        self,
        indexes: tuple[int, ...],
        items: tuple[ResumeItem, ...],
        requirements: tuple[JDRequirement, ...],
        budget: int,
    ) -> float:
        selected = [items[index] for index in indexes]
        value = sum(self._item_value(item, budget) for item in selected)
        covered = {identity for item in selected for identity in item.requirement_ids}
        value += sum(
            self.weights.coverage * (1.0 if requirement.required else 0.5)
            for requirement in requirements
            if requirement.requirement_id in covered
        )
        return value

    def _item_value(self, item: ResumeItem, budget: int) -> float:
        return (
            self.weights.jd_relevance * item.features["jd_relevance"]
            + self.weights.evidence_strength * item.features["evidence_strength"]
            + self.weights.title_and_brand_value * item.features["title_and_brand_value"]
            + self.weights.result_strength * item.features["result_strength"]
            + self.weights.narrative_value * item.features["narrative_value"]
            - self.weights.length_cost * item.length_cost / budget
        )

    @staticmethod
    def _duplicate_groups(items: tuple[ResumeItem, ...]) -> tuple[tuple[int, ...], ...]:
        groups: dict[str, set[int]] = {}
        for index, item in enumerate(items):
            for fact_id in item.evidence_fact_ids:
                groups.setdefault(f"fact:{fact_id}", set()).add(index)
            if item.redundancy_group:
                groups.setdefault(f"group:{item.redundancy_group}", set()).add(index)
        return tuple(tuple(sorted(indexes)) for indexes in groups.values())

    def _is_feasible(
        self,
        indexes: tuple[int, ...],
        items: tuple[ResumeItem, ...],
        budget: int,
    ) -> bool:
        selected = set(indexes)
        if sum(items[index].length_cost for index in selected) > budget:
            return False
        return all(len(selected.intersection(group)) <= 1 for group in self._duplicate_groups(items))

    @staticmethod
    def _tie_break(indexes: tuple[int, ...], items: tuple[ResumeItem, ...]) -> tuple[int, tuple[int, ...]]:
        return (sum(items[index].length_cost for index in indexes), indexes)

    @staticmethod
    def _section_order(
        requested: tuple[str, ...] | list[str],
        items: tuple[ResumeItem, ...],
        selected_indexes: tuple[int, ...],
    ) -> tuple[str, ...]:
        del requested, items, selected_indexes
        return RESUME_SECTION_ORDER

    def _exclusion_reason(
        self,
        index: int,
        selected_indexes: tuple[int, ...],
        items: tuple[ResumeItem, ...],
        budget: int,
    ) -> str:
        item = items[index]
        if item.length_cost > budget:
            return "Item exceeds the total content budget."
        selected = [items[value] for value in selected_indexes]
        if any(set(item.evidence_fact_ids).intersection(other.evidence_fact_ids) for other in selected):
            return "A selected item already uses the same evidence fact."
        if item.redundancy_group and any(item.redundancy_group == other.redundancy_group for other in selected):
            return "A higher-value item was selected from the same redundancy group."
        if sum(other.length_cost for other in selected) + item.length_cost > budget:
            return "The item did not fit within the remaining content budget."
        return "The item had lower combined value under the coverage and length objective."

    def _comparison(
        self,
        items: tuple[ResumeItem, ...],
        requirements: tuple[JDRequirement, ...],
        constrained: tuple[int, ...],
        budget: int,
        llm_selected_item_ids: tuple[str, ...] | list[str] | None,
    ) -> PlanningComparison:
        ranked = sorted(
            range(len(items)),
            key=lambda index: (-items[index].features["jd_relevance"], index),
        )
        similarity: list[int] = []
        used = 0
        for index in ranked:
            if used + items[index].length_cost <= budget:
                similarity.append(index)
                used += items[index].length_cost
        llm_indexes: tuple[int, ...] | None = None
        if llm_selected_item_ids is not None:
            known = {item.item_id: index for index, item in enumerate(items)}
            unknown = set(llm_selected_item_ids) - set(known)
            if unknown:
                raise ValueError(f"llm_selected_item_ids contains unknown items: {sorted(unknown)}")
            llm_indexes = tuple(known[identity] for identity in dict.fromkeys(llm_selected_item_ids))
        return PlanningComparison(
            constrained=self._metrics(constrained, items, requirements, budget),
            similarity_baseline=self._metrics(tuple(similarity), items, requirements, budget),
            llm_baseline=(
                self._metrics(llm_indexes, items, requirements, budget)
                if llm_indexes is not None
                else None
            ),
        )

    def _metrics(
        self,
        indexes: tuple[int, ...],
        items: tuple[ResumeItem, ...],
        requirements: tuple[JDRequirement, ...],
        budget: int,
    ) -> PlanningMetrics:
        selected = [items[index] for index in indexes]
        length = sum(item.length_cost for item in selected)
        covered = {identity for item in selected for identity in item.requirement_ids}
        target = {requirement.requirement_id for requirement in requirements if requirement.active}
        duplicate_indexes: set[int] = set()
        for group in self._duplicate_groups(items):
            present = [index for index in indexes if index in group]
            duplicate_indexes.update(present[1:])
        return PlanningMetrics(
            selected_count=len(selected),
            total_length=length,
            length_violation=max(0, length - budget),
            jd_coverage_rate=round(len(covered.intersection(target)) / len(target), 6) if target else 1.0,
            duplicate_rate=round(len(duplicate_indexes) / len(selected), 6) if selected else 0.0,
            unsupported_content_rate=0.0,
            objective_value=round(self._objective(indexes, items, requirements, budget), 6),
        )

    @staticmethod
    def _plan_sources(
        jd: JDDocument,
        items: tuple[ResumeItem, ...],
        resume_source: ResumeSource | None,
    ) -> tuple[SourceReference, ...]:
        unique: dict[tuple[str, str, str], SourceReference] = {}
        sources = [*jd.source_refs, *(source for item in items for source in item.source_refs)]
        if resume_source is not None:
            sources.extend(resume_source.source_refs)
        for source in sources:
            key = (source.source_id, source.content_hash, json.dumps(source.locator, sort_keys=True))
            unique[key] = source
        return tuple(unique.values())
