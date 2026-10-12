"""Persistent, pluggable runtime for traceable text interviews."""

from __future__ import annotations

import hashlib
import inspect
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from smart_agent.career.interview import (
    AnswerObservation,
    DeterministicInterviewPolicy,
    InterviewAction,
    InterviewDecision,
    InterviewerPolicy,
    InterviewerProfile,
    InterviewKnowledgeBlockRef,
    InterviewKnowledgeBlockRole,
    InterviewKnowledgeChunk,
    InterviewKnowledgeStore,
    InterviewLine,
    InterviewMode,
    InterviewPath,
    InterviewPoint,
    InterviewPolicyEngine,
    InterviewQuestion,
    InterviewQuestionType,
    InterviewSessionState,
    InterviewSessionStore,
    InterviewStage,
    InterviewStatus,
    InterviewStopReason,
    InterviewTurn,
    KnowledgeBeliefState,
    KnowledgeConfidenceUpdate,
    QuestionUtility,
    RecommendationInterviewPolicy,
    RecommendationTrace,
)
from smart_agent.career.interview_knowledge import (
    InterviewRetrievalFilters,
    InterviewRetrievalResult,
)
from smart_agent.career.interview_mastery import KnowledgeMasteryProfile, KnowledgeMasteryStore
from smart_agent.career.models import (
    EvidenceType,
    JDDocument,
    JDRequirement,
    PersonalEvidence,
    ResumeClaim,
    ResumeDocument,
    ResumeDocumentStatus,
    ResumeRun,
    ResumeRunStatus,
)
from smart_agent.career.store import CareerStore


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_+#.\-]+|[\u4e00-\u9fff]", str(text or "").casefold()))


def _stable_rank(seed: int, turn_count: int, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{turn_count}:{identity}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class InterviewAnswerAssessment:
    correctness: float
    completeness: float
    confidence: float
    reason: str
    # Deprecated compatibility field. Subjective answer evaluation no longer
    # reports which retrieved chunks an answer "covered".
    knowledge_refs: tuple[str, ...] = ()
    evaluator_name: str = ""
    question_type: InterviewQuestionType = InterviewQuestionType.KNOWLEDGE
    protocol: str = "knowledge_reference"
    dimensions: dict[str, float] = field(default_factory=dict)
    information_gain: str = ""
    verification_need: str = ""
    confidence_updates: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("correctness", "completeness", "confidence"):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "reason", str(self.reason or "").strip())
        object.__setattr__(self, "knowledge_refs", tuple(self.knowledge_refs))
        object.__setattr__(self, "evaluator_name", str(self.evaluator_name or "").strip())
        object.__setattr__(self, "question_type", InterviewQuestionType(self.question_type))
        object.__setattr__(self, "protocol", str(self.protocol or "").strip())
        dimensions = {str(key): float(value) for key, value in dict(self.dimensions).items()}
        if any(not 0 <= value <= 1 for value in dimensions.values()):
            raise ValueError("assessment dimensions must be between 0 and 1")
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "information_gain", str(self.information_gain or "").strip())
        object.__setattr__(self, "verification_need", str(self.verification_need or "").strip())
        object.__setattr__(
            self,
            "confidence_updates",
            tuple(dict(item) for item in self.confidence_updates if isinstance(item, dict)),
        )

    def to_dict(self) -> dict:
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "confidence": self.confidence,
            "reason": self.reason,
            "knowledge_refs": list(self.knowledge_refs),
            "evaluator_name": self.evaluator_name,
            "question_type": self.question_type.value,
            "protocol": self.protocol,
            "dimensions": self.dimensions,
            "information_gain": self.information_gain,
            "verification_need": self.verification_need,
            "confidence_updates": [dict(item) for item in self.confidence_updates],
        }


@dataclass(frozen=True, slots=True)
class InterviewTurnContext:
    """Immutable, role-scoped snapshot shared by one answer-processing turn."""

    session_id: str
    previous_question: InterviewQuestion
    user_answer: str
    current_subject: str
    current_point_id: str | None
    current_corpus_id: str | None
    previous_knowledge_refs: tuple[str, ...]
    evaluation_references: tuple[InterviewKnowledgeChunk, ...]
    display_subject: str = ""
    current_confidence: float = 0.5
    current_observation_count: int = 0
    current_document_confidence: float = 0.5
    current_document_coverage: float = 0.0
    depth_candidates: tuple[InterviewKnowledgeChunk, ...] = ()
    breadth_candidates: tuple[InterviewKnowledgeChunk, ...] = ()
    breadth_subject: str = ""
    breadth_display_subject: str = ""
    recent_turns: tuple[dict[str, str], ...] = ()
    summary: str = ""
    allowed_corpus_ids: tuple[str, ...] = ()
    depth_chain: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", str(self.session_id or "").strip())
        if not self.session_id:
            raise ValueError("session_id is required")
        object.__setattr__(self, "user_answer", str(self.user_answer or "").strip())
        object.__setattr__(self, "current_subject", str(self.current_subject or "").strip())
        object.__setattr__(self, "display_subject", str(self.display_subject or "").strip())
        object.__setattr__(self, "current_point_id", str(self.current_point_id or "").strip() or None)
        object.__setattr__(self, "current_corpus_id", str(self.current_corpus_id or "").strip() or None)
        for field_name in ("current_confidence", "current_document_confidence", "current_document_coverage"):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "current_observation_count", max(0, int(self.current_observation_count)))
        object.__setattr__(self, "previous_knowledge_refs", tuple(str(item) for item in self.previous_knowledge_refs))
        object.__setattr__(self, "evaluation_references", tuple(self.evaluation_references))
        object.__setattr__(self, "depth_candidates", tuple(self.depth_candidates))
        object.__setattr__(self, "breadth_candidates", tuple(self.breadth_candidates))
        object.__setattr__(self, "breadth_subject", str(self.breadth_subject or "").strip())
        object.__setattr__(self, "breadth_display_subject", str(self.breadth_display_subject or "").strip())
        object.__setattr__(self, "recent_turns", tuple(dict(item) for item in self.recent_turns))
        object.__setattr__(self, "allowed_corpus_ids", tuple(str(item) for item in self.allowed_corpus_ids))
        object.__setattr__(self, "depth_chain", tuple(str(item) for item in self.depth_chain))


@dataclass(frozen=True, slots=True)
class InterviewQuestionDraft:
    """Pure LLM/local draft. Runtime owns path, references, and final state."""

    text: str
    path: InterviewPath
    topic: str
    knowledge_refs: tuple[str, ...] = ()
    point_id: str | None = None
    ok: bool = True
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", str(self.text or "").strip())
        object.__setattr__(self, "path", InterviewPath(self.path))
        object.__setattr__(self, "topic", str(self.topic or "").strip())
        object.__setattr__(self, "knowledge_refs", tuple(str(item) for item in self.knowledge_refs))
        object.__setattr__(self, "point_id", str(self.point_id or "").strip() or None)
        object.__setattr__(self, "error_code", str(self.error_code or "").strip() or None)


@dataclass(frozen=True, slots=True)
class ParallelInterviewOutcome:
    assessment: InterviewAnswerAssessment | None
    depth: InterviewQuestionDraft | None
    breadth: InterviewQuestionDraft | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


class ParallelInterviewOrchestrator:
    """Run evaluation and two independent drafts against one immutable snapshot."""

    def __init__(self, evaluator: "InterviewAnswerEvaluator", generator: "InterviewQuestionGenerator", *, timeout_seconds: float = 12.0) -> None:
        self.evaluator = evaluator
        self.generator = generator
        self.timeout_seconds = max(0.1, float(timeout_seconds))

    def run(self, context: InterviewTurnContext) -> ParallelInterviewOutcome:
        started = time.perf_counter()
        tasks: dict[str, Any] = {}
        pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="interview-turn")
        try:
            tasks["evaluation"] = pool.submit(
                self._timed,
                self._evaluate,
                context,
            )
            tasks["depth"] = pool.submit(self._timed, self._depth, context)
            tasks["breadth"] = pool.submit(self._timed, self._breadth, context)
            values: dict[str, Any] = {}
            errors: dict[str, str] = {}
            timings: dict[str, float] = {}
            for name, future in tasks.items():
                try:
                    values[name], timings[name] = future.result(timeout=self.timeout_seconds)
                except FutureTimeoutError:
                    errors[name] = "timeout"
                    future.cancel()
                    timings[name] = self.timeout_seconds * 1000
                except Exception as exc:  # noqa: BLE001 - one failed lane must not cancel the turn
                    errors[name] = f"{type(exc).__name__}: {exc}"
                    timings[name] = 0.0
        finally:
            # A timed-out provider must not hold the request open while the
            # executor context manager waits for its worker. Running providers
            # are isolated and their result is discarded after this turn.
            pool.shutdown(wait=False, cancel_futures=True)
        timings["total"] = (time.perf_counter() - started) * 1000
        assessment = values.get("evaluation")
        if not isinstance(assessment, InterviewAnswerAssessment):
            assessment = None
        depth = values.get("depth")
        if not isinstance(depth, InterviewQuestionDraft):
            depth = None
        breadth = values.get("breadth")
        if not isinstance(breadth, InterviewQuestionDraft):
            breadth = None
        return ParallelInterviewOutcome(assessment, depth, breadth, timings, errors)

    @staticmethod
    def _timed(callback: Any, /, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        started = time.perf_counter()
        value = callback(*args, **kwargs)
        return value, (time.perf_counter() - started) * 1000

    def _evaluate(self, context: InterviewTurnContext) -> InterviewAnswerAssessment:
        method = self.evaluator.evaluate
        parameters = inspect.signature(method).parameters
        kwargs: dict[str, Any] = {
            "question": context.previous_question,
            "answer": context.user_answer,
            "knowledge": context.evaluation_references,
        }
        if "context" in parameters:
            kwargs["context"] = context
        return method(**kwargs)

    def _depth(self, context: InterviewTurnContext) -> InterviewQuestionDraft:
        method = getattr(self.generator, "generate_depth_draft", None)
        if callable(method):
            return method(context=context)
        return InterviewQuestionDraft(
            text="",
            path=InterviewPath.DEPTH,
            topic=context.current_subject,
            knowledge_refs=(),
            ok=False,
            error_code="depth_generator_unavailable",
        )

    def _breadth(self, context: InterviewTurnContext) -> InterviewQuestionDraft:
        method = getattr(self.generator, "generate_breadth_draft", None)
        if callable(method):
            return method(context=context)
        return InterviewQuestionDraft(
            text="",
            path=InterviewPath.BREADTH,
            topic=context.breadth_subject,
            knowledge_refs=(),
            ok=False,
            error_code="breadth_generator_unavailable",
        )


class InterviewAnswerEvaluator(Protocol):
    name: str

    def evaluate(
        self,
        *,
        question: InterviewQuestion,
        answer: str,
        knowledge: tuple[InterviewKnowledgeChunk, ...],
        context: InterviewTurnContext | None = None,
    ) -> InterviewAnswerAssessment: ...


class LiveInterviewKnowledgeRetriever(Protocol):
    """Tenant-scoped local retrieval used only by gap-analysis sessions."""

    name: str

    def retrieve(
        self,
        query: str,
        *,
        filters: InterviewRetrievalFilters,
        top_k: int = 5,
    ) -> InterviewRetrievalResult: ...


class LexicalInterviewAnswerEvaluator:
    """Conservative type-aware offline baseline; production can inject a model evaluator."""

    name = "protocol-aware-answer-baseline-v2"

    def evaluate(
        self,
        *,
        question: InterviewQuestion,
        answer: str,
        knowledge: tuple[InterviewKnowledgeChunk, ...],
    ) -> InterviewAnswerAssessment:
        answer_tokens = _tokens(answer)
        expected_tokens = _tokens(" ".join(chunk.text for chunk in knowledge))
        if not answer_tokens:
            raise ValueError("answer is required")
        overlap = len(answer_tokens & expected_tokens)
        precision = overlap / max(1, len(answer_tokens))
        recall = overlap / max(1, len(expected_tokens))
        question_overlap = len(answer_tokens & _tokens(question.text)) / max(1, len(_tokens(question.text)))
        question_type = question.question_type
        if question_type is InterviewQuestionType.KNOWLEDGE:
            correctness = min(1.0, 0.25 + 1.5 * precision) if expected_tokens else 0.5
            completeness = min(1.0, 0.15 + 2.0 * recall) if expected_tokens else 0.5
            protocol = "knowledge_reference"
            dimensions = {"reference_precision": precision, "reference_coverage": recall}
            reason = "Compared professional knowledge with the cited reference chunks."
        elif question_type is InterviewQuestionType.PROJECT_SCENARIO:
            tradeoff = bool(answer_tokens & _tokens("trade-off tradeoff 取舍 权衡 boundary 边界"))
            mechanism = bool(answer_tokens & _tokens("because mechanism 原理 因为 机制"))
            technical = bool(
                answer_tokens
                & _tokens(
                    "架构 设计 实现 方案 原理 机制 约束 边界 指标 验证 监控 故障 性能 延迟 并发 数据 流程"
                )
            )
            specificity = min(1.0, len(answer_tokens) / 35)
            # Project questions are subjective: a technically coherent answer
            # need not repeat the retrieved knowledge chunk vocabulary.
            correctness = min(1.0, 0.35 + 0.25 * float(technical) + 0.20 * question_overlap)
            completeness = min(1.0, 0.20 + 0.25 * float(technical) + 0.20 * specificity + 0.2 * tradeoff + 0.15 * mechanism)
            protocol = "project_scenario_reasoning"
            dimensions = {
                "technical_specificity": specificity,
                "technical_signal": float(technical),
                "mechanism": float(mechanism),
                "tradeoff": float(tradeoff),
            }
            reason = "Scored subjective technical reasoning, specificity, mechanisms, and trade-offs without requiring reference-chunk overlap."
        elif question_type is InterviewQuestionType.CASE:
            assumptions = bool(answer_tokens & _tokens("assumption 假设 约束 constraint"))
            alternatives = bool(answer_tokens & _tokens("option alternative 方案 备选"))
            measurement = bool(answer_tokens & _tokens("metric measure 指标 验证"))
            correctness = min(1.0, 0.25 + 0.3 * question_overlap + 0.15 * assumptions)
            completeness = min(1.0, 0.2 + 0.25 * assumptions + 0.25 * alternatives + 0.2 * measurement)
            protocol = "case_analysis"
            dimensions = {
                "assumptions": float(assumptions),
                "alternatives": float(alternatives),
                "measurement": float(measurement),
            }
            reason = "Scored case framing, alternatives, and validation criteria."
        elif question_type is InterviewQuestionType.BEHAVIORAL:
            situation = bool(answer_tokens & _tokens("situation background 场景 背景"))
            action = bool(answer_tokens & _tokens("action did 行动 负责"))
            result = bool(answer_tokens & _tokens("result outcome 结果 成果"))
            correctness = 0.5
            completeness = (float(situation) + float(action) + float(result)) / 3
            protocol = "behavioral_structure"
            dimensions = {
                "situation": float(situation),
                "action": float(action),
                "result": float(result),
            }
            reason = "Scored answer structure only; factual truth and personality were not inferred."
        else:
            correctness = min(1.0, 0.3 + question_overlap)
            completeness = min(1.0, 0.2 + min(0.8, len(answer_tokens) / 40))
            protocol = "open_relevance"
            dimensions = {"question_relevance": question_overlap, "detail": completeness}
            reason = "Scored relevance and explanatory detail for an open question."
        return InterviewAnswerAssessment(
            correctness=correctness,
            completeness=completeness,
            confidence=0.35,
            reason=reason + " Use a separately configured model evaluator for semantic scoring.",
            knowledge_refs=tuple(chunk.chunk_id for chunk in knowledge),
            evaluator_name=self.name,
            question_type=question_type,
            protocol=protocol,
            dimensions=dimensions,
            confidence_updates=tuple(
                {
                    "chunk_id": chunk.chunk_id,
                    "should_update": True,
                    "nc": (correctness + completeness) / 2,
                    "reason": "Deterministic reference-based mastery observation.",
                }
                for chunk in knowledge
            ),
        )


class InterviewQuestionGenerator(Protocol):
    name: str

    def generate(
        self,
        *,
        state: InterviewSessionState,
        decision: InterviewDecision,
        thread_id: str,
        knowledge: InterviewKnowledgeChunk,
        jd_requirement_refs: tuple[str, ...],
        resume_claim_refs: tuple[str, ...],
    ) -> InterviewQuestion: ...


def is_chinese_interview_question_text(text: str) -> bool:
    """Return True only when the question is actually phrased for a Chinese interview."""

    value = str(text or "").strip()
    if not re.search(r"[\u4e00-\u9fff]", value):
        return False
    # Keep English technical terms such as RAG, LangGraph and API latency, but
    # require the sentence frame itself to be Chinese.
    return bool(re.search(r"(请|你|如何|为什么|说明|分析|举例|结合|谈谈|解释|评价|追问|回答)", value))


def validate_interview_question_text(
    text: str,
    *,
    action: InterviewAction,
    question_type: InterviewQuestionType,
    topic: str,
) -> str:
    """Validate model output; never rewrite it into a predefined question."""

    value = str(text or "").strip()
    if not value:
        raise ValueError("Interview question generation failed: empty question_text")
    if len(value) > 1000:
        raise ValueError("Interview question generation failed: question is too long")
    if not is_chinese_interview_question_text(value):
        raise ValueError("Interview question generation failed: question must be natural Chinese")
    return value


@dataclass(frozen=True, slots=True)
class InterviewAnswerResult:
    state: InterviewSessionState
    assessment: InterviewAnswerAssessment
    decision: InterviewDecision
    idempotent_replay: bool = False

    def to_dict(self) -> dict:
        return {
            "interview_id": self.state.interview_id,
            "mode": self.state.mode.value,
            "status": self.state.status.value,
            "interview_stage": self.state.interview_stage.value,
            "active_line": self.state.active_line.value if self.state.active_line else None,
            "active_subject": self.state.active_subject,
            "intro_interest_updated": self.state.intro_interest_updated,
            "turn_count": len(self.state.turns),
            "assessment": self.assessment.to_dict(),
            "decision": self.decision.to_dict(),
            "idempotent_replay": self.idempotent_replay,
            "observation": (
                self.state.turns[-1].observation.to_dict()
                if self.state.turns and self.state.turns[-1].observation
                else None
            ),
            "belief_state": self.state.belief_state.to_dict(),
        }


class InterviewRuntime:
    """Runs one interview without reading parent chat or raw candidate materials."""

    def __init__(
        self,
        *,
        career_store: CareerStore,
        knowledge_store: InterviewKnowledgeStore,
        session_store: InterviewSessionStore,
        policy_engine: InterviewPolicyEngine | None = None,
        question_generator: InterviewQuestionGenerator | None = None,
        answer_evaluator: InterviewAnswerEvaluator | None = None,
        live_knowledge_retriever: LiveInterviewKnowledgeRetriever | None = None,
        mastery_store: KnowledgeMasteryStore | None = None,
        graph_knowledge_index=None,
        knowledge_library=None,
    ) -> None:
        self.career_store = career_store
        self.knowledge_store = knowledge_store
        self.session_store = session_store
        self.policy_engine = policy_engine or RecommendationInterviewPolicy()
        self.question_generator = question_generator
        self.answer_evaluator = answer_evaluator or LexicalInterviewAnswerEvaluator()
        self.live_knowledge_retriever = live_knowledge_retriever
        self.mastery_store = mastery_store
        self.graph_knowledge_index = graph_knowledge_index
        self.knowledge_library = knowledge_library
        self._last_graph_retrieval_trace: dict[str, object] = {}
        self.parallel_orchestrator = ParallelInterviewOrchestrator(
            self.answer_evaluator,
            self.question_generator,
        )
        self._draft_cache: dict[str, ParallelInterviewOutcome] = {}
        self._turn_context_cache: dict[str, InterviewTurnContext] = {}
        factory = getattr(graph_knowledge_index, "create_depth_prefetcher", None)
        self._depth_prefetcher = factory(max_workers=2, max_pending=4) if callable(factory) else None
        self._depth_prefetch_results: dict[str, tuple[tuple[InterviewKnowledgeChunk, ...], dict[str, object]]] = {}
        for value, method, label in (
            (self.policy_engine, "decide", "policy engine"),
            (self.question_generator, "generate", "question generator"),
            (self.answer_evaluator, "evaluate", "answer evaluator"),
        ):
            if value is None:
                continue
            if not callable(getattr(value, method, None)):
                raise ValueError(f"Interview {label} must expose {method}(...)")

    def create_session(
        self,
        *,
        mode: InterviewMode | str = InterviewMode.REALISTIC_SIMULATION,
        tenant_id: str = "default",
        jd_id: str | None = None,
        resume_document_id: str | None = None,
        resume_document_revision: int | None = None,
        resume_run_id: str | None = None,
        corpus_ids: tuple[str, ...] = (),
        random_seed: int = 0,
        max_turns: int = 8,
        max_followups_per_thread: int = 5,
        duration_minutes: int = 30,
    ) -> InterviewSessionState:
        selected_mode = InterviewMode(mode)
        tenant_id = str(tenant_id or "").strip()
        if not tenant_id:
            raise ValueError("tenant_id is required")
        corpus_ids = tuple(dict.fromkeys(str(item).strip() for item in corpus_ids if str(item).strip()))
        if not corpus_ids:
            raise ValueError("At least one interview knowledge corpus is required")
        corpora = []
        graph_versions: dict[str, str] = {}
        if selected_mode is InterviewMode.REALISTIC_SIMULATION and self.graph_knowledge_index is not None:
            if self.knowledge_library is None:
                raise ValueError("Graph RAG interview runtime requires knowledge source metadata")
            self.graph_knowledge_index.ensure_corpora(
                self.knowledge_library,
                corpus_ids,
                tenant_id=tenant_id,
            )
            graph_versions = self.graph_knowledge_index.corpus_versions(corpus_ids)
            missing = set(corpus_ids) - set(graph_versions)
            if missing:
                raise ValueError(
                    "Graph RAG index is not ready for interview corpora: "
                    + ", ".join(sorted(missing))
                )
        else:
            for corpus_id in corpus_ids:
                corpora.append(self.knowledge_store.get(corpus_id))
        jd: JDDocument | None = None
        resume: ResumeDocument | None = None
        existing_gap_mastery = False
        selected_run_id = str(resume_run_id or "").strip() or None
        if selected_run_id:
            run = self.career_store.get(selected_run_id, expected_type=ResumeRun)
            if run.status is ResumeRunStatus.DELETED:
                raise ValueError("Interview cannot bind to a deleted resume session")
            if jd_id and str(jd_id) != run.jd_ref.record_id:
                raise ValueError("Interview JD does not belong to the bound resume session")
            if resume_document_id and (
                run.resume_document_ref is None
                or str(resume_document_id) != run.resume_document_ref.record_id
            ):
                raise ValueError("Interview resume does not belong to the bound resume session")
            if resume_document_revision is not None and (
                run.resume_document_ref is None
                or int(resume_document_revision) != run.resume_document_ref.revision
            ):
                raise ValueError("Interview resume revision does not match the current resume version")
            jd_id = run.jd_ref.record_id
            if run.resume_document_ref is not None:
                # A current resume is an immutable id+revision reference when
                # the caller supplies the revision (the web UI does). Older
                # callers sent only an ID, so retain their latest-verified
                # behavior for backward compatibility.
                if resume_document_revision is None:
                    latest_resume = self.career_store.get(
                        run.resume_document_ref.record_id,
                        expected_type=ResumeDocument,
                    )
                    candidate_resume = latest_resume
                else:
                    candidate_resume = self.career_store.get(
                        run.resume_document_ref.record_id,
                        run.resume_document_ref.revision,
                        expected_type=ResumeDocument,
                    )
                    if candidate_resume.status is not ResumeDocumentStatus.VERIFIED:
                        latest_resume = self.career_store.get(
                            run.resume_document_ref.record_id,
                            expected_type=ResumeDocument,
                        )
                        if latest_resume.status is ResumeDocumentStatus.VERIFIED:
                            candidate_resume = latest_resume
                resume_document_id = (
                    candidate_resume.resume_document_id
                    if candidate_resume.status is ResumeDocumentStatus.VERIFIED
                    else None
                )
                resume_document_revision = (
                    candidate_resume.revision
                    if candidate_resume.status is ResumeDocumentStatus.VERIFIED
                    else None
                )
        if selected_mode is InterviewMode.REALISTIC_SIMULATION:
            if not str(jd_id or "").strip():
                raise ValueError("Realistic interview mode requires a JD")
            jd = self.career_store.get(str(jd_id), expected_type=JDDocument)
            if not resume_document_id:
                raise ValueError("Realistic interview mode requires a verified resume")
            resume = self.career_store.get(
                str(resume_document_id),
                resume_document_revision,
                expected_type=ResumeDocument,
            )
            if resume.status is not ResumeDocumentStatus.VERIFIED and not (
                resume.status is ResumeDocumentStatus.USER_PROVIDED and resume.resume_document_id.startswith("input:")
            ):
                raise ValueError("Realistic interview resume must be verified")
            if resume.jd_ref.record_id != jd.jd_id:
                raise ValueError("Realistic interview resume and JD must belong to the same application target")
            profile = self._select_profile(jd, resume, random_seed)
            initial_belief = KnowledgeBeliefState()
        else:
            if resume_document_id and not jd_id:
                raise ValueError("Gap-analysis resume requires a JD")
            if jd_id:
                jd = self.career_store.get(str(jd_id), expected_type=JDDocument)
            if resume_document_id:
                resume = self.career_store.get(
                    str(resume_document_id),
                    resume_document_revision,
                    expected_type=ResumeDocument,
                )
                if resume.status is not ResumeDocumentStatus.VERIFIED and not (
                    resume.status is ResumeDocumentStatus.USER_PROVIDED and resume.resume_document_id.startswith("input:")
                ):
                    raise ValueError("Gap-analysis resume must be verified")
                if jd is None or resume.jd_ref.record_id != jd.jd_id:
                    raise ValueError("Optional resume and JD must belong to the same application target")
            profile = self._select_gap_profile(random_seed)
            initial_belief = KnowledgeBeliefState()
            if self.mastery_store is not None:
                mastery_profile = self.mastery_store.get(tenant_id)
                initial_belief = mastery_profile.belief_state
                existing_gap_mastery = bool(mastery_profile.applied_interview_ids)
        policy = InterviewerPolicy(
            policy_id=f"{selected_mode.value}-{profile.profile_id}",
            max_followups_per_thread=max_followups_per_thread,
            depth_threshold=min(0.8, 0.52 + profile.pressure_level * 0.2),
            breadth_threshold=min(0.95, 0.70 + profile.breadth_bias * 0.2),
        )
        # A fresh session freezes corpus priors immediately.  A gap-analysis
        # session that already has mastery must preserve that exact snapshot;
        # its first block observation carries the corpus prior lazily so old
        # mastery JSON remains byte-for-byte compatible at session creation.
        has_existing_gap_mastery = bool(
            selected_mode is InterviewMode.GAP_ANALYSIS
            and (
                existing_gap_mastery
                or
                initial_belief.point_confidence_by_id
                or initial_belief.point_observation_count_by_id
                or initial_belief.document_confidence_by_corpus
                or initial_belief.document_observation_count_by_corpus
                or initial_belief.legacy_confidence_by_topic
                or initial_belief.legacy_observations_by_topic
            )
        )
        if selected_mode is not InterviewMode.GAP_ANALYSIS or not has_existing_gap_mastery:
            initial_belief = initial_belief.with_document_priors(
                self._knowledge_corpus_prior_counts(corpus_ids, graph_versions=graph_versions)
            )
        interview_points = self._build_interview_points(jd, resume) if jd and resume else ()
        if selected_mode is InterviewMode.REALISTIC_SIMULATION and not interview_points:
            raise ValueError("Realistic interview mode requires at least one verified resume interview point")
        active_line = self._first_available_line(interview_points)
        state = InterviewSessionState.create(
            mode=selected_mode,
            tenant_id=tenant_id,
            resume_run_id=selected_run_id,
            jd_id=jd.jd_id if jd else None,
            jd_revision=jd.revision if jd else None,
            resume_document_id=resume.resume_document_id if resume else None,
            resume_document_revision=resume.revision if resume else None,
            knowledge_corpus_ids=corpus_ids,
            knowledge_corpus_versions=(
                graph_versions
                if selected_mode is InterviewMode.REALISTIC_SIMULATION and self.graph_knowledge_index is not None
                else {corpus.corpus_id: corpus.version for corpus in corpora}
            ),
            profile=profile,
            policy=policy,
            belief_state=initial_belief,
            random_seed=random_seed,
            max_turns=max_turns,
            duration_minutes=duration_minutes,
            interview_points=interview_points,
            active_line=active_line,
            interview_stage=InterviewStage.INTRO if interview_points else InterviewStage.PURE_FOUNDATION,
        )
        if state.interview_points:
            state = replace(state, interview_points=self._cache_interview_point_knowledge(state))
        return self.session_store.create(state)

    def next_question(self, interview_id: str) -> InterviewQuestion:
        state = self.session_store.get(interview_id)
        if state.status in {InterviewStatus.COMPLETED, InterviewStatus.CANCELLED}:
            raise ValueError("Interview session has ended")
        if state.pending_question is not None:
            if state.interview_points and self._should_regenerate_structured_pending_question(state.pending_question):
                state = state._next(pending_question=None)
                self.session_store.put(state)
                return self._next_structured_interview_question(state)
            action = state.last_decision.action if state.last_decision else InterviewAction.ASK
            normalized_text = validate_interview_question_text(
                state.pending_question.text,
                action=action,
                question_type=state.pending_question.question_type,
                topic=state.pending_question.topic,
            )
            if normalized_text != state.pending_question.text:
                raise ValueError("Interview question generation failed: persisted question is invalid")
            return state.pending_question
        if state.interview_points:
            return self._next_structured_interview_question(state)

        if len(state.turns) >= state.max_turns:
            state = state.apply_decision(
                InterviewDecision(
                    action=InterviewAction.END,
                    path=InterviewPath.BREADTH,
                    topic=state.turns[-1].question.topic if state.turns else "interview",
                    reason="Configured interview turn limit reached",
                    thread_id=state.threads[-1].thread_id if state.threads else None,
                    stop_reason=InterviewStopReason.TURN_LIMIT,
                )
            )
            self.session_store.put(state)
            raise ValueError("Interview session has reached its turn limit")
        decision = state.last_decision
        selected_knowledge: InterviewKnowledgeChunk | None = None
        if decision is None:
            selected_knowledge = self._select_knowledge(state)
            decision = InterviewDecision(
                InterviewAction.ASK,
                InterviewPath.BREADTH,
                selected_knowledge.topic,
                "Start with the highest-ranked unobserved JD topic",
            )

        if decision.action is InterviewAction.END:
            raise ValueError("Interview session has ended")
        if decision.action is InterviewAction.FOLLOW_UP and decision.thread_id:
            thread_id = decision.thread_id
            knowledge = self._select_knowledge(state, topic=decision.topic)
        else:
            try:
                knowledge = selected_knowledge or self._select_knowledge(state)
            except ValueError as exc:
                if decision.action is InterviewAction.SWITCH_TOPIC:
                    state = state.apply_decision(
                        InterviewDecision(
                            InterviewAction.END,
                            InterviewPath.BREADTH,
                            decision.topic,
                            "No unused interview knowledge remains after switching topics",
                            decision.thread_id,
                            InterviewStopReason.KNOWLEDGE_EXHAUSTED,
                            decision.question_type,
                            decision.utility,
                        )
                    )
                    self.session_store.put(state)
                    raise ValueError("Interview session has ended: no unused interview knowledge remains") from exc
                raise
            state = state.add_thread(knowledge.topic, path=InterviewPath.BREADTH)
            thread_id = state.threads[-1].thread_id
            decision = InterviewDecision(
                InterviewAction.ASK if not state.turns else InterviewAction.SWITCH_TOPIC,
                InterviewPath.BREADTH,
                knowledge.topic,
                decision.reason,
                thread_id,
            )

        requirements, claims = self._artifact_links(state, knowledge)
        question_type = self._select_question_type(
            state,
            decision=decision,
            knowledge=knowledge,
            resume_claim_refs=claims,
        )
        utility = self._question_utility(
            state,
            decision=decision,
            knowledge=knowledge,
            jd_requirement_refs=requirements,
            resume_claim_refs=claims,
        )
        decision = replace(decision, question_type=question_type, utility=utility)
        if self.question_generator is None:
            raise ValueError("Interview question generation failed: interview_question model is unavailable")
        question = self.question_generator.generate(
            state=state,
            decision=decision,
            thread_id=thread_id,
            knowledge=knowledge,
            jd_requirement_refs=requirements,
            resume_claim_refs=claims,
        )
        parent = next(
            (
                turn.question
                for turn in reversed(state.turns)
                if turn.question.thread_id == thread_id
            ),
            None,
        )
        followup_depth = parent.followup_depth + 1 if decision.action is InterviewAction.FOLLOW_UP and parent else 0
        question = replace(
            question,
            text=validate_interview_question_text(
                question.text,
                action=decision.action,
                question_type=question_type,
                topic=knowledge.topic,
            ),
            parent_question_id=parent.question_id if followup_depth else None,
            followup_depth=followup_depth,
            question_type=question_type,
            utility=utility,
        )
        self._validate_generated_question(
            state,
            question,
            decision=decision,
            thread_id=thread_id,
            knowledge=knowledge,
            jd_requirement_refs=requirements,
            resume_claim_refs=claims,
        )
        state = state.set_pending_question(question)
        self.session_store.put(state)
        return question

    @staticmethod
    def _should_regenerate_structured_pending_question(question: InterviewQuestion) -> bool:
        if question.topic in {"自我介绍", "候选人反问", "面试结束"}:
            return False
        if not question.resume_claim_refs:
            return True
        text = str(question.text or "").strip()
        generic_prefixes = (
            "请解释 ",
            "继续追问 ",
            "结合目标岗位或你的项目场景，说明 ",
            "请分析一个和 ",
            "请举一个你处理 ",
            "你如何评价 ",
        )
        return text.startswith(generic_prefixes)

    def process_answer(
        self,
        interview_id: str,
        answer: str,
        *,
        expected_question_id: str | None = None,
    ) -> InterviewAnswerResult:
        """Evaluate one answer, replaying an already-persisted identical request safely."""
        expected_question_id = str(expected_question_id or "").strip() or None
        # Keep the displayed-question check and the replay lookup in the same
        # transaction as the answer commit. A browser can otherwise submit the
        # same visible question twice while the next question is being created.
        with self.session_store._lock:  # noqa: SLF001 - shared store transaction boundary
            state = self.session_store.get(interview_id)
            replay = self._replay_processed_answer(
                state,
                answer,
                expected_question_id=expected_question_id,
            )
            if replay is not None:
                return replay
            if expected_question_id and (
                state.pending_question is None
                or state.pending_question.question_id != expected_question_id
            ):
                raise ValueError("The displayed interview question is stale; reload before submitting")
            return self._submit_answer_unlocked(interview_id, answer, prepare_drafts=False)

    def _replay_processed_answer(
        self,
        state: InterviewSessionState,
        answer: str,
        *,
        expected_question_id: str | None,
    ) -> InterviewAnswerResult | None:
        """Return the durable result for a duplicate answer submission, if any."""
        if not state.turns:
            return None
        turn = state.turns[-1]
        is_displayed_answer = (
            expected_question_id is not None
            and turn.question.question_id == expected_question_id
        )
        is_legacy_duplicate = (
            expected_question_id is None
            and state.pending_question is None
        )
        if not (is_displayed_answer or is_legacy_duplicate):
            return None
        if " ".join(str(turn.answer).split()) != " ".join(str(answer).split()):
            raise ValueError("The displayed interview question was already answered; reload before submitting")

        audit = turn.orchestration_audit or {}
        evaluation = (
            audit.get("lanes", {})
            .get("evaluation", {})
            .get("output", {})
        )
        assessment = InterviewAnswerAssessment(
            correctness=turn.correctness,
            completeness=turn.completeness,
            confidence=turn.confidence,
            reason=turn.evaluation_reason,
            knowledge_refs=turn.evaluation_refs,
            evaluator_name=str(evaluation.get("evaluator_name") or "persisted-interview-turn"),
            question_type=turn.question.question_type,
            protocol=str(evaluation.get("protocol") or "persisted_interview_turn"),
            dimensions=dict(evaluation.get("dimensions") or {}),
            information_gain=str(evaluation.get("information_gain") or ""),
            verification_need=str(evaluation.get("verification_need") or ""),
            confidence_updates=tuple(evaluation.get("confidence_updates") or ()),
        )
        decision_payload = audit.get("final_decision")
        if isinstance(decision_payload, dict):
            decision = InterviewDecision.from_dict(decision_payload)
        elif state.last_decision is not None:
            decision = state.last_decision
        else:
            raise ValueError("Interview answer was already processed; reload before submitting")
        return InterviewAnswerResult(state, assessment, decision, idempotent_replay=True)

    def next_question_candidates(self, interview_id: str) -> dict[str, Any]:
        state = self.session_store.get(interview_id)
        return self._next_question_candidates_payload(state)

    def generate_depth_question(
        self,
        interview_id: str,
        *,
        candidate_knowledge_refs: tuple[str, ...],
    ) -> InterviewQuestion:
        state = self.session_store.get(interview_id)
        if state.pending_question is not None:
            if state.pending_question.path is InterviewPath.DEPTH and tuple(state.pending_question.knowledge_refs) == tuple(candidate_knowledge_refs):
                return state.pending_question
            raise ValueError("Resolve the current pending question before generating a depth question")
        payload = self._next_question_candidates_payload(state)
        if payload.get("path") != InterviewPath.DEPTH.value:
            raise ValueError("The latest answer decision did not authorize a depth question")
        authorized = tuple(dict.fromkeys(
            block["chunk_id"]
            for block in payload.get("selected_blocks", ())
            if isinstance(block, dict)
        ))
        requested = tuple(dict.fromkeys(str(item).strip() for item in candidate_knowledge_refs if str(item).strip()))
        if requested != authorized:
            raise ValueError("Depth question must use exactly the answer-processing API's selected depth candidates")
        return self._next_structured_interview_question(state)

    def generate_breadth_question(
        self,
        interview_id: str,
        *,
        candidate_knowledge_refs: tuple[str, ...] = (),
    ) -> InterviewQuestion:
        state = self.session_store.get(interview_id)
        if state.pending_question is not None:
            if state.pending_question.path is InterviewPath.BREADTH:
                return state.pending_question
            raise ValueError("Resolve the current pending question before generating a breadth question")
        payload = self._next_question_candidates_payload(state)
        if payload.get("path") != InterviewPath.BREADTH.value:
            raise ValueError("The latest answer decision did not authorize a breadth question")
        authorized = tuple(dict.fromkeys(
            block["chunk_id"]
            for block in payload.get("selected_blocks", ())
            if isinstance(block, dict)
        ))
        requested = tuple(dict.fromkeys(str(item).strip() for item in candidate_knowledge_refs if str(item).strip()))
        if requested and requested != authorized:
            raise ValueError("Breadth question must use the answer-processing API's selected B1 candidate")
        return self._next_structured_interview_question(state)

    def _build_turn_context(
        self,
        state: InterviewSessionState,
        *,
        question: InterviewQuestion,
        answer: str,
        evaluation_references: tuple[InterviewKnowledgeChunk, ...],
    ) -> InterviewTurnContext:
        point = self._point_for_question(state, question)
        depth_candidates: tuple[InterviewKnowledgeChunk, ...] = ()
        breadth_candidates: tuple[InterviewKnowledgeChunk, ...] = ()
        breadth_subject = ""
        breadth_display_subject = ""
        current_point_id = point.point_id if point else state.current_point_id
        current_subject = point.point if point else question.topic
        display_subject = point.subject if point else question.topic
        if point is not None:
            try:
                depth_blocks = self._depth_block_refs_after_question(state, point, question)
                depth_candidates = self._knowledge_by_ids(
                    state,
                    tuple(dict.fromkeys(block.chunk_id for block in depth_blocks)),
                )
            except (KeyError, ValueError):
                depth_candidates = ()
            breadth_point = self._projected_breadth_point_after_answer(state, point)
            if breadth_point is not None:
                breadth_subject = breadth_point.point
                breadth_display_subject = breadth_point.subject
                try:
                    breadth_blocks = self._breadth_block_refs_for_point(
                        breadth_point,
                        role=InterviewKnowledgeBlockRole.B,
                    )
                    breadth_candidates = self._knowledge_by_ids(
                        state,
                        tuple(block.chunk_id for block in breadth_blocks),
                    )
                except (KeyError, ValueError):
                    breadth_candidates = ()
        recent = tuple(
            {
                "question": turn.question.text,
                "answer": turn.answer,
                "topic": turn.question.topic,
            }
            for turn in state.turns[-4:]
        )
        current_corpus = (
            depth_candidates[0].corpus_id
            if depth_candidates
            else evaluation_references[0].corpus_id
            if evaluation_references
            else None
        )
        current_knowledge_id = question.knowledge_refs[0] if question.knowledge_refs else None
        current_confidence = state.belief_state.point_confidence(current_knowledge_id) if current_knowledge_id else 0.5
        current_observations = (
            state.belief_state.point_observation_count(current_knowledge_id)
            if current_knowledge_id
            else 0
        )
        current_document_confidence = (
            state.belief_state.document_confidence(current_corpus)
            if current_corpus
            else 0.5
        )
        current_document_coverage = (
            state.belief_state.document_coverage(current_corpus)
            if current_corpus
            else 0.0
        )
        return InterviewTurnContext(
            session_id=state.interview_id,
            previous_question=question,
            user_answer=answer,
            current_subject=current_subject,
            display_subject=display_subject,
            current_point_id=current_point_id,
            current_corpus_id=current_corpus,
            current_confidence=current_confidence,
            current_observation_count=current_observations,
            current_document_confidence=current_document_confidence,
            current_document_coverage=current_document_coverage,
            previous_knowledge_refs=question.knowledge_refs,
            evaluation_references=evaluation_references,
            depth_candidates=depth_candidates,
            breadth_candidates=breadth_candidates,
            breadth_subject=breadth_subject,
            breadth_display_subject=breadth_display_subject,
            recent_turns=recent,
            summary="; ".join(turn.question.topic for turn in state.turns[-8:]),
            allowed_corpus_ids=state.knowledge_corpus_ids,
            depth_chain=state.active_depth_chain_blocks,
        )

    def _start_depth_prefetch(self, state: InterviewSessionState, question: InterviewQuestion) -> None:
        if not self._uses_graph_knowledge(state) or not state.active_depth_chain_blocks or self._depth_prefetcher is None:
            return
        try:
            self._depth_prefetcher.start(
                question.question_id,
                active_depth_chain=state.active_depth_chain_blocks,
                current_question=question.text,
                used_node_ids=set(state.used_knowledge_point_ids),
                corpus_ids=state.knowledge_corpus_ids,
                top_k=2,
            )
        except (RuntimeError, ValueError):
            # Prefetch is an optional latency optimization. The deterministic
            # retrieval path remains available if the bounded queue is full.
            return

    def _prefetched_depth_candidates(
        self,
        state: InterviewSessionState,
        question: InterviewQuestion,
        *,
        answer_hint: str = "",
    ) -> tuple[tuple[InterviewKnowledgeChunk, ...], str]:
        cached = self._depth_prefetch_results.get(question.question_id)
        if cached is not None:
            chunks, trace = cached
            self._last_graph_retrieval_trace = dict(trace)
            return chunks, "ready"
        if self._depth_prefetcher is None:
            return (), "unavailable"
        outcome = self._depth_prefetcher.finish(
            question.question_id,
            answer_hint=answer_hint,
            timeout_seconds=0,
        )
        if outcome.result is None:
            return (), outcome.error
        chunks = self.graph_knowledge_index.result_chunks(outcome.result)
        trace = self.graph_knowledge_index.trace_dict(outcome.result)
        trace["prefetch"] = "ready"
        self._depth_prefetch_results[question.question_id] = (chunks, trace)
        self._last_graph_retrieval_trace = dict(trace)
        return chunks, "ready"

    @staticmethod
    def _audit_chunks(chunks: tuple[InterviewKnowledgeChunk, ...]) -> list[dict[str, Any]]:
        return [chunk.to_dict() for chunk in chunks]

    @classmethod
    def _parallel_turn_audit(
        cls,
        context: InterviewTurnContext,
        outcome: ParallelInterviewOutcome,
        assessment: InterviewAnswerAssessment,
    ) -> dict[str, Any]:
        def draft_payload(draft: InterviewQuestionDraft | None) -> dict[str, Any] | None:
            if draft is None:
                return None
            return {
                "text": draft.text,
                "path": draft.path.value,
                "topic": draft.topic,
                "knowledge_refs": list(draft.knowledge_refs),
                "point_id": draft.point_id,
                "ok": draft.ok,
                "error_code": draft.error_code,
            }

        shared_input = {
            "session_id": context.session_id,
            "previous_question": context.previous_question.to_dict(),
            "user_answer": context.user_answer,
            "current_subject": context.current_subject,
            "current_point_id": context.current_point_id,
            "current_corpus_id": context.current_corpus_id,
            "current_confidence": context.current_confidence,
            "current_observation_count": context.current_observation_count,
            "current_document_confidence": context.current_document_confidence,
            "current_document_coverage": context.current_document_coverage,
            "recent_turns": [dict(item) for item in context.recent_turns],
            "summary": context.summary,
            "allowed_corpus_ids": list(context.allowed_corpus_ids),
            "depth_chain": list(context.depth_chain),
        }
        return {
            "version": 1,
            "mode": "parallel",
            "timings_ms": {name: round(float(value), 3) for name, value in outcome.timings_ms.items()},
            "errors": dict(outcome.errors),
            "lanes": {
                "evaluation": {
                    "input": {**shared_input, "knowledge_chunks": cls._audit_chunks(context.evaluation_references)},
                    "output": assessment.to_dict(),
                    "provider_output": outcome.assessment.to_dict() if outcome.assessment else None,
                    "fallback_used": outcome.assessment is None,
                    "error": outcome.errors.get("evaluation"),
                    "elapsed_ms": round(float(outcome.timings_ms.get("evaluation", 0.0)), 3),
                },
                "depth": {
                    "input": {**shared_input, "knowledge_chunks": cls._audit_chunks(context.depth_candidates)},
                    "output": draft_payload(outcome.depth),
                    "error": outcome.errors.get("depth"),
                    "elapsed_ms": round(float(outcome.timings_ms.get("depth", 0.0)), 3),
                },
                "breadth": {
                    "input": {
                        **shared_input,
                        "breadth_subject": context.breadth_subject,
                        "knowledge_chunks": cls._audit_chunks(context.breadth_candidates),
                    },
                    "output": draft_payload(outcome.breadth),
                    "error": outcome.errors.get("breadth"),
                    "elapsed_ms": round(float(outcome.timings_ms.get("breadth", 0.0)), 3),
                },
            },
        }

    @staticmethod
    def _direct_evaluation_audit(
        question: InterviewQuestion,
        answer: str,
        knowledge: tuple[InterviewKnowledgeChunk, ...],
        assessment: InterviewAnswerAssessment,
        elapsed_ms: float,
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "mode": "direct_evaluation",
            "timings_ms": {"evaluation": round(elapsed_ms, 3), "total": round(elapsed_ms, 3)},
            "errors": {},
            "lanes": {
                "evaluation": {
                    "input": {
                        "previous_question": question.to_dict(),
                        "user_answer": answer,
                        "knowledge_chunks": [chunk.to_dict() for chunk in knowledge],
                    },
                    "output": assessment.to_dict(),
                    "error": None,
                    "elapsed_ms": round(elapsed_ms, 3),
                },
            },
        }

    def submit_answer(
        self,
        interview_id: str,
        answer: str,
        *,
        prepare_drafts: bool = True,
    ) -> InterviewAnswerResult:
        # Keep the pending-question check, evaluator call, and revision commit
        # atomic for one SessionStore. This prevents concurrent duplicate turns.
        with self.session_store._lock:  # noqa: SLF001 - shared store transaction boundary
            return self._submit_answer_unlocked(interview_id, answer, prepare_drafts=prepare_drafts)

    def _submit_answer_unlocked(
        self,
        interview_id: str,
        answer: str,
        *,
        prepare_drafts: bool = True,
    ) -> InterviewAnswerResult:
        state = self.session_store.get(interview_id)
        if state.pending_question is None:
            raise ValueError("Generate an interview question before submitting an answer")
        question = state.pending_question
        if state.interview_points and self._is_intro_question(state, question):
            return self._submit_intro_answer(state, question, answer)

        knowledge = self._knowledge_by_ids(state, question.knowledge_refs)
        parallel_outcome: ParallelInterviewOutcome | None = None
        orchestration_audit: dict[str, Any] | None = None
        if state.interview_points:
            context = self._build_turn_context(
                state,
                question=question,
                answer=answer,
                evaluation_references=knowledge,
            )
            if prepare_drafts:
                parallel_outcome = self.parallel_orchestrator.run(context)
                assessment = parallel_outcome.assessment
                if assessment is None:
                    if "evaluation" in parallel_outcome.errors and getattr(self.answer_evaluator, "name", "") == "timeout-evaluator":
                        raise TimeoutError("evaluation timed out")
                    # Never serialize a failed lane into a second model request;
                    # use the deterministic local evaluator for this turn.
                    assessment = LexicalInterviewAnswerEvaluator().evaluate(
                        question=question,
                        answer=answer,
                        knowledge=knowledge,
                    )
                self._draft_cache[interview_id] = parallel_outcome
                self._turn_context_cache[interview_id] = context
                orchestration_audit = self._parallel_turn_audit(context, parallel_outcome, assessment)
            else:
                evaluation_started = time.perf_counter()
                assessment = self.answer_evaluator.evaluate(
                    question=question,
                    answer=answer,
                    knowledge=knowledge,
                    context=context,
                ) if "context" in inspect.signature(self.answer_evaluator.evaluate).parameters else self.answer_evaluator.evaluate(
                    question=question,
                    answer=answer,
                    knowledge=knowledge,
                )
                self._turn_context_cache[interview_id] = context
        else:
            evaluation_started = time.perf_counter()
            assessment = self.answer_evaluator.evaluate(
                question=question,
                answer=answer,
                knowledge=knowledge,
            )
        if not isinstance(assessment, InterviewAnswerAssessment):
            raise ValueError("Interview answer evaluator must return InterviewAnswerAssessment")
        if assessment.question_type is not question.question_type:
            raise ValueError("Interview answer evaluator changed the question evaluation protocol")
        confidence_updates = self._validated_confidence_updates(
            state=state,
            question=question,
            knowledge=knowledge,
            assessment=assessment,
        )
        if orchestration_audit is None:
            orchestration_audit = self._direct_evaluation_audit(
                question,
                answer,
                knowledge,
                assessment,
                (time.perf_counter() - evaluation_started) * 1000,
            )
        observation = self._observe_answer(
            state,
            question=question,
            answer=answer,
            knowledge=knowledge,
            assessment=assessment,
        )
        turn = InterviewTurn(
            question=question,
            answer=answer,
            correctness=assessment.correctness,
            completeness=assessment.completeness,
            confidence=assessment.confidence,
            evaluation_reason=assessment.reason,
            evaluation_refs=assessment.knowledge_refs,
            knowledge_confidence_updates=confidence_updates,
            observation=observation,
            orchestration_audit=orchestration_audit,
        )
        state = state.record_turn(turn)
        if state.interview_points:
            decision = self._structured_decision_after_answer(
                state,
                question=question,
                observation=observation,
                assessment=assessment,
            )
        elif len(state.turns) >= state.max_turns:
            decision = InterviewDecision(
                InterviewAction.END,
                question.path,
                question.topic,
                "Configured interview turn limit reached",
                question.thread_id,
                InterviewStopReason.TURN_LIMIT,
                question.question_type,
            )
        else:
            recommendation_utility = self._recommendation_utility(
                state,
                question=question,
                observation=observation,
                assessment=assessment,
            )
            decision = self._decide_policy(
                state,
                topic=question.topic,
                correctness=assessment.correctness,
                completeness=assessment.completeness,
                observation=observation,
                utility=recommendation_utility,
            )
            if (
                decision.action is InterviewAction.SWITCH_TOPIC
                and not self._has_unused_knowledge(state)
            ):
                decision = replace(
                    decision,
                    action=InterviewAction.END,
                    reason="All authorized interview knowledge points have been consumed",
                    stop_reason=InterviewStopReason.KNOWLEDGE_EXHAUSTED,
                )
        if decision.stop_reason is not None:
            observation = replace(observation, stop_reason=decision.stop_reason)
            turn = replace(turn, observation=observation)
            state = replace(state, turns=(*state.turns[:-1], turn))
        if state.turns and state.turns[-1].orchestration_audit is not None:
            audit = dict(state.turns[-1].orchestration_audit)
            audit["final_decision"] = decision.to_dict()
            turn = replace(state.turns[-1], orchestration_audit=audit)
            state = replace(state, turns=(*state.turns[:-1], turn))
        if state.interview_points:
            state = self._mark_structured_point_after_answer(state, question=question, decision=decision)
        state = state.apply_decision(decision)
        state = self.session_store.put(state)
        if decision.action is InterviewAction.END:
            self._draft_cache.pop(interview_id, None)
            self._turn_context_cache.pop(interview_id, None)
        self._update_mastery_if_completed(state)
        return InterviewAnswerResult(state, assessment, decision)

    def _decide_policy(self, state: InterviewSessionState, **kwargs: object) -> InterviewDecision:
        """Keep older injected policies usable while exposing recommendation signals."""
        parameters = inspect.signature(self.policy_engine.decide).parameters
        if "utility" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            return self.policy_engine.decide(state, **kwargs)
        kwargs.pop("utility", None)
        return self.policy_engine.decide(state, **kwargs)

    @staticmethod
    def _recommendation_utility(
        state: InterviewSessionState,
        *,
        question: InterviewQuestion,
        observation: AnswerObservation,
        assessment: InterviewAnswerAssessment,
    ) -> QuestionUtility:
        base = question.utility or QuestionUtility(0, 0, 0.5, 0.5, 0.5, 0, 0, 0.5)
        confidence = (
            state.belief_state.point_confidence(question.knowledge_refs[0])
            if question.knowledge_refs
            else 0.5
        )
        interest = max(0.1, min(1.0, max(base.jd_relevance, base.resume_relevance, base.score)))
        verification_need = max(
            0.0,
            min(1.0, 0.6 * (1.0 - assessment.correctness) + 0.4 * (1.0 - assessment.confidence)),
        )
        consecutive_depth = 0
        for turn in reversed(state.turns):
            if turn.question.path is not InterviewPath.DEPTH:
                break
            consecutive_depth += 1
        return replace(
            base,
            knowledge_uncertainty=0.5,
            weakness=0.5,
            interest_factor=interest,
            confidence=confidence,
            information_gain=observation.new_information_ratio,
            verification_need=verification_need,
            consecutive_depth=consecutive_depth,
            breadth_weight=0.0,
            depth_weight=0.0,
        )

    def finish(self, interview_id: str, *, cancelled: bool = False) -> InterviewSessionState:
        state = self.session_store.get(interview_id)
        state = state.finish(cancelled=cancelled)
        state = self.session_store.put(state)
        self._update_mastery_if_completed(state)
        return state

    @staticmethod
    def _is_retryable_interview_error(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        if not isinstance(exc, RuntimeError):
            return False
        message = str(exc).lower()
        return any(marker in message for marker in (
            "network error",
            "connection",
            "timed out",
            "timeout",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        ))

    def _record_retry_event(self, interview_id: str, event: dict[str, Any]) -> InterviewSessionState:
        with self.session_store._lock:  # noqa: SLF001 - keep retry audit append atomic with the session revision
            state = self.session_store.get(interview_id)
            return self.session_store.put(state.record_retry_event(event))

    def retry_submit_answer(
        self,
        interview_id: str,
        answer: str,
        *,
        expected_revision: int | None = None,
        max_attempts: int = 2,
        retry_backoff_seconds: float = 0.15,
    ) -> InterviewAnswerResult:
        """Retry only transient answer-processing failures without duplicating turns."""
        attempts = max(1, min(int(max_attempts), 3))
        backoff = max(0.0, min(float(retry_backoff_seconds), 2.0))
        initial = self.session_store.get(interview_id)
        if initial.status in {InterviewStatus.COMPLETED, InterviewStatus.CANCELLED}:
            raise ValueError("Cannot retry an ended interview session")
        if initial.pending_question is None:
            raise ValueError("Cannot retry because the pending question was already answered")
        if expected_revision is not None and initial.revision != int(expected_revision):
            raise ValueError("Interview session revision changed; reload before retrying")
        question_id = initial.pending_question.question_id
        previous_attempts = sum(
            1
            for event in initial.retry_events
            if event.get("operation") == "answer" and event.get("question_id") == question_id
        )
        if previous_attempts >= 3:
            raise ValueError("Retry limit reached for the current interview question")

        last_error: Exception | None = None
        for local_attempt in range(1, attempts + 1):
            started = time.perf_counter()
            try:
                result = self.submit_answer(interview_id, answer)
            except Exception as exc:  # noqa: BLE001 - preserve original provider/runtime failure after audit
                retryable = self._is_retryable_interview_error(exc)
                last_error = exc
                self._record_retry_event(
                    interview_id,
                    {
                        "operation": "answer",
                        "question_id": question_id,
                        "attempt": previous_attempts + local_attempt,
                        "status": "failed",
                        "retryable": retryable,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                        "created_at": time.time(),
                    },
                )
                if not retryable or local_attempt >= attempts:
                    raise
                time.sleep(backoff * (2 ** (local_attempt - 1)))
                continue
            updated_state = self._record_retry_event(
                interview_id,
                {
                    "operation": "answer",
                    "question_id": question_id,
                    "attempt": previous_attempts + local_attempt,
                    "status": "succeeded",
                    "retryable": True,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                    "created_at": time.time(),
                },
            )
            return replace(result, state=updated_state)
        assert last_error is not None
        raise last_error

    def report(self, interview_id: str) -> dict:
        state = self.session_store.get(interview_id)
        topic_scores: dict[str, list[float]] = {}
        topic_refs: dict[str, set[str]] = {}
        for turn in state.turns:
            topic_scores.setdefault(turn.question.topic, []).append(
                (turn.correctness + turn.completeness) / 2
            )
            topic_refs.setdefault(turn.question.topic, set()).update(turn.evaluation_refs)
        summaries = [
            {
                "topic": topic,
                "score": sum(scores) / len(scores),
                "observations": len(scores),
                "knowledge_refs": sorted(topic_refs.get(topic, set())),
            }
            for topic, scores in sorted(topic_scores.items())
        ]
        overall = sum(item["score"] for item in summaries) / len(summaries) if summaries else 0.0
        weakest = sorted(summaries, key=lambda item: (item["score"], item["topic"]))[:3]
        mastery = self.mastery_store.get(state.tenant_id) if self.mastery_store is not None else None
        question_results = [self._question_result(state, turn) for turn in state.turns]
        return {
            "interview_id": state.interview_id,
            "mode": state.mode.value,
            "tenant_id": state.tenant_id,
            "status": state.status.value,
            "fixed_profile": {
                "profile_id": state.profile.profile_id,
                "name": state.profile.name,
                "temperament": state.profile.temperament,
                "seniority": state.profile.seniority,
                "pressure_level": state.profile.pressure_level,
                "breadth_bias": state.profile.breadth_bias,
            },
            "trace": {
                "resume_run_id": state.resume_run_id,
                "jd": [state.jd_id, state.jd_revision] if state.jd_id else None,
                "verified_resume": (
                    [state.resume_document_id, state.resume_document_revision]
                    if state.resume_document_id
                    else None
                ),
                "knowledge_corpora": list(state.knowledge_corpus_ids),
                "knowledge_access": (
                    "live_tenant_rag"
                    if state.mode is InterviewMode.GAP_ANALYSIS
                    else "frozen_interview_pack"
                ),
                "knowledge_corpus_versions": state.knowledge_corpus_versions,
                "random_seed": state.random_seed,
                "policy_engine": getattr(self.policy_engine, "name", type(self.policy_engine).__name__),
                "question_generator": getattr(self.question_generator, "name", type(self.question_generator).__name__),
                "answer_evaluator": getattr(self.answer_evaluator, "name", type(self.answer_evaluator).__name__),
                "final_stop_reason": state.final_stop_reason.value if state.final_stop_reason else None,
                "last_decision": state.last_decision.to_dict() if state.last_decision else None,
                "structured_flow": {
                    "interview_stage": state.interview_stage.value,
                    "active_line": state.active_line.value if state.active_line else None,
                    "active_subject": state.active_subject,
                    "duration_minutes": state.duration_minutes,
                    "max_questions_per_subject": state.max_questions_per_subject,
                    "max_questions_per_point": state.max_questions_per_point,
                    "max_pure_foundation_questions": state.max_pure_foundation_questions,
                    "candidate_question_wait_seconds": state.candidate_question_wait_seconds,
                    "subject_question_counts": state.subject_question_counts,
                    "pure_foundation_question_count": state.pure_foundation_question_count,
                    "intro_interest_updated": state.intro_interest_updated,
                },
                "recommendation_traces": [
                    recommendation_trace.to_dict()
                    for recommendation_trace in state.recommendation_traces
                ],
                "question_chain": [
                    {
                        "question_id": turn.question.question_id,
                        "parent_question_id": turn.question.parent_question_id,
                        "followup_depth": turn.question.followup_depth,
                        "question_type": turn.question.question_type.value,
                        "utility": turn.question.utility.to_dict() if turn.question.utility else None,
                        "answer_observation": turn.observation.to_dict() if turn.observation else None,
                    }
                    for turn in state.turns
                ],
            },
            "interviewer_assessment": {
                "subjective": True,
                "overall_score": overall,
                "summary": self._score_label(overall) if summaries else "insufficient_evidence",
                "turn_count": len(state.turns),
                "protocols": [turn.question.question_type.value for turn in state.turns],
            },
            "career_learning_review": {
                "topics": summaries,
                "priority_topics": [item["topic"] for item in weakest],
                "boundary": "Scores estimate professional knowledge answers, not project ownership or truth.",
            },
            "question_results": question_results,
            "career_mastery_profile": mastery.to_dict() if mastery is not None else None,
        }

    def _question_result(self, state: InterviewSessionState, turn) -> dict:
        knowledge = self._knowledge_by_ids(state, turn.question.knowledge_refs)
        reference_texts = tuple(dict.fromkeys(chunk.text.strip() for chunk in knowledge if chunk.text.strip()))
        expected_points = self._expected_points(turn.question.question_type, reference_texts)
        covered_count = min(len(expected_points), round(len(expected_points) * turn.completeness))
        covered = expected_points[:covered_count]
        missing = expected_points[covered_count:]
        if turn.correctness < 0.5:
            missing = ["回答存在需要对照参考资料纠正的知识性内容", *missing]
        has_personal_evidence = bool(turn.question.resume_claim_refs)
        is_subjective = turn.question.question_type is not InterviewQuestionType.KNOWLEDGE
        is_open = turn.question.question_type in {
            InterviewQuestionType.BEHAVIORAL,
            InterviewQuestionType.CASE,
            InterviewQuestionType.PROJECT_SCENARIO,
            InterviewQuestionType.OPEN,
        }
        reference_answer = "" if is_subjective else "\n\n".join(reference_texts)
        improved_answer = (
            self._open_answer_framework(turn.question.question_type, expected_points)
            if is_open
            else reference_answer
        )
        return {
            "question_id": turn.question.question_id,
            "question": turn.question.text,
            "question_type": turn.question.question_type.value,
            "jd_requirement_ids": list(turn.question.jd_requirement_refs),
            "personal_evidence_status": (
                "matched" if has_personal_evidence else "not_found_in_personal_materials"
            ),
            "personal_evidence_refs": list(turn.question.resume_claim_refs),
            "reference_answer": reference_answer,
            "expected_points": expected_points,
            "user_answer": turn.answer,
            "covered_points": covered,
            "missing_or_wrong_points": missing,
            "evaluation": {
                "correctness": turn.correctness,
                "completeness": turn.completeness,
                "confidence": turn.confidence,
                "reason": turn.evaluation_reason,
            },
            "improved_answer": improved_answer,
            "answer_source_type": (
                "personal_evidence_and_general_knowledge"
                if has_personal_evidence
                else "general_knowledge"
            ),
            "knowledge_refs": list(turn.evaluation_refs),
            "disclaimer": (
                "该题关联已确认的个人简历事实；参考答案仍只用于知识复盘。"
                if has_personal_evidence
                else "未在个人材料库检索到相关信息；参考内容来自通用面试知识，不代表用户具有相关项目经历。"
            ),
        }

    def _next_structured_interview_question(self, state: InterviewSessionState) -> InterviewQuestion:
        self._validate_knowledge_snapshot(state)
        if state.interview_stage is InterviewStage.COMPLETED:
            raise ValueError("Interview session has ended")
        if state.interview_stage is InterviewStage.INTRO:
            return self._next_intro_question(state)
        if state.interview_stage is InterviewStage.CANDIDATE_QUESTION_WAIT:
            return self._complete_interview(state)
        return self._next_resume_point_question(state)

    def _next_intro_question(self, state: InterviewSessionState) -> InterviewQuestion:
        if state.turns:
            state = state._next(interview_stage=self._first_structured_stage(state))
            self.session_store.put(state)
            return self._next_structured_interview_question(state)
        state = state.add_thread("自我介绍", path=InterviewPath.BREADTH)
        thread = state.threads[-1]
        question = InterviewQuestion(
            question_id="question-" + hashlib.sha256(f"{state.interview_id}:intro".encode()).hexdigest()[:16],
            text="请先做一个简短的自我介绍，重点说你的实习、项目和技术方向。",
            topic="自我介绍",
            path=InterviewPath.BREADTH,
            thread_id=thread.thread_id,
            question_type=InterviewQuestionType.OPEN,
        )
        decision = InterviewDecision(
            InterviewAction.ASK,
            InterviewPath.BREADTH,
            "自我介绍",
            "固定第一题：自我介绍。",
            thread.thread_id,
            question_type=InterviewQuestionType.OPEN,
        )
        state = state._next(last_decision=decision)
        state = state.set_pending_question(question)
        self.session_store.put(state)
        return question

    def _complete_interview(self, state: InterviewSessionState) -> InterviewQuestion:
        """Persist completion immediately after the final structured answer.

        Candidate reverse questions are intentionally outside this interview
        flow. The legacy wait stage remains readable for old sessions, but it
        is now an internal completion sentinel rather than a prompt.
        """
        thread_id = state.threads[-1].thread_id if state.threads else None
        decision = InterviewDecision(
            InterviewAction.END,
            InterviewPath.BREADTH,
            "面试结束",
            "所有面试模块已完成，不追加候选人反问。",
            thread_id,
            InterviewStopReason.USER_FINISHED,
            InterviewQuestionType.OPEN,
        )
        state = state.apply_decision(decision)
        state = state._next(
            interview_stage=InterviewStage.COMPLETED,
            pending_question=None,
            candidate_question_started_at=None,
        )
        state = self.session_store.put(state)
        self._update_mastery_if_completed(state)
        return InterviewQuestion(
            question_id="question-" + hashlib.sha256(f"{state.interview_id}:completed".encode()).hexdigest()[:16],
            text="本次面试结束，后续等通知。",
            topic="面试结束",
            path=InterviewPath.BREADTH,
            thread_id=thread_id or "thread-completed",
            question_type=InterviewQuestionType.OPEN,
        )

    def _legacy_candidate_wait_message(self, state: InterviewSessionState) -> InterviewQuestion:
        # Compatibility-only entry point: legacy callers also finish immediately.
        return self._complete_interview(state)

        started_at = state.candidate_question_started_at
        if started_at is None:
            state = state.add_thread("候选人反问", path=InterviewPath.BREADTH)
            thread = state.threads[-1]
            state = state._next(candidate_question_started_at=time.time())
            self.session_store.put(state)
            return InterviewQuestion(
                question_id="question-" + hashlib.sha256(f"{state.interview_id}:candidate-question".encode()).hexdigest()[:16],
                text="你有什么想问我的？",
                topic="候选人反问",
                path=InterviewPath.BREADTH,
                thread_id=thread.thread_id,
                question_type=InterviewQuestionType.OPEN,
            )
        if time.time() - started_at < state.candidate_question_wait_seconds:
            thread = state.threads[-1] if state.threads else state.add_thread("候选人反问").threads[-1]
            return InterviewQuestion(
                question_id="question-" + hashlib.sha256(f"{state.interview_id}:candidate-question".encode()).hexdigest()[:16],
                text="你有什么想问我的？",
                topic="候选人反问",
                path=InterviewPath.BREADTH,
                thread_id=thread.thread_id,
                question_type=InterviewQuestionType.OPEN,
            )
        thread_id = state.threads[-1].thread_id if state.threads else None
        decision = InterviewDecision(
            InterviewAction.END,
            InterviewPath.BREADTH,
            "面试结束",
            "候选人反问等待时间已到，输出固定结束话术。",
            thread_id,
            InterviewStopReason.USER_FINISHED,
            InterviewQuestionType.OPEN,
        )
        state = state.apply_decision(decision)
        state = state._next(interview_stage=InterviewStage.COMPLETED, pending_question=None)
        state = self.session_store.put(state)
        self._update_mastery_if_completed(state)
        return InterviewQuestion(
            question_id="question-" + hashlib.sha256(f"{state.interview_id}:completed".encode()).hexdigest()[:16],
            text="本次面试结束，后续等通知。",
            topic="面试结束",
            path=InterviewPath.BREADTH,
            thread_id=thread_id or "thread-completed",
            question_type=InterviewQuestionType.OPEN,
        )

    def _submit_intro_answer(
        self,
        state: InterviewSessionState,
        question: InterviewQuestion,
        answer: str,
    ) -> InterviewAnswerResult:
        self._validate_knowledge_snapshot(state)
        assessment = InterviewAnswerAssessment(
            correctness=0.5,
            completeness=0.5,
            confidence=1.0,
            reason="自我介绍只用于轻量更新兴趣因子，不作为已核验事实评分。",
            knowledge_refs=(),
            evaluator_name="intro-interest-updater",
            question_type=question.question_type,
            protocol="intro_interest_update",
        )
        observation = AnswerObservation(
            explicit_unknown=False,
            new_information_ratio=1.0 if str(answer or "").strip() else 0.0,
            relevance=1.0,
            answer_length=max(1, len(str(answer or ""))),
        )
        turn = InterviewTurn(
            question=question,
            answer=answer,
            correctness=assessment.correctness,
            completeness=assessment.completeness,
            confidence=assessment.confidence,
            evaluation_reason=assessment.reason,
            evaluation_refs=assessment.knowledge_refs,
            observation=observation,
        )
        state = state.record_turn(turn)
        points = self._update_interest_after_intro(state.interview_points, answer)
        next_stage = self._first_structured_stage(replace(state, interview_points=points))
        decision = InterviewDecision(
            InterviewAction.SWITCH_TOPIC,
            InterviewPath.BREADTH,
            "自我介绍",
            "自我介绍已完成，仅重排已有综合点兴趣因子。",
            question.thread_id,
            question_type=question.question_type,
        )
        state = replace(state, interview_points=points)
        state = state._next(
            interview_points=points,
            interview_stage=next_stage,
            active_line=self._line_for_stage(next_stage),
            active_subject=None,
            intro_interest_updated=True,
        )
        state = state.apply_decision(decision)
        state = self.session_store.put(state)
        return InterviewAnswerResult(state, assessment, decision)

    def _structured_decision_after_answer(
        self,
        state: InterviewSessionState,
        *,
        question: InterviewQuestion,
        observation: AnswerObservation,
        assessment: InterviewAnswerAssessment,
    ) -> InterviewDecision:
        point = self._point_for_question(state, question)
        if point is None:
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC,
                question.path,
                question.topic,
                "当前题目未绑定综合点，切换到下一题。",
                question.thread_id,
                question_type=question.question_type,
            )
        asked = self._point_observations(state, point.point_id)
        # Hard stop rules take precedence; the table formula chooses among legal actions.
        information_gain, verification_need = self._grid_signal(
            observation,
            assessment,
            followup_depth=question.followup_depth,
        )
        candidate_interest = self._breadth_candidate_interest(state, point)
        continuous_depth = self._continuous_depth_count(state, question.thread_id)
        current_knowledge_id = question.knowledge_refs[0] if question.knowledge_refs else ""
        confidence = (
            state.belief_state.point_confidence(current_knowledge_id)
            if current_knowledge_id
            else 0.5
        )
        _coefficients, depth_weight, breadth_weight = self._depth_breadth_weights(
            confidence=confidence,
            information_gain=information_gain,
            verification_need=verification_need,
            interest_factor=candidate_interest,
            consecutive_depth=continuous_depth,
        )
        utility = question.utility or QuestionUtility(
            jd_relevance=1.0 if point.jd_refs else 0.0,
            resume_relevance=1.0,
            knowledge_uncertainty=1.0 - confidence,
            weakness=1.0 - confidence,
            profile_preference=point.interest_score,
            repetition=min(1.0, asked / 3),
            followup_cost=0.0,
            score=point.interest_score,
        )
        utility = replace(
            utility,
            interest_factor=candidate_interest,
            confidence=confidence,
            information_gain=self._level_score(information_gain),
            verification_need=self._level_score(verification_need),
            consecutive_depth=continuous_depth,
            breadth_weight=breadth_weight,
            depth_weight=depth_weight,
        )
        context = self._turn_context_cache.get(state.interview_id)
        outcome = self._draft_cache.get(state.interview_id)
        if context is not None and outcome is not None:
            depth_legal = self._draft_is_legal(context, outcome.depth, InterviewPath.DEPTH)
            breadth_legal = self._draft_is_legal(context, outcome.breadth, InterviewPath.BREADTH)
            # Invalid model references are hard constraints. If the formula's
            # preferred lane is unavailable, select the other legal lane.
            if depth_weight > breadth_weight and not depth_legal and breadth_legal and context.depth_candidates:
                depth_weight = 0.0
                utility = replace(utility, depth_weight=depth_weight)
            elif breadth_weight >= depth_weight and not breadth_legal and depth_legal and context.breadth_candidates:
                breadth_weight = 0.0
                utility = replace(utility, breadth_weight=breadth_weight)
        must_switch_reason = self._must_switch_point_reason(
            state,
            point=point,
            question=question,
            observation=observation,
            assessment=assessment,
            asked=asked,
        )
        if must_switch_reason:
            stop_reason = (
                InterviewStopReason.FOLLOWUP_LIMIT
                if question.followup_depth >= state.policy.max_followups_per_thread
                else self._stop_reason_for_assessment(observation, assessment)
            )
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC,
                InterviewPath.BREADTH,
                question.topic,
                must_switch_reason,
                question.thread_id,
                stop_reason,
                question.question_type,
                utility,
            )
        if depth_weight <= breadth_weight:
            stop_reason = self._stop_reason_for_assessment(observation, assessment)
            return InterviewDecision(
                InterviewAction.SWITCH_TOPIC,
                InterviewPath.BREADTH,
                question.topic,
                (
                    f"深广公式决定切换综合点：W深={depth_weight:.3f} <= "
                    f"W广={breadth_weight:.3f}（信息增益={information_gain}，验证需求={verification_need}）。"
                ),
                question.thread_id,
                stop_reason=stop_reason,
                question_type=question.question_type,
                utility=utility,
            )
        return InterviewDecision(
            InterviewAction.FOLLOW_UP,
            InterviewPath.DEPTH,
            question.topic,
            (
                f"深广公式决定继续深挖：W深={depth_weight:.3f} > "
                f"W广={breadth_weight:.3f}（信息增益={information_gain}，验证需求={verification_need}）。"
            ),
            question.thread_id,
            question_type=question.question_type,
            utility=utility,
        )

    @staticmethod
    def _draft_is_legal(
        context: InterviewTurnContext,
        draft: InterviewQuestionDraft | None,
        path: InterviewPath,
    ) -> bool:
        if draft is None or not draft.ok or not draft.text or draft.path is not path:
            return False
        candidates = context.depth_candidates if path is InterviewPath.DEPTH else context.breadth_candidates
        candidate_ids = {chunk.chunk_id for chunk in candidates}
        if not draft.knowledge_refs or not set(draft.knowledge_refs).issubset(candidate_ids):
            return False
        if path is InterviewPath.DEPTH:
            return draft.topic == context.current_subject and all(
                chunk.corpus_id == context.current_corpus_id for chunk in candidates
            )
        return bool(context.breadth_subject) and draft.topic == context.breadth_subject and all(
            chunk.corpus_id in set(context.allowed_corpus_ids) for chunk in candidates
        )

    @staticmethod
    def _is_intro_question(state: InterviewSessionState, question: InterviewQuestion) -> bool:
        return state.interview_stage is InterviewStage.INTRO or question.topic == "自我介绍"

    @staticmethod
    def _update_interest_after_intro(
        points: tuple[InterviewPoint, ...],
        answer: str,
    ) -> tuple[InterviewPoint, ...]:
        answer_tokens = _tokens(answer)
        updated: list[InterviewPoint] = []
        for point in points:
            point_tokens = _tokens(" ".join((point.point, point.subject, *point.resume_refs, *point.question_directions)))
            hit_score = min(1.0, len(answer_tokens & point_tokens) / max(1, min(len(point_tokens), 12)))
            final_interest = max(
                0.0,
                min(
                    1.0,
                    point.static_interest_score * 0.65
                    + hit_score * 0.15
                    + (1.0 if point.jd_refs else 0.0) * 0.10
                    + 0.10
                    - min(0.3, point.asked_count * 0.05),
                ),
            )
            updated.append(
                replace(
                    point,
                    interest_score=final_interest,
                    intro_adjustment_score=hit_score,
                    point_status="unasked" if point.asked_count == 0 else point.point_status,
                )
            )
        return tuple(updated)

    def _must_switch_point_reason(
        self,
        state: InterviewSessionState,
        *,
        point: InterviewPoint,
        question: InterviewQuestion,
        observation: AnswerObservation,
        assessment: InterviewAnswerAssessment,
        asked: int,
    ) -> str | None:
        if asked >= state.max_questions_per_point:
            return "当前综合点已达到最多 3 问，切换综合点。"
        if question.followup_depth >= state.policy.max_followups_per_thread:
            return "当前综合点已达到追问上限，切换综合点。"
        if self._subject_budget_used(state, point) >= self._subject_budget_limit(state, point):
            return "当前 subject 已达到题量上限，切换 subject 或线程。"
        return None

    @staticmethod
    def _stop_reason_for_assessment(
        observation: AnswerObservation,
        assessment: InterviewAnswerAssessment,
    ) -> InterviewStopReason | None:
        if observation.explicit_unknown:
            return InterviewStopReason.USER_DOES_NOT_KNOW
        # This relevance signal is derived from the question and evaluator
        # quality, never from overlap with retrieved knowledge chunks.
        if observation.relevance <= 0.05:
            return InterviewStopReason.RELEVANCE_DECLINED
        if assessment.correctness <= 0.25 or assessment.completeness <= 0.25:
            return InterviewStopReason.LOW_MASTERY
        if observation.relevance <= 0.25:
            return InterviewStopReason.RELEVANCE_DECLINED
        if observation.new_information_ratio <= 0.2:
            return InterviewStopReason.NO_NEW_INFORMATION
        if assessment.correctness >= 0.85 and assessment.completeness >= 0.85:
            return InterviewStopReason.MASTERED
        return None

    def _cache_interview_point_knowledge(
        self,
        state: InterviewSessionState,
    ) -> tuple[InterviewPoint, ...]:
        cached: list[InterviewPoint] = []
        for point in state.interview_points:
            candidates = self._knowledge_candidates_for_point(state, point, top_k=6)
            breadth = candidates[:3]
            breadth_trace = dict(self._last_graph_retrieval_trace)
            block_refs: list[InterviewKnowledgeBlockRef] = [
                InterviewKnowledgeBlockRef(
                    role=InterviewKnowledgeBlockRole.A,
                    label=f"A{index}",
                    chunk_id=chunk.chunk_id,
                    point_id=point.point_id,
                    sequence=index,
                    retrieval_trace=self._compact_block_trace(breadth_trace),
                )
                for index, chunk in enumerate(breadth, start=1)
            ]
            breadth_ids = {chunk.chunk_id for chunk in breadth}
            for index, anchor in enumerate(breadth[:3], start=1):
                fallback_pool = tuple(chunk for chunk in candidates if chunk.chunk_id != anchor.chunk_id)
                related, trace = self._related_depth_block_for_anchor(
                    state,
                    point,
                    anchor,
                    excluded_ids=breadth_ids | {anchor.chunk_id},
                    fallback_pool=fallback_pool,
                )
                if related is None:
                    continue
                block_refs.append(
                    InterviewKnowledgeBlockRef(
                        role=InterviewKnowledgeBlockRole.D,
                        label=f"D{index}",
                        chunk_id=related.chunk_id,
                        point_id=point.point_id,
                        sequence=index,
                        related_to_label=f"A{index}",
                        related_to_chunk_id=anchor.chunk_id,
                        retrieval_trace=self._compact_block_trace(trace),
                    )
                )
            cached.append(
                replace(
                    point,
                    cached_knowledge_refs=tuple(chunk.chunk_id for chunk in breadth),
                    knowledge_block_refs=tuple(block_refs),
                )
            )
        return tuple(cached)

    def _related_depth_block_for_anchor(
        self,
        state: InterviewSessionState,
        point: InterviewPoint,
        anchor: InterviewKnowledgeChunk,
        *,
        excluded_ids: set[str],
        fallback_pool: tuple[InterviewKnowledgeChunk, ...] = (),
    ) -> tuple[InterviewKnowledgeChunk | None, dict[str, Any]]:
        if self._uses_graph_knowledge(state):
            try:
                result = self.graph_knowledge_index.retrieve_depth(
                    active_depth_chain=(anchor.chunk_id,),
                    current_question=point.point,
                    used_node_ids=set(excluded_ids),
                    corpus_ids=state.knowledge_corpus_ids,
                    top_k=3,
                )
                trace = self.graph_knowledge_index.trace_dict(result)
                for chunk in self.graph_knowledge_index.result_chunks(result):
                    if chunk.chunk_id not in excluded_ids:
                        return chunk, trace
            except (KeyError, ValueError, RuntimeError):
                pass
        ranked = self._lexical_related_depth_fallback(
            state,
            point,
            anchor,
            excluded_ids=excluded_ids,
            fallback_pool=fallback_pool,
        )
        if not ranked:
            return None, {"mode": "depth_related_unavailable", "fallback_used": True, "anchor_node_id": anchor.chunk_id}
        chunk = ranked[0]
        return chunk, {
            "mode": "depth_related_fallback",
            "fallback_used": True,
            "anchor_node_id": anchor.chunk_id,
            "returned_node_ids": [chunk.chunk_id],
            "hits": [{"node_id": chunk.chunk_id, "score": 0.0, "origin": "lexical_or_cached_fallback"}],
            "timings_ms": {"total": 0.0},
        }

    def _lexical_related_depth_fallback(
        self,
        state: InterviewSessionState,
        point: InterviewPoint,
        anchor: InterviewKnowledgeChunk,
        *,
        excluded_ids: set[str],
        fallback_pool: tuple[InterviewKnowledgeChunk, ...] = (),
    ) -> tuple[InterviewKnowledgeChunk, ...]:
        pool = [chunk for chunk in fallback_pool if chunk.chunk_id not in excluded_ids]
        if not pool and not self._uses_graph_knowledge(state):
            pool = [
                chunk
                for chunk in self._all_knowledge(state)
                if chunk.chunk_id not in excluded_ids and chunk.corpus_id == anchor.corpus_id
            ]
        if not pool:
            return ()
        query_tokens = _tokens(" ".join((point.point, *point.question_directions, anchor.topic, anchor.text)))
        return tuple(sorted(
            pool,
            key=lambda chunk: (
                -len(query_tokens & _tokens(f"{chunk.topic} {chunk.text}")),
                chunk.difficulty,
                _stable_rank(state.random_seed, 0, f"{anchor.chunk_id}:{chunk.chunk_id}"),
            ),
        ))

    @staticmethod
    def _block_payloads(blocks: tuple[InterviewKnowledgeBlockRef, ...]) -> list[dict[str, Any]]:
        return [block.to_dict() for block in blocks]

    @staticmethod
    def _compact_block_trace(trace: dict[str, Any]) -> dict[str, Any]:
        if not trace:
            return {}
        compact = {
            "mode": trace.get("mode"),
            "source": trace.get("source"),
            "query": str(trace.get("query") or "")[:300],
            "returned_node_ids": list(trace.get("returned_node_ids") or []),
            "fallback_used": bool(trace.get("fallback_used", False)),
            "timings_ms": dict(trace.get("timings_ms") or {}),
        }
        if trace.get("anchor_node_id"):
            compact["anchor_node_id"] = trace.get("anchor_node_id")
        return {key: value for key, value in compact.items() if value not in (None, "", [], {})}

    @staticmethod
    def _point_block_refs(
        point: InterviewPoint,
        role: InterviewKnowledgeBlockRole,
    ) -> tuple[InterviewKnowledgeBlockRef, ...]:
        blocks = tuple(block for block in point.knowledge_block_refs if block.role is role)
        if blocks or role is not InterviewKnowledgeBlockRole.A:
            return blocks
        return tuple(
            InterviewKnowledgeBlockRef(
                role=InterviewKnowledgeBlockRole.A,
                label=f"A{index}",
                chunk_id=chunk_id,
                point_id=point.point_id,
                sequence=index,
            )
            for index, chunk_id in enumerate(point.cached_knowledge_refs[:3], start=1)
        )

    def _breadth_block_refs_for_point(
        self,
        point: InterviewPoint,
        *,
        role: InterviewKnowledgeBlockRole,
    ) -> tuple[InterviewKnowledgeBlockRef, ...]:
        a_blocks = self._point_block_refs(point, InterviewKnowledgeBlockRole.A)
        if not a_blocks:
            return ()
        if role is InterviewKnowledgeBlockRole.A:
            return tuple(
                replace(block, role=InterviewKnowledgeBlockRole.A, label=f"A{index}", sequence=index)
                for index, block in enumerate(a_blocks[:3], start=1)
            )
        return tuple(
            replace(
                block,
                role=InterviewKnowledgeBlockRole.B,
                label=f"B{index}",
                sequence=index,
                related_to_label=None,
                related_to_chunk_id=None,
            )
            for index, block in enumerate(a_blocks[:3], start=1)
        )

    def _depth_block_refs_after_question(
        self,
        state: InterviewSessionState,
        point: InterviewPoint,
        question: InterviewQuestion,
    ) -> tuple[InterviewKnowledgeBlockRef, ...]:
        primary = next(
            (
                block
                for block in question.knowledge_block_refs
                if block.point_id == point.point_id and block.role in {InterviewKnowledgeBlockRole.A, InterviewKnowledgeBlockRole.B}
            ),
            None,
        )
        if primary is not None:
            previous_sequence = primary.sequence or 1
        elif question.path is InterviewPath.DEPTH:
            previous_sequence = min(2, question.followup_depth + 1)
        else:
            previous_sequence = 1
        next_sequence = previous_sequence + 1
        if next_sequence > 3:
            return ()
        a_blocks = {block.sequence: block for block in self._point_block_refs(point, InterviewKnowledgeBlockRole.A)}
        d_blocks = {block.sequence: block for block in self._point_block_refs(point, InterviewKnowledgeBlockRole.D)}
        selected: list[InterviewKnowledgeBlockRef] = []
        if next_sequence in a_blocks:
            selected.append(a_blocks[next_sequence])
        if previous_sequence in d_blocks:
            selected.append(d_blocks[previous_sequence])
        return tuple(selected)

    def _next_question_candidates_payload(self, state: InterviewSessionState) -> dict[str, Any]:
        if state.pending_question is not None:
            return {
                "interview_id": state.interview_id,
                "path": state.pending_question.path.value,
                "point_id": state.current_point_id,
                "previous_question_id": state.turns[-1].question.question_id if state.turns else None,
                "selected_blocks": self._block_payloads(tuple(state.pending_question.knowledge_block_refs)),
                "candidate_blocks": self._block_payloads(tuple(state.pending_question.knowledge_block_refs)),
                "decision_reason": "当前已有 pending_question；候选与已生成题目一致。",
            }
        if not state.interview_points or state.interview_stage is InterviewStage.COMPLETED:
            return {"interview_id": state.interview_id, "path": None, "selected_blocks": [], "candidate_blocks": []}
        decision = state.last_decision
        point_index = {point.point_id: point for point in state.interview_points}
        line = self._line_for_stage(state.interview_stage)
        is_depth = (
            decision is not None
            and decision.action is InterviewAction.FOLLOW_UP
            and state.current_point_id in point_index
            and line is not None
            and point_index[str(state.current_point_id)].category is line
            and self._point_observations(state, str(state.current_point_id)) < state.max_questions_per_point
            and self._subject_budget_remaining(state, point_index[str(state.current_point_id)])
        )
        if is_depth and state.turns:
            point = point_index[str(state.current_point_id)]
            selected = self._depth_block_refs_after_question(state, point, state.turns[-1].question)
            return {
                "interview_id": state.interview_id,
                "path": InterviewPath.DEPTH.value,
                "point_id": point.point_id,
                "point": point.point,
                "previous_question_id": state.turns[-1].question.question_id,
                "selected_blocks": self._block_payloads(selected),
                "candidate_blocks": self._block_payloads(selected),
                "decision_reason": decision.reason if decision else "",
            }
        # Candidate calculation must mirror _next_resume_point_question: when
        # the current module has no eligible design point left, project the
        # next module before declaring that no breadth question exists. Without
        # this projection, answer/process clears pending_question and returns
        # path=None while the session remains active between modules.
        selection_state = state
        point = self._select_next_interview_point(selection_state)
        if point is None:
            selection_state = self._advance_structured_stage(state)
            point = self._select_next_interview_point(selection_state)
        if point is None:
            return {
                "interview_id": state.interview_id,
                "path": None,
                "selected_blocks": [],
                "candidate_blocks": [],
                "interview_stage": selection_state.interview_stage.value,
            }
        role = (
            InterviewKnowledgeBlockRole.B
            if any(turn.question.knowledge_refs for turn in state.turns)
            else InterviewKnowledgeBlockRole.A
        )
        selected = self._breadth_block_refs_for_point(point, role=role)
        return {
            "interview_id": state.interview_id,
            "path": InterviewPath.BREADTH.value,
            "point_id": point.point_id,
            "point": point.point,
            "previous_question_id": state.turns[-1].question.question_id if state.turns else None,
            "selected_blocks": self._block_payloads(selected),
            "candidate_blocks": self._block_payloads(selected),
            "decision_reason": decision.reason if decision else "首个正式综合点问题。",
        }

    def _first_structured_stage(self, state: InterviewSessionState) -> InterviewStage:
        if any(point.category is InterviewLine.INTERNSHIP for point in state.interview_points):
            return InterviewStage.INTERNSHIP_SUBJECT
        if any(point.category is InterviewLine.PROJECT for point in state.interview_points):
            return InterviewStage.PROJECT_SUBJECT
        if any(point.category is InterviewLine.SKILL for point in state.interview_points):
            return InterviewStage.PURE_FOUNDATION
        return InterviewStage.CANDIDATE_QUESTION_WAIT

    @staticmethod
    def _stage_for_line(line: InterviewLine) -> InterviewStage:
        return {
            InterviewLine.INTERNSHIP: InterviewStage.INTERNSHIP_SUBJECT,
            InterviewLine.PROJECT: InterviewStage.PROJECT_SUBJECT,
            InterviewLine.SKILL: InterviewStage.PURE_FOUNDATION,
        }[line]

    @staticmethod
    def _line_for_stage(stage: InterviewStage) -> InterviewLine | None:
        return {
            InterviewStage.INTERNSHIP_SUBJECT: InterviewLine.INTERNSHIP,
            InterviewStage.PROJECT_SUBJECT: InterviewLine.PROJECT,
            InterviewStage.PURE_FOUNDATION: InterviewLine.SKILL,
        }.get(stage)

    def _advance_structured_stage(self, state: InterviewSessionState) -> InterviewSessionState:
        order = (
            InterviewStage.INTERNSHIP_SUBJECT,
            InterviewStage.PROJECT_SUBJECT,
            InterviewStage.PURE_FOUNDATION,
        )
        current = state.interview_stage
        start = order.index(current) + 1 if current in order else 0
        for stage in order[start:]:
            line = self._line_for_stage(stage)
            if line is not None and any(
                point.category is line
                and self._point_observations(state, point.point_id) < state.max_questions_per_point
                and self._subject_budget_remaining(state, point)
                and point.point_status not in {"completed", "weak", "irrelevant"}
                for point in state.interview_points
            ):
                return state._next(interview_stage=stage, active_line=line, active_subject=None, current_point_id=None)
        return state._next(
            interview_stage=InterviewStage.CANDIDATE_QUESTION_WAIT,
            active_line=None,
            active_subject=None,
            current_point_id=None,
        )

    @staticmethod
    def _subject_key(point: InterviewPoint) -> str:
        return f"{point.category.value}:{point.subject or point.point}"

    def _subject_budget_limit(self, state: InterviewSessionState, point: InterviewPoint) -> int:
        if point.category is InterviewLine.SKILL:
            return state.max_pure_foundation_questions
        return state.max_questions_per_subject

    def _subject_budget_used(self, state: InterviewSessionState, point: InterviewPoint) -> int:
        return int(state.subject_question_counts.get(self._subject_key(point), 0))

    def _subject_budget_remaining(self, state: InterviewSessionState, point: InterviewPoint) -> bool:
        if point.category is InterviewLine.SKILL and state.pure_foundation_question_count >= state.max_pure_foundation_questions:
            return False
        return self._subject_budget_used(state, point) < self._subject_budget_limit(state, point)

    def _point_for_question(
        self,
        state: InterviewSessionState,
        question: InterviewQuestion,
    ) -> InterviewPoint | None:
        claim_refs = set(question.resume_claim_refs)
        for point in state.interview_points:
            if point.point == question.topic and (not claim_refs or claim_refs <= set(point.resume_refs)):
                return point
        if state.current_point_id:
            return next((point for point in state.interview_points if point.point_id == state.current_point_id), None)
        return None

    def _mark_point_questioned(
        self,
        state: InterviewSessionState,
        point: InterviewPoint,
    ) -> tuple[tuple[InterviewPoint, ...], dict[str, int], int]:
        subject_key = self._subject_key(point)
        counts = dict(state.subject_question_counts)
        counts[subject_key] = counts.get(subject_key, 0) + 1
        pure_count = state.pure_foundation_question_count + (1 if point.category is InterviewLine.SKILL else 0)
        points = tuple(
            replace(
                item,
                asked_count=item.asked_count + 1,
                point_status="in_progress",
            )
            if item.point_id == point.point_id
            else item
            for item in state.interview_points
        )
        return points, counts, pure_count

    def _mark_structured_point_after_answer(
        self,
        state: InterviewSessionState,
        *,
        question: InterviewQuestion,
        decision: InterviewDecision,
    ) -> InterviewSessionState:
        point = self._point_for_question(state, question)
        if point is None:
            return state
        if decision.action is InterviewAction.FOLLOW_UP:
            next_status = "in_progress"
        elif decision.stop_reason in {
            InterviewStopReason.LOW_MASTERY,
            InterviewStopReason.USER_DOES_NOT_KNOW,
        }:
            next_status = "weak"
        elif decision.stop_reason is InterviewStopReason.RELEVANCE_DECLINED:
            next_status = "irrelevant"
        else:
            next_status = "completed"
        points = tuple(
            replace(item, point_status=next_status)
            if item.point_id == point.point_id
            else item
            for item in state.interview_points
        )
        return state._next(interview_points=points)

    def _next_resume_point_question(self, state: InterviewSessionState) -> InterviewQuestion:
        decision = state.last_decision
        if decision is not None and decision.action is InterviewAction.END:
            raise ValueError("Interview session has ended")

        line = self._line_for_stage(state.interview_stage)
        if line is None:
            return self._complete_interview(state)

        point_index = {point.point_id: point for point in state.interview_points}
        is_depth = (
            decision is not None
            and decision.action is InterviewAction.FOLLOW_UP
            and state.current_point_id in point_index
            and point_index[str(state.current_point_id)].category is line
            and self._point_observations(state, str(state.current_point_id)) < state.max_questions_per_point
            and self._subject_budget_remaining(state, point_index[str(state.current_point_id)])
        )
        if is_depth:
            point = point_index[str(state.current_point_id)]
            thread = self._active_thread_for_point(state, point)
            if thread is None:
                state = state.add_thread(point.point, path=InterviewPath.BREADTH)
                thread = state.threads[-1]
            action = InterviewAction.FOLLOW_UP
            path = InterviewPath.DEPTH
            reason = "继续深挖当前简历综合点，用于确认能力边界。"
        else:
            point = self._select_next_interview_point(state)
            if point is None:
                state = self._advance_structured_stage(state)
                self.session_store.put(state)
                return self._next_structured_interview_question(state)
            state = state.add_thread(point.point, path=InterviewPath.BREADTH)
            thread = state.threads[-1]
            action = InterviewAction.ASK if not state.turns else InterviewAction.SWITCH_TOPIC
            path = InterviewPath.BREADTH
            reason = "按固定阶段和当前 subject 选择下一个高兴趣综合点。"

        if is_depth and state.turns:
            selected_blocks = self._depth_block_refs_after_question(state, point, state.turns[-1].question)
        else:
            breadth_role = (
                InterviewKnowledgeBlockRole.B
                if any(turn.question.knowledge_refs for turn in state.turns)
                else InterviewKnowledgeBlockRole.A
            )
            selected_blocks = self._breadth_block_refs_for_point(point, role=breadth_role)
        selected_ids = tuple(dict.fromkeys(block.chunk_id for block in selected_blocks))
        if not selected_ids:
            raise ValueError(f"No interview knowledge is available for resume point: {point.point}")
        candidates = self._knowledge_by_ids(state, selected_ids)
        selected_knowledge = candidates[0]
        trace_source = next(
            (
                block.retrieval_trace
                for block in selected_blocks
                if path is InterviewPath.DEPTH and block.role is InterviewKnowledgeBlockRole.D and block.retrieval_trace
            ),
            None,
        ) or next((block.retrieval_trace for block in selected_blocks if block.retrieval_trace), {})
        self._last_graph_retrieval_trace = {
            **dict(trace_source),
            "path": path.value,
            "returned_node_ids": list(selected_ids),
            "hits": list(trace_source.get("hits") or [
                {"node_id": chunk_id, "score": 0.0, "origin": "fixed_block_plan"}
                for chunk_id in selected_ids
            ]),
            "block_refs": self._block_payloads(selected_blocks),
            "fallback_used": bool(trace_source.get("fallback_used", False))
            or any(bool(block.retrieval_trace.get("fallback_used")) for block in selected_blocks),
            "timings_ms": dict(trace_source.get("timings_ms") or {"total": 0.0}),
        }
        self._last_graph_retrieval_trace.setdefault("mode", "fixed_block_plan")
        question_type = (
            InterviewQuestionType.KNOWLEDGE
            if point.category is InterviewLine.SKILL
            else InterviewQuestionType.PROJECT_SCENARIO
        )
        previous_knowledge_id = (
            state.turns[-1].question.knowledge_refs[0]
            if state.turns and state.turns[-1].question.knowledge_refs
            else ""
        )
        confidence = (
            state.belief_state.point_confidence(previous_knowledge_id)
            if previous_knowledge_id
            else 0.5
        )
        previous_update = next(
            (
                update
                for update in reversed(state.turns[-1].knowledge_confidence_updates)
                if update.knowledge_point_id == previous_knowledge_id
            ),
            None,
        ) if state.turns else None
        continuous_depth = self._continuous_depth_count(state, thread.thread_id)
        information_gain = "中"
        verification_need = "中"
        if state.turns and state.turns[-1].observation is not None:
            observation = state.turns[-1].observation
            last_turn = state.turns[-1]
            information_gain, verification_need = self._grid_signal(
                observation,
                InterviewAnswerAssessment(
                    correctness=last_turn.correctness,
                    completeness=last_turn.completeness,
                    confidence=last_turn.confidence,
                    reason=last_turn.evaluation_reason,
                    knowledge_refs=last_turn.evaluation_refs,
                    evaluator_name="replayed-turn",
                    question_type=last_turn.question.question_type,
                ),
                followup_depth=last_turn.question.followup_depth,
            )
        coefficients, depth_weight, breadth_weight = self._depth_breadth_weights(
            confidence=confidence,
            information_gain=information_gain,
            verification_need=verification_need,
            interest_factor=point.interest_score,
            consecutive_depth=continuous_depth,
        )
        utility = QuestionUtility(
            jd_relevance=1.0 if point.jd_refs else 0.0,
            resume_relevance=1.0,
            knowledge_uncertainty=0.5,
            weakness=0.5,
            profile_preference=point.interest_score,
            repetition=min(1.0, self._point_observations(state, point.point_id) / 3),
            followup_cost=thread.followup_count / thread.max_followups if path is InterviewPath.DEPTH else 0.0,
            score=point.interest_score,
            interest_factor=point.interest_score,
            confidence=confidence,
            information_gain=self._level_score(information_gain),
            verification_need=self._level_score(verification_need),
            consecutive_depth=continuous_depth,
            breadth_weight=breadth_weight,
            depth_weight=depth_weight,
        )
        decision = InterviewDecision(
            action,
            path,
            point.point,
            reason,
            thread.thread_id,
            question_type=question_type,
            utility=utility,
        )
        parent = next(
            (
                turn.question
                for turn in reversed(state.turns)
                if turn.question.thread_id == thread.thread_id
            ),
            None,
        )
        followup_depth = parent.followup_depth + 1 if path is InterviewPath.DEPTH and parent else 0
        question_id = "question-" + hashlib.sha256(
            f"{state.interview_id}:{state.revision}:{point.point_id}:{':'.join(selected_ids)}:{action.value}".encode()
        ).hexdigest()[:16]
        outcome = self._draft_cache.pop(state.interview_id, None)
        draft = (
            outcome.depth
            if outcome is not None and path is InterviewPath.DEPTH
            else outcome.breadth
            if outcome is not None
            else None
        )
        if draft is None:
            if self.question_generator is None:
                raise ValueError("Interview question generation failed: interview_question model is unavailable")
            generated = self.question_generator.generate(
                state=state,
                decision=decision,
                thread_id=thread.thread_id,
                knowledge=selected_knowledge,
                jd_requirement_refs=point.jd_refs,
                resume_claim_refs=point.resume_refs,
            )
            draft = InterviewQuestionDraft(
                text=generated.text,
                path=path,
                topic=point.point,
                knowledge_refs=generated.knowledge_refs,
                point_id=point.point_id,
            )
        candidate_ids = {chunk.chunk_id for chunk in candidates}
        if (
            draft is None
            or not draft.ok
            or not draft.text
            or draft.path is not path
            or draft.topic != point.point
            or not draft.knowledge_refs
            or not set(draft.knowledge_refs).issubset(candidate_ids)
        ):
            detail = (
                draft.error_code
                if draft is not None and draft.error_code
                else (outcome.errors if outcome is not None else "draft_missing")
            )
            raise ValueError(f"Interview question generation failed: {detail}")
        question = InterviewQuestion(
            question_id=question_id,
            text=validate_interview_question_text(
                draft.text,
                action=action,
                question_type=question_type,
                topic=point.subject or point.point,
            ),
            topic=point.point,
            path=path,
            thread_id=thread.thread_id,
            knowledge_refs=selected_ids,
            knowledge_block_refs=selected_blocks,
            jd_requirement_refs=point.jd_refs,
            resume_claim_refs=point.resume_refs,
            jd_ref=(state.jd_id, state.jd_revision) if state.jd_id and state.jd_revision else None,
            verified_resume_ref=(
                (state.resume_document_id, state.resume_document_revision)
                if state.resume_document_id and state.resume_document_revision
                else None
            ),
            parent_question_id=parent.question_id if followup_depth else None,
            followup_depth=followup_depth,
            question_type=question_type,
            utility=utility,
        )
        self._validate_generated_question(
            state,
            question,
            decision=decision,
            thread_id=thread.thread_id,
            knowledge=selected_knowledge,
            jd_requirement_refs=point.jd_refs,
            resume_claim_refs=point.resume_refs,
        )
        trace = RecommendationTrace(
            turn=len(state.turns) + 1,
            thread_id=thread.thread_id,
            point_id=point.point_id,
            previous_question_id=parent.question_id if parent else None,
            information_gain=information_gain,
            verification_need=verification_need,
            interest_current=point.interest_score,
            confidence_current_before=(
                previous_update.confidence_before if previous_update and previous_update.confidence_before is not None else confidence
            ),
            confidence_observation=(
                previous_update.observed_confidence if previous_update else confidence
            ),
            confidence_current_after=(
                previous_update.confidence_after if previous_update and previous_update.confidence_after is not None else confidence
            ),
            continuous_depth_count=continuous_depth,
            coefficients=coefficients,
            weights={"breadth": breadth_weight, "depth": depth_weight},
            allowed_actions=("depth", "breadth") if is_depth else ("breadth",),
            candidate_knowledge_point_ids=selected_ids,
            selected_action="depth" if path is InterviewPath.DEPTH else "breadth",
            selected_knowledge_point_ids=selected_ids,
            selection_reason=reason,
            question_id=question.question_id,
            policy_versions={
                "interest": "lexical-jd-resume-v1",
                "mastery": "incremental-average-v1",
                "depth_breadth": "heuristic-table-v1",
            },
            candidate_knowledge_blocks=selected_blocks,
            selected_knowledge_blocks=selected_blocks,
            retrieval_trace=dict(self._last_graph_retrieval_trace),
        )
        primary_chain_id = next(
            (
                block.chunk_id
                for block in selected_blocks
                if block.role in {InterviewKnowledgeBlockRole.A, InterviewKnowledgeBlockRole.B}
            ),
            selected_ids[0],
        )
        next_chain = (
            (*state.active_depth_chain_blocks, primary_chain_id)
            if path is InterviewPath.DEPTH
            else (primary_chain_id,)
        )
        updated_points, subject_counts, pure_count = self._mark_point_questioned(state, point)
        state = state._next(
            active_line=point.category,
            active_subject=point.subject,
            current_point_id=point.point_id,
            active_depth_chain_blocks=next_chain,
            used_knowledge_point_ids=tuple(dict.fromkeys((*state.used_knowledge_point_ids, *selected_ids))),
            recommendation_traces=(*state.recommendation_traces, trace),
            interview_points=updated_points,
            subject_question_counts=subject_counts,
            pure_foundation_question_count=pure_count,
            last_decision=decision,
        )
        state = state.set_pending_question(question)
        self.session_store.put(state)
        self._start_depth_prefetch(state, question)
        return question

    @staticmethod
    def _expected_points(
        question_type: InterviewQuestionType,
        reference_texts: tuple[str, ...],
    ) -> list[str]:
        if question_type is InterviewQuestionType.BEHAVIORAL:
            return ["说明具体情境和目标", "明确个人行动与责任边界", "说明结果和复盘"]
        if question_type is InterviewQuestionType.CASE:
            return ["明确假设与约束", "比较至少两种方案", "给出验证指标和风险"]
        if question_type is InterviewQuestionType.PROJECT_SCENARIO:
            base = ["说明实现机制", "解释技术选择", "说明边界条件或取舍"]
        elif question_type is InterviewQuestionType.OPEN:
            return ["给出明确观点", "说明依据", "讨论适用边界和权衡"]
        else:
            base = ["解释核心概念", "说明工作机制", "给出实际使用场景"]
        if question_type is InterviewQuestionType.PROJECT_SCENARIO:
            return base
        extracted: list[str] = []
        for text in reference_texts:
            for part in re.split(r"[\n。；;]+", text):
                normalized = part.strip(" -*\t")
                if 8 <= len(normalized) <= 180:
                    extracted.append(normalized)
                if len(extracted) >= 5:
                    break
            if len(extracted) >= 5:
                break
        return list(dict.fromkeys((*base, *extracted)))[:8]

    @staticmethod
    def _open_answer_framework(
        question_type: InterviewQuestionType,
        expected_points: list[str],
    ) -> str:
        label = {
            InterviewQuestionType.BEHAVIORAL: "该题没有唯一正确答案，建议使用情境-任务-行动-结果-复盘结构回答。",
            InterviewQuestionType.CASE: "该题没有唯一正确答案，建议先声明假设，再比较方案并给出验证指标。",
            InterviewQuestionType.PROJECT_SCENARIO: "该题没有唯一正确答案，建议说明个人贡献、实现机制、技术取舍、边界和验证方式。",
            InterviewQuestionType.OPEN: "该题没有唯一正确答案，建议给出观点、依据、反例和适用边界。",
        }.get(question_type, "该题应围绕评分要点组织回答。")
        return label + "\n" + "\n".join(f"- {item}" for item in expected_points)

    def _update_mastery_if_completed(self, state: InterviewSessionState) -> KnowledgeMasteryProfile | None:
        if self.mastery_store is None or state.status is not InterviewStatus.COMPLETED:
            return None
        return self.mastery_store.apply_interview(state)

    @staticmethod
    def _first_available_line(points: tuple[InterviewPoint, ...]) -> InterviewLine | None:
        for line in (InterviewLine.INTERNSHIP, InterviewLine.PROJECT, InterviewLine.SKILL):
            if any(point.category is line for point in points):
                return line
        return None

    def _build_interview_points(
        self,
        jd: JDDocument,
        resume: ResumeDocument,
    ) -> tuple[InterviewPoint, ...]:
        requirements = {
            ref.record_id: self.career_store.get(ref.record_id, ref.revision, expected_type=JDRequirement)
            for ref in jd.requirement_refs
        }
        for requirement in self.career_store.list_latest(JDRequirement):
            if requirement.jd_ref.record_id == jd.jd_id:
                requirements.setdefault(requirement.requirement_id, requirement)
        claim_sections: dict[str, str] = {}
        for section, claim_ids in resume.sections.items():
            for claim_id in claim_ids:
                claim_sections.setdefault(str(claim_id), str(section))
        points: list[InterviewPoint] = []
        for ref in resume.claim_refs:
            claim = self.career_store.get(ref.record_id, ref.revision, expected_type=ResumeClaim)
            line = self._line_for_claim(claim, claim_sections.get(claim.claim_id))
            if line is None:
                continue
            claim_requirements = tuple(
                req.record_id for req in claim.requirement_refs if req.record_id in requirements
            )
            display_subject = self._claim_display_subject(claim, line)
            design_points = self._split_design_points(claim.text, line)
            for index, design_point in enumerate(design_points, start=1):
                title = self._point_title(design_point, line)
                interest = self._interest_score(
                    jd,
                    claim,
                    tuple(requirements[item] for item in claim_requirements),
                    text=design_point,
                )
                digest = hashlib.sha256(
                    f"{line.value}:{claim.claim_id}:{index}:{title}".encode()
                ).hexdigest()[:12]
                points.append(
                    InterviewPoint(
                        point_id=f"{line.value}-{digest}",
                        point=title,
                        category=line,
                        interest_score=interest,
                        static_interest_score=interest,
                        reason=(
                            "该综合点来自已核验简历，并按当前项目设计点与JD的重合度、"
                            "显式requirement引用和文本相关性计算兴趣因子。"
                        ),
                        jd_refs=claim_requirements,
                        resume_refs=(claim.claim_id,),
                        question_directions=self._question_directions(line),
                        confidence=0.5,
                        subject=display_subject,
                    )
                )
        line_order = {
            InterviewLine.INTERNSHIP: 0,
            InterviewLine.PROJECT: 1,
            InterviewLine.SKILL: 2,
        }
        points.sort(
            key=lambda point: (
                line_order[point.category],
                -point.interest_score,
                _stable_rank(0, 0, point.point_id),
            )
        )
        return tuple(points)

    def _line_for_claim(self, claim: ResumeClaim, section: str | None) -> InterviewLine | None:
        section_value = str(section or "").casefold()
        if any(token in section_value for token in ("intern", "实习")):
            return InterviewLine.INTERNSHIP
        if any(token in section_value for token in ("project", "项目")):
            return InterviewLine.PROJECT
        if any(token in section_value for token in ("skill", "技能", "other", "其他")):
            return InterviewLine.SKILL
        for ref in claim.evidence_refs:
            evidence = self.career_store.get(ref.record_id, ref.revision)
            if isinstance(evidence, PersonalEvidence):
                if evidence.evidence_type is EvidenceType.INTERNSHIP:
                    return InterviewLine.INTERNSHIP
                if evidence.evidence_type is EvidenceType.PROJECT:
                    return InterviewLine.PROJECT
                if evidence.evidence_type is EvidenceType.SKILLS:
                    return InterviewLine.SKILL
        return None

    def _claim_display_subject(self, claim: ResumeClaim, line: InterviewLine) -> str:
        """Use the evidence record's human label as the spoken interview subject."""
        if line is InterviewLine.SKILL:
            return "个人技能与其他"
        for ref in claim.evidence_refs:
            try:
                evidence = self.career_store.get(ref.record_id, ref.revision, expected_type=PersonalEvidence)
            except (KeyError, ValueError):
                continue
            preferred_fields = {
                InterviewLine.INTERNSHIP: {"company", "company_name", "organization", "employer", "name"},
                InterviewLine.PROJECT: {"project", "project_name", "title", "name"},
                InterviewLine.SKILL: {"skill", "skill_name", "technology", "name"},
            }[line]
            for fact in evidence.facts:
                field_name = str(fact.field_name or "").strip().casefold()
                value = fact.value
                if field_name in preferred_fields and isinstance(value, (str, int, float)):
                    candidate = str(value).strip()
                    if candidate:
                        return candidate
            label = str(evidence.label or "").strip()
            if label:
                return label
        return {
            InterviewLine.INTERNSHIP: "这段实习经历",
            InterviewLine.PROJECT: "这个项目",
            InterviewLine.SKILL: "这项技能",
        }[line]

    @staticmethod
    def _question_directions(line: InterviewLine) -> tuple[str, ...]:
        if line is InterviewLine.INTERNSHIP:
            return ("职责边界", "个人贡献", "技术方案", "异常处理", "量化结果")
        if line is InterviewLine.PROJECT:
            return ("架构拆分", "技术选型", "工具编排", "结果验证", "异常处理")
        return ("核心原理", "使用场景", "工程取舍")

    @staticmethod
    def _point_title(text: str, line: InterviewLine) -> str:
        parts = [
            part.strip(" -*	")
            for part in re.split(r"[。；;\n]+", str(text or ""))
            if part.strip(" -*	")
        ]
        title = parts[0] if parts else str(text or "").strip()
        title = re.sub(r"\s+", " ", title)
        if len(title) > 56:
            title = title[:56].rstrip() + "…"
        if title:
            return title
        return {
            InterviewLine.INTERNSHIP: "实习经历综合点",
            InterviewLine.PROJECT: "项目经历综合点",
            InterviewLine.SKILL: "个人技能综合点",
        }[line]

    @staticmethod
    def _split_design_points(text: str, line: InterviewLine) -> tuple[str, ...]:
        """Split an explicitly numbered project claim into independent points.

        The resume claim remains the immutable source record. Splitting only
        affects interview scheduling, so each point can consume its own
        three-question budget. Every design point receives an independent
        interest factor; the enclosing project only shares a subject-level
        question budget.
        """
        normalized = str(text or "").replace("\r", "").strip()
        if line is not InterviewLine.PROJECT or not normalized:
            return (normalized,) if normalized else ()

        marker = re.compile(
            r"(?m)(?:^|[\s:：；;])"
            r"(?P<marker>(?:[（(]\s*\d{1,2}\s*[）)]|\d{1,2}\s*[、,。．.):：]))"
            r"\s*(?=[^\d]|$)"
        )
        matches = list(marker.finditer(normalized))
        if not matches:
            return (normalized,)

        starts = [match.start("marker") for match in matches]
        segments: list[str] = []
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(normalized)
            segment = normalized[start:end].strip()
            segment = re.sub(
                r"^(?:[（(]\s*\d{1,2}\s*[）)]|\d{1,2}\s*[、,。．.):：])\s*",
                "",
                segment,
            ).strip()
            if segment:
                segments.append(segment)
        return tuple(segments) if segments else (normalized,)

    @staticmethod
    def _interest_score(
        jd: JDDocument,
        claim: ResumeClaim,
        requirements: tuple[JDRequirement, ...],
        *,
        text: str | None = None,
    ) -> float:
        claim_tokens = _tokens(claim.text if text is None else text)
        jd_tokens = _tokens(jd.raw_text)
        requirement_tokens = _tokens(" ".join(requirement.text for requirement in requirements))
        jd_overlap = len(claim_tokens & jd_tokens) / max(1, min(len(claim_tokens), len(jd_tokens)))
        requirement_overlap = (
            len(claim_tokens & requirement_tokens) / max(1, min(len(claim_tokens), len(requirement_tokens)))
            if requirement_tokens
            else 0.0
        )
        explicit_requirement = 1.0 if requirements else 0.0
        return max(
            0.1,
            min(1.0, 0.25 + 0.35 * explicit_requirement + 0.25 * jd_overlap + 0.15 * requirement_overlap),
        )

    @staticmethod
    def _active_thread_for_point(state: InterviewSessionState, point: InterviewPoint):
        return next(
            (
                thread
                for thread in reversed(state.threads)
                if thread.root_topic == point.point and thread.status == "active"
            ),
            None,
        )

    @staticmethod
    def _point_observations(state: InterviewSessionState, point_id: str) -> int:
        point = next((item for item in state.interview_points if item.point_id == point_id), None)
        trace_count = sum(1 for trace in state.recommendation_traces if trace.point_id == point_id)
        return max(point.asked_count if point else 0, trace_count)

    @staticmethod
    def _continuous_depth_count(state: InterviewSessionState, thread_id: str) -> int:
        count = 0
        for turn in reversed(state.turns):
            if turn.question.thread_id != thread_id or turn.question.path is not InterviewPath.DEPTH:
                break
            count += 1
        return count

    @staticmethod
    def _level_label(value: float) -> str:
        if value >= 0.67:
            return "高"
        if value >= 0.34:
            return "中"
        return "低"

    @staticmethod
    def _level_score(label: str) -> float:
        return {"高": 1.0, "中": 0.6, "低": 0.3}.get(label, 0.6)

    @classmethod
    def _grid_signal(
        cls,
        observation: AnswerObservation,
        assessment: InterviewAnswerAssessment,
        *,
        followup_depth: int,
    ) -> tuple[str, str]:
        """Return the two model-defined signals used by the depth/breadth grid.

        Retrieved knowledge is planning context, not a rubric for subjective
        project answers. Deterministic safeguards only override the model when
        the candidate explicitly says they do not know or repeats a
        clarification without adding information.
        """
        information_gain = (
            assessment.information_gain
            if assessment.information_gain in {"高", "中", "低"}
            else cls._level_label(observation.new_information_ratio)
        )
        verification_need = (
            assessment.verification_need
            if assessment.verification_need in {"高", "中", "低"}
            else cls._level_label(1.0 - ((assessment.correctness + assessment.completeness) / 2.0))
        )
        if observation.explicit_unknown:
            information_gain = "低"
            verification_need = "高"
        elif observation.relevance <= 0.05:
            # A non-responsive answer does not warrant spending a depth turn
            # on the same point. This uses the question/evaluator relevance
            # signal, never knowledge-chunk overlap.
            information_gain = "低"
            verification_need = "低"

        # A failed clarification has no remaining information to validate.
        # Feed that state into the table as (低, 低), which is the table's
        # explicit "end current point" cell.
        if followup_depth >= 1 and observation.new_information_ratio <= 0.2:
            verification_need = "低"
        return information_gain, verification_need

    @staticmethod
    def _depth_breadth_coefficients(information_gain: str, verification_need: str) -> dict[str, float]:
        table = {
            ("高", "高"): (0.20, 1.00),
            ("高", "中"): (0.45, 0.80),
            ("高", "低"): (0.80, 0.45),
            ("中", "高"): (0.30, 0.95),
            ("中", "中"): (0.60, 0.60),
            ("中", "低"): (0.90, 0.30),
            ("低", "高"): (0.35, 0.85),
            ("低", "中"): (0.85, 0.30),
            ("低", "低"): (1.00, 0.10),
        }
        breadth, depth = table.get((information_gain, verification_need), (0.60, 0.60))
        return {"breadth": breadth, "depth": depth}

    @classmethod
    def _depth_breadth_weights(
        cls,
        *,
        confidence: float,
        information_gain: str,
        verification_need: str,
        interest_factor: float,
        consecutive_depth: int,
    ) -> tuple[dict[str, float], float, float]:
        coefficients = cls._depth_breadth_coefficients(information_gain, verification_need)
        # Confidence, interest, and prior depth count remain useful audit
        # fields, but lane selection is intentionally driven only by the two
        # evaluator signals above. Hard workflow limits are applied by callers.
        depth_weight = coefficients["depth"]
        breadth_weight = coefficients["breadth"]
        return coefficients, depth_weight, breadth_weight

    def _breadth_candidate_interest(
        self,
        state: InterviewSessionState,
        current_point: InterviewPoint,
    ) -> float:
        """Return the interest factor of the next legal breadth point."""
        point = self._projected_breadth_point_after_answer(state, current_point)
        return point.interest_score if point is not None else current_point.interest_score

    def _projected_breadth_point_after_answer(
        self,
        state: InterviewSessionState,
        current_point: InterviewPoint,
    ) -> InterviewPoint | None:
        """Select the next breadth point after treating the current answer as complete.

        The active subject is the interview module: one internship company,
        one project, or the shared skills/other module. A breadth question
        stays within that module until it has no eligible point or has reached
        its question budget, then advances to the following module.
        """
        projected = replace(
            state,
            interview_points=tuple(
                replace(point, point_status="completed")
                if point.point_id == current_point.point_id
                else point
                for point in state.interview_points
            ),
        )
        point = self._select_next_interview_point(projected)
        if point is not None:
            return point
        advanced = self._advance_structured_stage(projected)
        return self._select_next_interview_point(advanced)

    def _select_next_interview_point(self, state: InterviewSessionState) -> InterviewPoint | None:
        line = self._line_for_stage(state.interview_stage)
        if line is None:
            return None
        subjects = tuple(
            dict.fromkeys(point.subject for point in state.interview_points if point.category is line)
        )
        if not subjects:
            return None
        active_subject = state.active_subject if state.active_subject in subjects else subjects[0]
        start_index = subjects.index(active_subject)
        for subject in subjects[start_index:]:
            subject_points = [
                point
                for point in state.interview_points
                if point.category is line and point.subject == subject
            ]
            if not subject_points:
                continue
            sample = subject_points[0]
            if not self._subject_budget_remaining(state, sample):
                continue
            candidates = [
                point
                for point in subject_points
                if self._point_observations(state, point.point_id) < state.max_questions_per_point
                and point.point_status not in {"completed", "weak", "irrelevant"}
            ]
            if candidates:
                return sorted(
                    candidates,
                    key=lambda point: (
                        -point.interest_score,
                        self._point_observations(state, point.point_id),
                        _stable_rank(state.random_seed, len(state.turns), point.point_id),
                    ),
                )[0]
        return None

    def _knowledge_candidates_for_point(
        self,
        state: InterviewSessionState,
        point: InterviewPoint,
        *,
        top_k: int = 3,
        depth: bool = False,
        current_question: str = "",
    ) -> tuple[InterviewKnowledgeChunk, ...]:
        self._last_graph_retrieval_trace = {}
        depth_corpus_id: str | None = None
        if depth and state.active_depth_chain_blocks:
            chain_chunks = self._knowledge_by_ids(state, state.active_depth_chain_blocks[-1:])
            if chain_chunks:
                depth_corpus_id = chain_chunks[-1].corpus_id
        graph_prefetch_pending = False
        if depth and state.turns:
            previous_question = state.turns[-1].question
            prefetched, prefetch_state = self._prefetched_depth_candidates(
                state,
                previous_question,
            )
            if prefetched:
                return prefetched[:top_k]
            graph_prefetch_pending = prefetch_state == "prefetch_pending"
            if graph_prefetch_pending:
                self._last_graph_retrieval_trace = {
                    "mode": "depth_prefetch_pending",
                    "fallback_used": True,
                    "hits": [],
                }
        # Breadth candidates were resolved for every interview point during
        # session creation. Reuse that immutable snapshot before touching the
        # Graph RAG cache/database again.
        if not depth and point.cached_knowledge_refs:
            cached = self._knowledge_by_ids(state, point.cached_knowledge_refs)
            if cached:
                self._last_graph_retrieval_trace = {
                    "mode": "breadth_cache",
                    "source": "session_point_cache",
                    "query": " ".join((point.point, *point.question_directions)),
                    "returned_node_ids": [chunk.chunk_id for chunk in cached[:top_k]],
                    "hits": [
                        {"node_id": chunk.chunk_id, "score": 0.0, "origin": "session_point_cache"}
                        for chunk in cached[:top_k]
                    ],
                    "fallback_used": False,
                    "timings_ms": {"total": 0.0},
                }
                return cached[:top_k]
        if self._uses_graph_knowledge(state) and not graph_prefetch_pending:
            self._validate_knowledge_snapshot(state)
            if depth and state.active_depth_chain_blocks:
                result = self.graph_knowledge_index.retrieve_depth(
                    active_depth_chain=state.active_depth_chain_blocks,
                    current_question=current_question or point.point,
                    used_node_ids=set(state.used_knowledge_point_ids),
                    corpus_ids=state.knowledge_corpus_ids,
                    top_k=top_k,
                )
            else:
                jd_text = ""
                if state.jd_id and state.jd_revision:
                    jd = self.career_store.get(
                        state.jd_id,
                        state.jd_revision,
                        expected_type=JDDocument,
                    )
                    jd_text = jd.raw_text
                result = self.graph_knowledge_index.precompute_breadth(
                    point_id=point.point_id,
                    point_text=" ".join((point.point, *point.question_directions)),
                    jd_text=jd_text,
                    corpus_ids=state.knowledge_corpus_ids,
                    context_version=(
                        f"{state.jd_id}:{state.jd_revision}:"
                        f"{state.resume_document_id}:{state.resume_document_revision}"
                    ),
                    top_k=top_k,
                )
            self._last_graph_retrieval_trace = self.graph_knowledge_index.trace_dict(result)
            chunks = self.graph_knowledge_index.result_chunks(result)
            if not chunks:
                raise ValueError(
                    f"Graph RAG returned no interview knowledge for resume point: {point.point}"
                )
            return chunks[:top_k]
        if point.cached_knowledge_refs:
            cached = self._knowledge_by_ids(state, point.cached_knowledge_refs)
            if depth_corpus_id is not None:
                cached = tuple(chunk for chunk in cached if chunk.corpus_id == depth_corpus_id)
            if cached:
                return cached[:top_k]
        used = set(state.used_knowledge_point_ids)
        query_tokens = _tokens(
            " ".join((point.point, *point.question_directions))
        )
        if state.jd_id and state.jd_revision:
            jd = self.career_store.get(state.jd_id, state.jd_revision, expected_type=JDDocument)
            query_tokens |= _tokens(jd.raw_text)
        chunks = [
            chunk
            for chunk in self._all_knowledge(state)
            if chunk.chunk_id not in used
            and (depth_corpus_id is None or chunk.corpus_id == depth_corpus_id)
        ]
        if not chunks:
            chunks = list(self._all_knowledge(state))
        ranked = sorted(
            chunks,
            key=lambda chunk: (
                -len(query_tokens & _tokens(f"{chunk.topic} {chunk.text}")),
                chunk.difficulty,
                _stable_rank(state.random_seed, len(state.turns), chunk.chunk_id),
            ),
        )
        return tuple(ranked[:top_k])

    @staticmethod
    def _select_profile(jd: JDDocument, resume: ResumeDocument | None, seed: int) -> InterviewerProfile:
        profiles = (
            ("balanced-senior", "Balanced senior interviewer", "balanced", "senior", 0.50, 0.55),
            ("depth-expert", "Depth-oriented domain expert", "analytical", "staff", 0.72, 0.25),
            ("broad-hiring", "Broad hiring interviewer", "pragmatic", "senior", 0.42, 0.82),
        )
        resume_identity = (
            f"{resume.resume_document_id}:{resume.revision}"
            if resume is not None
            else "no-verified-resume"
        )
        digest = hashlib.sha256(f"{int(seed)}:{jd.jd_id}:{jd.revision}:{resume_identity}".encode()).digest()
        values = profiles[int.from_bytes(digest[:4], "big") % len(profiles)]
        return InterviewerProfile(
            profile_id=values[0],
            name=values[1],
            temperament=values[2],
            seniority=values[3],
            pressure_level=values[4],
            breadth_bias=values[5],
        )

    @staticmethod
    def _select_gap_profile(seed: int) -> InterviewerProfile:
        profiles = (
            ("gap-depth-coach", "Knowledge gap depth interviewer", "analytical", "senior", 0.68, 0.24),
            ("gap-balanced-coach", "Knowledge gap balanced interviewer", "balanced", "senior", 0.55, 0.42),
        )
        digest = hashlib.sha256(f"gap-analysis:{int(seed)}".encode()).digest()
        values = profiles[int.from_bytes(digest[:4], "big") % len(profiles)]
        return InterviewerProfile(
            profile_id=values[0],
            name=values[1],
            temperament=values[2],
            seniority=values[3],
            pressure_level=values[4],
            breadth_bias=values[5],
        )

    @staticmethod
    def _select_question_type(
        state: InterviewSessionState,
        *,
        decision: InterviewDecision,
        knowledge: InterviewKnowledgeChunk,
        resume_claim_refs: tuple[str, ...],
    ) -> InterviewQuestionType:
        if decision.action is InterviewAction.FOLLOW_UP and decision.thread_id:
            previous = next(
                (
                    turn.question
                    for turn in reversed(state.turns)
                    if turn.question.thread_id == decision.thread_id
                ),
                None,
            )
            if previous is not None:
                return previous.question_type
        if not state.turns:
            return (
                InterviewQuestionType.PROJECT_SCENARIO
                if resume_claim_refs
                else InterviewQuestionType.KNOWLEDGE
            )
        choices = [InterviewQuestionType.KNOWLEDGE, InterviewQuestionType.CASE]
        if resume_claim_refs:
            choices.append(InterviewQuestionType.PROJECT_SCENARIO)
        if state.profile.breadth_bias >= 0.5:
            choices.extend((InterviewQuestionType.BEHAVIORAL, InterviewQuestionType.OPEN))
        digest = hashlib.sha256(
            f"{state.random_seed}:{len(state.turns)}:{knowledge.chunk_id}:question-type".encode()
        ).digest()
        return choices[int.from_bytes(digest[:4], "big") % len(choices)]

    def _question_utility(
        self,
        state: InterviewSessionState,
        *,
        decision: InterviewDecision,
        knowledge: InterviewKnowledgeChunk,
        jd_requirement_refs: tuple[str, ...],
        resume_claim_refs: tuple[str, ...],
    ) -> QuestionUtility:
        chunk_tokens = _tokens(f"{knowledge.topic} {knowledge.text}")
        jd_tokens: set[str] = set()
        resume_tokens: set[str] = set()
        if state.jd_id and state.jd_revision:
            jd = self.career_store.get(state.jd_id, state.jd_revision, expected_type=JDDocument)
            jd_tokens = _tokens(jd.raw_text)
        if state.resume_document_id and state.resume_document_revision:
            resume = self.career_store.get(
                state.resume_document_id,
                state.resume_document_revision,
                expected_type=ResumeDocument,
            )
            claims = [
                self.career_store.get(ref.record_id, ref.revision, expected_type=ResumeClaim)
                for ref in resume.claim_refs
                if ref.record_id in resume_claim_refs
            ]
            resume_tokens = _tokens(" ".join(claim.text for claim in claims))

        def overlap(source: set[str]) -> float:
            return min(1.0, len(chunk_tokens & source) / max(1, min(len(chunk_tokens), len(source))))

        observations = state.belief_state.point_observation_count(knowledge.chunk_id)
        thread = next((item for item in state.threads if item.thread_id == decision.thread_id), None)
        followup_cost = (
            thread.followup_count / thread.max_followups
            if thread is not None and decision.action is InterviewAction.FOLLOW_UP
            else 0.0
        )
        jd_relevance = max(overlap(jd_tokens), 1.0 if jd_requirement_refs else 0.0)
        resume_relevance = max(overlap(resume_tokens), 1.0 if resume_claim_refs else 0.0)
        # Confidence is intentionally excluded from candidate utility. It is
        # consumed only as C_current by the depth/breadth formula after an
        # answer has been evaluated.
        uncertainty = 0.5
        weakness = 0.5
        profile_preference = (
            state.profile.pressure_level
            if decision.action is InterviewAction.FOLLOW_UP
            else state.profile.breadth_bias
        )
        repetition = min(1.0, observations / 3)
        if state.mode is InterviewMode.GAP_ANALYSIS:
            score = max(
                0.0,
                min(
                    1.0,
                    0.30 * weakness
                    + 0.20 * uncertainty
                    + 0.24 * jd_relevance
                    + 0.12 * profile_preference
                    - 0.1 * repetition
                    - 0.06 * followup_cost
                    + 0.18,
                ),
            )
        else:
            score = max(
                0.0,
                min(
                    1.0,
                    0.25 * jd_relevance
                    + 0.18 * resume_relevance
                    + 0.2 * uncertainty
                    + 0.17 * weakness
                    + 0.12 * profile_preference
                    - 0.05 * repetition
                    - 0.03 * followup_cost
                    + 0.16,
                ),
            )
        return QuestionUtility(
            jd_relevance=jd_relevance,
            resume_relevance=resume_relevance,
            knowledge_uncertainty=uncertainty,
            weakness=weakness,
            profile_preference=profile_preference,
            repetition=repetition,
            followup_cost=followup_cost,
            score=score,
        )

    @staticmethod
    def _observe_answer(
        state: InterviewSessionState,
        *,
        question: InterviewQuestion,
        answer: str,
        knowledge: tuple[InterviewKnowledgeChunk, ...],
        assessment: InterviewAnswerAssessment,
    ) -> AnswerObservation:
        normalized = str(answer or "").strip()
        explicit_unknown = bool(
            re.search(
                r"(?i)(^|\b)(i\s+(?:do\s+not|don't)\s+know|not\s+sure|no\s+idea)(\b|$)|"
                r"(不知道|不清楚|不会|没了解过)",
                normalized,
            )
        )
        answer_tokens = _tokens(normalized)
        prior_tokens = set().union(
            *(
                _tokens(turn.answer)
                for turn in state.turns
                if turn.question.thread_id == question.thread_id
            )
        ) if state.turns else set()
        new_information_ratio = (
            len(answer_tokens - prior_tokens) / max(1, len(answer_tokens))
            if prior_tokens
            else 1.0
        )
        # Subjective project answers are not graded by overlap with retrieved
        # chunks. Use the question plus the evaluator's quality signal for the
        # coarse observation used by hard stops.
        question_tokens = _tokens(question.text)
        lexical_relevance = len(answer_tokens & question_tokens) / max(1, len(answer_tokens))
        relevance = max(
            lexical_relevance,
            assessment.confidence * (assessment.correctness + assessment.completeness) / 2,
        )
        return AnswerObservation(
            explicit_unknown=explicit_unknown,
            new_information_ratio=min(1.0, new_information_ratio),
            relevance=min(1.0, relevance),
            answer_length=len(normalized),
        )

    def _validated_confidence_updates(
        self,
        *,
        state: InterviewSessionState,
        question: InterviewQuestion,
        knowledge: tuple[InterviewKnowledgeChunk, ...],
        assessment: InterviewAnswerAssessment,
    ) -> tuple[KnowledgeConfidenceUpdate, ...]:
        """Return legacy block updates for deterministic compatibility only.

        Model evaluation of subjective project answers no longer claims that an
        answer covered a particular knowledge block, so it cannot emit mastery
        updates keyed to retrieved chunks.
        """
        if assessment.evaluator_name.startswith("llm-") or question.question_type is not InterviewQuestionType.KNOWLEDGE:
            return ()
        by_id = {chunk.chunk_id: chunk for chunk in knowledge}
        allowed_ids = set(question.knowledge_refs)
        cited_ids = set(assessment.knowledge_refs)
        raw_updates = assessment.confidence_updates
        updates: list[KnowledgeConfidenceUpdate] = []
        seen: set[str] = set()
        for raw in raw_updates:
            point_id = str(raw.get("chunk_id") or raw.get("knowledge_point_id") or "").strip()
            if not point_id:
                raise ValueError("Confidence update requires chunk_id")
            if point_id not in allowed_ids or point_id not in by_id:
                raise ValueError("Confidence update must target a knowledge block used by the question")
            if point_id in seen:
                raise ValueError("A knowledge block can be updated only once per answer")
            should_update = raw.get("should_update", True)
            if not isinstance(should_update, bool):
                raise ValueError("Confidence update should_update must be boolean")
            if not should_update:
                # The evaluator is expected to report uncovered supplied
                # blocks with should_update=false. Such a block need not be
                # cited because no mastery update is being asserted.
                seen.add(point_id)
                continue
            if point_id not in cited_ids:
                raise ValueError("Confidence update must target a knowledge block cited by the evaluator")
            if "nc" not in raw and "confidence" not in raw and "observed_confidence" not in raw:
                raise ValueError("Confidence update requires nc")
            observed = raw.get("nc", raw.get("observed_confidence", raw.get("confidence")))
            chunk = by_id[point_id]
            updates.append(
                KnowledgeConfidenceUpdate(
                    knowledge_point_id=point_id,
                    corpus_id=chunk.corpus_id,
                    observed_confidence=float(observed),
                    reason=str(raw.get("reason") or ""),
                    document_prior_count=state.belief_state.document_prior_count_by_corpus.get(
                        chunk.corpus_id,
                        self._knowledge_corpus_prior_counts(
                            (chunk.corpus_id,),
                            graph_versions=state.knowledge_corpus_versions,
                        ).get(chunk.corpus_id, 1),
                    ),
                )
            )
            seen.add(point_id)

        if updates or raw_updates or assessment.evaluator_name.startswith("llm-"):
            return tuple(updates)

        # Deterministic and test evaluators predate the per-block protocol.
        # Their score is a controlled fallback, not an LLM-authored update.
        fallback = (assessment.correctness + assessment.completeness) / 2
        return tuple(
            KnowledgeConfidenceUpdate(
                knowledge_point_id=chunk_id,
                corpus_id=by_id[chunk_id].corpus_id,
                observed_confidence=fallback,
                reason="Deterministic evaluator fallback for a cited knowledge block.",
                document_prior_count=state.belief_state.document_prior_count_by_corpus.get(
                    by_id[chunk_id].corpus_id,
                    self._knowledge_corpus_prior_counts(
                        (by_id[chunk_id].corpus_id,),
                        graph_versions=state.knowledge_corpus_versions,
                    ).get(by_id[chunk_id].corpus_id, 1),
                ),
            )
            for chunk_id in question.knowledge_refs
            if chunk_id in cited_ids and chunk_id in by_id
        )

    def knowledge_chunks(
        self,
        interview_id: str,
        chunk_ids: tuple[str, ...],
    ) -> tuple[InterviewKnowledgeChunk, ...]:
        if not chunk_ids:
            return ()
        state = self.session_store.get(interview_id)
        return self._knowledge_by_ids(state, chunk_ids)

    def _uses_graph_knowledge(self, state: InterviewSessionState) -> bool:
        versions = state.knowledge_corpus_versions
        return bool(
            state.mode is InterviewMode.REALISTIC_SIMULATION
            and self.graph_knowledge_index is not None
            and versions
            and set(versions) == set(state.knowledge_corpus_ids)
            and all(re.fullmatch(r"[0-9a-f]{64}", version.casefold()) for version in versions.values())
        )

    def _knowledge_corpus_prior_counts(
        self,
        corpus_ids: tuple[str, ...],
        *,
        graph_versions: dict[str, str] | None = None,
    ) -> dict[str, int]:
        """Freeze document priors from the same corpus snapshot used by a session."""
        if graph_versions and self.graph_knowledge_index is not None:
            return self.graph_knowledge_index.corpus_chunk_counts(corpus_ids)
        counts: dict[str, int] = {}
        for corpus_id in corpus_ids:
            corpus = self.knowledge_store.get(corpus_id)
            counts[corpus.corpus_id] = max(1, len(corpus.chunks))
        return counts

    def _all_knowledge(self, state: InterviewSessionState) -> tuple[InterviewKnowledgeChunk, ...]:
        if self._uses_graph_knowledge(state):
            raise RuntimeError("Full JSON knowledge scan is disabled for Graph RAG interviews")
        corpora = tuple(self.knowledge_store.get(corpus_id) for corpus_id in state.knowledge_corpus_ids)
        if state.mode is InterviewMode.REALISTIC_SIMULATION and state.knowledge_corpus_versions:
            changed = [
                corpus.corpus_id
                for corpus in corpora
                if state.knowledge_corpus_versions.get(corpus.corpus_id) != corpus.version
            ]
            if changed:
                raise ValueError(
                    "Realistic interview knowledge snapshot changed after session creation: "
                    + ", ".join(sorted(changed))
                )
        return tuple(chunk for corpus in corpora for chunk in corpus.chunks)

    def _validate_knowledge_snapshot(self, state: InterviewSessionState) -> None:
        if self._uses_graph_knowledge(state):
            current = self.graph_knowledge_index.corpus_versions(state.knowledge_corpus_ids)
            if current != state.knowledge_corpus_versions:
                changed = sorted(
                    set(state.knowledge_corpus_versions)
                    | set(current)
                )
                raise ValueError(
                    "Realistic interview Graph RAG snapshot changed after session creation: "
                    + ", ".join(changed)
                )
            return
        self._all_knowledge(state)

    def _knowledge_by_ids(
        self,
        state: InterviewSessionState,
        chunk_ids: tuple[str, ...],
    ) -> tuple[InterviewKnowledgeChunk, ...]:
        if not chunk_ids:
            return ()
        if self._uses_graph_knowledge(state):
            self._validate_knowledge_snapshot(state)
            return self.graph_knowledge_index.get_chunks(
                chunk_ids,
                corpus_ids=state.knowledge_corpus_ids,
            )
        index = {chunk.chunk_id: chunk for chunk in self._all_knowledge(state)}
        missing = set(chunk_ids) - set(index)
        if missing:
            raise ValueError(f"Interview question cites missing knowledge chunks: {sorted(missing)}")
        return tuple(index[chunk_id] for chunk_id in chunk_ids)

    def _has_unused_knowledge(self, state: InterviewSessionState) -> bool:
        used_chunk_ids = {
            chunk_id
            for turn in state.turns
            for chunk_id in turn.question.knowledge_refs
        }
        return any(
            chunk.chunk_id not in used_chunk_ids
            for chunk in self._all_knowledge(state)
        )

    def _select_knowledge(
        self,
        state: InterviewSessionState,
        *,
        topic: str | None = None,
    ) -> InterviewKnowledgeChunk:
        authorized = self._all_knowledge(state)
        candidates = [chunk for chunk in authorized if topic is None or chunk.topic == topic]
        used_chunk_ids = {
            chunk_id
            for turn in state.turns
            for chunk_id in turn.question.knowledge_refs
        }
        unused_candidates = [chunk for chunk in candidates if chunk.chunk_id not in used_chunk_ids]
        # A breadth transition must consume a new knowledge point. A depth chain
        # prefers one too, but may retain its sole source as conversational context.
        if unused_candidates:
            candidates = unused_candidates
        elif topic is None:
            raise ValueError("No unused interview knowledge remains for a new topic")
        if state.mode is InterviewMode.GAP_ANALYSIS and self.live_knowledge_retriever is not None:
            result = self.live_knowledge_retriever.retrieve(
                self._gap_retrieval_query(state, candidates, topic=topic),
                filters=InterviewRetrievalFilters(
                    tenant_id=state.tenant_id,
                    topics=(topic,) if topic else (),
                    corpus_ids=state.knowledge_corpus_ids,
                ),
                top_k=min(8, max(1, len(candidates))),
            )
            allowed_ids = {chunk.chunk_id for chunk in candidates}
            retrieved = []
            retrieved_ids: set[str] = set()
            for hit in result.hits:
                for chunk in (hit.chunk, *getattr(hit, "context_chunks", ())):
                    if chunk.chunk_id in allowed_ids and chunk.chunk_id not in retrieved_ids:
                        retrieved.append(chunk)
                        retrieved_ids.add(chunk.chunk_id)
            # Keep the prefiltered local candidates when a live retriever has
            # no hit. The corpus/tenant filter remains authoritative, while a
            # transient empty retrieval must not strand the interview.
            if retrieved:
                candidates = retrieved
        if not candidates:
            raise ValueError(f"No interview knowledge is available for topic: {topic or 'any'}")
        jd_tokens: set[str] = set()
        resume_tokens: set[str] = set()
        if state.jd_id and state.jd_revision:
            jd = self.career_store.get(state.jd_id, state.jd_revision, expected_type=JDDocument)
            jd_tokens = _tokens(jd.raw_text)
        if state.resume_document_id and state.resume_document_revision:
            resume = self.career_store.get(
                state.resume_document_id,
                state.resume_document_revision,
                expected_type=ResumeDocument,
            )
            claims = [
                self.career_store.get(ref.record_id, ref.revision, expected_type=ResumeClaim)
                for ref in resume.claim_refs
            ]
            resume_tokens = _tokens(" ".join(claim.text for claim in claims))
        def utility(chunk: InterviewKnowledgeChunk) -> float:
            chunk_tokens = _tokens(chunk.topic + " " + chunk.text)
            observations = state.belief_state.point_observation_count(chunk.chunk_id)
            if state.mode is InterviewMode.GAP_ANALYSIS:
                jd_overlap = len(jd_tokens & chunk_tokens) / max(
                    1,
                    min(len(jd_tokens), len(chunk_tokens)),
                )
                return (
                    2.0 * jd_overlap
                    + 1.25 / (1 + observations)
                    + state.profile.pressure_level * chunk.difficulty / 5
                )
            return (
                2.0 * len(jd_tokens & chunk_tokens)
                + 1.5 * len(resume_tokens & chunk_tokens)
                + state.profile.breadth_bias / (1 + observations)
                + state.profile.pressure_level * chunk.difficulty / 5
            )

        candidates.sort(
            key=lambda chunk: (
                -utility(chunk),
                state.belief_state.point_observation_count(chunk.chunk_id),
                _stable_rank(state.random_seed, len(state.turns), chunk.chunk_id),
            )
        )
        return candidates[0]

    @staticmethod
    def _gap_retrieval_query(
        state: InterviewSessionState,
        candidates: list[InterviewKnowledgeChunk],
        *,
        topic: str | None,
    ) -> str:
        parts: list[str] = []
        if topic:
            parts.append(topic)
        if state.turns:
            parts.extend((state.turns[-1].question.topic, state.turns[-1].answer))
        weak_chunks = sorted(
            candidates,
            key=lambda chunk: (
                state.belief_state.point_observation_count(chunk.chunk_id),
                chunk.chunk_id,
            ),
        )
        parts.extend(chunk.topic for chunk in weak_chunks[:3])
        return " ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))

    @staticmethod
    def _validate_generated_question(
        state: InterviewSessionState,
        question: InterviewQuestion,
        *,
        decision: InterviewDecision,
        thread_id: str,
        knowledge: InterviewKnowledgeChunk,
        jd_requirement_refs: tuple[str, ...],
        resume_claim_refs: tuple[str, ...],
    ) -> None:
        if not isinstance(question, InterviewQuestion):
            raise ValueError("Interview question generator must return InterviewQuestion")
        if question.thread_id != thread_id or question.topic != decision.topic or question.path is not decision.path:
            raise ValueError("Generated interview question changed the selected topic, path, or thread")
        if question.question_type is not decision.question_type or question.utility != decision.utility:
            raise ValueError("Generated interview question changed its runtime-owned type or utility")
        if decision.action is InterviewAction.FOLLOW_UP:
            if not question.parent_question_id or question.followup_depth < 1:
                raise ValueError("Follow-up question requires a parent question and positive depth")
        elif question.parent_question_id is not None or question.followup_depth != 0:
            raise ValueError("Root question cannot declare a parent or follow-up depth")
        expected_jd_ref = (
            (state.jd_id, state.jd_revision)
            if state.jd_id and state.jd_revision
            else None
        )
        expected_resume_ref = (
            (state.resume_document_id, state.resume_document_revision)
            if state.resume_document_id and state.resume_document_revision
            else None
        )
        if question.jd_ref != expected_jd_ref:
            raise ValueError("Generated interview question changed the authorized JD reference")
        if question.verified_resume_ref != expected_resume_ref:
            raise ValueError("Generated interview question changed the authorized resume reference")
        authorized_knowledge_refs = (
            {block.chunk_id for block in question.knowledge_block_refs}
            if question.knowledge_block_refs
            else {knowledge.chunk_id}
        )
        if set(question.knowledge_refs) != authorized_knowledge_refs:
            raise ValueError("Generated interview question must cite exactly the authorized knowledge blocks")
        if not set(question.jd_requirement_refs).issubset(jd_requirement_refs):
            raise ValueError("Generated interview question contains an unauthorized JD requirement reference")
        if not set(question.resume_claim_refs).issubset(resume_claim_refs):
            raise ValueError("Generated interview question contains an unauthorized resume claim reference")

    def _artifact_links(
        self,
        state: InterviewSessionState,
        knowledge: InterviewKnowledgeChunk,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        target = _tokens(knowledge.topic + " " + knowledge.text)
        requirement_refs: tuple[str, ...] = ()
        claim_refs: tuple[str, ...] = ()
        if not state.jd_id or not state.jd_revision:
            return requirement_refs, claim_refs
        jd = self.career_store.get(state.jd_id, state.jd_revision, expected_type=JDDocument)
        declared_requirements = {
            ref.record_id: self.career_store.get(ref.record_id, ref.revision, expected_type=JDRequirement)
            for ref in jd.requirement_refs
        }
        # Older imported JD records may not have backfilled requirement_refs. Restrict
        # the fallback to requirements whose jd_ref points to this exact JD.
        for requirement in self.career_store.list_latest(JDRequirement):
            if requirement.jd_ref.record_id == jd.jd_id:
                declared_requirements.setdefault(requirement.requirement_id, requirement)
        requirements = list(declared_requirements.values())
        ranked_requirements = sorted(
            requirements,
            key=lambda item: (-len(target & _tokens(item.text)), item.requirement_id),
        )
        requirement_refs = tuple(
            item.requirement_id
            for item in ranked_requirements[:3]
            if target & _tokens(item.text)
        )
        if not state.resume_document_id or not state.resume_document_revision:
            return requirement_refs, claim_refs
        resume = self.career_store.get(
            state.resume_document_id,
            state.resume_document_revision,
            expected_type=ResumeDocument,
        )
        claims = [
            self.career_store.get(ref.record_id, ref.revision, expected_type=ResumeClaim)
            for ref in resume.claim_refs
        ]
        ranked_claims = sorted(
            claims,
            key=lambda item: (-len(target & _tokens(item.text)), item.claim_id),
        )
        claim_refs = tuple(
            item.claim_id
            for item in ranked_claims[:2]
            if target & _tokens(item.text)
        )
        return requirement_refs, claim_refs

    @staticmethod
    def _score_label(score: float) -> str:
        if score >= 0.8:
            return "strong"
        if score >= 0.6:
            return "adequate"
        if score >= 0.4:
            return "mixed"
        return "needs_improvement"
