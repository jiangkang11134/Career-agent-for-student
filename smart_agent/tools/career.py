"""Deterministic support tools for the Career Agent domain."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

from smart_agent.career import (
    RESUME_SECTION_ORDER,
    AuthorityAwareReranker,
    CandidateFactStatus,
    CandidateFactStore,
    CareerGoal,
    CareerGraphRAGIndex,
    CareerStore,
    CareerWorkflowEngine,
    CareerWorkspaceIndexer,
    ConstraintResumePlanner,
    HashingInterviewEmbedding,
    HeuristicCareerFactExtractor,
    HeuristicJDCapabilityTreeBuilder,
    HybridInterviewKnowledgeRetriever,
    InterviewKnowledgeChunk,
    InterviewKnowledgeCorpus,
    InterviewKnowledgeLibrary,
    InterviewKnowledgeSourceType,
    InterviewKnowledgeStore,
    InterviewKnowledgeType,
    InterviewMode,
    InterviewRetrievalFilters,
    InterviewRuntime,
    InterviewSessionStore,
    JDDocument,
    JDRequirement,
    KnowledgeMasteryStore,
    LineProjectChunker,
    LLMCareerFactExtractor,
    LLMInterviewAnswerEvaluator,
    LLMInterviewQuestionGenerator,
    LLMJDAnalyzer,
    LLMProjectCandidateSelector,
    LocalBGEInterviewEmbedding,
    LocalBGEInterviewReranker,
    MaterialCategory,
    MaterialStore,
    ModelAPIIdentity,
    OpenAICompatibleInterviewReranker,
    PersonalEvidence,
    PersonalMaterialWorkflow,
    PlannerWeights,
    ProjectAnalysisRecord,
    ProjectAnalysisRunStore,
    ProjectAnalysisStore,
    ProjectDeepDiveService,
    ProjectDeepDiveStore,
    ProjectDescriptionDocument,
    ProjectDescriptionService,
    ProjectExperienceService,
    ProjectLibraryStore,
    ProjectSearchGate,
    ProjectSourceStatus,
    ProjectSourceType,
    ProjectUnderstanding,
    ResumeCandidateBuilder,
    ResumeDocumentBuilder,
    ResumeHeader,
    ResumePDFRenderer,
    ResumePlan,
    ResumeRevisionManager,
    ResumeSource,
    ResumeVerificationAgent,
    VerifierModelConfig,
    create_career_client,
)
from smart_agent.career.jd import JDAnalyzer
from smart_agent.career.retrieval import (
    HybridEvidenceRetriever,
    OpenAICompatibleEmbeddingProvider,
    RetrievalConfig,
)
from smart_agent.review.career import extract_tech_skills, review_resume
from smart_agent.tooling import ToolDefinition, ToolResult


def _require_text(input_data: dict) -> dict:
    text = input_data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text is required")
    return {"text": text.strip()}


def _career_store(context) -> CareerStore:
    runtime = context._runtime or {}
    injected = runtime.get("_careerStore")
    if isinstance(injected, CareerStore):
        return injected
    workspace = Path(context.cwd).resolve()
    configured = runtime.get("careerStorePath")
    root = Path(configured).expanduser() if configured else workspace / ".runtime" / "career"
    if not root.is_absolute():
        root = workspace / root
    root = root.resolve()
    if not root.is_relative_to(workspace):
        raise ValueError("careerStorePath must stay inside the active workspace.")
    return CareerStore(root)


def _material_store(context) -> MaterialStore:
    runtime = context._runtime or {}
    injected = runtime.get("_materialStore")
    if isinstance(injected, MaterialStore):
        return injected
    store = _career_store(context)
    extractor = runtime.get("_materialTextExtractor")
    chunker = runtime.get("_projectChunker")
    if extractor is not None and not callable(getattr(extractor, "extract", None)):
        raise ValueError("_materialTextExtractor must expose extract(path)")
    if chunker is not None and not callable(getattr(chunker, "chunk", None)):
        raise ValueError("_projectChunker must expose chunk(text, ...)" )
    configured_max_chars = int(runtime.get("careerChunkMaxChars", 1400))
    configured_overlap = int(runtime.get("careerChunkOverlapChars", 160))
    return MaterialStore(
        store.root / "materials",
        workspace=context.cwd,
        text_extractor=extractor,
        project_chunker=chunker or LineProjectChunker(
            max_chars=configured_max_chars,
            overlap_chars=configured_overlap,
        ),
    )


def _project_library_store(context) -> ProjectLibraryStore:
    store = _career_store(context)
    return ProjectLibraryStore(store.root / "project-library", workspace=context.cwd)


def _project_analysis_store(context) -> ProjectAnalysisStore:
    store = _career_store(context)
    return ProjectAnalysisStore(store.root / "project-library" / "analyses")


def _project_experience_service(context) -> ProjectExperienceService:
    store = _career_store(context)
    return ProjectExperienceService(
        career_store=store,
        project_store=_project_library_store(context),
        analysis_store=_project_analysis_store(context),
        run_store=ProjectAnalysisRunStore(store.root / "project-library" / "runs"),
    )


def _project_deep_dive_service(context) -> ProjectDeepDiveService:
    runtime = context._runtime or {}
    store = _career_store(context)
    client = create_career_client(
        runtime,
        "project_matcher",
        audit_root=store.root / "audit",
    )
    return ProjectDeepDiveService(
        project_store=_project_library_store(context),
        decision_store=ProjectDeepDiveStore(store.root / "project-library" / "deep-dives"),
        gate=ProjectSearchGate(
            selector=LLMProjectCandidateSelector(client) if client is not None else None,
            high_threshold=float(runtime.get("careerProjectSearchHighThreshold", 0.65)),
            medium_threshold=float(runtime.get("careerProjectSearchMediumThreshold", 0.40)),
            sufficient_evidence_score=float(runtime.get("careerProjectSufficientEvidenceScore", 0.72)),
            max_candidate_projects=int(runtime.get("careerProjectSearchMaxCandidates", 3)),
        ),
    )


def _interview_knowledge_store(context) -> InterviewKnowledgeStore:
    runtime = context._runtime or {}
    injected = runtime.get("_interviewKnowledgeStore")
    if isinstance(injected, InterviewKnowledgeStore):
        return injected
    store = _career_store(context)
    return InterviewKnowledgeStore(store.root / "interview-knowledge")


def _interview_knowledge_library(context) -> InterviewKnowledgeLibrary:
    runtime = context._runtime or {}
    injected = runtime.get("_interviewKnowledgeLibrary")
    if injected is not None:
        required = ("import_source", "list_sources", "verify_source_integrity")
        if not all(callable(getattr(injected, name, None)) for name in required):
            raise ValueError(f"_interviewKnowledgeLibrary must expose {', '.join(required)}")
        return injected
    store = _career_store(context)
    return InterviewKnowledgeLibrary(
        store.root / "interview-knowledge",
        max_chunk_chars=int(runtime.get("careerInterviewChunkMaxChars", 1200)),
    )


def _career_graph_rag_index(context) -> CareerGraphRAGIndex:
    runtime = context._runtime or {}
    injected = runtime.get("_careerGraphRAGIndex")
    if injected is not None:
        required = (
            "sync_corpus",
            "sync_library",
            "ensure_corpora",
            "corpus_versions",
            "get_chunks",
            "precompute_breadth",
            "retrieve_depth",
            "result_chunks",
            "trace_dict",
        )
        if not all(callable(getattr(injected, name, None)) for name in required):
            raise ValueError(f"_careerGraphRAGIndex must expose {', '.join(required)}")
        return injected
    store = _career_store(context)
    # Use the same configured local BGE-M3 provider for graph ingestion and
    # graph retrieval; fall back to the deterministic provider only when BGE
    # is not configured in the runtime/environment.
    index = CareerGraphRAGIndex(
        store.root / "graph-rag",
        embedding_provider=_interview_embedding_provider(runtime),
    )
    runtime["_careerGraphRAGIndex"] = index
    return index


def _ready_career_graph_rag_index(context):
    library = _interview_knowledge_library(context)
    summary = library.metadata_summary()
    if not summary["source_count"]:
        return None
    return _career_graph_rag_index(context)


def _interview_session_store(context) -> InterviewSessionStore:
    runtime = context._runtime or {}
    injected = runtime.get("_interviewSessionStore")
    if isinstance(injected, InterviewSessionStore):
        return injected
    store = _career_store(context)
    return InterviewSessionStore(store.root / "interviews")


def _interview_mastery_store(context) -> KnowledgeMasteryStore:
    runtime = context._runtime or {}
    injected = runtime.get("_knowledgeMasteryStore")
    if isinstance(injected, KnowledgeMasteryStore):
        return injected
    store = _career_store(context)
    return KnowledgeMasteryStore(store.root / "knowledge-mastery")


def _interview_embedding_provider(runtime: dict, *, required: bool = True):
    injected = runtime.get("_interviewEmbeddingProvider")
    if injected is not None:
        return injected
    if not required:
        return None
    local = LocalBGEInterviewEmbedding.from_runtime(runtime)
    if local is not None:
        runtime["_interviewEmbeddingProvider"] = local
        return local
    provider = HashingInterviewEmbedding(int(runtime.get("careerInterviewEmbeddingDimension", 256)))
    runtime["_interviewEmbeddingProvider"] = provider
    return provider


def _interview_reranker(context, runtime: dict):
    injected = runtime.get("_interviewReranker")
    if injected is not None:
        return injected
    local = LocalBGEInterviewReranker.from_runtime(runtime)
    if local is not None:
        runtime["_interviewReranker"] = local
        return local
    remote = OpenAICompatibleInterviewReranker.from_runtime(
        runtime,
        audit_path=_career_store(context).root / "audit" / "reranker-calls.jsonl",
    )
    if remote is not None:
        runtime["_interviewReranker"] = remote
        return remote
    authority = AuthorityAwareReranker()
    runtime["_interviewReranker"] = authority
    return authority


def _interview_live_retriever(context):
    runtime = context._runtime or {}
    injected = runtime.get("_interviewLiveKnowledgeRetriever")
    if injected is not None:
        if not callable(getattr(injected, "retrieve", None)):
            raise ValueError("_interviewLiveKnowledgeRetriever must expose retrieve(...)")
        return injected
    library = _interview_knowledge_library(context)
    if not library.list_sources():
        return None
    variant = str(runtime.get("careerInterviewLiveRetrievalVariant") or "hybrid").strip().lower()
    valid_variants = {"bm25", "vector", "hybrid", "hybrid_rerank", "hybrid_rrf", "hybrid_rrf_rerank"}
    if variant not in valid_variants:
        raise ValueError("careerInterviewLiveRetrievalVariant is invalid")
    embedding = _interview_embedding_provider(runtime, required=variant != "bm25")
    reranker = runtime.get("_interviewReranker")
    if variant.endswith("_rerank") and reranker is None:
        reranker = _interview_reranker(context, runtime)
    return HybridInterviewKnowledgeRetriever(
        library,
        embedding_provider=embedding,
        reranker=reranker if variant.endswith("_rerank") else None,
        variant=variant,
        bm25_weight=float(runtime.get("careerInterviewBM25Weight", 0.55)),
        vector_weight=float(runtime.get("careerInterviewVectorWeight", 0.45)),
        rrf_k=int(runtime.get("careerInterviewRRFK", 60)),
        neighbor_window=int(runtime.get("careerInterviewNeighborWindow", 1)),
        rerank_candidate_limit=int(runtime.get("careerInterviewRerankCandidateLimit", 30)),
        random_seed=int(runtime.get("careerInterviewRetrievalSeed", 0)),
    )


def _interview_runtime(context) -> InterviewRuntime:
    runtime = context._runtime or {}
    injected = runtime.get("_interviewRuntime")
    if injected is not None:
        required = (
            "create_session",
            "next_question",
            "submit_answer",
            "process_answer",
            "next_question_candidates",
            "generate_depth_question",
            "generate_breadth_question",
            "finish",
            "report",
        )
        if not all(callable(getattr(injected, name, None)) for name in required):
            raise ValueError(f"_interviewRuntime must expose {', '.join(required)}")
        return injected
    career_store = _career_store(context)
    if runtime.get("careerUserId"):
        from smart_agent.career.interview_inputs import InterviewInputStore
        career_store = InterviewInputStore(career_store, runtime["careerUserId"])
    audit_root = career_store.root / "audit"
    question_generator = runtime.get("_interviewQuestionGenerator")
    if question_generator is None:
        question_client = create_career_client(
            runtime,
            "interview_question",
            audit_root=audit_root,
            required=True,
        )
        if question_client is not None:
            question_generator = LLMInterviewQuestionGenerator(question_client, career_store)
    answer_evaluator = runtime.get("_interviewAnswerEvaluator")
    if answer_evaluator is None:
        evaluator_client = create_career_client(
            runtime,
            "interview_evaluator",
            audit_root=audit_root,
        )
        if evaluator_client is not None:
            answer_evaluator = LLMInterviewAnswerEvaluator(evaluator_client)
    knowledge_library = _interview_knowledge_library(context)
    return InterviewRuntime(
        career_store=career_store,
        knowledge_store=_interview_knowledge_store(context),
        session_store=_interview_session_store(context),
        policy_engine=runtime.get("_interviewPolicyEngine"),
        question_generator=question_generator,
        answer_evaluator=answer_evaluator,
        live_knowledge_retriever=_interview_live_retriever(context),
        mastery_store=_interview_mastery_store(context),
        graph_knowledge_index=_ready_career_graph_rag_index(context),
        knowledge_library=knowledge_library,
    )


def _candidate_fact_store(context) -> CandidateFactStore:
    runtime = context._runtime or {}
    injected = runtime.get("_candidateFactStore")
    if isinstance(injected, CandidateFactStore):
        return injected
    if injected is not None:
        required = ("list", "upsert", "import_to_evidence", "reject")
        if not all(callable(getattr(injected, name, None)) for name in required):
            raise ValueError("_candidateFactStore must expose list, upsert, import_to_evidence, and reject")
        return injected
    store = _career_store(context)
    return CandidateFactStore(store.root / "candidate-facts")


def _career_fact_extractor(context):
    runtime = context._runtime or {}
    extractor = runtime.get("_careerFactExtractor")
    if extractor is None:
        client = create_career_client(
            runtime,
            "fact_extractor",
            audit_root=_career_store(context).root / "audit",
        )
        extractor = LLMCareerFactExtractor(client) if client is not None else HeuristicCareerFactExtractor()
    if not callable(getattr(extractor, "extract", None)):
        raise ValueError("_careerFactExtractor must expose extract(manifest, text, ...)")
    return extractor


def _career_workspace_indexer(context):
    runtime = context._runtime or {}
    injected = runtime.get("_careerWorkspaceIndexer")
    if injected is not None:
        if not callable(getattr(injected, "build", None)):
            raise ValueError("_careerWorkspaceIndexer must expose build()")
        return injected
    return CareerWorkspaceIndexer(
        career_store=_career_store(context),
        material_store=_material_store(context),
        candidate_fact_store=_candidate_fact_store(context),
        interview_knowledge_store=_interview_knowledge_store(context),
        interview_knowledge_library=_interview_knowledge_library(context),
        project_library_store=_project_library_store(context),
    )


def _find_material_manifest(context, material_id: str):
    material_id = str(material_id or "").strip()
    manifest = next(
        (item for item in _material_store(context).list_manifests() if item.material_id == material_id),
        None,
    )
    if manifest is None:
        raise ValueError(f"Unknown material_id: {material_id}")
    return manifest


def _validate_inspect_career_workspace(input_data: dict) -> dict:
    goal = str(input_data.get("goal") or CareerGoal.RESUME.value).strip().lower()
    try:
        return {"goal": CareerGoal(goal)}
    except ValueError as exc:
        raise ValueError(f"goal must be one of: {', '.join(item.value for item in CareerGoal)}") from exc


def _inspect_career_workspace(input_data: dict, context) -> ToolResult:
    index = _career_workspace_indexer(context).build()
    runtime = context._runtime or {}
    engine = runtime.get("_careerWorkflowEngine") or CareerWorkflowEngine()
    if not callable(getattr(engine, "inspect", None)):
        raise ValueError("_careerWorkflowEngine must expose inspect(index, goal=...)")
    state = engine.inspect(index, goal=input_data["goal"])
    previous_stage = str(runtime.get("domainStage") or "default")
    runtime["careerGoal"] = state.goal.value
    runtime["careerWorkflowStage"] = state.recommended_stage.value
    runtime["domainStage"] = state.recommended_domain_stage
    if previous_stage != state.recommended_domain_stage:
        runtime["_capabilitiesDirty"] = True
    session = getattr(context, "session", None)
    if session is not None and callable(getattr(session, "set_domain_stage", None)):
        session.set_domain_stage(state.recommended_domain_stage)
    payload = state.to_dict()
    payload["previous_domain_stage"] = previous_stage
    payload["applied_domain_stage"] = state.recommended_domain_stage
    payload["capabilities_reloaded"] = previous_stage != state.recommended_domain_stage
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


inspect_career_workspace_tool = ToolDefinition(
    name="inspect_career_workspace",
    description=(
        "Inspect Career artifacts and recommend the next evidence-safe action for a resume, interview, or "
        "application goal. This is advisory: the Agent chooses the action, while hard gates report what cannot "
        "be skipped."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "goal": {"type": "string", "enum": [item.value for item in CareerGoal]},
        },
    },
    validator=_validate_inspect_career_workspace,
    run=_inspect_career_workspace,
)


def _validate_import_material(input_data: dict) -> dict:
    category = str(input_data.get("category") or "").strip().lower()
    try:
        material_category = MaterialCategory(category)
    except ValueError as exc:
        raise ValueError(f"category must be one of: {', '.join(item.value for item in MaterialCategory)}") from exc
    if material_category is MaterialCategory.PROJECT:
        raise ValueError(
            "Project sources must use register_project_source or the project ZIP endpoint; "
            "project files are not imported into the material RAG index."
        )
    label = str(input_data.get("label") or "").strip()
    if not label:
        raise ValueError("label is required")
    text = input_data.get("text")
    source_path = input_data.get("source_path")
    if (text is None) == (source_path is None):
        raise ValueError("Provide exactly one of text or source_path")
    if text is not None and (not isinstance(text, str) or not text.strip()):
        raise ValueError("text must be a non-empty string")
    if source_path is not None and (not isinstance(source_path, str) or not source_path.strip()):
        raise ValueError("source_path must be a non-empty workspace path")
    if source_path is not None and material_category is not MaterialCategory.RESUME:
        raise ValueError("Internship, education, research, and award experiences must use structured text fields")
    project_id = str(input_data.get("project_id") or "").strip()
    facts = input_data.get("facts", [])
    if not isinstance(facts, list):
        raise ValueError("facts must be an array")
    if facts:
        raise ValueError("facts must be extracted as candidate facts before they can enter PersonalEvidence")
    normalized_facts: list[dict] = []
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError("each fact must be an object")
        field_name = str(fact.get("field_name") or "").strip()
        if not field_name:
            raise ValueError("fact.field_name is required")
        if "value" not in fact:
            raise ValueError("fact.value is required")
        confidence = float(fact.get("confidence", 0.0))
        if not 0 <= confidence <= 1:
            raise ValueError("fact.confidence must be between 0 and 1")
        normalized_facts.append(
            {"field_name": field_name, "value": fact["value"], "confidence": confidence}
        )
    if material_category is MaterialCategory.RESUME and normalized_facts:
        raise ValueError("resume is a read-only source and cannot import facts")
    return {
        "category": material_category,
        "label": label,
        "text": text.strip() if isinstance(text, str) else None,
        "source_path": source_path.strip() if isinstance(source_path, str) else None,
        "project_id": project_id,
        "facts": normalized_facts,
    }


def _import_material(input_data: dict, context) -> ToolResult:
    source_path = input_data["source_path"]
    if source_path:
        workspace = Path(context.cwd).resolve()
        resolved = Path(source_path).expanduser()
        if not resolved.is_absolute():
            resolved = workspace / resolved
        resolved = resolved.resolve()
        if not resolved.is_relative_to(workspace):
            raise ValueError("source_path must stay inside the active workspace")
        source_path = resolved
    workflow = PersonalMaterialWorkflow(
        _material_store(context),
        _candidate_fact_store(context),
        career_store=_career_store(context),
        fact_extractor=_career_fact_extractor(context),
    )
    result = workflow.organize(
        category=input_data["category"],
        label=input_data["label"],
        text=input_data["text"],
        source_path=source_path,
        project_id=input_data["project_id"],
    )
    payload = result.import_result.to_dict(include_extracted_text=False)
    payload["candidate_facts"] = [fact.to_dict() for fact in result.candidate_facts]
    payload["candidate_fact_count"] = len(result.candidate_facts)
    payload["structured_path"] = result.structured_path
    payload["workflow_status"] = result.status
    payload["text_preview"] = result.import_result.extracted_text[:500]
    payload["text_length"] = len(result.import_result.extracted_text)
    payload["fact_confirmation_required"] = bool(result.candidate_facts)
    payload["next_step"] = (
        "Review candidate facts, then call import_candidate_facts before confirm_career_facts."
        if result.candidate_facts
        else "Material is stored as a read-only source."
    )
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


import_career_material_tool = ToolDefinition(
    name="import_career_material",
    description=(
        "Import one Career material by category. Resume sources may use a file path; education, internship, research, "
        "and award experiences must be provided as structured text fields. Register real projects through "
        "the separate project library so project code, datasets, environments, and model assets are not embedded. "
        "Extracted facts are staged as CandidateFact records and require explicit review before resume use. "
        "A resume is a read-only reference source and cannot create facts."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [item.value for item in MaterialCategory],
            },
            "label": {"type": "string"},
            "text": {"type": "string"},
            "source_path": {"type": "string"},
            "project_id": {"type": "string"},
        },
        "required": ["category", "label"],
    },
    validator=_validate_import_material,
    run=_import_material,
)


def _validate_extract_candidate_facts(input_data: dict) -> dict:
    material_id = str(input_data.get("material_id") or "").strip()
    if not material_id:
        raise ValueError("material_id is required")
    max_facts = int(input_data.get("max_facts", 12))
    if not 1 <= max_facts <= 50:
        raise ValueError("max_facts must be between 1 and 50")
    return {"material_id": material_id, "max_facts": max_facts}


def _extract_candidate_facts(input_data: dict, context) -> ToolResult:
    manifest = _find_material_manifest(context, input_data["material_id"])
    if manifest.category is MaterialCategory.RESUME:
        raise ValueError("Resume sources are read-only references and cannot produce candidate personal facts.")
    text = _material_store(context).read_text(manifest.material_id, max_chars=200_000)
    extractor = _career_fact_extractor(context)
    candidates = extractor.extract(manifest, text, max_facts=input_data["max_facts"])
    stored = _candidate_fact_store(context).upsert(candidates)
    payload = {
        "material_id": manifest.material_id,
        "category": manifest.category.value,
        "extractor": str(getattr(extractor, "name", "custom")),
        "candidate_facts": [fact.to_dict() for fact in stored],
        "count": len(stored),
        "next_step": "Review candidate facts, then call import_candidate_facts before confirm_career_facts.",
    }
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


extract_career_facts_tool = ToolDefinition(
    name="extract_career_facts",
    description=(
        "Extract traceable candidate facts from an imported Career material. Candidate facts are only a staging "
        "artifact; they cannot support resume claims until imported into PersonalEvidence and then confirmed."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "material_id": {"type": "string"},
            "max_facts": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["material_id"],
    },
    validator=_validate_extract_candidate_facts,
    run=_extract_candidate_facts,
)


def _validate_list_candidate_facts(input_data: dict) -> dict:
    category = input_data.get("category")
    status = input_data.get("status")
    material_id = str(input_data.get("material_id") or "").strip()
    return {
        "material_id": material_id or None,
        "category": MaterialCategory(str(category).strip().lower()) if category not in (None, "") else None,
        "status": CandidateFactStatus(str(status).strip().lower()) if status not in (None, "") else None,
    }


def _list_candidate_facts(input_data: dict, context) -> ToolResult:
    facts = _candidate_fact_store(context).list(
        material_id=input_data["material_id"],
        category=input_data["category"],
        status=input_data["status"],
    )
    payload = {
        "candidate_facts": [fact.to_dict() for fact in facts],
        "count": len(facts),
    }
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


list_candidate_facts_tool = ToolDefinition(
    name="list_candidate_facts",
    description="List staged Career candidate facts. These are not usable resume evidence until imported and confirmed.",
    input_schema={
        "type": "object",
        "properties": {
            "material_id": {"type": "string"},
            "category": {"type": "string", "enum": [item.value for item in MaterialCategory if item is not MaterialCategory.RESUME]},
            "status": {"type": "string", "enum": [item.value for item in CandidateFactStatus]},
        },
    },
    validator=_validate_list_candidate_facts,
    run=_list_candidate_facts,
)


def _validate_candidate_fact_ids(input_data: dict) -> dict:
    values = input_data.get("candidate_fact_ids")
    if not isinstance(values, list) or not values:
        raise ValueError("candidate_fact_ids must be a non-empty array")
    ids = tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))
    if not ids:
        raise ValueError("candidate_fact_ids must contain at least one ID")
    return {"candidate_fact_ids": ids}


def _import_candidate_facts(input_data: dict, context) -> ToolResult:
    records, candidates = _candidate_fact_store(context).import_to_evidence(
        input_data["candidate_fact_ids"],
        career_store=_career_store(context),
    )
    payload = {
        "records": [record.to_dict() for record in records],
        "candidate_facts": [fact.to_dict() for fact in candidates],
        "record_count": len(records),
        "candidate_fact_count": len(candidates),
        "fact_status": "unverified",
        "next_step": "Call confirm_career_facts with the returned evidence_id and fact_ids after user verification.",
    }
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


import_candidate_facts_tool = ToolDefinition(
    name="import_candidate_facts",
    description=(
        "Import selected candidate facts into PersonalEvidence as unverified facts. This does not confirm them; "
        "resume claims still require confirm_career_facts."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "candidate_fact_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["candidate_fact_ids"],
    },
    validator=_validate_candidate_fact_ids,
    run=_import_candidate_facts,
)


def _validate_reject_candidate_facts(input_data: dict) -> dict:
    parsed = _validate_candidate_fact_ids(input_data)
    reason = str(input_data.get("reason") or "").strip()
    if not reason:
        raise ValueError("reason is required")
    return {**parsed, "reason": reason}


def _reject_candidate_facts(input_data: dict, context) -> ToolResult:
    rejected = _candidate_fact_store(context).reject(
        input_data["candidate_fact_ids"],
        reason=input_data["reason"],
    )
    payload = {
        "candidate_facts": [fact.to_dict() for fact in rejected],
        "count": len(rejected),
    }
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


reject_candidate_facts_tool = ToolDefinition(
    name="reject_candidate_facts",
    description="Reject staged candidate facts before they enter PersonalEvidence.",
    input_schema={
        "type": "object",
        "properties": {
            "candidate_fact_ids": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["candidate_fact_ids", "reason"],
    },
    validator=_validate_reject_candidate_facts,
    run=_reject_candidate_facts,
)


def _validate_list_materials(input_data: dict) -> dict:
    category = input_data.get("category")
    if category in (None, ""):
        return {"category": None}
    try:
        return {"category": MaterialCategory(str(category).strip().lower())}
    except ValueError as exc:
        raise ValueError(f"Unknown material category: {category}") from exc


def _list_materials(input_data: dict, context) -> ToolResult:
    manifests = _material_store(context).list_manifests(input_data["category"])
    payload = {
        "materials": [manifest.to_dict() for manifest in manifests],
        "count": len(manifests),
    }
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


list_career_materials_tool = ToolDefinition(
    name="list_career_materials",
    description="List imported Career material manifests without returning full document text.",
    input_schema={
        "type": "object",
        "properties": {"category": {"type": "string", "enum": [item.value for item in MaterialCategory]}},
    },
    validator=_validate_list_materials,
    run=_list_materials,
)


def _validate_read_material(input_data: dict) -> dict:
    material_id = str(input_data.get("material_id") or "").strip()
    if not material_id:
        raise ValueError("material_id is required")
    max_chars = int(input_data.get("max_chars", 50_000))
    if not 1 <= max_chars <= 200_000:
        raise ValueError("max_chars must be between 1 and 200000")
    return {"material_id": material_id, "max_chars": max_chars}


def _read_material(input_data: dict, context) -> ToolResult:
    text = _material_store(context).read_text(input_data["material_id"], max_chars=input_data["max_chars"])
    payload = {
        "material_id": input_data["material_id"],
        "text": text,
        "text_length": len(text),
        "truncated": len(text) >= input_data["max_chars"],
    }
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


read_career_material_tool = ToolDefinition(
    name="read_career_material",
    description="Read the cached extracted text for one imported material with an explicit character limit.",
    input_schema={
        "type": "object",
        "properties": {
            "material_id": {"type": "string"},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": 200000},
        },
        "required": ["material_id"],
    },
    validator=_validate_read_material,
    run=_read_material,
)


def _validate_fact_transition(input_data: dict) -> dict:
    evidence_id = str(input_data.get("evidence_id") or "").strip()
    fact_ids = input_data.get("fact_ids")
    if not evidence_id:
        raise ValueError("evidence_id is required")
    if not isinstance(fact_ids, list) or not fact_ids:
        raise ValueError("fact_ids must be a non-empty array")
    normalized = tuple(dict.fromkeys(str(item).strip() for item in fact_ids if str(item).strip()))
    if not normalized:
        raise ValueError("fact_ids must contain at least one ID")
    reason = str(input_data.get("reason") or "User confirmed imported fact").strip()
    if not reason:
        raise ValueError("reason cannot be empty")
    return {"evidence_id": evidence_id, "fact_ids": normalized, "reason": reason}


def _validate_fact_invalidation(input_data: dict) -> dict:
    if not str(input_data.get("reason") or "").strip():
        raise ValueError("reason is required when invalidating facts")
    return _validate_fact_transition(input_data)


def _transition_facts(input_data: dict, context, *, operation: str) -> ToolResult:
    store = _career_store(context)
    records = []
    for fact_id in input_data["fact_ids"]:
        record = (
            store.confirm_fact(
                input_data["evidence_id"],
                fact_id,
                actor="user",
                reason=input_data["reason"],
            )
            if operation == "confirm"
            else store.invalidate_fact(
                input_data["evidence_id"],
                fact_id,
                actor="user",
                reason=input_data["reason"],
            )
        )
        records.append(record.to_dict())
    return ToolResult(
        ok=True,
        output=json.dumps(
            {"evidence_id": input_data["evidence_id"], "operation": operation, "records": records},
            ensure_ascii=False,
            indent=2,
        ),
    )


confirm_career_facts_tool = ToolDefinition(
    name="confirm_career_facts",
    description="Mark selected imported personal facts as user-confirmed so they can support resume claims.",
    input_schema={
        "type": "object",
        "properties": {
            "evidence_id": {"type": "string"},
            "fact_ids": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["evidence_id", "fact_ids"],
    },
    validator=_validate_fact_transition,
    run=lambda input_data, context: _transition_facts(input_data, context, operation="confirm"),
)


invalidate_career_facts_tool = ToolDefinition(
    name="invalidate_career_facts",
    description="Invalidate imported personal facts that are incorrect or should not be used in a resume.",
    input_schema={
        "type": "object",
        "properties": {
            "evidence_id": {"type": "string"},
            "fact_ids": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["evidence_id", "fact_ids", "reason"],
    },
    validator=_validate_fact_invalidation,
    run=lambda input_data, context: _transition_facts(input_data, context, operation="invalidate"),
)


def _validate_register_project_source(input_data: dict) -> dict:
    project_id = str(input_data.get("project_id") or "").strip()
    source_path = str(input_data.get("source_path") or "").strip()
    if not project_id or not source_path:
        raise ValueError("project_id and source_path are required")
    return {
        "project_id": project_id,
        "display_name": str(input_data.get("display_name") or project_id).strip(),
        "source_path": source_path,
        "source_type": ProjectSourceType(
            str(input_data.get("source_type") or ProjectSourceType.LOCAL_PATH.value)
        ),
    }


def _register_project_source(input_data: dict, context) -> ToolResult:
    workspace = Path(context.cwd).resolve()
    source_path = Path(input_data["source_path"]).expanduser()
    if not source_path.is_absolute():
        source_path = workspace / source_path
    source_path = source_path.resolve()
    if not source_path.is_relative_to(workspace):
        raise ValueError("source_path must stay inside the active workspace")
    record = _project_library_store(context).register(
        project_id=input_data["project_id"],
        display_name=input_data["display_name"],
        source_type=input_data["source_type"],
        source_location=source_path,
        managed_copy=False,
    )
    if record.status is ProjectSourceStatus.NEEDS_REANALYSIS:
        ProjectDescriptionService(_career_store(context)).mark_source_changed(
            record.project_id,
            current_project_version=record.project_version,
        )
    return ToolResult(ok=True, output=json.dumps({"project_source": record.to_dict()}, ensure_ascii=False, indent=2))


register_project_source_tool = ToolDefinition(
    name="register_project_source",
    description=(
        "Register a workspace project path in the project library. This stores only the path, version, and readable "
        "artifact manifest; it does not chunk or embed project content."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "display_name": {"type": "string"},
            "source_path": {"type": "string"},
            "source_type": {"type": "string", "enum": [item.value for item in ProjectSourceType]},
        },
        "required": ["project_id", "source_path"],
    },
    validator=_validate_register_project_source,
    run=_register_project_source,
)


def _list_project_sources(_input_data: dict, context) -> ToolResult:
    library = _project_library_store(context)
    analyses = _project_analysis_store(context)
    projects = []
    for source in library.list():
        analysis = analyses.latest(source.project_id)
        projects.append(
            {
                "project_source": source.to_dict(),
                "latest_analysis": analysis.to_dict() if analysis is not None else None,
            }
        )
    return ToolResult(ok=True, output=json.dumps({"projects": projects}, ensure_ascii=False, indent=2))


list_project_sources_tool = ToolDefinition(
    name="list_project_sources",
    description="List registered project paths, versions, analysis status, and latest structured analysis.",
    input_schema={"type": "object", "properties": {}},
    validator=lambda _value: {},
    run=_list_project_sources,
)


def _validate_save_project_understanding(input_data: dict) -> dict:
    project_id = str(input_data.get("project_id") or "").strip()
    analysis_run_id = str(input_data.get("analysis_run_id") or "").strip()
    understanding = input_data.get("understanding")
    if not project_id or not analysis_run_id or not isinstance(understanding, dict):
        raise ValueError("project_id, analysis_run_id, and understanding are required")
    return {
        "project_id": project_id,
        "analysis_run_id": analysis_run_id,
        "intent": str(input_data.get("intent") or "overview").strip(),
        "generator": str(input_data.get("generator") or "project-analyst-agent-v1").strip(),
        "model": str(input_data.get("model") or "").strip(),
        "understanding": ProjectUnderstanding.from_dict(understanding),
    }


def _save_project_understanding(input_data: dict, context) -> ToolResult:
    result = _project_experience_service(context).persist_understanding(
        project_id=input_data["project_id"],
        analysis_run_id=input_data["analysis_run_id"],
        intent=input_data["intent"],
        generator=input_data["generator"],
        model=input_data["model"],
        understanding=input_data["understanding"],
        trigger="career_agent",
    )
    return ToolResult(
        ok=True,
        output=json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
    )


save_project_understanding_tool = ToolDefinition(
    name="save_project_understanding",
    description=(
        "Persist a validated project_understanding artifact returned by Project Analyst. Overview analysis becomes "
        "one user-confirmable project description; it never becomes usable PersonalEvidence automatically."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "project_id": {"type": "string"},
            "analysis_run_id": {"type": "string"},
            "intent": {"type": "string"},
            "generator": {"type": "string"},
            "model": {"type": "string"},
            "understanding": {"type": "object"},
        },
        "required": ["project_id", "analysis_run_id", "understanding"],
    },
    validator=_validate_save_project_understanding,
    run=_save_project_understanding,
)


def _validate_search_interview_knowledge(input_data: dict) -> dict:
    query = str(input_data.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    top_k = int(input_data.get("top_k", 5))
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    topic = str(input_data.get("topic") or "").strip()
    corpus_id = str(input_data.get("corpus_id") or "").strip()
    tenant_id = str(input_data.get("tenant_id") or "").strip()

    def strings(name: str) -> tuple[str, ...]:
        value = input_data.get(name, [])
        if not isinstance(value, list):
            raise ValueError(f"{name} must be an array")
        return tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))

    minimum_authority = int(input_data.get("minimum_authority", 2))
    maximum_difficulty = int(input_data.get("maximum_difficulty", 5))
    if not 1 <= minimum_authority <= 5 or not 1 <= maximum_difficulty <= 5:
        raise ValueError("minimum_authority and maximum_difficulty must be between 1 and 5")
    return {
        "query": query,
        "top_k": top_k,
        "topic": topic or None,
        "corpus_id": corpus_id or None,
        "tenant_id": tenant_id,
        "domains": strings("domains"),
        "subdomains": strings("subdomains"),
        "topics": strings("topics"),
        "minimum_authority": minimum_authority,
        "maximum_difficulty": maximum_difficulty,
    }


def _search_interview_knowledge(input_data: dict, context) -> ToolResult:
    runtime = context._runtime or {}
    corpus = runtime.get("_interviewKnowledgeCorpus")
    store = None if corpus is not None else _interview_knowledge_store(context)
    library = _interview_knowledge_library(context)
    retriever = runtime.get("_interviewKnowledgeRetriever")
    if retriever is None and library.list_sources():
        variant = str(runtime.get("careerInterviewRetrievalVariant") or "hybrid_rerank").strip().lower()
        valid_variants = {"bm25", "vector", "hybrid", "hybrid_rerank", "hybrid_rrf", "hybrid_rrf_rerank"}
        if variant not in valid_variants:
            raise ValueError("careerInterviewRetrievalVariant is invalid")
        embedding = _interview_embedding_provider(runtime, required=variant != "bm25")
        reranker = _interview_reranker(context, runtime) if variant.endswith("_rerank") else None
        retriever = HybridInterviewKnowledgeRetriever(
            library,
            embedding_provider=embedding,
            reranker=reranker,
            variant=variant,
            bm25_weight=float(runtime.get("careerInterviewBM25Weight", 0.55)),
            vector_weight=float(runtime.get("careerInterviewVectorWeight", 0.45)),
            rrf_k=int(runtime.get("careerInterviewRRFK", 60)),
            neighbor_window=int(runtime.get("careerInterviewNeighborWindow", 1)),
            rerank_candidate_limit=int(runtime.get("careerInterviewRerankCandidateLimit", 30)),
            random_seed=int(runtime.get("careerInterviewRetrievalSeed", 0)),
        )
    retriever = retriever or corpus or store
    if retriever is None or not callable(getattr(retriever, "search", None)):
        raise ValueError("An interview knowledge corpus or _interviewKnowledgeRetriever is required")
    if isinstance(retriever, HybridInterviewKnowledgeRetriever):
        result = retriever.retrieve(
            input_data["query"],
            filters=InterviewRetrievalFilters(
                tenant_id=input_data["tenant_id"]
                or str(runtime.get("careerTenantId") or "default"),
                domains=input_data["domains"],
                subdomains=input_data["subdomains"],
                topics=(*input_data["topics"], *((input_data["topic"],) if input_data["topic"] else ())),
                minimum_authority=input_data["minimum_authority"],
                maximum_difficulty=input_data["maximum_difficulty"],
                corpus_ids=((input_data["corpus_id"],) if input_data["corpus_id"] else ()),
            ),
            top_k=input_data["top_k"],
        )
        return ToolResult(ok=True, output=json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    hits = retriever.search(
        input_data["query"],
        top_k=input_data["top_k"],
        topic=input_data["topic"],
        **({"corpus_id": input_data["corpus_id"]} if retriever is store else {}),
    )
    payload = {
        "backend": str(getattr(retriever, "name", "lexical-interview-baseline")),
        "query": input_data["query"],
        "topic": input_data["topic"],
        "corpus_id": input_data["corpus_id"] or str(getattr(corpus, "corpus_id", "")),
        "hits": [hit.to_dict() for hit in hits],
    }
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


search_interview_knowledge_tool = ToolDefinition(
    name="search_interview_knowledge",
    description=(
        "Search only the explicitly supplied interview knowledge corpus. Return source IDs and locators for every "
        "hit. This corpus is not personal evidence and must never be used to create resume claims."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            "topic": {"type": "string"},
            "corpus_id": {"type": "string"},
            "tenant_id": {"type": "string"},
            "domains": {"type": "array", "items": {"type": "string"}},
            "subdomains": {"type": "array", "items": {"type": "string"}},
            "topics": {"type": "array", "items": {"type": "string"}},
            "minimum_authority": {"type": "integer", "minimum": 1, "maximum": 5},
            "maximum_difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["query"],
    },
    validator=_validate_search_interview_knowledge,
    run=_search_interview_knowledge,
)


def _validate_import_interview_knowledge_source(input_data: dict) -> dict:
    tenant_id = str(input_data.get("tenant_id") or "default").strip()
    name = str(input_data.get("name") or "").strip()
    text = input_data.get("text")
    source_path = str(input_data.get("source_path") or "").strip()
    if not name or not tenant_id:
        raise ValueError("tenant_id and name are required")
    if (not isinstance(text, str) or not text.strip()) == (not source_path):
        raise ValueError("Provide exactly one of text or source_path")
    pdf_mode = str(input_data.get("pdf_mode") or "normal").strip().casefold()
    if pdf_mode not in {"normal", "qa"}:
        raise ValueError("pdf_mode must be normal or qa")

    def strings(name: str, *, required: bool = False) -> tuple[str, ...]:
        value = input_data.get(name, [])
        if not isinstance(value, list):
            raise ValueError(f"{name} must be an array")
        result = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        if required and not result:
            raise ValueError(f"{name} must contain at least one value")
        return result

    return {
        "tenant_id": tenant_id,
        "name": name,
        "source_type": InterviewKnowledgeSourceType(
            input_data.get("source_type", InterviewKnowledgeSourceType.USER_NOTE)
        ),
        "authority_level": int(input_data.get("authority_level", 3)),
        "domains": strings("domains", required=True),
        "subdomains": strings("subdomains"),
        "topics": strings("topics", required=True),
        "difficulty": int(input_data.get("difficulty", 3)),
        "text": text.strip() if isinstance(text, str) and text.strip() else None,
        "source_path": source_path or None,
        "source_name": str(input_data.get("source_name") or "").strip(),
        "corpus_id": str(input_data.get("corpus_id") or "").strip(),
        "pdf_mode": pdf_mode,
    }


def _import_interview_knowledge_source(input_data: dict, context) -> ToolResult:
    if input_data["source_path"]:
        workspace = Path(context.cwd).resolve()
        path = Path(input_data["source_path"]).expanduser()
        if not path.is_absolute():
            path = workspace / path
        path = path.resolve()
        if not path.is_relative_to(workspace):
            raise ValueError("source_path must stay inside the active workspace")
        input_data = {**input_data, "source_path": path}
    library = _interview_knowledge_library(context)
    result = library.import_source(**input_data)
    graph_index = _career_graph_rag_index(context)
    try:
        graph_sync = graph_index.sync_corpus(
            result.corpus,
            content_hash=result.source.content_hash,
            source_path=result.source.source_path,
        )
    except Exception:
        if not result.deduplicated:
            with suppress(Exception):
                library.delete_source(result.source.source_id, tenant_id=result.source.tenant_id)
            with suppress(Exception):
                graph_index.delete_corpus(result.corpus.corpus_id)
        raise
    source = library.update_source_index_metadata(result.source.source_id, graph_sync)
    payload = {
        "source": source.to_dict(),
        "corpus": {
            "corpus_id": result.corpus.corpus_id,
            "version": result.corpus.version,
            "chunk_count": len(result.corpus.chunks),
        },
        "deduplicated": result.deduplicated,
        "parsed_blocks": result.parsed_blocks,
        "graph_rag": graph_sync.to_dict(),
    }
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


import_interview_knowledge_source_tool = ToolDefinition(
    name="import_interview_knowledge_source",
    description=(
        "Import one tenant-scoped TXT, Markdown, PDF, DOCX, IPYNB, image, or text interview knowledge source. "
        "PNG, JPG/JPEG, BMP, WEBP, and TIFF images are OCR-processed. The source is content-addressed, tagged, "
        "prompt-injection-scanned, and isolated from resume evidence."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "tenant_id": {"type": "string"},
            "name": {"type": "string"},
            "source_type": {
                "type": "string",
                "enum": [item.value for item in InterviewKnowledgeSourceType],
            },
            "authority_level": {"type": "integer", "minimum": 1, "maximum": 5},
            "domains": {"type": "array", "items": {"type": "string"}},
            "subdomains": {"type": "array", "items": {"type": "string"}},
            "topics": {"type": "array", "items": {"type": "string"}},
            "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
            "text": {"type": "string"},
            "source_path": {"type": "string"},
            "source_name": {"type": "string"},
            "corpus_id": {"type": "string"},
            "pdf_mode": {"type": "string", "enum": ["normal", "qa"]},
        },
        "required": ["name", "domains", "topics"],
    },
    validator=_validate_import_interview_knowledge_source,
    run=_import_interview_knowledge_source,
)


def _validate_build_jd_capability_tree(input_data: dict) -> dict:
    jd_id = str(input_data.get("jd_id") or "").strip()
    if not jd_id:
        raise ValueError("jd_id is required")
    return {"jd_id": jd_id}


def _build_jd_capability_tree(input_data: dict, context) -> ToolResult:
    store = _career_store(context)
    jd = store.get(input_data["jd_id"], expected_type=JDDocument)
    requirements = tuple(
        requirement
        for requirement in store.list_latest(JDRequirement)
        if requirement.jd_ref.record_id == jd.jd_id and requirement.active
    )
    if not requirements:
        raise ValueError("JD has no active requirements")
    builder = (context._runtime or {}).get("_jdCapabilityTreeBuilder") or HeuristicJDCapabilityTreeBuilder()
    if not callable(getattr(builder, "build", None)):
        raise ValueError("_jdCapabilityTreeBuilder must expose build(jd, requirements)")
    tree = builder.build(jd, requirements)
    return ToolResult(ok=True, output=json.dumps(tree.to_dict(), ensure_ascii=False, indent=2))


build_jd_capability_tree_tool = ToolDefinition(
    name="build_jd_capability_tree",
    description=(
        "Build a traceable JD capability tree whose leaf filters can constrain interview knowledge retrieval."
    ),
    input_schema={
        "type": "object",
        "properties": {"jd_id": {"type": "string"}},
        "required": ["jd_id"],
    },
    validator=_validate_build_jd_capability_tree,
    run=_build_jd_capability_tree,
)


def _validate_import_interview_knowledge(input_data: dict) -> dict:
    values = {
        "corpus_id": str(input_data.get("corpus_id") or "").strip(),
        "name": str(input_data.get("name") or "Interview knowledge").strip(),
        "domain": str(input_data.get("domain") or "general").strip(),
        "subdomain": str(input_data.get("subdomain") or "").strip(),
        "topic": str(input_data.get("topic") or "").strip(),
        "text": str(input_data.get("text") or "").strip(),
        "source_id": str(input_data.get("source_id") or "").strip(),
    }
    if not values["corpus_id"] or not values["topic"] or not values["text"] or not values["source_id"]:
        raise ValueError("corpus_id, topic, text, and source_id are required")
    difficulty = int(input_data.get("difficulty", 3))
    if not 1 <= difficulty <= 5:
        raise ValueError("difficulty must be between 1 and 5")
    knowledge_type = InterviewKnowledgeType(input_data.get("knowledge_type", InterviewKnowledgeType.FUNDAMENTAL))
    locator = input_data.get("locator", {})
    if not isinstance(locator, dict):
        raise ValueError("locator must be an object")
    return {**values, "difficulty": difficulty, "knowledge_type": knowledge_type, "locator": dict(locator)}


def _import_interview_knowledge(input_data: dict, context) -> ToolResult:
    import hashlib

    chunk_id = "knowledge-" + hashlib.sha256(
        f"{input_data['corpus_id']}:{input_data['source_id']}:{input_data['topic']}:{input_data['text']}".encode("utf-8")
    ).hexdigest()[:20]
    chunk = InterviewKnowledgeChunk(
        chunk_id=chunk_id,
        corpus_id=input_data["corpus_id"],
        domain=input_data["domain"],
        subdomain=input_data["subdomain"],
        topic=input_data["topic"],
        text=input_data["text"],
        source_id=input_data["source_id"],
        locator=input_data["locator"],
        difficulty=input_data["difficulty"],
        knowledge_type=input_data["knowledge_type"],
    )
    corpus = _interview_knowledge_store(context).append(
        chunk,
        name=input_data["name"],
        domain=input_data["domain"],
    )
    payload = {
        "corpus_id": corpus.corpus_id,
        "chunk": chunk.to_dict(),
        "chunk_count": len(corpus.chunks),
    }
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


import_interview_knowledge_tool = ToolDefinition(
    name="import_interview_knowledge",
    description=(
        "Store one traceable professional-knowledge chunk for Interviewer Agent RAG. This knowledge is isolated "
        "from PersonalEvidence and cannot support resume claims."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "corpus_id": {"type": "string"},
            "name": {"type": "string"},
            "domain": {"type": "string"},
            "subdomain": {"type": "string"},
            "topic": {"type": "string"},
            "text": {"type": "string"},
            "source_id": {"type": "string"},
            "locator": {"type": "object"},
            "difficulty": {"type": "integer", "minimum": 1, "maximum": 5},
            "knowledge_type": {"type": "string", "enum": [item.value for item in InterviewKnowledgeType]},
        },
        "required": ["corpus_id", "topic", "text", "source_id"],
    },
    validator=_validate_import_interview_knowledge,
    run=_import_interview_knowledge,
)


def _list_interview_knowledge(_input_data: dict, context) -> ToolResult:
    corpora = _interview_knowledge_store(context).list_metadata()
    sources = _interview_knowledge_library(context).list_sources()
    payload = {
        "corpora": [
            {
                "corpus_id": corpus.corpus_id,
                "name": corpus.name,
                "domain": corpus.domain,
                "version": corpus.version,
                "chunk_count": corpus.chunk_count,
            }
            for corpus in corpora
        ],
        "count": len(corpora),
        "sources": [source.to_dict() for source in sources],
        "source_count": len(sources),
    }
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


list_interview_knowledge_tool = ToolDefinition(
    name="list_interview_knowledge",
    description="List interview knowledge corpus metadata without returning full source text.",
    input_schema={"type": "object", "properties": {}},
    validator=lambda _input_data: {},
    run=_list_interview_knowledge,
)


def _validate_create_interview_session(input_data: dict) -> dict:
    mode = InterviewMode(input_data.get("mode", InterviewMode.REALISTIC_SIMULATION.value))
    tenant_id = str(input_data.get("tenant_id") or "default").strip()
    resume_run_id = str(input_data.get("resume_run_id") or "").strip()
    jd_id = str(input_data.get("jd_id") or "").strip()
    resume_document_id = str(input_data.get("resume_document_id") or "").strip()
    raw_resume_revision = input_data.get("resume_document_revision")
    resume_document_revision = None
    if raw_resume_revision not in (None, ""):
        try:
            resume_document_revision = int(raw_resume_revision)
        except (TypeError, ValueError) as exc:
            raise ValueError("resume_document_revision must be an integer") from exc
        if resume_document_revision < 1:
            raise ValueError("resume_document_revision must be at least 1")
    raw_corpus_ids = input_data.get("corpus_ids")
    if not tenant_id:
        raise ValueError("tenant_id is required")
    if mode is InterviewMode.REALISTIC_SIMULATION and not jd_id:
        raise ValueError("realistic_simulation requires jd_id")
    if mode is InterviewMode.GAP_ANALYSIS and (not resume_run_id or not jd_id) and not (
        jd_id.startswith("input:") and resume_document_id.startswith("input:")
    ):
        raise ValueError("gap_analysis requires resume_run_id and jd_id")
    if not isinstance(raw_corpus_ids, list) or not raw_corpus_ids:
        raise ValueError("corpus_ids must be a non-empty array")
    corpus_ids = tuple(dict.fromkeys(str(item).strip() for item in raw_corpus_ids if str(item).strip()))
    if not corpus_ids:
        raise ValueError("corpus_ids must contain at least one ID")
    max_turns = int(input_data.get("max_turns", 8))
    if not 1 <= max_turns <= 30:
        raise ValueError("max_turns must be between 1 and 30")
    duration_minutes = int(input_data.get("duration_minutes", 30))
    if duration_minutes not in {30, 60}:
        raise ValueError("duration_minutes must be 30 or 60")
    return {
        "mode": mode,
        "tenant_id": tenant_id,
        "resume_run_id": resume_run_id or None,
        "jd_id": jd_id or None,
        "resume_document_id": resume_document_id or None,
        "resume_document_revision": resume_document_revision,
        "corpus_ids": corpus_ids,
        "random_seed": int(input_data.get("random_seed", 0)),
        "max_turns": max_turns,
        "duration_minutes": duration_minutes,
    }


def _create_interview_session(input_data: dict, context) -> ToolResult:
    runtime = context._runtime or {}
    max_followups = int(runtime.get("interviewMaxFollowupsPerThread", 3))
    if not 1 <= max_followups <= 10:
        raise ValueError("interviewMaxFollowupsPerThread must be between 1 and 10")
    session_input = dict(input_data)
    configured_tenant = str(runtime.get("careerTenantId") or "").strip()
    if session_input.get("tenant_id") == "default" and configured_tenant:
        session_input["tenant_id"] = configured_tenant
    state = _interview_runtime(context).create_session(
        **session_input,
        max_followups_per_thread=max_followups,
    )
    return ToolResult(ok=True, output=json.dumps(state.to_dict(), ensure_ascii=False, indent=2))


create_interview_session_tool = ToolDefinition(
    name="create_interview_session",
    description=(
        "Create an isolated persistent Interviewer session. realistic_simulation requires a JD and interview "
        "knowledge corpora, and uses the current verified ResumeDocument (including its immutable revision) when "
        "provided. gap_analysis requires a bound ResumeRun and JD, may use its verified resume, and combines "
        "tenant-scoped RAG with historical mastery priors."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": [item.value for item in InterviewMode],
            },
            "tenant_id": {"type": "string"},
            "resume_run_id": {"type": "string"},
            "jd_id": {"type": "string"},
            "resume_document_id": {"type": "string"},
            "resume_document_revision": {"type": "integer", "minimum": 1},
            "corpus_ids": {"type": "array", "items": {"type": "string"}},
            "random_seed": {"type": "integer"},
            "max_turns": {"type": "integer", "minimum": 1, "maximum": 30},
            "duration_minutes": {"type": "integer", "enum": [30, 60]},
        },
        "required": ["corpus_ids"],
    },
    validator=_validate_create_interview_session,
    run=_create_interview_session,
)


def _validate_interview_id(input_data: dict) -> dict:
    interview_id = str(input_data.get("interview_id") or "").strip()
    if not interview_id:
        raise ValueError("interview_id is required")
    return {"interview_id": interview_id}


def _get_interview_session(input_data: dict, context) -> ToolResult:
    state = _interview_session_store(context).get(input_data["interview_id"])
    payload = state.to_dict()
    # The full lane audit is retained for the operations-console replay. Keep
    # the candidate-facing tool response below the transport output limit.
    for turn in payload.get("turns", []):
        if isinstance(turn, dict):
            turn.pop("orchestration_audit", None)
            question = turn.get("question")
            if isinstance(question, dict):
                for block in question.get("knowledge_block_refs", []) or []:
                    if isinstance(block, dict):
                        block.pop("retrieval_trace", None)
    pending_question = payload.get("pending_question")
    if isinstance(pending_question, dict):
        for block in pending_question.get("knowledge_block_refs", []) or []:
            if isinstance(block, dict):
                block.pop("retrieval_trace", None)
    for point in payload.get("interview_points", []) or []:
        if not isinstance(point, dict):
            continue
        for block in point.get("knowledge_block_refs", []) or []:
            if isinstance(block, dict):
                block.pop("retrieval_trace", None)
    for trace in payload.get("recommendation_traces", []) or []:
        if not isinstance(trace, dict):
            continue
        retrieval = trace.get("retrieval_trace")
        if isinstance(retrieval, dict):
            trace["retrieval_trace"] = {
                key: retrieval.get(key)
                for key in ("mode", "path", "returned_node_ids", "fallback_used", "timings_ms")
                if retrieval.get(key) not in (None, "", [], {})
            }
        for field_name in ("candidate_knowledge_blocks", "selected_knowledge_blocks"):
            for block in trace.get(field_name, []) or []:
                if isinstance(block, dict):
                    block.pop("retrieval_trace", None)
    report = _interview_runtime(context).report(state.interview_id)
    trace = report.get("trace") if isinstance(report, dict) else None
    if isinstance(trace, dict):
        compact_traces = []
        for recommendation_trace in trace.get("recommendation_traces", []) or []:
            if not isinstance(recommendation_trace, dict):
                continue
            compact_traces.append({
                "turn": recommendation_trace.get("turn"),
                "point_id": recommendation_trace.get("point_id"),
                "previous_question_id": recommendation_trace.get("previous_question_id"),
                "allowed_actions": recommendation_trace.get("allowed_actions", []),
                "candidate_knowledge_point_ids": recommendation_trace.get("candidate_knowledge_point_ids", []),
                "selected_action": recommendation_trace.get("selected_action"),
                "selected_knowledge_point_ids": recommendation_trace.get("selected_knowledge_point_ids", []),
                "selected_knowledge_blocks": [
                    {key: block.get(key) for key in ("role", "label", "chunk_id", "point_id", "sequence") if block.get(key) not in (None, "")}
                    for block in recommendation_trace.get("selected_knowledge_blocks", []) or []
                    if isinstance(block, dict)
                ],
                "selection_reason": recommendation_trace.get("selection_reason"),
                "question_id": recommendation_trace.get("question_id"),
            })
        trace["recommendation_traces"] = compact_traces
    payload["report"] = report
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


get_interview_session_tool = ToolDefinition(
    name="get_interview_session",
    description=(
        "Read one isolated interview session, including its fixed profile, pending question, transcript, "
        "knowledge-belief state, source references, and current report."
    ),
    input_schema={
        "type": "object",
        "properties": {"interview_id": {"type": "string"}},
        "required": ["interview_id"],
    },
    validator=_validate_interview_id,
    run=_get_interview_session,
)


def _question_sources(context, interview_id: str, knowledge_refs: tuple[str, ...]) -> list[dict]:
    chunks = _interview_runtime(context).knowledge_chunks(interview_id, knowledge_refs)
    return [
        {
            "chunk_id": chunk.chunk_id,
            "corpus_id": chunk.corpus_id,
            "topic": chunk.topic,
            "source_id": chunk.source_id,
            "locator": chunk.locator,
            "content_hash": chunk.content_hash,
        }
        for chunk in chunks
    ]


def _next_interview_question(input_data: dict, context) -> ToolResult:
    question = _interview_runtime(context).next_question(input_data["interview_id"])
    state = _interview_session_store(context).get(input_data["interview_id"])
    payload = question.to_dict()
    payload["interview_stage"] = state.interview_stage.value
    payload["active_line"] = state.active_line.value if state.active_line else None
    payload["active_subject"] = state.active_subject
    payload["candidate_question_wait_seconds"] = state.candidate_question_wait_seconds
    payload["status"] = state.status.value
    payload["knowledge_sources"] = _question_sources(
        context,
        input_data["interview_id"],
        question.knowledge_refs,
    )
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


next_interview_question_tool = ToolDefinition(
    name="next_interview_question",
    description=(
        "Return the current unanswered question or use the configured interview model to generate the next traceable question. "
        "Calling this repeatedly before an answer is idempotent."
    ),
    input_schema={
        "type": "object",
        "properties": {"interview_id": {"type": "string"}},
        "required": ["interview_id"],
    },
    validator=_validate_interview_id,
    run=_next_interview_question,
)


def _validate_interview_answer(input_data: dict) -> dict:
    parsed = _validate_interview_id(input_data)
    answer = str(input_data.get("answer") or "").strip()
    if not answer:
        raise ValueError("answer is required")
    if len(answer) > 20_000:
        raise ValueError("answer must not exceed 20000 characters")
    parsed["answer"] = answer
    question_id = str(input_data.get("question_id") or "").strip()
    if question_id:
        parsed["question_id"] = question_id
    return parsed


def _submit_interview_answer(input_data: dict, context) -> ToolResult:
    result = _interview_runtime(context).submit_answer(
        input_data["interview_id"],
        input_data["answer"],
    )
    payload = result.to_dict()
    payload["report"] = _interview_runtime(context).report(result.state.interview_id)
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


def _process_interview_answer(input_data: dict, context) -> ToolResult:
    runtime = _interview_runtime(context)
    result = runtime.process_answer(
        input_data["interview_id"],
        input_data["answer"],
        expected_question_id=input_data.get("question_id"),
    )
    payload = result.to_dict()
    payload["candidates"] = runtime.next_question_candidates(input_data["interview_id"])
    payload["report"] = runtime.report(result.state.interview_id)
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


process_interview_answer_tool = ToolDefinition(
    name="process_interview_answer",
    description=(
        "独立处理面试回答：读取上一题的 A/D/B 题目编排上下文，评估候选人的技术回答质量、"
        "信息增益和验证需求，并完成深广决策；只返回下一题候选块关系，不生成最终问题文本。"
    ),
    input_schema={
        "type": "object",
        "properties": {
            "interview_id": {"type": "string"},
            "answer": {"type": "string", "maxLength": 20000},
            "question_id": {"type": "string"},
        },
        "required": ["interview_id", "answer"],
    },
    validator=_validate_interview_answer,
    run=_process_interview_answer,
)


def _validate_question_generation_input(input_data: dict) -> dict:
    parsed = _validate_interview_id(input_data)
    refs = input_data.get("candidate_knowledge_refs", [])
    if refs is None:
        refs = []
    if not isinstance(refs, list):
        raise ValueError("candidate_knowledge_refs must be an array")
    parsed["candidate_knowledge_refs"] = tuple(dict.fromkeys(str(item).strip() for item in refs if str(item).strip()))
    return parsed


def _question_generation_payload(runtime, question, interview_id: str) -> dict:
    state = runtime.session_store.get(interview_id)
    payload = question.to_dict()
    payload["interview_stage"] = state.interview_stage.value
    payload["active_line"] = state.active_line.value if state.active_line else None
    payload["active_subject"] = state.active_subject
    payload["status"] = state.status.value
    payload["knowledge_sources"] = [
        {
            "chunk_id": chunk.chunk_id,
            "block_role": next(
                (block.role.value for block in question.knowledge_block_refs if block.chunk_id == chunk.chunk_id),
                None,
            ),
            "block_label": next(
                (block.label for block in question.knowledge_block_refs if block.chunk_id == chunk.chunk_id),
                None,
            ),
            "corpus_id": chunk.corpus_id,
            "topic": chunk.topic,
            "source_id": chunk.source_id,
            "locator": chunk.locator,
            "content_hash": chunk.content_hash,
        }
        for chunk in runtime.knowledge_chunks(interview_id, question.knowledge_refs)
    ]
    payload["candidates"] = [
        block.to_dict() for block in question.knowledge_block_refs
    ]
    return payload


def _generate_interview_depth_question(input_data: dict, context) -> ToolResult:
    runtime = _interview_runtime(context)
    question = runtime.generate_depth_question(
        input_data["interview_id"],
        candidate_knowledge_refs=input_data["candidate_knowledge_refs"],
    )
    payload = _question_generation_payload(runtime, question, input_data["interview_id"])
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


generate_interview_depth_question_tool = ToolDefinition(
    name="generate_interview_depth_question",
    description="只使用回答处理 API 授权的 A2+D1 或 A3+D2 候选块生成深度问题。",
    input_schema={
        "type": "object",
        "properties": {
            "interview_id": {"type": "string"},
            "candidate_knowledge_refs": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["interview_id", "candidate_knowledge_refs"],
    },
    validator=_validate_question_generation_input,
    run=_generate_interview_depth_question,
)


def _generate_interview_breadth_question(input_data: dict, context) -> ToolResult:
    runtime = _interview_runtime(context)
    question = runtime.generate_breadth_question(
        input_data["interview_id"],
        candidate_knowledge_refs=input_data["candidate_knowledge_refs"],
    )
    payload = _question_generation_payload(runtime, question, input_data["interview_id"])
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


generate_interview_breadth_question_tool = ToolDefinition(
    name="generate_interview_breadth_question",
    description="只使用下一个综合点的 B1 知识块生成广度问题。",
    input_schema={
        "type": "object",
        "properties": {
            "interview_id": {"type": "string"},
            "candidate_knowledge_refs": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["interview_id"],
    },
    validator=_validate_question_generation_input,
    run=_generate_interview_breadth_question,
)


submit_interview_answer_tool = ToolDefinition(
    name="submit_interview_answer",
    description=(
        "Record the answer to the pending question, evaluate it against cited interview knowledge, update only the "
        "session KnowledgeBeliefState, and choose FOLLOW_UP, SWITCH_TOPIC, or END."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "interview_id": {"type": "string"},
            "answer": {"type": "string", "maxLength": 20000},
        },
        "required": ["interview_id", "answer"],
    },
    validator=_validate_interview_answer,
    run=_submit_interview_answer,
)


def _validate_retry_interview_answer(input_data: dict) -> dict:
    parsed = _validate_interview_answer(input_data)
    revision = input_data.get("expected_revision")
    if revision is not None:
        parsed["expected_revision"] = int(revision)
    parsed["max_attempts"] = max(1, min(int(input_data.get("max_attempts", 2)), 3))
    return parsed


def _retry_interview_answer(input_data: dict, context) -> ToolResult:
    runtime = _interview_runtime(context)
    result = runtime.retry_submit_answer(
        input_data["interview_id"],
        input_data["answer"],
        expected_revision=input_data.get("expected_revision"),
        max_attempts=input_data["max_attempts"],
    )
    payload = result.to_dict()
    payload["retry_events"] = list(result.state.retry_events[-3:])
    payload["report"] = runtime.report(result.state.interview_id)
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


retry_interview_answer_tool = ToolDefinition(
    name="retry_interview_answer",
    description=(
        "Retry one transient interview answer-processing failure with bounded exponential backoff. "
        "It rejects stale revisions, completed sessions, duplicate committed answers, and business validation failures."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "interview_id": {"type": "string"},
            "answer": {"type": "string", "maxLength": 20000},
            "expected_revision": {"type": "integer", "minimum": 1},
            "max_attempts": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["interview_id", "answer"],
    },
    validator=_validate_retry_interview_answer,
    run=_retry_interview_answer,
)


def _validate_finish_interview(input_data: dict) -> dict:
    parsed = _validate_interview_id(input_data)
    parsed["cancelled"] = bool(input_data.get("cancelled", False))
    return parsed


def _finish_interview_session(input_data: dict, context) -> ToolResult:
    runtime = _interview_runtime(context)
    state = runtime.finish(input_data["interview_id"], cancelled=input_data["cancelled"])
    payload = {
        "interview_id": state.interview_id,
        "status": state.status.value,
        "report": runtime.report(state.interview_id),
    }
    return ToolResult(ok=True, output=json.dumps(payload, ensure_ascii=False, indent=2))


finish_interview_session_tool = ToolDefinition(
    name="finish_interview_session",
    description=(
        "Finish an interview after its pending question is answered, or cancel it and discard the pending question. "
        "Returns a subjective interviewer assessment and a source-traceable Career learning review."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "interview_id": {"type": "string"},
            "cancelled": {"type": "boolean"},
        },
        "required": ["interview_id"],
    },
    validator=_validate_finish_interview,
    run=_finish_interview_session,
)


def _validate_analyze_jd(input_data: dict) -> dict:
    parsed = _require_text(input_data)
    requirements = input_data.get("requirements")
    if requirements is not None and not isinstance(requirements, list):
        raise ValueError("requirements must be an array")
    source = input_data.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("source must be an object")
    parsed.update(
        {
            "role_title": str(input_data.get("role_title") or "").strip(),
            "company": str(input_data.get("company") or "").strip(),
            "requirements": requirements,
            "source": source,
        }
    )
    return parsed


def _analyze_jd(input_data: dict, context) -> ToolResult:
    source = input_data["source"]
    runtime = context._runtime or {}
    analyzer = runtime.get("_jdAnalyzer")
    if analyzer is None:
        client = create_career_client(
            runtime,
            "jd_analyzer",
            audit_root=_career_store(context).root / "audit",
        )
        analyzer = LLMJDAnalyzer(client) if client is not None else JDAnalyzer()
    if not callable(getattr(analyzer, "analyze", None)):
        raise ValueError("_jdAnalyzer must expose analyze(...)" )
    analysis = analyzer.analyze(
        input_data["text"],
        role_title=input_data["role_title"],
        company=input_data["company"],
        requirements=input_data["requirements"],
        source_id=str(source.get("source_id") or ""),
        source_type=str(source.get("source_type") or "user_input"),
        locator=source.get("locator") if isinstance(source.get("locator"), dict) else None,
    ).persist(_career_store(context))
    result = analysis.to_dict()
    result["requirement_count"] = len(analysis.requirements)
    return ToolResult(ok=True, output=json.dumps(result, ensure_ascii=False, indent=2))


analyze_jd_tool = ToolDefinition(
    name="analyze_jd",
    description=(
        "Normalize a text JD into traceable requirements and persist them. Supply requirements as structured "
        "objects after semantic analysis; every original_text must be copied exactly from the JD. If omitted, "
        "the tool runs a lower-confidence heuristic baseline. Never invent source text."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "role_title": {"type": "string"},
            "company": {"type": "string"},
            "requirements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "responsibility",
                                "hard_skill",
                                "soft_skill",
                                "experience",
                                "education",
                                "preference",
                            ],
                        },
                        "text": {"type": "string"},
                        "original_text": {"type": "string"},
                        "required": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "start": {"type": "integer", "minimum": 0},
                        "end": {"type": "integer", "minimum": 1},
                    },
                    "required": ["category", "text", "original_text"],
                },
            },
            "source": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "source_type": {
                        "type": "string",
                        "enum": ["file", "url", "user_input"],
                    },
                    "locator": {"type": "object"},
                },
            },
        },
        "required": ["text"],
    },
    validator=_validate_analyze_jd,
    run=_analyze_jd,
)


def _validate_retrieve_evidence(input_data: dict) -> dict:
    requirement_ids = input_data.get("requirement_ids")
    requirement_refs = input_data.get("requirement_refs")
    if requirement_refs is None and (not isinstance(requirement_ids, list) or not requirement_ids):
        raise ValueError("requirement_ids or requirement_refs must be a non-empty array")
    if requirement_refs is not None and (not isinstance(requirement_refs, list) or not requirement_refs):
        raise ValueError("requirement_refs must be a non-empty array")
    evidence_ids = input_data.get("evidence_ids")
    if evidence_ids is not None and not isinstance(evidence_ids, list):
        raise ValueError("evidence_ids must be an array")
    evidence_refs = input_data.get("evidence_refs")
    if evidence_refs is not None and not isinstance(evidence_refs, list):
        raise ValueError("evidence_refs must be an array")
    if evidence_ids is not None and evidence_refs is not None:
        raise ValueError("Provide evidence_ids or evidence_refs, not both")
    top_k = int(input_data.get("top_k_per_requirement", 3))
    if not 1 <= top_k <= 20:
        raise ValueError("top_k_per_requirement must be between 1 and 20")
    normalized_requirements = tuple(
        dict.fromkeys(str(item).strip() for item in (requirement_ids or []) if str(item).strip())
    )
    normalized_requirement_refs = tuple(
        (str(item.get("record_id") or "").strip(), int(item.get("revision", 0)))
        for item in (requirement_refs or [])
        if isinstance(item, dict)
    )
    normalized_evidence_refs = tuple(
        (str(item.get("record_id") or "").strip(), int(item.get("revision", 0)))
        for item in (evidence_refs or [])
        if isinstance(item, dict)
    )
    if requirement_refs is not None and (
        len(normalized_requirement_refs) != len(requirement_refs)
        or any(not identity or revision < 1 for identity, revision in normalized_requirement_refs)
    ):
        raise ValueError("Each requirement_refs item needs record_id and revision")
    if evidence_refs is not None and (
        len(normalized_evidence_refs) != len(evidence_refs)
        or any(not identity or revision < 1 for identity, revision in normalized_evidence_refs)
    ):
        raise ValueError("Each evidence_refs item needs record_id and revision")
    if not normalized_requirements and not normalized_requirement_refs:
        raise ValueError("JD requirements must contain at least one reference")
    return {
        "requirement_ids": normalized_requirements,
        "requirement_refs": normalized_requirement_refs or None,
        "evidence_ids": (
            tuple(dict.fromkeys(str(item).strip() for item in evidence_ids if str(item).strip()))
            if evidence_ids is not None
            else None
        ),
        "evidence_refs": normalized_evidence_refs if evidence_refs is not None else None,
        "top_k_per_requirement": top_k,
    }


def _retrieve_evidence(input_data: dict, context) -> ToolResult:
    store = _career_store(context)
    runtime = context._runtime or {}
    project_library = _project_library_store(context)
    project_service = ProjectDescriptionService(store)
    for source in project_library.list():
        refreshed = project_library.get(source.project_id, refresh=True)
        if refreshed.status is ProjectSourceStatus.NEEDS_REANALYSIS:
            project_service.mark_source_changed(
                refreshed.project_id,
                current_project_version=refreshed.project_version,
            )
    requirements = (
        tuple(
            store.get(identity, revision, expected_type=JDRequirement)
            for identity, revision in input_data["requirement_refs"]
        )
        if input_data["requirement_refs"] is not None
        else tuple(
            store.get(identity, expected_type=JDRequirement)
            for identity in input_data["requirement_ids"]
        )
    )
    if input_data["evidence_refs"] is not None:
        evidence = tuple(
            store.get(identity, revision, expected_type=PersonalEvidence)
            for identity, revision in input_data["evidence_refs"]
        )
    elif input_data["evidence_ids"] is None:
        evidence = store.list_latest(PersonalEvidence)
    else:
        evidence = tuple(
            store.get(identity, expected_type=PersonalEvidence)
            for identity in input_data["evidence_ids"]
        )
    provider = OpenAICompatibleEmbeddingProvider.from_runtime(context._runtime)
    config = RetrievalConfig(
        keyword_weight=float(runtime.get("careerRetrievalKeywordWeight", 0.6)),
        semantic_weight=float(runtime.get("careerRetrievalSemanticWeight", 0.4)),
        minimum_score=float(runtime.get("careerRetrievalMinimumScore", 0.18)),
        indirect_threshold=float(runtime.get("careerRetrievalIndirectThreshold", 0.25)),
        partial_threshold=float(runtime.get("careerRetrievalPartialThreshold", 0.48)),
        direct_threshold=float(runtime.get("careerRetrievalDirectThreshold", 0.72)),
        top_k_per_requirement=input_data["top_k_per_requirement"],
    )
    injected_retriever = runtime.get("_careerEvidenceRetriever")
    if injected_retriever is not None and not callable(getattr(injected_retriever, "retrieve", None)):
        raise ValueError("_careerEvidenceRetriever must expose retrieve(...)" )
    retriever = injected_retriever or HybridEvidenceRetriever(
        embedding_provider=provider,
        config=config,
    )
    embedding_error = ""
    try:
        batch = retriever.retrieve(requirements, evidence)
    except Exception as exc:  # noqa: BLE001 - configured embedding failure must not block lexical retrieval
        if provider is None:
            raise
        embedding_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        fallback = HybridEvidenceRetriever(config=config).retrieve(requirements, evidence)
        batch = type(fallback)(fallback.matches, "bm25_fallback", provider.name)
    batch = batch.persist(store)
    result = batch.to_dict()
    active_project_ids = {item.project_id.casefold() for item in project_library.list()}
    project_descriptions = tuple(
        item
        for item in store.list_latest(ProjectDescriptionDocument)
        if item.project_id.casefold() in active_project_ids
    )
    result["project_deep_scan_decisions"] = [
        decision.to_dict()
        for decision in _project_deep_dive_service(context).plan(
            requirements,
            matches=batch.matches,
            projects=project_descriptions,
            budget=int(runtime.get("careerProjectDeepScanBudget", 3)),
        )
    ]
    active_material_ids = {
        manifest.material_id
        for manifest in _material_store(context).list_manifests(include_deleted=False)
    }
    resume_sources = tuple(
        source
        for source in store.list_latest(ResumeSource)
        if any(ref.source_id in active_material_ids for ref in source.source_refs)
    )
    result["resume_source_candidates"] = _match_resume_sources(requirements, resume_sources)
    result.update(
        {
            "requirement_count": len(requirements),
            "evidence_record_count": len(evidence),
            "gap_count": len(batch.gaps),
            "embedding_error": embedding_error,
        }
    )
    return ToolResult(ok=True, output=json.dumps(result, ensure_ascii=False, indent=2))


def _match_resume_sources(
    requirements: tuple[JDRequirement, ...],
    resume_sources: tuple[ResumeSource, ...],
) -> list[dict]:
    """Find low-trust resume hints without promoting them to PersonalEvidence."""

    from smart_agent.career.retrieval import tokenize

    matches: list[dict] = []
    for requirement in requirements:
        query = {item for item in tokenize(requirement.text) if len(item) > 1}
        if not query:
            continue
        for source in resume_sources:
            scored_sections = []
            for section in source.sections:
                document = set(tokenize(section.text))
                overlap = query & document
                scored_sections.append((len(overlap) / len(query), section))
            score, matched_section = max(scored_sections, key=lambda item: item[0])
            if score < 0.25:
                continue
            matches.append(
                {
                    "requirement_id": requirement.requirement_id,
                    "resume_source_id": source.resume_source_id,
                    "resume_source_revision": source.revision,
                    "section_id": matched_section.section_id,
                    "section_content_hash": matched_section.content_hash,
                    "source_locator": matched_section.locator,
                    "score": round(score, 6),
                    "status": "candidate_only",
                    "reason": "原始通用简历出现相关内容，但尚未成为用户确认事实。",
                }
            )
    return matches


retrieve_evidence_tool = ToolDefinition(
    name="retrieve_career_evidence",
    description=(
        "Retrieve verified personal facts independently for each persisted JD requirement. Uses BM25 and, when "
        "configured, an OpenAI-compatible embedding endpoint such as Doubao Ark. Returns explicit evidence gaps "
        "instead of inventing missing experience."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "requirement_ids": {"type": "array", "items": {"type": "string"}},
            "requirement_refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "record_id": {"type": "string"},
                        "revision": {"type": "integer", "minimum": 1},
                    },
                    "required": ["record_id", "revision"],
                },
            },
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "evidence_refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "record_id": {"type": "string"},
                        "revision": {"type": "integer", "minimum": 1},
                    },
                    "required": ["record_id", "revision"],
                },
            },
            "top_k_per_requirement": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": [],
    },
    validator=_validate_retrieve_evidence,
    run=_retrieve_evidence,
)


def _validate_plan_resume(input_data: dict) -> dict:
    jd_id = str(input_data.get("jd_id") or "").strip()
    if not jd_id:
        raise ValueError("jd_id is required")
    candidates = input_data.get("candidate_items")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate_items must be a non-empty array")
    weights = input_data.get("weights")
    if weights is not None and not isinstance(weights, dict):
        raise ValueError("weights must be an object")
    required_item_ids = input_data.get("required_item_ids", [])
    if not isinstance(required_item_ids, list):
        raise ValueError("required_item_ids must be an array")
    content_budget = int(input_data.get("content_budget", 0))
    if content_budget < 1:
        raise ValueError("content_budget must be at least 1")
    return {
        "jd_id": jd_id,
        "candidate_items": candidates,
        "section_order": RESUME_SECTION_ORDER,
        "content_budget": content_budget,
        "resume_source_id": str(input_data.get("resume_source_id") or "").strip(),
        "weights": weights,
        "required_item_ids": tuple(str(item).strip() for item in required_item_ids if str(item).strip()),
    }


def _plan_resume(input_data: dict, context) -> ToolResult:
    store = _career_store(context)
    jd = store.get(input_data["jd_id"], expected_type=JDDocument)
    if not jd.requirement_refs:
        raise ValueError("JDDocument has no normalized requirements. Run analyze_jd first.")
    requirements = tuple(
        store.get(ref.record_id, ref.revision, expected_type=JDRequirement)
        for ref in jd.requirement_refs
    )
    items = ResumeCandidateBuilder(store).build(input_data["candidate_items"])
    llm_selected = tuple(
        item.item_id
        for item, draft in zip(items, input_data["candidate_items"], strict=True)
        if draft.get("selected_by_llm") is True
    )
    resume_source = (
        store.get(input_data["resume_source_id"], expected_type=ResumeSource)
        if input_data["resume_source_id"]
        else None
    )
    planner = (context._runtime or {}).get("_resumePlanner")
    if planner is None:
        planner = ConstraintResumePlanner(
            weights=PlannerWeights.from_dict(input_data["weights"]),
        )
    if not callable(getattr(planner, "plan", None)):
        raise ValueError("_resumePlanner must expose plan(...)" )
    result = planner.plan(
        jd=jd,
        requirements=requirements,
        items=items,
        content_budget=input_data["content_budget"],
        section_order=input_data["section_order"],
        resume_source=resume_source,
        llm_selected_item_ids=llm_selected,
        required_item_ids=input_data.get("required_item_ids", ()),
    ).persist(store)
    return ToolResult(ok=True, output=json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


plan_resume_tool = ToolDefinition(
    name="plan_resume_content",
    description=(
        "Select evidence-backed resume content under a hard length budget. First propose candidate items and score "
        "jd_relevance, evidence_strength, title_and_brand_value, result_strength, and narrative_value from 0 to 1. "
        "Mark selected_by_llm before optimization for the experiment baseline. The solver never invents or rewrites "
        "facts; it only selects cited candidates and returns an exclusion reason for every omitted item. Resume "
        "sections always use the fixed internships, projects, education, skills/other, research, awards order."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "jd_id": {"type": "string"},
            "resume_source_id": {"type": "string"},
            "content_budget": {"type": "integer", "minimum": 1},
            "candidate_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "section": {"type": "string"},
                        "proposed_text": {"type": "string"},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "evidence_fact_ids": {"type": "array", "items": {"type": "string"}},
                        "requirement_ids": {"type": "array", "items": {"type": "string"}},
                        "redundancy_group": {"type": "string"},
                        "length_cost": {"type": "integer", "minimum": 1},
                        "selected_by_llm": {"type": "boolean"},
                        "features": {
                            "type": "object",
                            "properties": {
                                "jd_relevance": {"type": "number", "minimum": 0, "maximum": 1},
                                "evidence_strength": {"type": "number", "minimum": 0, "maximum": 1},
                                "title_and_brand_value": {"type": "number", "minimum": 0, "maximum": 1},
                                "result_strength": {"type": "number", "minimum": 0, "maximum": 1},
                                "narrative_value": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": [
                                "jd_relevance",
                                "evidence_strength",
                                "title_and_brand_value",
                                "result_strength",
                                "narrative_value",
                            ],
                        },
                    },
                    "required": [
                        "section",
                        "proposed_text",
                        "evidence_ids",
                        "evidence_fact_ids",
                        "requirement_ids",
                        "length_cost",
                        "features",
                    ],
                },
            },
            "weights": {"type": "object"},
            "required_item_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["jd_id", "content_budget", "candidate_items"],
    },
    validator=_validate_plan_resume,
    run=_plan_resume,
)


def _validate_generate_resume(input_data: dict) -> dict:
    plan_id = str(input_data.get("plan_id") or "").strip()
    header = input_data.get("header")
    claim_drafts = input_data.get("claim_drafts")
    if not plan_id:
        raise ValueError("plan_id is required")
    if not isinstance(header, dict) or not str(header.get("full_name") or "").strip():
        raise ValueError("header.full_name is required")
    if not isinstance(claim_drafts, list) or not claim_drafts:
        raise ValueError("claim_drafts must be a non-empty array")
    links = header.get("links", [])
    if not isinstance(links, list):
        raise ValueError("header.links must be an array")
    return {
        "plan_id": plan_id,
        "header": {
            "full_name": str(header.get("full_name") or "").strip(),
            "headline": str(header.get("headline") or "").strip(),
            "email": str(header.get("email") or "").strip(),
            "phone": str(header.get("phone") or "").strip(),
            "location": str(header.get("location") or "").strip(),
            "links": tuple(str(item).strip() for item in links if str(item).strip()),
        },
        "claim_drafts": claim_drafts,
        "template_id": str(input_data.get("template_id") or "").strip(),
    }


def _generate_resume(input_data: dict, context) -> ToolResult:
    store = _career_store(context)
    runtime = context._runtime or {}
    injected_identity = runtime.get("_careerGenerationIdentity")
    generator = (
        injected_identity
        if isinstance(injected_identity, ModelAPIIdentity)
        else ModelAPIIdentity.from_runtime(runtime)
    )
    plan = store.get(input_data["plan_id"], expected_type=ResumePlan)
    template = None
    if input_data.get("template_id"):
        from smart_agent.career import ResumeTemplate

        template = store.get(input_data["template_id"], expected_type=ResumeTemplate)
    builder = runtime.get("_resumeBuilder")
    if builder is None:
        builder = ResumeDocumentBuilder(store)
    if not callable(getattr(builder, "build", None)):
        raise ValueError("_resumeBuilder must expose build(...)" )
    result = builder.build(
        plan=plan,
        header=ResumeHeader(**input_data["header"]),
        claim_drafts=input_data["claim_drafts"],
        generator=generator,
        template=template,
    ).persist(store)
    return ToolResult(ok=True, output=json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


generate_resume_tool = ToolDefinition(
    name="generate_resume_document",
    description=(
        "Create a structured ResumeDocument from one persisted ResumePlan. Supply exactly one concise claim for "
        "each selected ResumeItem. Evidence and JD references are inherited by the system and cannot be supplied "
        "or changed by this call. Do not introduce any metric absent from cited evidence."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "plan_id": {"type": "string"},
            "header": {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string"},
                    "headline": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"},
                    "location": {"type": "string"},
                    "links": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["full_name"],
            },
            "claim_drafts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["item_id", "text"],
                },
            },
            "template_id": {"type": "string", "description": "Optional format-only ResumeTemplate identity."},
        },
        "required": ["plan_id", "header", "claim_drafts"],
    },
    validator=_validate_generate_resume,
    run=_generate_resume,
)


def _validate_resume_identity(input_data: dict) -> dict:
    resume_document_id = str(input_data.get("resume_document_id") or "").strip()
    if not resume_document_id:
        raise ValueError("resume_document_id is required")
    return {"resume_document_id": resume_document_id}


def _verify_resume(input_data: dict, context) -> ToolResult:
    from smart_agent.agent_collection import build_default_registry

    store = _career_store(context)
    runtime = context._runtime or {}
    injected = runtime.get("_resumeVerificationAgent")
    agent = (
        injected
        if isinstance(injected, ResumeVerificationAgent)
        else build_default_registry().create(
            "resume_verification",
            store=store,
            cwd=context.cwd,
            verifier_config=VerifierModelConfig.from_runtime(runtime),
        )
    )
    session_id = str(
        getattr(context.session, "session_id", "")
        or runtime.get("parentSessionId")
        or runtime.get("sessionId")
        or "career-session"
    )
    result = agent.verify(input_data["resume_document_id"], parent_session_id=session_id)
    return ToolResult(ok=True, output=json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


verify_resume_tool = ToolDefinition(
    name="verify_resume_document",
    description=(
        "Run the isolated Resume Verification Agent against a structured resume. The verifier model and API are "
        "loaded only from dedicated server-side configuration and must differ from the generation model and API. "
        "The verifier returns structured issues and never edits the resume."
    ),
    input_schema={
        "type": "object",
        "properties": {"resume_document_id": {"type": "string"}},
        "required": ["resume_document_id"],
    },
    validator=_validate_resume_identity,
    run=_verify_resume,
)


def _validate_revise_resume(input_data: dict) -> dict:
    parsed = _validate_resume_identity(input_data)
    report_id = str(input_data.get("verification_report_id") or "").strip()
    revisions = input_data.get("claim_revisions")
    if not report_id:
        raise ValueError("verification_report_id is required")
    if not isinstance(revisions, list) or not revisions:
        raise ValueError("claim_revisions must be a non-empty array")
    parsed.update({"verification_report_id": report_id, "claim_revisions": revisions})
    return parsed


def _revise_resume(input_data: dict, context) -> ToolResult:
    result = ResumeRevisionManager(_career_store(context)).revise(
        input_data["resume_document_id"],
        verification_report_id=input_data["verification_report_id"],
        claim_revisions=input_data["claim_revisions"],
    )
    return ToolResult(ok=True, output=json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


revise_resume_tool = ToolDefinition(
    name="revise_resume_document",
    description=(
        "Apply one of at most two evidence-constrained correction rounds. Only claims explicitly named by the "
        "current VerificationReport may change. A revision may remove requirement associations explicitly named "
        "by that report; evidence references remain immutable."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "resume_document_id": {"type": "string"},
            "verification_report_id": {"type": "string"},
            "claim_revisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "text": {"type": "string"},
                        "remove_requirement_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["claim_id", "text"],
                },
            },
        },
        "required": ["resume_document_id", "verification_report_id", "claim_revisions"],
    },
    validator=_validate_revise_resume,
    run=_revise_resume,
)


def _validate_render_resume(input_data: dict) -> dict:
    parsed = _validate_resume_identity(input_data)
    max_pages = int(input_data.get("max_pages", 1))
    if not 1 <= max_pages <= 3:
        raise ValueError("max_pages must be between 1 and 3")
    parsed.update(
        {
            "output_dir": str(input_data.get("output_dir") or ".runtime/career/exports").strip(),
            "max_pages": max_pages,
            "theme_id": str(input_data.get("theme_id") or "professional").strip(),
        }
    )
    return parsed


def _render_resume(input_data: dict, context) -> ToolResult:
    workspace = Path(context.cwd).resolve()
    output_dir = Path(input_data["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = workspace / output_dir
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(workspace):
        raise ValueError("Resume output_dir must stay inside the active workspace.")
    runtime = context._runtime or {}
    injected = runtime.get("_resumePDFRenderer")
    renderer = injected if isinstance(injected, ResumePDFRenderer) else ResumePDFRenderer(_career_store(context))
    render_args = {
        "output_dir": output_dir,
        "max_pages": input_data["max_pages"],
        "theme_id": input_data["theme_id"],
    }
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        result = renderer.render(input_data["resume_document_id"], **render_args)
    else:
        # Playwright's Sync API cannot run in the event-loop thread. Keep the
        # renderer synchronous for CLI/tests, but isolate Web-triggered calls.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="career-pdf") as executor:
            result = executor.submit(
                renderer.render,
                input_data["resume_document_id"],
                **render_args,
            ).result()
    return ToolResult(ok=True, output=json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


render_resume_tool = ToolDefinition(
    name="render_resume_pdf",
    description=(
        "Render only a passed, independently verified ResumeDocument to semantic HTML and an ATS-readable PDF. "
        "The Chromium pipeline rejects page-limit, horizontal-overflow, font, or text-layer failures and writes a "
        "claim provenance sidecar."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "resume_document_id": {"type": "string"},
            "output_dir": {"type": "string"},
            "max_pages": {"type": "integer", "minimum": 1, "maximum": 3},
            "theme_id": {"type": "string", "enum": ["professional"]},
        },
        "required": ["resume_document_id"],
    },
    validator=_validate_render_resume,
    run=_render_resume,
)


def _plan_interview(input_data: dict, _context) -> ToolResult:
    text = input_data["text"]
    skills = extract_tech_skills(text)
    focus = skills[:5] or ["project depth", "problem solving", "communication"]
    plan = {
        "stages": [
            {"name": "background", "questions": 1},
            {"name": "project evidence", "questions": 3},
            {"name": "role skills", "questions": 3, "focus": focus},
            {"name": "reflection", "questions": 1},
        ],
        "evaluation": ["technical accuracy", "evidence", "clarity", "role fit"],
    }
    return ToolResult(ok=True, output=json.dumps(plan, ensure_ascii=False, indent=2))


build_interview_plan_tool = ToolDefinition(
    name="build_interview_plan",
    description="Build a structured mock-interview plan from a JD or role description.",
    input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    validator=_require_text,
    run=_plan_interview,
)


def _validate_resume_review(input_data: dict) -> dict:
    if not isinstance(input_data.get("jd"), str) or not input_data["jd"].strip():
        raise ValueError("jd is required")
    for field in ("resume_projects", "project_summaries"):
        if not isinstance(input_data.get(field), list):
            raise ValueError(f"{field} must be an array")
    known_skills = input_data.get("known_skills", [])
    if not isinstance(known_skills, list):
        raise ValueError("known_skills must be an array")
    return {
        "jd": input_data["jd"].strip(),
        "resume_projects": input_data["resume_projects"],
        "project_summaries": input_data["project_summaries"],
        "known_skills": known_skills,
    }


def _review_resume(input_data: dict, _context) -> ToolResult:
    result = review_resume(**input_data)
    return ToolResult(ok=True, output=json.dumps(result, ensure_ascii=False, indent=2))


review_resume_tool = ToolDefinition(
    name="review_resume_evidence",
    description=(
        "Review a generated resume against the JD, imported project summaries, and verified user skills. "
        "Use this after generating or revising resume project content."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "jd": {"type": "string"},
            "resume_projects": {"type": "array", "items": {"type": "object"}},
            "project_summaries": {"type": "array", "items": {"type": "object"}},
            "known_skills": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["jd", "resume_projects", "project_summaries"],
    },
    validator=_validate_resume_review,
    run=_review_resume,
)


CAREER_TOOLS = [
    inspect_career_workspace_tool,
    import_career_material_tool,
    extract_career_facts_tool,
    list_candidate_facts_tool,
    import_candidate_facts_tool,
    reject_candidate_facts_tool,
    list_career_materials_tool,
    read_career_material_tool,
    confirm_career_facts_tool,
    invalidate_career_facts_tool,
    register_project_source_tool,
    list_project_sources_tool,
    save_project_understanding_tool,
    search_interview_knowledge_tool,
    import_interview_knowledge_source_tool,
    import_interview_knowledge_tool,
    list_interview_knowledge_tool,
    build_jd_capability_tree_tool,
    create_interview_session_tool,
    get_interview_session_tool,
    next_interview_question_tool,
    process_interview_answer_tool,
    generate_interview_depth_question_tool,
    generate_interview_breadth_question_tool,
    submit_interview_answer_tool,
    retry_interview_answer_tool,
    finish_interview_session_tool,
    analyze_jd_tool,
    retrieve_evidence_tool,
    plan_resume_tool,
    generate_resume_tool,
    verify_resume_tool,
    revise_resume_tool,
    render_resume_tool,
    build_interview_plan_tool,
    review_resume_tool,
]
