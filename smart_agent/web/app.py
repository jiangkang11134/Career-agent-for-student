"""Smart Agent Web application powered by FastAPI and WebSocket.

启动：smart-agent-web
访问：http://localhost:8341
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

# Web 入口位于 smart_agent/web，工作区根目录是其上两级。
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from smart_agent.agent_events import agent_event_hub
from smart_agent.agent_loop import run_agent_turn
from smart_agent.auth import AuthError, AuthService, AuthUser, auth_health
from smart_agent.config import load_project_env, load_runtime_config
from smart_agent.context_manager import ContextManager
from smart_agent.domains import AgentDomain, get_domain_module, route_domain
from smart_agent.fact_ledger import FactLedger, parse_memory_operations, strip_memory_operations
from smart_agent.model_registry import create_model_adapter
from smart_agent.preflight import PreflightIntent, classify_with_optional_model
from smart_agent.tenant_model_config import TenantModelConfigStore
from smart_agent.tooling import ToolContext, ToolRegistry
from smart_agent.tools import create_default_tool_registry
from smart_agent.web.idempotency import idempotency_store
from smart_agent.web.metrics import metrics_registry
from smart_agent.web.metrics import now as metrics_now
from smart_agent.web.task_queue import PersistentTaskQueue

app = FastAPI(title="Smart Agent")
from smart_agent.web.workspace_context import router as workspace_context_router

app.include_router(workspace_context_router)
_AUTH_SERVICE: AuthService | None = None
_CURRENT_AUTH_USER: contextvars.ContextVar[AuthUser | None] = contextvars.ContextVar(
    "smart_agent_auth_user",
    default=None,
)
_CURRENT_TENANT_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "smart_agent_tenant_id",
    default="",
)
_TENANT_MODEL_CONFIG_STORE: TenantModelConfigStore | None = None


def _tenant_model_config_store() -> TenantModelConfigStore:
    global _TENANT_MODEL_CONFIG_STORE
    if _TENANT_MODEL_CONFIG_STORE is None:
        defaults = load_runtime_config(ROOT)
        _TENANT_MODEL_CONFIG_STORE = TenantModelConfigStore(
            engine=_auth_service().engine,
            redis_url=str(defaults.get("redisUrl") or ""),
        )
    return _TENANT_MODEL_CONFIG_STORE


def _tenant_runtime(tenant_id: str) -> dict[str, object]:
    """Load global defaults and overlay the authenticated tenant's settings."""
    runtime = dict(load_runtime_config(ROOT))
    configs = _tenant_model_config_store().get_all(tenant_id)
    main = configs.get("main_agent")
    if main:
        provider = main["provider"]
        runtime.update({"model": main["model"], "providerOverride": provider})
        if provider == "anthropic":
            runtime.update({"baseUrl": main["base_url"], "apiKey": main["api_key"]})
        elif provider == "openai":
            runtime.update({"openaiBaseUrl": main["base_url"], "openaiApiKey": main["api_key"]})
        elif provider == "openrouter":
            runtime.update({"openrouterBaseUrl": main["base_url"], "openrouterApiKey": main["api_key"]})
        else:
            runtime.update({"customBaseUrl": main["base_url"], "customApiKey": main["api_key"]})
    suffixes = {
        "fact_extractor": "FactExtractor",
        "jd_analyzer": "JDAnalyzer",
        "project_matcher": "ProjectMatcher",
        "project_analyst": "ProjectAnalyst",
        "resume_generator": "ResumeGenerator",
        "jd_critic": "JDCritic",
        "interview_question": "InterviewQuestion",
        "interview_evaluator": "InterviewEvaluator",
        "verifier": "Verifier",
    }
    for role, suffix in suffixes.items():
        item = configs.get(role)
        if item:
            runtime.update({
                f"career{suffix}Provider": item["provider"],
                f"career{suffix}Model": item["model"],
                f"career{suffix}BaseUrl": item["base_url"],
                f"career{suffix}ApiKey": item["api_key"],
            })
    return runtime


def _apply_frontend_model_override(
    runtime: dict,
    *,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
) -> dict:
    """Apply one-request model settings sent by the Web UI.

    The WebSocket chat settings are per turn. Keep them in runtime instead of
    writing process-wide environment variables, otherwise one user's temporary
    key can leak into another request.
    """
    next_runtime = dict(runtime)
    model = str(model or "").strip()
    base_url = str(base_url or "").strip().rstrip("/")
    api_key = str(api_key or "").strip()
    if model:
        next_runtime["model"] = model
    if not base_url and not api_key:
        return next_runtime
    if base_url:
        next_runtime["baseUrl"] = base_url
        next_runtime["openaiBaseUrl"] = base_url
        next_runtime["customBaseUrl"] = base_url
    if api_key:
        next_runtime["apiKey"] = api_key
        next_runtime["openaiApiKey"] = api_key
        next_runtime["openrouterApiKey"] = api_key
        next_runtime["customApiKey"] = api_key
        next_runtime.pop("authToken", None)
    if base_url and "anthropic.com" not in base_url.lower():
        next_runtime["providerOverride"] = "custom"
    elif base_url:
        next_runtime["providerOverride"] = "anthropic"
    elif api_key:
        existing_provider = str(next_runtime.get("providerOverride") or "").strip().lower()
        if not existing_provider:
            if next_runtime.get("customBaseUrl"):
                existing_provider = "custom"
            elif next_runtime.get("openrouterBaseUrl"):
                existing_provider = "openrouter"
            elif next_runtime.get("openaiBaseUrl"):
                existing_provider = "openai"
        if existing_provider:
            next_runtime["providerOverride"] = existing_provider
        else:
            try:
                from smart_agent.model_registry import detect_provider

                next_runtime["providerOverride"] = detect_provider(
                    str(next_runtime.get("model") or ""),
                    next_runtime,
                ).value
            except Exception:
                next_runtime["providerOverride"] = "anthropic"
    return next_runtime


def _is_unsupported_model_error(exc: BaseException) -> bool:
    """识别供应商明确返回的模型不支持错误，避免吞掉普通请求失败。"""
    message = str(exc or "").lower()
    markers = (
        "not supported by any configured account",
        "model is not supported",
        "unsupported model",
        "model_not_supported",
    )
    return any(marker in message for marker in markers)


def _sanitize_user_facing_text(value: object) -> str:
    """Remove runtime control/guard lines that must never reach end users."""
    text = str(value or "")
    if not text:
        return ""
    kept: list[str] = []
    for line in text.splitlines():
        normalized = line.strip().casefold()
        if normalized.startswith((
            "runtime phase:",
            "execution guard:",
            "verification guard:",
            "completion guard:",
            "widening is now available",
            "widened mode is active",
        )):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _user_visible_progress_message(value: object) -> str:
    """Translate internal progress markers to concise UI status text."""
    text = str(value or "").strip()
    lowered = text.casefold()
    if lowered.startswith("runtime phase:"):
        match = re.search(r"runtime phase:\s*([a-z_-]+)", text, flags=re.IGNORECASE)
        phase = (match.group(1).lower() if match else "")
        return {
            "explore": "正在分析任务结构",
            "execute": "正在执行任务",
            "verify": "正在验证结果",
        }.get(phase, "正在处理任务")
    if lowered.startswith(("execution guard:", "verification guard:", "completion guard:")):
        return "正在补充验证并整理结果"
    if "widening is now available" in lowered or "widened mode is active" in lowered:
        return "正在扩大检查范围"
    return _sanitize_user_facing_text(text)


def _run_agent_turn_with_model_fallback(
    *,
    model: Any,
    tools: Any,
    fallback_model_name: str,
    current_model_name: str,
    runtime: dict[str, Any],
    **kwargs: Any,
) -> list[dict]:
    """Run one turn and retry once with the configured default model."""
    kwargs.pop("runtime", None)
    try:
        return run_agent_turn(model=model, tools=tools, runtime=runtime, **kwargs)
    except RuntimeError as exc:
        if not (
            _is_unsupported_model_error(exc)
            and fallback_model_name
            and fallback_model_name != current_model_name
        ):
            raise
        runtime["model"] = fallback_model_name
        fallback = create_model_adapter(model=fallback_model_name, tools=tools, runtime=runtime)
        return run_agent_turn(model=fallback, tools=tools, runtime=runtime, **kwargs)


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    started = metrics_now()
    response = await call_next(request)
    route = getattr(request.scope.get("route"), "path", request.url.path)
    elapsed = metrics_now() - started
    metrics_registry.observe_request(
        request.method,
        route,
        response.status_code,
        elapsed,
    )
    response.headers["X-Request-Duration-Ms"] = str(round(elapsed * 1000, 2))
    return response


def _hydrate_project_integrations() -> None:
    """Expose project .env integration settings to legacy clients.

    ``load_project_env`` intentionally does not mutate ``os.environ``. The
    FactLedger, Redis queue and idempotency clients still read these three
    infrastructure keys directly, so hydrate only those keys at startup.
    Existing process-level values always win.
    """
    project_env = load_project_env(ROOT)
    for name in (
        "SMART_AGENT_DATABASE_URL",
        "SMART_AGENT_REDIS_URL",
        "SMART_AGENT_FACT_LEDGER_BACKEND",
    ):
        if not os.environ.get(name) and project_env.get(name):
            os.environ[name] = str(project_env[name]).strip()


def _auth_service() -> AuthService:
    global _AUTH_SERVICE
    if _AUTH_SERVICE is None:
        try:
            _hydrate_project_integrations()
        except Exception:
            pass
        _AUTH_SERVICE = AuthService.from_env()
    return _AUTH_SERVICE


def _extract_bearer_token(value: str) -> str:
    prefix = "bearer "
    text = str(value or "").strip()
    return text[len(prefix):].strip() if text.lower().startswith(prefix) else ""


def _request_bearer_token(request: Request) -> str:
    return _extract_bearer_token(request.headers.get("authorization", ""))


def _websocket_token(websocket: WebSocket) -> str:
    query_token = str(websocket.query_params.get("token") or "").strip()
    if query_token:
        return query_token
    return _extract_bearer_token(websocket.headers.get("authorization", ""))


def _require_user_for_token(token: str) -> AuthUser:
    user = _auth_service().user_from_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录")
    return user


def _career_tenant_id() -> str:
    user = _CURRENT_AUTH_USER.get()
    if user is not None:
        return user.tenant_id
    return _CURRENT_TENANT_ID.get()


def _set_request_user(user: AuthUser):
    user_token = _CURRENT_AUTH_USER.set(user)
    tenant_token = _CURRENT_TENANT_ID.set(user.tenant_id)
    return user_token, tenant_token


def _reset_request_user(tokens) -> None:
    user_token, tenant_token = tokens
    _CURRENT_TENANT_ID.reset(tenant_token)
    _CURRENT_AUTH_USER.reset(user_token)


def _career_auth_required(path: str) -> bool:
    return path.startswith("/api/career")


def _tenant_api_auth_required(path: str) -> bool:
    """Protect tenant and administrative APIs, including legacy file routes."""
    return (
        _career_auth_required(path)
        or path.startswith("/api/admin")
        or path in {"/api/files", "/api/file", "/api/save", "/api/upload", "/api/report", "/api/metrics"}
    )


@app.middleware("http")
async def career_auth_middleware(request: Request, call_next):
    if not _tenant_api_auth_required(request.url.path):
        return await call_next(request)
    try:
        user = _require_user_for_token(_request_bearer_token(request))
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"detail": f"认证服务不可用: {exc}"})
    tokens = _set_request_user(user)
    request.state.current_user = user
    try:
        if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            response = await idempotency_store.execute(
                request,
                call_next,
                f"{user.tenant_id}:{request.url.path}",
            )
        else:
            response = await call_next(request)
        if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            _invalidate_career_library_cache(user.tenant_id)
        return response
    finally:
        _reset_request_user(tokens)


def _auth_payload(user: AuthUser, token: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"user": user.to_dict()}
    if token is not None:
        payload["token"] = token
    return payload


@app.post("/api/auth/register")
async def auth_register(data: dict):
    try:
        user, token = await asyncio.to_thread(
            _auth_service().register,
            str(data.get("account") or data.get("email") or ""),
            str(data.get("password") or ""),
            str(data.get("display_name") or ""),
        )
        return _auth_payload(user, token)
    except AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"auth unavailable: {exc}") from exc


@app.post("/api/auth/login")
async def auth_login(data: dict):
    try:
        user, token = await asyncio.to_thread(
            _auth_service().authenticate,
            str(data.get("account") or data.get("email") or ""),
            str(data.get("password") or ""),
        )
        return _auth_payload(user, token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"auth unavailable: {exc}") from exc


@app.get("/api/auth/me")
async def auth_me(request: Request):
    try:
        return _auth_payload(_require_user_for_token(_request_bearer_token(request)))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"auth unavailable: {exc}") from exc


@app.post("/api/auth/logout")
async def auth_logout(request: Request):
    token = _request_bearer_token(request)
    try:
        if token:
            await asyncio.to_thread(_auth_service().revoke, token)
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"auth unavailable: {exc}") from exc


@app.get("/api/auth/health")
async def auth_health_check():
    try:
        _hydrate_project_integrations()
        service = _auth_service()
        await asyncio.to_thread(service.init_schema)
        from smart_agent.fact_ledger import FactLedger

        ledger = FactLedger(ROOT / ".runtime" / "career" / "fact-ledger", tenant_id="health-check")
        redis_url = str(os.environ.get("SMART_AGENT_REDIS_URL") or "").strip()
        redis_client = ledger._redis_client()
        return {
            "ok": True,
            "database": service.engine.url.get_backend_name(),
            "fact_ledger_backend": ledger.backend,
            "redis": {
                "configured": bool(redis_url),
                "connected": redis_client is not None,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"ok": False, "detail": str(exc)})


@dataclass(frozen=True, slots=True)
class ResumeEvidencePolicy:
    """Internal resume evidence policy; unsafe variants are benchmark-only."""

    enforce_project_prompt: bool = True
    enforce_project_claim_validation: bool = True
    enforce_project_user_requirement_filter: bool = True
    apply_critic_revisions: bool = True
    verification_mode: str = "enforce"


FULL_RESUME_EVIDENCE_POLICY = ResumeEvidencePolicy()
PROJECT_SAFETY_ABLATION_POLICY = ResumeEvidencePolicy(
    enforce_project_prompt=False,
    enforce_project_claim_validation=False,
    enforce_project_user_requirement_filter=False,
    apply_critic_revisions=False,
    verification_mode="shadow",
)
# Bump this whenever a frontend-visible API contract changes.  The browser
# checks it during startup so a stale long-running backend cannot masquerade
# as a successfully updated service.
CAREER_API_VERSION = "2026-08-30.2"
CAREER_API_CAPABILITIES = (
    "material_batch_delete",
    "interview_knowledge_delete",
    "interview_knowledge_batch_delete",
    "interview_knowledge_structured_pdf_md_chunking",
    "interview_knowledge_pdf_mode_selection",
    "interview_knowledge_batch_import_diagnostics",
    "interview_knowledge_bge_m3_embeddings",
    "interview_knowledge_sparse_dense_embeddings",
    "project_library_batch_delete",
    "project_knowledge_bge_m3_sparse_dense",
    "project_analysis_background",
    "resume_version_delete",
    "resume_session_delete",
)
FRONTEND_DIST = ROOT / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    # Vite emits hashed assets; serving them separately lets the browser cache
    # the React bundle without re-reading the page or the legacy fallback.
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")
MAX_EDITABLE_FILE_BYTES = 1_000_000
MAX_LEGACY_UPLOAD_BYTES = 100 * 1024 * 1024
_LEGACY_UPLOAD_SUFFIXES = {
    ".zip", ".tar", ".gz", ".tgz", ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".cs",
    ".php", ".rb", ".swift", ".kt", ".sql", ".sh", ".ps1", ".html",
    ".css", ".scss", ".vue", ".svelte", ".md", ".txt", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".pdf", ".docx",
}
PROJECT_ARCHIVE_MAX_BYTES = 5 * 1024 * 1024 * 1024
PROJECT_ARCHIVE_CHUNK_BYTES = 8 * 1024 * 1024
PROJECT_ARCHIVE_MAX_MEMBERS = 200_000
INTERVIEW_KNOWLEDGE_UPLOAD_MAX_BYTES = 5 * 1024 * 1024 * 1024
INTERVIEW_KNOWLEDGE_CHUNK_BYTES = 8 * 1024 * 1024
INTERVIEW_KNOWLEDGE_MAX_MEMBERS = 200_000
INTERVIEW_KNOWLEDGE_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".docx",
    ".ipynb",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}
_VOICE_RUNTIMES: dict[str, object] = {}
_PROJECT_EMBEDDERS: dict[str, object] = {}
_PROJECT_INDEX_STORES: dict[str, object] = {}
_CAREER_STORE_INSTANCES: dict[str, tuple[object, object, object, object]] = {}
_PROJECT_ANALYSIS_TASKS: dict[str, asyncio.Task] = {}
_PROJECT_ANALYSIS_QUEUED: set[str] = set()
_PROJECT_TASK_QUEUE = PersistentTaskQueue(
    name="smart_agent:project_analysis",
    root=ROOT / ".runtime" / "tasks",
)
_PROJECT_TASK_STOP: asyncio.Event | None = None
_PROJECT_TASK_WORKER: asyncio.Task | None = None
_CAREER_ARTIFACTS_CACHE: dict[str, tuple[float, dict]] = {}
_CAREER_LIBRARY_CACHE_TTL = max(
    1.0, float(os.environ.get("SMART_AGENT_LIBRARY_CACHE_TTL_SECONDS", "30"))
)
_CAREER_MATERIAL_LIBRARY_CACHE: dict[str, tuple[float, dict]] = {}
_CAREER_PROJECT_LIBRARY_CACHE: dict[str, tuple[float, dict]] = {}


def _career_library_cache_key() -> str:
    return _career_tenant_id() or "legacy"


def _safe_tenant_fragment(tenant_id: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9_-]+", "_", str(tenant_id or "").strip())
    return fragment[:48] or "legacy"


def _career_library_cache_get(cache: dict[str, tuple[float, dict]], key: str) -> dict | None:
    entry = cache.get(key)
    if entry is None:
        return None
    created_at, payload = entry
    if time.monotonic() - created_at >= _CAREER_LIBRARY_CACHE_TTL:
        cache.pop(key, None)
        return None
    return payload


def _invalidate_career_library_cache(tenant_id: str | None = None) -> None:
    selected = tenant_id if tenant_id is not None else _career_tenant_id()
    if selected:
        key = selected or "legacy"
        _CAREER_ARTIFACTS_CACHE.pop(key, None)
        _CAREER_MATERIAL_LIBRARY_CACHE.pop(key, None)
        _CAREER_PROJECT_LIBRARY_CACHE.pop(key, None)
        return
    _CAREER_MATERIAL_LIBRARY_CACHE.clear()
    _CAREER_PROJECT_LIBRARY_CACHE.clear()
    _CAREER_ARTIFACTS_CACHE.clear()
_INTERVIEW_AGENT_EVENTS: deque[dict[str, object]] = deque(maxlen=500)
_AGENT_WORK_PATTERN = re.compile(
    r"(?i)(?:"
    r"write|create|implement|modify|edit|fix|debug|refactor|run|test|review|"
    r"upload|save|delete|rename|move|commit|"
    r"写|创建|实现|修改|编辑|修复|调试|重构|运行|测试|审查|"
    r"读取?文件|查看.*(?:文件|仓库|代码)|删除|重命名|移动|提交|"
    r"简历|JD|岗位|求职|面试|投递"
    r")"
)


@app.exception_handler(Exception)
async def unhandled_api_exception(_request, exc: Exception):
    """Keep API failures machine-readable instead of returning plain-text 500 pages."""
    try:
        from smart_agent.career import ReferenceIntegrityError

        if isinstance(exc, ReferenceIntegrityError):
            return JSONResponse(
                status_code=409,
                content={
                    "detail": (
                        "简历引用的个人事实已未确认或已失效，请先确认材料事实并重新分析岗位。"
                    ),
                    "code": "resume_reference_integrity_error",
                },
            )
    except ImportError:
        pass
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


def _workspace_path(path: str) -> Path:
    target = (ROOT / path).resolve()
    try:
        target.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path outside project root") from exc
    return target


def _tenant_project_path(project_id: str, relative_path: str = ".") -> Path:
    """Resolve a path below one of the current tenant's registered projects."""
    selected = str(project_id or "").strip()
    if not selected:
        raise HTTPException(status_code=400, detail="project_id is required")
    try:
        source = _project_library_stores()[0].get(selected, refresh=False)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
    root = Path(source.source_location).resolve()
    if root.is_file():
        root = root.parent
    candidate = (root / str(relative_path or ".")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="path outside registered project") from exc
    return candidate


def _tenant_owned_path(path: Path) -> Path:
    """Require a local project source to live inside the current tenant root."""
    tenant_root = _career_root().resolve()
    try:
        path.resolve().relative_to(tenant_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="project source must be inside tenant storage") from exc
    return path.resolve()


async def _read_limited_upload(file: UploadFile, *, limit: int) -> bytes:
    """Read an upload without allowing an unbounded body to reach memory."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(min(1024 * 1024, limit - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=f"upload exceeds {limit // (1024 * 1024)} MB")
        chunks.append(chunk)
    return b"".join(chunks)


def _requires_agent_loop(user_input: str) -> bool:
    """Return whether a request needs tools and the multi-step agent loop."""
    return bool(_AGENT_WORK_PATTERN.search(user_input or ""))


def _should_use_agent_loop(preflight, user_input: str) -> bool:
    """Select the tool loop from the conservative preflight decision.

    The legacy keyword matcher is only a compatibility fallback: inspection
    requests can lack an execution verb even though preflight marks them as
    tool-backed tasks.
    """
    if getattr(preflight, "needs_tools", False):
        return True
    intent = getattr(preflight, "intent", None)
    if intent in {PreflightIntent.TASK, PreflightIntent.PROJECT_INFO}:
        return True
    return _requires_agent_loop(user_input)


def _is_read_only_workspace_request(user_input: str) -> bool:
    """Recognize inspection requests that may use the current repo by default."""
    value = " ".join(str(user_input or "").strip().split())
    return bool(
        _READ_ONLY_WORKSPACE_PATTERN.search(value)
        and not _MUTATION_REQUEST_PATTERN.search(value)
    )


def _career_root() -> Path:
    tenant_id = _career_tenant_id()
    if tenant_id:
        root = (ROOT / ".runtime" / "career" / "tenants" / tenant_id).resolve()
    else:
        root = (ROOT / ".runtime" / "career").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _project_knowledge_store(root: Path):
    """Build the project index with the configured local BGE-M3 provider.

    The provider is optional: environments without a local model continue to
    use the deterministic hashing backend, preserving offline compatibility.
    """
    from smart_agent.career import LocalBGEInterviewEmbedding, ProjectKnowledgeStore, QdrantProjectKnowledgeStore

    try:
        runtime_config = load_runtime_config(ROOT)
    except (OSError, ValueError, RuntimeError):
        runtime_config = {}
    key = str(root.resolve())
    provider = _PROJECT_EMBEDDERS.get(key)
    if provider is None:
        try:
            provider = LocalBGEInterviewEmbedding.from_runtime(runtime_config)
        except (OSError, ValueError, RuntimeError):
            provider = None
        _PROJECT_EMBEDDERS[key] = provider
    backend = str(
        runtime_config.get("careerVectorBackend")
        or os.environ.get("CAREER_VECTOR_BACKEND")
        or os.environ.get("CAREER_VECTOR_STORE")
        or "chromadb"
    ).strip().casefold()
    qdrant_url = str(runtime_config.get("qdrantUrl") or os.environ.get("QDRANT_URL") or "").strip()
    qdrant_prefix = str(
        runtime_config.get("qdrantCollectionPrefix")
        or os.environ.get("QDRANT_COLLECTION_PREFIX", "smart_agent_project")
    )
    tenant_id = _career_tenant_id()
    if tenant_id:
        qdrant_prefix = f"{qdrant_prefix}_{_safe_tenant_fragment(tenant_id)}"
    provider_name = str(getattr(provider, "name", "hashing-384"))
    store_key = "|".join((key, backend, qdrant_url, qdrant_prefix, provider_name))
    cached = _PROJECT_INDEX_STORES.get(store_key)
    if cached is not None:
        return cached
    if backend == "qdrant":
        store = QdrantProjectKnowledgeStore(
            root,
            embedding_backend=provider,
            url=qdrant_url or None,
            api_key=str(runtime_config.get("qdrantApiKey") or os.environ.get("QDRANT_API_KEY") or "").strip() or None,
            collection_prefix=qdrant_prefix,
            timeout_seconds=float(runtime_config.get("qdrantTimeoutSeconds") or os.environ.get("QDRANT_TIMEOUT_SECONDS", 10)),
        )
    elif backend == "chromadb":
        store = ProjectKnowledgeStore(root, embedding_backend=provider)
    else:
        raise ValueError(f"Unsupported CAREER_VECTOR_BACKEND: {backend}")
    _PROJECT_INDEX_STORES[store_key] = store
    return store


def _career_stores():
    from smart_agent.career import CandidateFactStore, CareerStore, MaterialStore

    root = _career_root()
    cache_key = str(root.resolve())
    cached = _CAREER_STORE_INSTANCES.get(cache_key)
    if cached is not None:
        return cached
    stores = (
        CareerStore(root),
        MaterialStore(root / "materials", workspace=str(ROOT)),
        CandidateFactStore(root / "candidate-facts"),
    )
    result = (root, *stores)
    _CAREER_STORE_INSTANCES[cache_key] = result
    return result


def _voice_session_runtime():
    from smart_agent.career import VoiceRecordingStore, VoiceSessionRuntime

    root = _career_root()
    key = str(root)
    runtime = _VOICE_RUNTIMES.get(key)
    if not isinstance(runtime, VoiceSessionRuntime):
        runtime = VoiceSessionRuntime(recording_store=VoiceRecordingStore(root / "voice-recordings"))
        _VOICE_RUNTIMES[key] = runtime
    return runtime


def _split_resume_into_materials(
    *,
    root: Path,
    career_store,
    material_store,
    candidate_store,
    label: str,
    resume_source,
) -> dict[str, dict]:
    """Persist non-empty section materials pinned to one resume revision."""
    from smart_agent.career import MaterialCategory, PersonalMaterialWorkflow

def _localize_interview_question_payload(question: dict | None, *, action: str = "ask") -> None:
    """Preserve recorded LLM output; presentation must not synthesize questions."""


def _localize_interview_report_question(item: dict) -> None:
    """Preserve historical question text instead of masking it with a template."""

    section_categories = {
        "internships": MaterialCategory.INTERNSHIP,
        "projects": MaterialCategory.PROJECT,
        "education": MaterialCategory.EDUCATION,
        "skills": MaterialCategory.SKILLS,
        "research": MaterialCategory.RESEARCH,
        "awards": MaterialCategory.AWARD,
    }
    project_store = _project_knowledge_store(root / "project-knowledge")
    results: dict[str, dict] = {}
    for section in resume_source.sections:
        if not section.text:
            continue
        category = section_categories[section.section_id]
        section_label = f"{label} · {category.value}"
        workflow = PersonalMaterialWorkflow(
            material_store,
            candidate_store,
            career_store=career_store,
            project_store=project_store,
        )
        result = workflow.organize(
            category=category,
            label=section_label,
            text=section.text,
            project_id=f"{resume_source.resume_source_id}-{category.value}",
            max_candidate_facts=100,
            resume_section=section,
        )
        results[category.value] = {
            "material": result.import_result.to_dict(include_extracted_text=False),
            "candidate_facts": [fact.to_dict() for fact in result.candidate_facts],
            "structured_path": result.structured_path,
        }
    return results


def _interview_runtime_direct():
    from smart_agent.career import (
        CareerGraphRAGIndex,
        CareerStore,
        InterviewKnowledgeLibrary,
        InterviewKnowledgeStore,
        InterviewRuntime,
        InterviewSessionStore,
        KnowledgeMasteryStore,
        LLMInterviewQuestionGenerator,
        LocalBGEInterviewEmbedding,
        create_career_client,
    )

    root = _career_root()
    knowledge_library = InterviewKnowledgeLibrary(root / "interview-knowledge")
    career_store = CareerStore(root)
    question_client = create_career_client(
        load_runtime_config(ROOT),
        "interview_question",
        audit_root=root / "audit",
        required=True,
    )
    runtime = load_runtime_config(ROOT)
    from smart_agent.career.interview_inputs import InterviewInputStore
    user = _CURRENT_AUTH_USER.get()
    if user:
        career_store = InterviewInputStore(career_store, user.user_id)
    embedding_provider = LocalBGEInterviewEmbedding.from_runtime(runtime)
    return InterviewRuntime(
        career_store=career_store,
        knowledge_store=InterviewKnowledgeStore(root / "interview-knowledge"),
        session_store=InterviewSessionStore(root / "interviews"),
        question_generator=LLMInterviewQuestionGenerator(question_client, career_store),
        mastery_store=KnowledgeMasteryStore(root / "interview-mastery"),
        graph_knowledge_index=CareerGraphRAGIndex(
            root / "graph-rag", embedding_provider=embedding_provider
        ),
        knowledge_library=knowledge_library,
    )


def _interview_question_payload_direct(interview_id: str, question) -> dict:
    from smart_agent.career import InterviewSessionStore

    root = _career_root()
    state = InterviewSessionStore(root / "interviews").get(interview_id)
    return _interview_question_payload_from_state(state, question)


@lru_cache(maxsize=128)
def _interview_persisted_corpus_chunks(
    root: str,
    corpus_id: str,
    corpus_version: str,
) -> tuple[object, ...]:
    from smart_agent.career import InterviewKnowledgeStore

    return tuple(InterviewKnowledgeStore(root).get(corpus_id).chunks)


def _interview_question_payload_from_state(
    state,
    question,
    *,
    runtime=None,
    use_graph_runtime: bool = True,
) -> dict:
    payload = question.to_dict()
    block_by_chunk = {
        block.chunk_id: block
        for block in getattr(question, "knowledge_block_refs", ())
    }
    payload["interview_stage"] = state.interview_stage.value
    payload["active_line"] = state.active_line.value if state.active_line else None
    payload["active_subject"] = state.active_subject
    payload["candidate_question_wait_seconds"] = state.candidate_question_wait_seconds
    payload["status"] = state.status.value
    knowledge_sources: list[dict[str, object]] = []
    knowledge_errors: list[dict[str, str]] = []
    try:
        if use_graph_runtime:
            chunks = (runtime or _interview_runtime_direct()).knowledge_chunks(
                state.interview_id,
                tuple(question.knowledge_refs),
            )
        else:
            chunks_by_id = {
                chunk.chunk_id: chunk
                for corpus_id in state.knowledge_corpus_ids
                for chunk in _interview_persisted_corpus_chunks(
                    str(_career_root() / "interview-knowledge"),
                    corpus_id,
                    str(state.knowledge_corpus_versions.get(corpus_id, "")),
                )
            }
            chunks = tuple(chunks_by_id[item] for item in question.knowledge_refs)
    except Exception as exc:
        # Audit reads must survive a locked/unavailable Graph RAG SQLite file.
        # The immutable JSON corpus is the authoritative fallback for the
        # already-persisted chunk references.
        try:
            from smart_agent.career import InterviewKnowledgeStore

            store = InterviewKnowledgeStore(_career_root() / "interview-knowledge")
            chunks_by_id = {
                chunk.chunk_id: chunk
                for corpus_id in state.knowledge_corpus_ids
                for chunk in store.get(corpus_id).chunks
            }
            chunks = tuple(chunks_by_id[item] for item in question.knowledge_refs)
        except Exception as fallback_exc:
            chunks = ()
            knowledge_errors.append({"corpus_id": "", "error": str(fallback_exc or exc)})
    for chunk in chunks:
        knowledge_sources.append({
            "chunk_id": chunk.chunk_id,
            "block_role": block_by_chunk.get(chunk.chunk_id).role.value if chunk.chunk_id in block_by_chunk else None,
            "block_label": block_by_chunk.get(chunk.chunk_id).label if chunk.chunk_id in block_by_chunk else None,
            "corpus_id": chunk.corpus_id,
            "topic": chunk.topic,
            "difficulty": chunk.difficulty,
            "source_id": chunk.source_id,
            "locator": chunk.locator,
            "content_hash": chunk.content_hash,
            "text": chunk.text,
            "text_summary": chunk.text.strip()[:320],
        })
    payload["knowledge_sources"] = knowledge_sources
    payload["knowledge_load_errors"] = knowledge_errors
    return payload


def _interview_recommendation_trace(state, question_id: str) -> dict[str, object]:
    for trace in reversed(state.recommendation_traces):
        if trace.question_id == question_id:
            return trace.to_dict()
    return {}


def _interview_round_audit(
    state,
    *,
    question,
    turn=None,
    runtime=None,
    use_graph_runtime: bool = True,
) -> dict[str, object]:
    """Build one durable, readable audit record from the session state."""
    question_payload = question.to_dict()
    recommendation = _interview_recommendation_trace(state, question.question_id)
    retrieval = dict(recommendation.get("retrieval_trace") or {})
    knowledge_errors: list[dict[str, str]] = []
    try:
        knowledge_blocks = _interview_question_payload_from_state(
            state,
            question,
            runtime=runtime,
            use_graph_runtime=use_graph_runtime,
        ).get("knowledge_sources", [])
    except Exception as exc:
        knowledge_blocks = []
        knowledge_errors.append({"error": str(exc)})

    orchestration = dict(turn.orchestration_audit or {}) if turn is not None else {}
    final_decision = dict(orchestration.get("final_decision") or {})
    selected_action = str(
        recommendation.get("selected_action")
        or ("depth" if question.path.value == "depth" else "breadth")
    )
    selection_reason = str(
        recommendation.get("selection_reason")
        or ("首个问题，按面试流程开始提问。" if turn is None and not recommendation else "当前问题未保存独立策略说明。")
    )
    evaluation = None
    answer = None
    if turn is not None:
        answer = {
            "text": turn.answer,
            "created_at": turn.created_at,
            "submitted": True,
        }
        evaluation = {
            "correctness": turn.correctness,
            "completeness": turn.completeness,
            "confidence": turn.confidence,
            "reason": turn.evaluation_reason,
            "evaluation_refs": list(turn.evaluation_refs),
            "knowledge_confidence_updates": [
                item.to_dict() for item in turn.knowledge_confidence_updates
            ],
            "observation": turn.observation.to_dict() if turn.observation else None,
        }
    return {
        "turn": recommendation.get("turn") or (len(state.turns) + 1 if turn is None else len(state.turns)),
        "question_id": question.question_id,
        "question": question_payload,
        "answer": answer,
        "user_answer": turn.answer if turn is not None else None,
        "knowledge_refs": list(question.knowledge_refs),
        "evaluation_refs": list(turn.evaluation_refs) if turn is not None else [],
        "knowledge_blocks": knowledge_blocks,
        "knowledge_load_errors": knowledge_errors,
        "selection": {
            "mode": selected_action,
            "path": question.path.value,
            "action": selected_action,
            "reason": selection_reason,
            "followup_depth": question.followup_depth,
            "recommendation_trace": recommendation,
        },
        "retrieval": {
            "mode": retrieval.get("mode") or ("not_applicable" if not question.knowledge_refs else "reference_lookup"),
            "query": retrieval.get("query") or "",
            "candidate_knowledge_refs": list(recommendation.get("candidate_knowledge_point_ids") or []),
            "selected_knowledge_refs": list(recommendation.get("selected_knowledge_point_ids") or question.knowledge_refs),
            "returned_node_ids": list(retrieval.get("returned_node_ids") or []),
            "hits": list(retrieval.get("hits") or []),
            "fallback_used": bool(retrieval.get("fallback_used", False)),
            "timings_ms": dict(retrieval.get("timings_ms") or {}),
            "trace": retrieval,
        },
        "evaluation": evaluation,
        "decision_after_answer": final_decision or None,
        "orchestration_audit": orchestration or None,
    }


def _interview_audit_payload(state) -> dict[str, object]:
    runtime = None
    rounds = [
        _interview_round_audit(
            state,
            question=turn.question,
            turn=turn,
            runtime=runtime,
            use_graph_runtime=False,
        )
        for turn in state.turns
    ]
    answered_ids = {str(item["question_id"]) for item in rounds}
    if state.pending_question is not None and state.pending_question.question_id not in answered_ids:
        rounds.append(
            _interview_round_audit(
                state,
                question=state.pending_question,
                runtime=runtime,
                use_graph_runtime=False,
            )
        )
    depth_count = sum(item["selection"]["mode"] == "depth" for item in rounds)
    breadth_count = sum(item["selection"]["mode"] == "breadth" for item in rounds)
    answered_count = sum(item["answer"] is not None for item in rounds)
    return {
        "interview_id": state.interview_id,
        "generated_at": time.time(),
        "session": {
            "mode": state.mode.value,
            "status": state.status.value,
            "resume_run_id": state.resume_run_id,
            "jd_id": state.jd_id,
            "jd_revision": state.jd_revision,
            "resume_document_id": state.resume_document_id,
            "resume_document_revision": state.resume_document_revision,
            "knowledge_corpus_ids": list(state.knowledge_corpus_ids),
            "knowledge_corpus_versions": dict(state.knowledge_corpus_versions),
            "interview_stage": state.interview_stage.value,
            "active_line": state.active_line.value if state.active_line else None,
            "active_subject": state.active_subject,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
        },
        "summary": {
            "round_count": len(rounds),
            "answered_rounds": answered_count,
            "pending_rounds": len(rounds) - answered_count,
            "depth_rounds": depth_count,
            "breadth_rounds": breadth_count,
            "knowledge_grounded_rounds": sum(bool(item["knowledge_refs"]) for item in rounds),
        },
        "rounds": rounds,
    }


def _is_stale_generic_interview_question(question) -> bool:
    from smart_agent.career.interview_runtime import InterviewRuntime

    return InterviewRuntime._should_regenerate_structured_pending_question(question)


def _repair_stale_interview_pending_for_payload(session) -> dict:
    if session.pending_question is None:
        return session.to_dict()
    if not _is_stale_generic_interview_question(session.pending_question):
        return session.to_dict()
    if session.interview_points:
        try:
            runtime = _interview_runtime_direct()
            runtime.next_question(session.interview_id)
            return runtime.session_store.get(session.interview_id).to_dict()
        except (KeyError, ValueError, RuntimeError):
            pass
    payload = session.to_dict()
    payload["pending_question"] = None
    payload["stale_pending_question_hidden"] = True
    return payload


def _interview_preview(value: object, limit: int = 320) -> str:
    text_value = re.sub(r"\\s+", " ", str(value or "").strip())
    if len(text_value) > limit:
        return text_value[: max(0, limit - 1)].rstrip() + "…"
    return text_value


def _interview_decision_snapshot(interview_id: str, question_payload: dict) -> dict[str, object]:
    from smart_agent.career import InterviewSessionStore

    try:
        state = InterviewSessionStore(_career_root() / "interviews").get(interview_id)
        decision = state.last_decision.to_dict() if state.last_decision else None
        stage = state.interview_stage.value
        active_subject = state.active_subject
    except Exception:
        decision = None
        stage = ""
        active_subject = None
    utility = question_payload.get("utility") if isinstance(question_payload.get("utility"), dict) else {}
    return {
        "interview_stage": stage,
        "active_subject": active_subject,
        "path": question_payload.get("path") or (decision or {}).get("path") or "",
        "action": (decision or {}).get("action") or ("follow_up" if question_payload.get("parent_question_id") else "ask"),
        "topic": question_payload.get("topic") or (decision or {}).get("topic") or "",
        "question_type": question_payload.get("question_type") or (decision or {}).get("question_type") or "",
        "reason": (decision or {}).get("reason") or "first question or idempotent pending question",
        "followup_depth": int(question_payload.get("followup_depth") or 0),
        "depth_weight": utility.get("depth_weight"),
        "breadth_weight": utility.get("breadth_weight"),
        "utility": utility,
    }


def _interview_session_admin_input(state) -> dict[str, object]:
    return {
        "interview_id": state.interview_id,
        "mode": state.mode.value,
        "status": state.status.value,
        "resume_run_id": state.resume_run_id,
        "jd_id": state.jd_id,
        "jd_revision": state.jd_revision,
        "resume_document_id": state.resume_document_id,
        "resume_document_revision": state.resume_document_revision,
        "knowledge_corpus_ids": list(state.knowledge_corpus_ids),
        "knowledge_corpus_versions": dict(state.knowledge_corpus_versions),
        "turn_count": len(state.turns),
        "max_turns": state.max_turns,
        "interview_stage": state.interview_stage.value,
        "active_line": state.active_line.value if state.active_line else None,
        "active_subject": state.active_subject,
        "duration_minutes": state.duration_minutes,
        "max_questions_per_subject": state.max_questions_per_subject,
        "max_questions_per_point": state.max_questions_per_point,
        "max_pure_foundation_questions": state.max_pure_foundation_questions,
        "candidate_question_wait_seconds": state.candidate_question_wait_seconds,
        "subject_question_counts": dict(state.subject_question_counts),
        "pure_foundation_question_count": state.pure_foundation_question_count,
        "intro_interest_updated": state.intro_interest_updated,
        "candidate_question_started_at": state.candidate_question_started_at,
        "recommendation_traces": [
            recommendation_trace.to_dict()
            for recommendation_trace in state.recommendation_traces
        ],
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "previous_turns": [turn.to_dict() for turn in state.turns],
        "pending_question": state.pending_question.to_dict() if state.pending_question else None,
        "last_decision": state.last_decision.to_dict() if state.last_decision else None,
    }


def _interviewer_event_id(interview_id: str, question_id: str) -> str:
    safe_interview_id = re.sub(r"[^A-Za-z0-9_-]+", "-", interview_id)[:80]
    safe_question_id = re.sub(r"[^A-Za-z0-9_-]+", "-", question_id)[:80]
    return f"interview-agent-{safe_interview_id}-{safe_question_id}"


def _build_interviewer_question_event(
    *,
    interview_id: str,
    question_payload: dict | None,
    elapsed_ms: int,
    source: str,
    status: str = "completed",
    error: str = "",
    created_at: float | None = None,
    session_input: dict[str, object] | None = None,
    decision_payload: dict[str, object] | None = None,
    orchestration_audit: dict[str, object] | None = None,
    persisted: bool = False,
) -> dict[str, object]:
    event_created_at = time.time() if created_at is None else float(created_at)
    payload = dict(question_payload or {})
    question_id = str(payload.get("question_id") or f"question-{uuid.uuid4().hex[:12]}")
    knowledge = [dict(item) for item in payload.get("knowledge_sources", []) if isinstance(item, dict)][:3]
    audit = dict(orchestration_audit or {})
    decision = dict(
        decision_payload
        or audit.get("final_decision")
        or (_interview_decision_snapshot(interview_id, payload) if payload else {})
    )
    full_input = {
        "source": source,
        "interview_id": interview_id,
        "session": session_input or {"interview_id": interview_id},
        "decision": decision,
        "knowledge_refs": list(payload.get("knowledge_refs") or []),
        "jd_requirement_refs": list(payload.get("jd_requirement_refs") or []),
        "resume_claim_refs": list(payload.get("resume_claim_refs") or []),
        "knowledge_chunks": knowledge,
        "knowledge_load_errors": list(payload.get("knowledge_load_errors") or []),
        "orchestration": {
            "mode": audit.get("mode"),
            "timings_ms": dict(audit.get("timings_ms") or {}),
            "errors": dict(audit.get("errors") or {}),
        } if audit else None,
    }
    full_output = {
        "question": payload if payload else None,
        "error": error,
    }
    round_audit = payload.get("round_audit") if isinstance(payload.get("round_audit"), dict) else None
    tool_trace = [
        {
            "tool": "get_interview_session",
            "ok": True,
            "input": {"interview_id": interview_id},
            "output": session_input or {"interview_id": interview_id},
            "output_summary": f"interview_id={interview_id}",
        },
        {
            "tool": "retrieve_interview_knowledge",
            "ok": True,
            "input": {
                "corpus_ids": (session_input or {}).get("knowledge_corpus_ids", []),
                "knowledge_refs": list(payload.get("knowledge_refs") or []),
            },
            "output": {
                "knowledge_chunks": knowledge,
                "knowledge_load_errors": list(payload.get("knowledge_load_errors") or []),
            },
            "output_summary": f"sent_chunks={len(knowledge)}; ids={', '.join(str(item.get('chunk_id')) for item in knowledge)}",
            "knowledge_chunks": knowledge,
        },
        {
            "tool": "choose_depth_or_breadth",
            "ok": status == "completed",
            "input": {
                "session": session_input or {"interview_id": interview_id},
                "question_context": {
                    "parent_question_id": payload.get("parent_question_id"),
                    "followup_depth": payload.get("followup_depth") or 0,
                    "topic": payload.get("topic") or "",
                    "path": payload.get("path") or "",
                },
            },
            "output": decision,
            "output_summary": f"path={decision.get('path', '')}; action={decision.get('action', '')}; reason={decision.get('reason', '')}",
            "decision": decision,
        },
        {
            "tool": "question_generator",
            "ok": status == "completed",
            "input": full_input,
            "output": full_output,
            "output_summary": _interview_preview(payload.get("text") or error),
        },
    ]
    for lane_name, lane in dict(audit.get("lanes") or {}).items():
        if not isinstance(lane, dict):
            continue
        tool_trace.append({
            "tool": f"parallel_{lane_name}",
            "ok": not bool(lane.get("error")),
            "input": lane.get("input"),
            "output": lane.get("output"),
            "output_summary": _interview_preview(str(lane.get("error") or lane.get("output") or "")),
            "elapsed_ms": lane.get("elapsed_ms") or 0,
            "error": lane.get("error"),
        })
    return {
        "id": _interviewer_event_id(interview_id, question_id),
        "kind": "interview_question",
        "agent": "Interviewer Agent",
        "status": status,
        "session_id": interview_id,
        "created_at": event_created_at,
        "elapsed_ms": elapsed_ms,
        "tool_call_count": len(tool_trace),
        "token_count": 0,
        "error": error,
        "input_summary": f"source={source}; interview_id={interview_id}",
        "output_summary": _interview_preview(payload.get("text") or error),
        "input": full_input,
        "output": full_output,
        "trace": tool_trace,
        "question": {
            "question_id": question_id,
            "text": payload.get("text") or "",
            "text_summary": _interview_preview(payload.get("text")),
            "topic": payload.get("topic") or "",
            "question_type": payload.get("question_type") or "",
            "path": payload.get("path") or "",
            "followup_depth": payload.get("followup_depth") or 0,
        },
        "decision": decision,
        "knowledge_chunks": knowledge,
        "latency_breakdown": {
            "total_ms": elapsed_ms,
            "question_generation_ms": elapsed_ms,
            "source": source,
            "parallel_lanes_ms": dict(audit.get("timings_ms") or {}),
        },
        "orchestration_audit": audit or None,
        "round_audit": round_audit,
        "question_answer": {
            "question_id": question_id,
            "question_text": payload.get("text") or "",
            "user_answer": payload.get("candidate_answer"),
            "knowledge_refs": list(payload.get("knowledge_refs") or []),
            "evaluation_refs": list((payload.get("evaluation") or {}).get("refs") or []),
            "knowledge_chunks": knowledge,
            "selection": decision,
            "retrieval": dict(payload.get("retrieval_trace") or {}),
            "evaluation": payload.get("evaluation"),
        },
        "persisted": persisted,
        "llm_audit": "不展示隐藏思维链；这里展示可审计的策略摘要、检索证据、工具调用和模型最终输出。",
    }


def _record_interviewer_question_event(
    *,
    interview_id: str,
    question_payload: dict | None,
    elapsed_ms: int,
    source: str,
    status: str = "completed",
    error: str = "",
) -> None:
    session_input = None
    state = None
    try:
        from smart_agent.career import InterviewSessionStore

        state = InterviewSessionStore(_career_root() / "interviews").get(interview_id)
        session_input = _interview_session_admin_input(state)
    except Exception:
        session_input = {"interview_id": interview_id}
    if state is not None and question_payload and state.pending_question is not None:
        question_payload = dict(question_payload)
        round_audit = _interview_round_audit(state, question=state.pending_question)
        question_payload["round_audit"] = round_audit
        question_payload["retrieval_trace"] = round_audit["retrieval"]["trace"]
    event = _build_interviewer_question_event(
        interview_id=interview_id,
        question_payload=question_payload,
        elapsed_ms=elapsed_ms,
        source=source,
        status=status,
        error=error,
        session_input=session_input,
    )
    _INTERVIEW_AGENT_EVENTS.append(event)
    agent_event_hub.publish(
        "interview_question_finished" if status == "completed" else "interview_question_failed",
        id=event["id"],
        kind=event["kind"],
        agent=event["agent"],
        status=status,
        session_id=interview_id,
        elapsed_ms=elapsed_ms,
        tool_call_count=event["tool_call_count"],
        input_summary=event["input_summary"],
        output_summary=event["output_summary"],
        message=event["llm_audit"],
    )


def _persisted_interviewer_question_events() -> list[dict[str, object]]:
    from smart_agent.career import InterviewSessionStore

    store = InterviewSessionStore(_career_root() / "interviews")
    # Rebuild audit events from durable session/corpus data. Opening the live
    # Graph RAG index here can fail while another worker holds its SQLite file.
    runtime = None
    events: list[dict[str, object]] = []
    for state in store.list():
        session_input = _interview_session_admin_input(state)
        for turn in state.turns:
            payload = _interview_question_payload_from_state(
                state,
                turn.question,
                runtime=runtime,
                use_graph_runtime=False,
            )
            output_payload = dict(payload)
            output_payload["candidate_answer"] = turn.answer
            output_payload["evaluation"] = {
                "correctness": turn.correctness,
                "completeness": turn.completeness,
                "confidence": turn.confidence,
                "reason": turn.evaluation_reason,
                "refs": list(turn.evaluation_refs),
                "observation": turn.observation.to_dict() if turn.observation else None,
            }
            round_audit = _interview_round_audit(
                state,
                question=turn.question,
                turn=turn,
                runtime=runtime,
                use_graph_runtime=False,
            )
            output_payload["round_audit"] = round_audit
            output_payload["retrieval_trace"] = round_audit["retrieval"]["trace"]
            event = _build_interviewer_question_event(
                interview_id=state.interview_id,
                question_payload=output_payload,
                elapsed_ms=round(float((turn.orchestration_audit or {}).get("timings_ms", {}).get("total", 0))),
                source="persisted_turn",
                created_at=turn.created_at,
                session_input=session_input,
                decision_payload=(turn.orchestration_audit or {}).get("final_decision") or {
                    "path": payload.get("path") or "",
                    "action": "follow_up" if payload.get("parent_question_id") else "ask",
                    "topic": payload.get("topic") or "",
                    "question_type": payload.get("question_type") or "",
                    "reason": "从已持久化的面试会话回放；历史文件未保存隐藏思维链。",
                    "followup_depth": int(payload.get("followup_depth") or 0),
                    "depth_weight": (payload.get("utility") or {}).get("depth_weight") if isinstance(payload.get("utility"), dict) else None,
                    "breadth_weight": (payload.get("utility") or {}).get("breadth_weight") if isinstance(payload.get("utility"), dict) else None,
                    "utility": payload.get("utility") if isinstance(payload.get("utility"), dict) else {},
                },
                orchestration_audit=turn.orchestration_audit,
                persisted=True,
            )
            event["output"]["turn_result"] = output_payload.get("evaluation")
            events.append(event)
        if state.pending_question is not None:
            payload = _interview_question_payload_from_state(
                state,
                state.pending_question,
                runtime=runtime,
                use_graph_runtime=False,
            )
            round_audit = _interview_round_audit(
                state,
                question=state.pending_question,
                runtime=runtime,
                use_graph_runtime=False,
            )
            payload["round_audit"] = round_audit
            payload["retrieval_trace"] = round_audit["retrieval"]["trace"]
            event = _build_interviewer_question_event(
                interview_id=state.interview_id,
                question_payload=payload,
                elapsed_ms=0,
                source="persisted_pending_question",
                created_at=state.updated_at,
                session_input=session_input,
                decision_payload=state.last_decision.to_dict() if state.last_decision else None,
                persisted=True,
            )
            events.append(event)
    return events


def _execute_next_interview_question_with_trace(interview_id: str, *, source: str) -> dict:
    started = time.perf_counter()
    try:
        question = _execute_career_tool(
            "next_interview_question",
            {"interview_id": interview_id},
            stage="interview",
            enable_model_runtime=True,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        _record_interviewer_question_event(
            interview_id=interview_id,
            question_payload=question,
            elapsed_ms=elapsed_ms,
            source=source,
        )
        return question
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        _record_interviewer_question_event(
            interview_id=interview_id,
            question_payload=None,
            elapsed_ms=elapsed_ms,
            source=source,
            status="failed",
            error=str(exc),
        )
        raise


def _direct_next_interview_question_with_trace(runtime, interview_id: str, *, source: str):
    started = time.perf_counter()
    try:
        question = runtime.next_question(interview_id)
        payload = _interview_question_payload_direct(interview_id, question)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        _record_interviewer_question_event(
            interview_id=interview_id,
            question_payload=payload,
            elapsed_ms=elapsed_ms,
            source=source,
        )
        return question
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        _record_interviewer_question_event(
            interview_id=interview_id,
            question_payload=None,
            elapsed_ms=elapsed_ms,
            source=source,
            status="failed",
            error=str(exc),
        )
        raise


def _voice_adapters():
    from smart_agent.career import (
        BufferedSpeechToTextAdapter,
        BufferedTextToSpeechAdapter,
        CosyVoiceTextToSpeechAdapter,
        GPTSoVITSTextToSpeechAdapter,
        HttpSpeechToTextAdapter,
        HttpTextToSpeechAdapter,
        VolcengineTextToSpeechAdapter,
    )

    asr_base_url = os.environ.get("SMART_AGENT_ASR_BASE_URL", "").strip()
    asr_api_key = os.environ.get("SMART_AGENT_ASR_API_KEY", "").strip()
    asr_model = os.environ.get("SMART_AGENT_ASR_MODEL", "").strip()
    tts_base_url = os.environ.get("SMART_AGENT_TTS_BASE_URL", "").strip()
    tts_api_key = os.environ.get("SMART_AGENT_TTS_API_KEY", "").strip()
    tts_model = os.environ.get("SMART_AGENT_TTS_MODEL", "").strip()
    tts_provider = os.environ.get("SMART_AGENT_TTS_PROVIDER", "").strip().lower()
    tts_voice = os.environ.get("SMART_AGENT_TTS_VOICE", "").strip()
    tts_style = os.environ.get("SMART_AGENT_TTS_STYLE", "").strip()
    tts_timeout = float(os.environ.get("SMART_AGENT_TTS_TIMEOUT_SECONDS", "30"))
    tts_max_retries = int(os.environ.get("SMART_AGENT_TTS_MAX_RETRIES", "1"))
    tts_retry_backoff = float(os.environ.get("SMART_AGENT_TTS_RETRY_BACKOFF_SECONDS", "0.1"))
    asr = (
        HttpSpeechToTextAdapter(base_url=asr_base_url, api_key=asr_api_key, model=asr_model)
        if asr_base_url and asr_api_key and asr_model
        else BufferedSpeechToTextAdapter()
    )
    if tts_provider in {"official", "volcengine", "volcengine-tts", "doubao", "bytedance"}:
        tts = VolcengineTextToSpeechAdapter(
            app_id=os.environ.get("SMART_AGENT_VOLCENGINE_TTS_APP_ID", "").strip(),
            access_token=tts_api_key or os.environ.get("SMART_AGENT_VOLCENGINE_TTS_ACCESS_TOKEN", "").strip(),
            voice_type=tts_voice or os.environ.get("SMART_AGENT_VOLCENGINE_TTS_VOICE_TYPE", "").strip(),
            base_url=tts_base_url or "https://openspeech.bytedance.com/api/v1/tts",
            encoding=os.environ.get("SMART_AGENT_VOLCENGINE_TTS_ENCODING", "mp3").strip(),
            speed_ratio=float(os.environ.get("SMART_AGENT_VOLCENGINE_TTS_SPEED_RATIO", "1.0")),
            timeout_seconds=tts_timeout,
            max_retries=tts_max_retries,
            retry_backoff_seconds=tts_retry_backoff,
        )
    elif tts_provider in {"cosyvoice", "cosyvoice3", "local-cosyvoice", "local_cosyvoice"}:
        tts = CosyVoiceTextToSpeechAdapter(
            base_url=tts_base_url or "http://127.0.0.1:9881",
            api_key=tts_api_key,
            model=tts_model or "cosyvoice3-0.5b",
            voice=tts_voice,
            style=tts_style,
            timeout_seconds=tts_timeout,
            max_retries=tts_max_retries,
            retry_backoff_seconds=tts_retry_backoff,
        )
    elif tts_provider in {"gpt-sovits", "gpt_sovits"}:
        tts = GPTSoVITSTextToSpeechAdapter(
            base_url=tts_base_url,
            ref_audio_path=os.environ.get("SMART_AGENT_GPT_SOVITS_REF_AUDIO_PATH", "").strip(),
            prompt_text=os.environ.get("SMART_AGENT_GPT_SOVITS_PROMPT_TEXT", "").strip(),
            text_lang=os.environ.get("SMART_AGENT_GPT_SOVITS_TEXT_LANG", "zh").strip(),
            prompt_lang=os.environ.get("SMART_AGENT_GPT_SOVITS_PROMPT_LANG", "zh").strip(),
            speed_factor=float(os.environ.get("SMART_AGENT_GPT_SOVITS_SPEED_FACTOR", "1.0")),
            timeout_seconds=tts_timeout,
            max_retries=tts_max_retries,
            retry_backoff_seconds=tts_retry_backoff,
        )
    elif tts_base_url and tts_api_key and tts_model:
        tts = HttpTextToSpeechAdapter(
            base_url=tts_base_url,
            api_key=tts_api_key,
            model=tts_model,
            voice=tts_voice,
            style=tts_style,
            timeout_seconds=tts_timeout,
            max_retries=tts_max_retries,
            retry_backoff_seconds=tts_retry_backoff,
        )
    else:
        tts = BufferedTextToSpeechAdapter()
    return asr, tts


def _interview_answer_result_payload_direct(runtime, result) -> dict:
    payload = result.to_dict()
    state_payload = result.state.to_dict()
    for turn in state_payload.get("turns", []):
        if isinstance(turn, dict):
            turn.pop("orchestration_audit", None)
    payload["state"] = state_payload
    if result.state.status.value == "completed":
        payload["report"] = runtime.report(result.state.interview_id)
    return payload


def _project_library_stores():
    from smart_agent.career import ProjectAnalysisStore, ProjectLibraryStore

    root = _career_root()
    project_root = root / "project-library"
    return (
        ProjectLibraryStore(project_root, workspace=ROOT),
        ProjectAnalysisStore(project_root / "analyses"),
    )


def _project_experience_service():
    from smart_agent.career import (
        CareerStore,
        ProjectAnalysisRunStore,
        ProjectExperienceService,
    )

    project_store, analysis_store = _project_library_stores()
    root = _career_root()
    return ProjectExperienceService(
        career_store=CareerStore(root),
        project_store=project_store,
        analysis_store=analysis_store,
        run_store=ProjectAnalysisRunStore(root / "project-library" / "runs"),
    )


def _project_analysis_task_status(project_id: str) -> str:
    """Return the in-process status of an automatically scheduled analysis."""

    selected = str(project_id or "").strip()
    if selected in _PROJECT_ANALYSIS_QUEUED:
        return "queued"
    task = _PROJECT_ANALYSIS_TASKS.get(selected)
    if task is None:
        return ""
    if task.done():
        _PROJECT_ANALYSIS_TASKS.pop(selected, None)
        return ""
    return "running"


async def _run_project_analysis_background(project_id: str) -> None:
    """Run automatic project analysis without keeping the upload request open."""

    try:
        await career_project_library_analyze(
            project_id,
            {"mode": "agent", "intent": "overview", "refresh": True},
        )
    except asyncio.CancelledError:
        raise
    except Exception:
        # career_project_library_analyze records the failure with its exact
        # error. The wrapper consumes it so one failed model request cannot
        # produce an unhandled-task warning or affect other uploads.
        return


async def _handle_project_task(payload: dict) -> None:
    if str(payload.get("type") or "") != "project_analysis":
        return
    selected = str(payload.get("project_id") or "").strip()
    tenant_id = str(payload.get("tenant_id") or "").strip()
    if not selected:
        return
    _PROJECT_ANALYSIS_QUEUED.discard(selected)
    existing = _PROJECT_ANALYSIS_TASKS.get(selected)
    if existing is not None and not existing.done():
        return
    tenant_token = _CURRENT_TENANT_ID.set(tenant_id)
    try:
        task = asyncio.current_task()
        if task is not None:
            _PROJECT_ANALYSIS_TASKS[selected] = task
        await _run_project_analysis_background(selected)
    finally:
        if _PROJECT_ANALYSIS_TASKS.get(selected) is asyncio.current_task():
            _PROJECT_ANALYSIS_TASKS.pop(selected, None)
        _CURRENT_TENANT_ID.reset(tenant_token)


async def _schedule_project_analysis(project_id: str) -> bool:
    """Persist one analysis task per project and report whether it was queued."""

    selected = str(project_id or "").strip()
    if not selected:
        return False
    existing = _PROJECT_ANALYSIS_TASKS.get(selected)
    if existing is not None and not existing.done():
        return False
    if selected in _PROJECT_ANALYSIS_QUEUED:
        return False
    await _PROJECT_TASK_QUEUE.enqueue({
        "type": "project_analysis",
        "project_id": selected,
        "tenant_id": _career_tenant_id(),
    })
    _PROJECT_ANALYSIS_QUEUED.add(selected)
    return True


@app.on_event("startup")
async def _start_project_task_worker() -> None:
    global _PROJECT_TASK_STOP, _PROJECT_TASK_WORKER
    _hydrate_project_integrations()
    if _PROJECT_TASK_WORKER is not None and not _PROJECT_TASK_WORKER.done():
        return
    _PROJECT_TASK_STOP = asyncio.Event()
    _PROJECT_TASK_WORKER = asyncio.create_task(
        _PROJECT_TASK_QUEUE.worker(_handle_project_task, _PROJECT_TASK_STOP),
        name="smart-agent-project-worker",
    )


@app.on_event("shutdown")
async def _stop_project_task_worker() -> None:
    global _PROJECT_TASK_STOP, _PROJECT_TASK_WORKER
    if _PROJECT_TASK_STOP is not None:
        _PROJECT_TASK_STOP.set()
    if _PROJECT_TASK_WORKER is not None:
        try:
            await asyncio.wait_for(_PROJECT_TASK_WORKER, timeout=3)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            _PROJECT_TASK_WORKER.cancel()
    _PROJECT_TASK_STOP = None
    _PROJECT_TASK_WORKER = None


def _project_deep_dive_service():
    from smart_agent.career import (
        LLMProjectCandidateSelector,
        ProjectDeepDiveService,
        ProjectDeepDiveStore,
        ProjectSearchGate,
        create_career_client,
    )

    runtime = load_runtime_config(ROOT)
    project_store, _analysis_store = _project_library_stores()
    client = create_career_client(
        runtime,
        "project_matcher",
        audit_root=_career_root() / "audit",
    )
    return ProjectDeepDiveService(
        project_store=project_store,
        decision_store=ProjectDeepDiveStore(_career_root() / "project-library" / "deep-dives"),
        gate=ProjectSearchGate(
            selector=LLMProjectCandidateSelector(client) if client is not None else None,
            high_threshold=float(runtime.get("careerProjectSearchHighThreshold", 0.65)),
            medium_threshold=float(runtime.get("careerProjectSearchMediumThreshold", 0.40)),
            sufficient_evidence_score=float(runtime.get("careerProjectSufficientEvidenceScore", 0.72)),
            max_candidate_projects=int(runtime.get("careerProjectSearchMaxCandidates", 3)),
        ),
    )


def _career_material_snapshot():
    from smart_agent.career import material_catalog_version

    _root, career_store, material_store, candidate_store = _career_stores()
    manifests = material_store.list_manifests()
    candidates = candidate_store.list()
    from smart_agent.career import PersonalEvidence, ProjectDescriptionDocument

    evidence = career_store.list_latest(PersonalEvidence)
    descriptions = career_store.list_latest(ProjectDescriptionDocument)
    return (
        career_store,
        material_store,
        candidate_store,
        manifests,
        candidates,
        evidence,
        descriptions,
        material_catalog_version(manifests, evidence, candidates, descriptions),
    )


def _is_project_library_evidence(record) -> bool:
    """Identify evidence created from a registered project description."""
    return any(
        str(ref.source_id or "").startswith("project:")
        or bool((ref.locator or {}).get("project_id"))
        for ref in getattr(record, "source_refs", ())
    )


def _project_evidence_ids(record) -> set[str]:
    return {
        str((ref.locator or {}).get("project_id") or "").strip()
        for ref in getattr(record, "source_refs", ())
        if str((ref.locator or {}).get("project_id") or "").strip()
    }


def _refresh_resume_project_snapshot(root: Path, store, run):
    """Synchronize registered projects into the resume run before generation."""
    from smart_agent.career import (
        PersonalEvidence,
        ProjectDescriptionDocument,
        ProjectDescriptionService,
        ProjectLibraryStore,
        ProjectSourceStatus,
        ResumeRunService,
        material_catalog_version,
        version_ref,
    )

    project_store = ProjectLibraryStore(root / "project-library", workspace=ROOT)
    description_service = ProjectDescriptionService(store)
    for source in project_store.list():
        refreshed = project_store.get(source.project_id, refresh=True)
        if refreshed.status is ProjectSourceStatus.NEEDS_REANALYSIS:
            description_service.mark_source_changed(
                refreshed.project_id,
                current_project_version=refreshed.project_version,
            )

    descriptions = store.list_latest(ProjectDescriptionDocument)
    active_sources = {
        source.project_id.casefold()
        for source in project_store.list()
        if source.status not in {ProjectSourceStatus.INVALIDATED, ProjectSourceStatus.UNAVAILABLE}
    }
    confirmed_descriptions = {
        item.project_id: item
        for item in descriptions
        if item.project_id.casefold() in active_sources
        and item.status.value == "user_confirmed"
        and item.evidence_id
    }
    desired_project_ids = {
        item.evidence_id
        for item in confirmed_descriptions.values()
        if item.evidence_id
    }
    confirmed_project_ids = {item.casefold() for item in confirmed_descriptions}

    snapshot_records: list[PersonalEvidence] = []
    seen_ids: set[str] = set()
    for ref in run.evidence_snapshot_refs:
        record = store.get(ref.record_id, ref.revision, expected_type=PersonalEvidence)
        if _is_project_library_evidence(record) and not (
            {item.casefold() for item in _project_evidence_ids(record)} & confirmed_project_ids
        ):
            continue
        snapshot_records.append(record)
        seen_ids.add(record.evidence_id)
    for evidence_id in sorted(desired_project_ids):
        if evidence_id in seen_ids:
            continue
        record = store.get(evidence_id, expected_type=PersonalEvidence)
        if any(fact.is_usable for fact in record.facts):
            snapshot_records.append(record)
            seen_ids.add(record.evidence_id)

    _root, _career_store, material_store, candidate_store = _career_stores()
    manifests = material_store.list_manifests()
    candidates = candidate_store.list()
    all_evidence = store.list_latest(PersonalEvidence)
    catalog = material_catalog_version(
        manifests,
        all_evidence,
        candidates,
        descriptions,
    )
    refreshed_evidence = tuple(snapshot_records)
    project_snapshot_changed = tuple(
        (ref.record_id, ref.revision) for ref in run.evidence_snapshot_refs
    ) != tuple((ref.record_id, ref.revision) for ref in (version_ref(item) for item in refreshed_evidence))
    return ResumeRunService(store).refresh_project_evidence_snapshot(
        run.resume_run_id,
        evidence=refreshed_evidence,
        material_catalog_version=catalog if project_snapshot_changed else run.material_catalog_version,
    )


def _resolve_resume_run(store, resume_run_id: str = ""):
    from smart_agent.career import RecordNotFoundError, ResumeRun, ResumeRunStatus

    identity = str(resume_run_id or "").strip()
    if identity:
        try:
            run = store.get(identity, expected_type=ResumeRun)
        except RecordNotFoundError as exc:
            raise HTTPException(status_code=404, detail="当前岗位会话已失效，请重新提交 JD。") from exc
        if run.status is ResumeRunStatus.DELETED:
            raise HTTPException(status_code=410, detail="The resume session has been deleted.")
        return run
    runs = tuple(
        run
        for run in store.list_latest(ResumeRun)
        if run.status is not ResumeRunStatus.DELETED
    )
    if len(runs) == 1:
        return runs[0]
    if not runs:
        raise HTTPException(status_code=409, detail="Analyze a JD to create a resume_run first.")
    raise HTTPException(status_code=400, detail="resume_run_id is required when multiple resume tasks exist.")


def _validate_resume_run_evidence_snapshot(store, run) -> None:
    """Reject runs whose pinned evidence is stale or contains no usable facts."""
    from smart_agent.career import PersonalEvidence, RecordNotFoundError

    for ref in run.evidence_snapshot_refs:
        try:
            evidence = store.get(
                ref.record_id,
                ref.revision,
                expected_type=PersonalEvidence,
            )
            latest_revision = store.latest_revision(ref.record_id)
        except RecordNotFoundError as exc:
            raise HTTPException(
                status_code=409,
                detail="个人材料已失效，请重新确认材料事实并重新分析岗位。",
            ) from exc
        if latest_revision != ref.revision:
            raise HTTPException(
                status_code=409,
                detail="个人材料已更新，当前岗位分析已过期，请重新分析岗位。",
            )
        if not any(fact.is_usable for fact in evidence.facts):
            raise HTTPException(
                status_code=409,
                detail="当前岗位会话没有已确认的个人事实，请先确认材料事实。",
            )


def _validate_resume_run_matches(store, run) -> None:
    """Ensure cached requirement matches cite only the pinned usable facts."""
    from smart_agent.career import EvidenceMatch, PersonalEvidence, SupportLevel

    snapshot_refs = {
        (ref.record_id, ref.revision): ref for ref in run.evidence_snapshot_refs
    }
    for ref in run.match_refs:
        match = store.get(ref.record_id, ref.revision, expected_type=EvidenceMatch)
        if match.support_level is SupportLevel.UNSUPPORTED or match.evidence_ref is None:
            continue
        evidence_key = (match.evidence_ref.record_id, match.evidence_ref.revision)
        if evidence_key not in snapshot_refs:
            raise HTTPException(
                status_code=409,
                detail="岗位匹配结果已过期，请重新分析岗位。",
            )
        evidence = store.get(
            match.evidence_ref.record_id,
            match.evidence_ref.revision,
            expected_type=PersonalEvidence,
        )
        usable_fact_ids = {fact.fact_id for fact in evidence.facts if fact.is_usable}
        if not set(match.evidence_fact_ids).issubset(usable_fact_ids):
            raise HTTPException(
                status_code=409,
                detail="岗位匹配引用了未确认或已失效事实，请重新确认材料并重新分析岗位。",
            )


def _career_tool_context(
    stage: str = "resume",
    *,
    enable_model_runtime: bool = False,
) -> tuple[ToolRegistry, ToolContext]:
    """Create a request-scoped Career tool context over the persistent store.

    HTTP handlers deliberately use the same ToolRegistry contracts as the agent
    loop. This keeps the UI protocol from growing a second, subtly different
    implementation of resume and interview operations.
    """
    from smart_agent.career import CareerModelConfig, CareerStore, ModelAPIIdentity

    root = _career_root()
    store = CareerStore(root)
    tenant_id = _career_tenant_id() or "legacy-web-user"
    runtime = dict(load_runtime_config(ROOT)) if enable_model_runtime else {}
    runtime.update({
        "activeDomain": "career",
        "domainStage": stage,
        "careerTenantId": tenant_id,
        "careerUserId": _CURRENT_AUTH_USER.get().user_id if _CURRENT_AUTH_USER.get() else "",
        "careerStorePath": str(root),
        "_careerStore": store,
        # ResumeDocumentBuilder needs an explicit identity. When model runtime
        # is enabled, preserve the actual configured Career model/API channel
        # in the generated document; offline contexts retain the test baseline.
        "_careerGenerationIdentity": (
            (
                lambda config: ModelAPIIdentity(config.provider, config.model, config.api_id)
            )(CareerModelConfig.from_runtime(runtime, "resume_generator", required=True))
            if enable_model_runtime and stage == "resume"
            else ModelAPIIdentity("offline", "offline-generator-v1", "api:offline-generation")
        ),
    })
    return create_default_tool_registry(str(ROOT), runtime=runtime), ToolContext(
        cwd=str(ROOT),
        _runtime=runtime,
    )


def _execute_career_tool(
    tool_name: str,
    payload: dict,
    *,
    stage: str = "resume",
    enable_model_runtime: bool = False,
    runtime_overrides: dict[str, object] | None = None,
) -> dict:
    registry, context = _career_tool_context(stage, enable_model_runtime=enable_model_runtime)
    if runtime_overrides:
        context._runtime.update(runtime_overrides)
    # HTTP handlers consume the tool result as a machine-readable contract.
    # Context-window truncation inserts plain text and would corrupt large JSON.
    result = registry.execute(tool_name, payload, context, truncate_output=False)
    if not result.ok:
        raise HTTPException(status_code=409, detail=result.output)
    try:
        return json.loads(result.output)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Career tool returned invalid JSON: {tool_name}") from exc

# ── API ──

_CAREER_AGENT_MODEL_ROLES = (
    ("fact_extractor", "材料事实抽取", "FACT_EXTRACTOR", "FactExtractor"),
    ("jd_analyzer", "岗位 JD 分析", "JD_ANALYZER", "JDAnalyzer"),
    ("project_matcher", "项目匹配", "PROJECT_MATCHER", "ProjectMatcher"),
    ("project_analyst", "项目代码分析", "PROJECT_ANALYST", "ProjectAnalyst"),
    ("resume_generator", "简历生成", "RESUME_GENERATOR", "ResumeGenerator"),
    ("jd_critic", "JD 覆盖审查", "JD_CRITIC", "JDCritic"),
    ("interview_question", "面试出题", "INTERVIEW_QUESTION", "InterviewQuestion"),
    ("interview_evaluator", "面试评估", "INTERVIEW_EVALUATOR", "InterviewEvaluator"),
    ("verifier", "简历独立核验", "VERIFIER", "Verifier"),
)
_CAREER_AGENT_MODEL_ROLE_MAP = {item[0]: item for item in _CAREER_AGENT_MODEL_ROLES}
_SUPPORTED_AGENT_MODEL_PROVIDERS = frozenset({"anthropic", "openai", "openrouter", "custom"})


def _main_agent_model_config(runtime: dict[str, object]) -> dict[str, object]:
    """Return the effective main-agent connection without exposing its key."""
    from smart_agent.model_registry import build_provider_config

    active = build_provider_config(str(runtime.get("model") or ""), runtime)
    return {
        "label": "主 Agent",
        "provider": active.provider.value,
        "model": active.model,
        "base_url": active.base_url.rstrip("/"),
        "api_key_configured": bool(active.api_key),
    }


def _career_agent_model_config(runtime: dict[str, object]) -> dict[str, object]:
    """Return effective Career role configuration without exposing credentials."""
    from smart_agent.career import CareerModelConfig

    roles: list[dict[str, object]] = []
    for role, label, _env_prefix, setting_prefix in _CAREER_AGENT_MODEL_ROLES:
        if role == "verifier":
            provider = str(runtime.get("careerVerifierProvider") or "custom").strip().lower()
            model = str(runtime.get("careerVerifierModel") or "").strip()
            base_url = str(runtime.get("careerVerifierBaseUrl") or "").strip().rstrip("/")
            key_configured = bool(str(runtime.get("careerVerifierApiKey") or "").strip())
        else:
            provider = str(runtime.get(f"career{setting_prefix}Provider") or "custom").strip().lower()
            model = str(runtime.get(f"career{setting_prefix}Model") or "").strip()
            base_url = str(runtime.get(f"career{setting_prefix}BaseUrl") or "").strip().rstrip("/")
            key_configured = bool(str(runtime.get(f"career{setting_prefix}ApiKey") or "").strip())
            try:
                resolved = CareerModelConfig.from_runtime(runtime, role, required=False)
            except ValueError:
                resolved = None
            if resolved is not None:
                provider = resolved.provider
                model = resolved.model
                base_url = resolved.base_url
                key_configured = bool(resolved.api_key)
            elif role == "project_analyst":
                # The existing Project Analyst uses the general sub-agent
                # channel until the user saves a dedicated Career override.
                from smart_agent.model_registry import build_provider_config

                active = build_provider_config(str(runtime.get("model") or ""), runtime)
                provider = active.provider.value
                model = active.model
                base_url = active.base_url
                key_configured = bool(active.api_key)
        roles.append(
            {
                "role": role,
                "label": label,
                "provider": provider if provider in _SUPPORTED_AGENT_MODEL_PROVIDERS else "custom",
                "model": model,
                "base_url": base_url,
                "api_key_configured": key_configured,
            }
        )
    return {"main_agent": _main_agent_model_config(runtime), "roles": roles}


def _validate_main_agent_connection(
    data: dict[str, object],
    runtime: dict[str, object],
    *,
    require_api_key: bool = False,
) -> dict[str, str]:
    base_url = str(data.get("base_url") or "").strip().rstrip("/")
    api_key = str(data.get("api_key") or "").strip()
    parsed_url = urllib.parse.urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("main_agent.base_url must be an http(s) URL")
    current = _main_agent_model_config(runtime)
    if not api_key and (require_api_key or not current["api_key_configured"]):
        raise ValueError("main_agent.api_key is required")
    provider = str(current["provider"])
    fields = {
        # Persist the same environment keys that load_runtime_config reads
        # with prefer_settings_env=True. Writing top-level baseUrl/apiKey is
        # insufficient because Claude's global env would override them.
        "custom": ("CUSTOM_API_BASE_URL", "CUSTOM_API_KEY"),
        "openai": ("OPENAI_BASE_URL", "OPENAI_API_KEY"),
        "openrouter": ("OPENROUTER_BASE_URL", "OPENROUTER_API_KEY"),
        "anthropic": ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY"),
    }
    base_field, key_field = fields[provider]
    updates = {base_field: base_url}
    if api_key:
        updates[key_field] = api_key
        if provider == "anthropic":
            # Do not let an old Claude auth token win over the newly entered
            # API key on the next runtime load.
            updates["ANTHROPIC_AUTH_TOKEN"] = ""
    return updates


def _validate_career_agent_model_updates(
    data: dict[str, object],
    runtime: dict[str, object],
) -> tuple[dict[str, str], list[str]]:
    raw_roles = data.get("roles")
    if not isinstance(raw_roles, list) or not raw_roles:
        raise ValueError("roles must be a non-empty array")
    current_roles = {
        str(item["role"]): item
        for item in _career_agent_model_config(runtime)["roles"]
        if isinstance(item, dict)
    }
    updates: dict[str, str] = {}
    saved_roles: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_roles):
        if not isinstance(item, dict):
            raise ValueError(f"roles[{index}] must be an object")
        role = str(item.get("role") or "").strip()
        descriptor = _CAREER_AGENT_MODEL_ROLE_MAP.get(role)
        if descriptor is None:
            raise ValueError(f"Unsupported Career agent role: {role or index}")
        if role in seen:
            raise ValueError(f"Duplicate Career agent role: {role}")
        seen.add(role)
        provider = str(item.get("provider") or "").strip().lower()
        model = str(item.get("model") or "").strip()
        base_url = str(item.get("base_url") or "").strip().rstrip("/")
        api_key = str(item.get("api_key") or "").strip()
        if provider not in _SUPPORTED_AGENT_MODEL_PROVIDERS:
            raise ValueError(f"roles[{index}].provider must be anthropic, openai, openrouter, or custom")
        if not model:
            raise ValueError(f"roles[{index}].model is required")
        parsed_url = urllib.parse.urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(f"roles[{index}].base_url must be an http(s) URL")
        existing = current_roles[role]
        if not api_key and (
            not bool(existing.get("api_key_configured"))
            or provider != str(existing.get("provider") or "")
        ):
            raise ValueError(f"roles[{index}].api_key is required for this provider")
        _role, _label, env_prefix, _setting_prefix = descriptor
        env_root = f"CAREER_{env_prefix}_"
        updates.update(
            {
                f"{env_root}PROVIDER": provider,
                f"{env_root}MODEL": model,
                f"{env_root}BASE_URL": base_url,
            }
        )
        if api_key:
            updates[f"{env_root}API_KEY"] = api_key
        saved_roles.append(role)
    return updates, saved_roles


def _validate_shared_career_agent_connection(
    data: dict[str, object],
    runtime: dict[str, object],
) -> tuple[dict[str, str], list[str]]:
    base_url = str(data.get("base_url") or "").strip().rstrip("/")
    api_key = str(data.get("api_key") or "").strip()
    parsed_url = urllib.parse.urlparse(base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("base_url must be an http(s) URL")
    if not api_key:
        raise ValueError("api_key is required")

    effective_roles = {
        str(item["role"]): item
        for item in _career_agent_model_config(runtime)["roles"]
        if isinstance(item, dict)
    }
    updates: dict[str, str] = {}
    roles: list[str] = []
    for role, _label, env_prefix, setting_prefix in _CAREER_AGENT_MODEL_ROLES:
        env_root = f"CAREER_{env_prefix}_"
        updates[f"{env_root}BASE_URL"] = base_url
        updates[f"{env_root}API_KEY"] = api_key
        # Project Analyst ordinarily inherits the general sub-agent channel.
        # Once its connection is made role-specific, preserve that effective
        # provider/model so the dedicated config stays complete.
        if role == "project_analyst" and not str(
            runtime.get(f"career{setting_prefix}Model") or ""
        ).strip():
            effective = effective_roles[role]
            model = str(effective.get("model") or "").strip()
            provider = str(effective.get("provider") or "").strip().lower()
            if model:
                updates[f"{env_root}MODEL"] = model
            if provider in _SUPPORTED_AGENT_MODEL_PROVIDERS:
                updates[f"{env_root}PROVIDER"] = provider
        roles.append(role)
    return updates, roles


@app.get("/api/career/agent-model-config")
async def career_agent_model_config_get():
    tenant_id = _career_tenant_id()
    if not tenant_id:
        raise HTTPException(status_code=401, detail="请先登录")
    return _career_agent_model_config(_tenant_runtime(tenant_id))


@app.post("/api/career/agent-model-config")
async def career_agent_model_config_save(data: dict):
    tenant_id = _career_tenant_id()
    if not tenant_id:
        raise HTTPException(status_code=401, detail="请先登录")
    runtime = _tenant_runtime(tenant_id)
    try:
        _env_updates, saved_roles = _validate_career_agent_model_updates(data, runtime)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store = _tenant_model_config_store()
    for item in data["roles"]:
        if str(item.get("role") or "") not in saved_roles:
            continue
        role = str(item["role"])
        _role, _label, _env_prefix, setting_prefix = _CAREER_AGENT_MODEL_ROLE_MAP[role]
        fallback_key = str(runtime.get(f"career{setting_prefix}ApiKey") or "")
        store.upsert(
            tenant_id,
            role,
            provider=str(item.get("provider") or "custom"),
            model=str(item.get("model") or ""),
            base_url=str(item.get("base_url") or ""),
            api_key=str(item.get("api_key") or fallback_key) or None,
        )
    return {**_career_agent_model_config(_tenant_runtime(tenant_id)), "saved_roles": saved_roles}


@app.post("/api/career/agent-model-config/main-agent")
async def main_agent_model_config_save(data: dict):
    tenant_id = _career_tenant_id()
    if not tenant_id:
        raise HTTPException(status_code=401, detail="请先登录")
    runtime = _tenant_runtime(tenant_id)
    try:
        updates = _validate_main_agent_connection(data, runtime)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current = _main_agent_model_config(runtime)
    provider = str(current["provider"])
    fallback_key = {
        "anthropic": runtime.get("apiKey"),
        "openai": runtime.get("openaiApiKey"),
        "openrouter": runtime.get("openrouterApiKey"),
        "custom": runtime.get("customApiKey"),
    }.get(provider)
    _tenant_model_config_store().upsert(
        tenant_id,
        "main_agent",
        provider=provider,
        model=str(runtime.get("model") or current.get("model") or ""),
        base_url=str(data.get("base_url") or ""),
        api_key=str(data.get("api_key") or fallback_key or "") or None,
    )
    return _career_agent_model_config(_tenant_runtime(tenant_id))


@app.post("/api/career/agent-model-config/shared-connection")
async def career_agent_model_config_shared_connection(data: dict):
    tenant_id = _career_tenant_id()
    if not tenant_id:
        raise HTTPException(status_code=401, detail="请先登录")
    runtime = _tenant_runtime(tenant_id)
    try:
        _env_updates, saved_roles = _validate_shared_career_agent_connection(data, runtime)
        _main_updates = _validate_main_agent_connection(data, runtime, require_api_key=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store = _tenant_model_config_store()
    current_main = _main_agent_model_config(runtime)
    store.upsert(
        tenant_id,
        "main_agent",
        provider=str(current_main["provider"]),
        model=str(runtime.get("model") or current_main.get("model") or ""),
        base_url=str(data.get("base_url") or ""),
        api_key=str(data.get("api_key") or ""),
    )
    effective_roles = {str(item["role"]): item for item in _career_agent_model_config(runtime)["roles"]}
    for role in saved_roles:
        item = effective_roles[role]
        store.upsert(
            tenant_id,
            role,
            provider=str(item.get("provider") or "custom"),
            model=str(item.get("model") or ""),
            base_url=str(data.get("base_url") or ""),
            api_key=str(data.get("api_key") or ""),
        )
    return {**_career_agent_model_config(_tenant_runtime(tenant_id)), "saved_roles": saved_roles}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/metrics")
async def metrics():
    return Response(content=metrics_registry.render_prometheus(), media_type="text/plain; version=0.0.4")


@app.get("/api/config")
async def get_config():
    """返回当前模型配置和审查模式。"""
    try:
        runtime = load_runtime_config()
        return {
            "model": runtime.get("model", "unknown"),
            "api_version": CAREER_API_VERSION,
            "api_capabilities": list(CAREER_API_CAPABILITIES),
            "review_mode": os.environ.get("SMART_AGENT_REVIEW_MODE", "off"),
            "cost_limit": os.environ.get("SMART_AGENT_API_COST_LIMIT", "unlimited"),
            "tts_provider": os.environ.get("SMART_AGENT_TTS_PROVIDER", "browser") or "browser",
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/career/voice/synthesize")
async def career_voice_synthesize(data: dict):
    text = str(data.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    _asr, tts = _voice_adapters()
    if tts.name == "buffered-tts":
        raise HTTPException(status_code=503, detail="Real TTS is not configured")
    try:
        playback = tts.synthesize(text)
    except (OSError, TimeoutError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"tts": tts.name, "playback": playback.to_dict()}


@app.get("/api/career/workspace")
async def career_workspace(goal: str = "resume"):
    """Return a compact Career artifact index for the UI."""
    from smart_agent.career import (
        CandidateFactStore,
        CareerGoal,
        CareerStore,
        CareerWorkflowEngine,
        CareerWorkspaceIndexer,
        InterviewKnowledgeLibrary,
        InterviewKnowledgeStore,
        MaterialStore,
    )

    try:
        selected_goal = CareerGoal(goal)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="goal must be resume, interview, or application") from exc
    root = _career_root()
    indexer = CareerWorkspaceIndexer(
        career_store=CareerStore(root),
        material_store=MaterialStore(root / "materials", workspace=str(ROOT)),
        candidate_fact_store=CandidateFactStore(root / "candidate-facts"),
        interview_knowledge_store=InterviewKnowledgeStore(root / "interview-knowledge"),
        interview_knowledge_library=InterviewKnowledgeLibrary(root / "interview-knowledge"),
        project_library_store=_project_library_stores()[0],
    )
    index = indexer.build()
    return CareerWorkflowEngine().inspect(index, goal=selected_goal).to_dict()


@app.get("/api/career/facts")
async def career_facts(status: str = ""):
    """List staged candidate facts with their source references."""
    from smart_agent.career import CandidateFactStatus

    try:
        _root, _career_store, _material_store, candidate_store = _career_stores()
        selected = CandidateFactStatus(status) if status else None
        facts = candidate_store.list(status=selected)
        return {"facts": [fact.to_dict() for fact in facts]}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _public_interview_knowledge_sources(root: Path) -> list[dict]:
    """Return interview-only knowledge sources for the personal library UI.

    These records are intentionally kept separate from material manifests and
    PersonalEvidence, so they can power interview RAG without entering resume
    generation or fact confirmation.
    """
    from smart_agent.career import InterviewKnowledgeLibrary

    library = InterviewKnowledgeLibrary(root / "interview-knowledge")
    records = []
    tenant_id = _career_tenant_id() or "legacy-web-user"
    for source in library.list_sources(tenant_id=tenant_id):
        corpus_metadata = library.corpus_store.get_metadata(source.corpus_id)
        source_record = source.to_dict()
        source_record.pop("source_path", None)
        records.append(
            {
                **source_record,
                "corpus": {
                    "corpus_id": source.corpus_id,
                    "version": source.index_version or source.content_hash,
                    "chunk_count": source.chunk_count or corpus_metadata.chunk_count,
                },
            }
        )
    return records


@app.get("/api/career/material-library")
async def career_material_library():
    """Return the persistent material library independently of any JD task."""
    cache_key = _career_library_cache_key()
    cached = _career_library_cache_get(_CAREER_MATERIAL_LIBRARY_CACHE, cache_key)
    if cached is not None:
        return cached
    from smart_agent.career import (
        CandidateFactStatus,
        MaterialCategory,
        PersonalEvidence,
        ProjectDescriptionDocument,
        material_catalog_version,
        record_to_dict,
    )

    try:
        root, career_store, material_store, candidate_store = _career_stores()
        manifests = material_store.list_manifests()
        candidates = candidate_store.list()
        evidence = career_store.list_latest(PersonalEvidence)
        descriptions = career_store.list_latest(ProjectDescriptionDocument)
        project_store, analysis_store = _project_library_stores()
        project_sources = project_store.list()
        interview_knowledge_sources = _public_interview_knowledge_sources(root)
        interview_knowledge_chunk_count = sum(
            int((item.get("corpus") or {}).get("chunk_count") or 0)
            for item in interview_knowledge_sources
        )
        active_project_ids = {item.project_id for item in project_sources}
        descriptions = tuple(item for item in descriptions if item.project_id in active_project_ids)
        category_counts = {
            category.value: sum(item.category is category for item in manifests)
            for category in MaterialCategory
        }
        confirmed_fact_count = sum(
            fact.is_usable
            for record in evidence
            for fact in record.facts
        )
        pending_fact_count = sum(
            item.status is CandidateFactStatus.CANDIDATE for item in candidates
        )
        timestamp_candidates = [
            *(item.updated_at for item in candidates),
            *(item.updated_at for item in evidence),
            *(item.updated_at for item in descriptions),
            *(item.updated_at for item in project_sources),
            *(float(item.get("imported_at") or 0.0) for item in interview_knowledge_sources),
        ]
        if material_store.manifest_path.exists():
            timestamp_candidates.append(material_store.manifest_path.stat().st_mtime)
        payload = {
            "material_catalog_version": material_catalog_version(
                manifests,
                evidence,
                candidates,
                descriptions,
            ),
            "summary": {
                "material_count": len(manifests),
                "project_count": len(project_sources),
                "interview_knowledge_count": len(interview_knowledge_sources),
                "interview_knowledge_chunk_count": interview_knowledge_chunk_count,
                "pending_fact_count": pending_fact_count,
                "confirmed_fact_count": confirmed_fact_count,
                "rejected_fact_count": sum(
                    item.status is CandidateFactStatus.REJECTED for item in candidates
                ),
                "latest_updated_at": max(timestamp_candidates, default=0.0),
            },
            "category_counts": category_counts,
            "materials": [item.to_dict() for item in manifests],
            "project_sources": [
                {
                    **item.to_dict(),
                    "latest_analysis": latest.to_dict() if latest is not None else None,
                }
                for item in project_sources
                for latest in (analysis_store.latest(item.project_id),)
            ],
            "project_descriptions": [record_to_dict(item) for item in descriptions],
            "interview_knowledge_sources": interview_knowledge_sources,
        }
        _CAREER_MATERIAL_LIBRARY_CACHE[cache_key] = (time.monotonic(), payload)
        return payload
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/career/agent/turn")
async def career_agent_turn(data: dict):
    """Run one bounded, evidence-safe Career Agent turn.

    This endpoint exposes the backend single-agent contract without coupling
    it to the current browser WebSocket protocol.
    """
    from smart_agent.career import CareerAgent, CareerAgentRequest, CareerAgentRunStore

    message = str(data.get("message") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    goal = str(data.get("goal") or "resume").strip().lower()
    session_id = str(data.get("session_id") or "").strip()
    history = data.get("history") or []
    if not isinstance(history, list):
        raise HTTPException(status_code=400, detail="history must be an array")
    try:
        request = CareerAgentRequest(
            message=message,
            goal=goal,
            session_id=session_id,
            history=tuple(history),
        )
        runtime = load_runtime_config()
        runtime["activeDomain"] = "career"
        runtime["supervisorMode"] = "off"
        runtime["careerSingleAgent"] = True
        tools = create_default_tool_registry(str(ROOT), runtime=runtime)
        try:
            model = create_model_adapter(
                model=str(data.get("model") or runtime.get("model") or ""),
                tools=tools,
                runtime=runtime,
            )
            from smart_agent.agent_collection import build_default_registry

            result = build_default_registry().create("career",
                model=model,
                cwd=str(ROOT),
                tools=tools,
                runtime=runtime,
                run_store=CareerAgentRunStore(_career_root() / "agent-runs"),
                max_steps=int(data.get("max_steps") or 12),
            ).run(request)
            return result.to_dict()
        finally:
            tools.dispose()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/career/agent/metrics")
async def career_agent_metrics():
    from smart_agent.career import CareerAgentRunStore

    store = CareerAgentRunStore(_career_root() / "agent-runs")
    return {**store.metrics(), "runs": [item.to_dict() for item in store.list()]}


def _admin_trace_item(item: dict[str, object]) -> dict[str, object]:
    """Expose agent movement without leaking prompts, material text, or secrets."""
    tool = str(item.get("tool") or item.get("tool_name") or "").strip()
    safe: dict[str, object] = {"tool": tool, "ok": bool(item.get("ok", True))}
    path = str(item.get("path") or "").strip()
    if path:
        safe["path"] = path
    return safe


def _agent_stage_root() -> Path:
    return _career_root() / "agent-stages"


def _record_agent_stage(
    *,
    resume_run_id: str,
    agent_type: str,
    agent_name: str,
    stage: str,
    status: str,
    started_at: float,
    error: str = "",
    token_count: int = 0,
    tool_call_count: int = 0,
) -> dict[str, object]:
    """Persist a redacted Career pipeline stage and publish it to the live hub."""
    finished_at = time.time()
    payload: dict[str, object] = {
        "id": f"career-stage-{uuid.uuid4().hex}",
        "kind": "career_agent_stage",
        "agent_type": agent_type,
        "agent_name": agent_name,
        "stage": stage,
        "status": status,
        "resume_run_id": resume_run_id,
        "session_id": resume_run_id,
        "created_at": started_at,
        "elapsed_ms": max(0.0, (finished_at - started_at) * 1000),
        "tool_call_count": max(0, int(tool_call_count)),
        "token_count": max(0, int(token_count)),
        "error": str(error or "")[:500],
        "trace": [],
    }
    root = _agent_stage_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{payload['id']}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    try:
        agent_event_hub.publish(
            "agent_finished" if status in {"completed", "failed"} else "agent_started",
            agent_type=agent_type,
            agent_name=agent_name,
            session_id=resume_run_id,
            child_session_id=payload["id"],
            status=status,
            stage=stage,
            resume_run_id=resume_run_id,
            elapsed_ms=payload["elapsed_ms"],
            tool_call_count=payload["tool_call_count"],
            token_count=payload["token_count"],
            error=payload["error"],
        )
    except Exception:
        pass
    return payload


def _read_agent_stages() -> list[dict[str, object]]:
    root = _agent_stage_root()
    if not root.is_dir():
        return []
    result: list[dict[str, object]] = []
    for path in root.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


@app.get("/api/admin/agent-flow")
async def admin_agent_flow(limit: int = 100):
    """Read-only operational view of parent and child Agent execution."""
    from smart_agent.career import CareerAgentRunStore, ProjectAnalysisRunStore

    bounded = max(1, min(int(limit or 100), 500))
    career_runs = CareerAgentRunStore(_career_root() / "agent-runs").list()
    project_runs = ProjectAnalysisRunStore(_career_root() / "project-library" / "runs").list()
    career_stage_events = _read_agent_stages()
    persisted_interview_events = _persisted_interviewer_question_events()
    interview_events_by_id = {
        str(item.get("id")): item
        for item in persisted_interview_events
    }
    for item in _INTERVIEW_AGENT_EVENTS:
        event_id = str(item.get("id"))
        persisted = interview_events_by_id.get(event_id)
        if persisted and persisted.get("persisted") and not item.get("persisted"):
            continue
        interview_events_by_id[event_id] = item
    interview_events = list(interview_events_by_id.values())
    events: list[dict[str, object]] = []
    for run in career_runs:
        events.append({
            "id": run.run_id,
            "kind": "career_agent",
            "agent": "Career Agent",
            "status": run.status,
            "session_id": run.session_id,
            "project_id": "",
            "created_at": run.created_at,
            "elapsed_ms": run.elapsed_ms,
            "tool_call_count": run.tool_call_count,
            "token_count": run.token_count,
            "error": run.error,
            "trace": [_admin_trace_item(item) for item in run.tool_trace],
        })
    for run in project_runs:
        events.append({
            "id": run.run_id,
            "kind": "project_analyst",
            "agent": "Project Analyst",
            "status": run.status,
            "session_id": "",
            "project_id": run.project_id,
            "created_at": run.created_at,
            "elapsed_ms": run.elapsed_ms,
            "tool_call_count": run.tool_call_count,
            "token_count": run.token_count,
            "error": run.error,
            "trace": [_admin_trace_item(item) for item in run.agent_trace],
        })
    events.extend(interview_events)
    events.extend(career_stage_events)
    events.sort(key=lambda item: (float(item.get("created_at") or 0), str(item.get("id") or "")), reverse=True)
    return {
        "generated_at": time.time(),
        "summary": {
            "career_runs": len(career_runs),
            "project_analysis_runs": len(project_runs),
            "interview_questions": len(interview_events),
            "failed": sum(item.get("status") == "failed" for item in events),
            "tool_calls": sum(int(item.get("tool_call_count") or 0) for item in events),
            "tokens": sum(int(item.get("token_count") or 0) for item in events),
        },
        "events": events[:bounded],
    }


@app.get("/api/admin/agent-events")
async def admin_agent_events(limit: int = 200, after: int = 0):
    """Return the recent live Agent event buffer for the operations console."""
    bounded = max(1, min(int(limit or 200), 2_000))
    cursor = max(0, int(after or 0))
    return {
        "generated_at": time.time(),
        "after": cursor,
        "events": agent_event_hub.events_for(after=cursor, limit=bounded),
    }


@app.get("/api/admin/agent-events/stream")
async def admin_agent_events_stream(request: Request, after: int = 0):
    """Stream Agent events as Server-Sent Events.

    The stream is intentionally read-only.  Events are small redacted
    summaries and can be replayed by passing the last received sequence in
    ``after`` after a reconnect.
    """
    import queue

    cursor = max(0, int(after or 0))
    token, subscriber = agent_event_hub.subscribe()

    async def generate():
        nonlocal cursor
        try:
            # Replay first so a page opened during an active run has context.
            for item in agent_event_hub.events_for(after=cursor, limit=200):
                cursor = max(cursor, int(item.get("sequence") or 0))
                yield f"id: {item.get('event_id', '')}\nevent: agent\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
            last_heartbeat = time.monotonic()
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.to_thread(subscriber.get, True, 1.0)
                except queue.Empty:
                    if time.monotonic() - last_heartbeat >= 15:
                        yield ": heartbeat\n\n"
                        last_heartbeat = time.monotonic()
                    continue
                if item.sequence <= cursor:
                    continue
                cursor = item.sequence
                payload = item.to_dict()
                yield f"id: {item.event_id}\nevent: agent\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        finally:
            agent_event_hub.unsubscribe(token)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/career/facts/decision")
async def career_fact_decision(data: dict):
    """Confirm or reject candidate facts in one explicit user decision."""
    from smart_agent.career import CandidateFactStatus, record_to_dict

    decision = str(data.get("decision") or "").strip().lower()
    candidate_ids = tuple(data.get("candidate_fact_ids") or ())
    reason = str(data.get("reason") or "user decision").strip()
    if decision not in {"confirm", "reject"}:
        raise HTTPException(status_code=400, detail="decision must be confirm or reject")
    try:
        _root, career_store, _material_store, candidate_store = _career_stores()
        if decision == "confirm":
            records, transitioned = candidate_store.import_to_evidence(candidate_ids, career_store=career_store)
            # The browser's single "confirm" action represents the owner's
            # explicit confirmation of the extracted fact, not merely the
            # import staging step exposed to the agent tool. Promote each
            # imported fact to usable evidence in the same request.
            confirmed_records = []
            for fact in transitioned:
                if fact.evidence_id and fact.fact_id:
                    confirmed_records.append(
                        career_store.confirm_fact(
                            fact.evidence_id,
                            fact.fact_id,
                            actor="user",
                            reason=reason,
                        )
                    )
            return {
                "decision": decision,
                "candidate_facts": [fact.to_dict() for fact in transitioned],
                "evidence": [record_to_dict(record) for record in (confirmed_records or records)],
            }
        transitioned = candidate_store.reject(candidate_ids, reason=reason)
        return {"decision": decision, "candidate_facts": [fact.to_dict() for fact in transitioned]}
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/career/jd/analyze")
async def career_jd_analyze(data: dict):
    """Normalize and persist a JD; heuristic parsing remains an explicit offline mode."""
    from smart_agent.career import JDAnalyzer

    text = str(data.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="JD text is required")
    role_title = str(data.get("role_title") or "").strip()
    if not role_title:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        first_line = lines[0] if lines else ""
        labeled_title = re.search(r"(?im)^\s*(?:职位|岗位|position|role)\s*[:：]\s*[^\n]+", text)
        looks_like_title = bool(
            first_line
            and len(first_line) <= 80
            and not re.search(r"[。；;]", first_line)
            and not re.search(r"负责|岗位职责|岗位说明|岗位要求|职位描述|职位要求|任职要求|工作内容", first_line)
        )
        if not labeled_title and not looks_like_title:
            raise HTTPException(status_code=400, detail="请填写岗位名称，或在 JD 正文首行提供岗位名称")
    try:
        _root, career_store, _material_store, _candidate_store = _career_stores()
        analysis = JDAnalyzer().analyze(
            text,
            requirements=data.get("requirements"),
            role_title=role_title,
            company=str(data.get("company") or ""),
            source_id=str(data.get("source_id") or ""),
            validate_metadata_presence=False,
        ).persist(career_store)
        from smart_agent.career import ResumeRunService, record_to_dict

        snapshot = _career_material_snapshot()
        run = ResumeRunService(career_store).create(
            jd=analysis.document,
            evidence=snapshot[5],
            material_catalog_version=snapshot[7],
        )
        return {
            **analysis.to_dict(),
            "resume_run": record_to_dict(run),
            "material_catalog_version": snapshot[7],
        }
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _latest_career_record(store, record_type):
    records = store.list_latest(record_type)
    return max(records, key=lambda item: (item.updated_at, item.revision, item.content_hash), default=None)


def _role_tokens(value: str) -> set[str]:
    text = str(value or "").casefold()
    tokens = set(re.findall(r"[a-z0-9+#.]+|[\u4e00-\u9fff]", text))
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        tokens.update(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return tokens


def _internship_fields(evidence) -> dict[str, str]:
    """Recover structured internship fields from confirmed fact text, including legacy rows."""
    fields = {"company": "", "department": "", "role": "", "summary": "", "content": "", "date": ""}
    labels = {
        "实习公司": "company", "实习公司名称": "company", "公司": "company",
        "实习岗位": "role", "岗位": "role", "职位": "role",
        "项目部门": "department", "部门": "department",
        "实习简述": "summary", "实习成果": "summary", "简述": "summary",
        "实习内容": "content", "内容": "content", "职责": "content",
        "实习日期": "date", "日期": "date", "时间": "date",
    }
    for fact in evidence.facts:
        if not fact.is_usable:
            continue
        raw = str(fact.value or "").strip()
        key, value = "", raw
        match = re.match(r"^\s*([^:：]{1,20})\s*[:：]\s*(.+)$", raw)
        if match:
            key, value = match.group(1).strip(), match.group(2).strip()
        target = labels.get(key)
        if target is None:
            field_name = str(getattr(fact, "field_name", "") or "").casefold()
            target = (
                "company" if field_name in {"company", "employer", "internship_company"}
                else "department" if field_name in {"department", "team", "internship_department"}
                else "role" if field_name in {"title", "role", "position", "internship_role"}
                else "date" if field_name in {"date", "period", "internship_date"}
                else "content" if field_name in {"responsibility", "content", "internship_content"}
                else "summary"
            )
        if value and not fields[target]:
            fields[target] = value
    fields["summary"] = fields["summary"] or fields["content"]
    return fields


def _internship_role_match_score(target_role: str, internship_role: str) -> float:
    target = _role_tokens(target_role)
    actual = _role_tokens(internship_role)
    if not target or not actual:
        return 0.0
    overlap = len(target & actual) / max(1, len(target))
    substring = 1.0 if str(internship_role).casefold() in str(target_role).casefold() or str(target_role).casefold() in str(internship_role).casefold() else 0.0
    return min(1.0, 0.75 * overlap + 0.25 * substring)


def _clean_resume_fact_value(section: str, value: str) -> str:
    """Remove an imported section heading that the renderer already owns."""
    text = str(value or "").strip()
    text = re.sub(r"^\s*(?:\d+[、)]\s*|\d+\.(?=\s)|[.;；]\s*)", "", text)
    text = re.sub(r"\s*[.;；]\s*$", "", text)
    text = re.sub(r"[,，]\s*使(?:用|[\u2f00-\u2fff])$", "", text)
    if section == "awards":
        text = re.sub(r"^(?:获奖经历|获奖)\s*[:：]\s*", "", text)
    elif section == "research":
        text = re.sub(r"^科研经历\s*[:：]\s*", "", text)
    elif section == "education":
        text = re.sub(r"^教育经历\s*[:：]\s*", "", text)
    return text.strip()


def _resume_fact_text(section: str, fact) -> str:
    text = _clean_resume_fact_value(section, str(fact.value).strip())
    field_name = str(getattr(fact, "field_name", "") or "").strip().casefold()
    if field_name in {"timeline", "project_date", "project_period"} and not re.match(r"^(?:项目)?(?:日期|时间)\s*[:：]", text):
        return f"项目时间：{text}"
    if section == "education" and field_name in {"date", "time", "period", "education_date"} and not re.match(r"^(?:教育)?(?:日期|时间)\s*[:：]", text):
        return f"教育时间：{text}"
    return text


def _free_text_block(value: object) -> str:
    """Preserve user-authored free-text layout while normalizing line endings."""
    text = str(value or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _strip_generated_skills_heading(text: str) -> str:
    """Remove only the section heading already rendered by the resume template."""
    return re.sub(
        r"^\s*(?:技能/其他|技能证书|专业技能|技能|其他|skills)\s*[:：]\s*",
        "",
        str(text or ""),
        count=1,
        flags=re.IGNORECASE,
    ).strip()


def _source_reference_text(ref) -> str:
    locator = getattr(ref, "locator", {}) or {}
    if not isinstance(locator, dict):
        return ""
    raw_path = str(locator.get("path") or "").strip()
    if not raw_path:
        return ""
    try:
        path = Path(raw_path).expanduser().resolve()
        root = ROOT.expanduser().resolve()
        if not path.is_relative_to(root):
            return ""
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ""


def _skills_resume_text(evidence, candidate_facts) -> tuple[str, list]:
    """Build skills/other as one verbatim block, not as semicolon-joined facts."""
    facts = [fact for fact in candidate_facts if getattr(fact, "is_usable", False)]
    for fact in facts:
        field_name = str(getattr(fact, "field_name", "") or "").strip().casefold()
        value = _free_text_block(getattr(fact, "value", ""))
        if value and (field_name in {"content", "skills", "skill", "full_text"} or "\n" in value):
            return _strip_generated_skills_heading(value), [fact]

    source_refs = list(getattr(evidence, "source_refs", ()) or [])
    for fact in facts:
        source_refs.extend(getattr(fact, "source_refs", ()) or [])
    for ref in source_refs:
        value = _free_text_block(_source_reference_text(ref))
        if value:
            return _strip_generated_skills_heading(value), facts

    lines = [_strip_generated_skills_heading(_resume_fact_text("skills", fact)) for fact in facts]
    text = "\n".join(line for line in lines if line.strip())
    return text.strip(), facts


def _project_fact_values(fact) -> list[str]:
    value = fact.value
    if isinstance(value, list):
        return [" ".join(str(item).split()).strip() for item in value if str(item).strip()]
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return []
    if "\n- " in str(value):
        return [
            " ".join(item.split()).strip()
            for item in re.split(r"\n-\s*", str(value))
            if " ".join(item.split()).strip()
        ]
    return [text]


def _project_technology_stack(evidence) -> str:
    """Extract a confirmed technology-stack hint for the project editor."""
    field_names = {"technology_stack", "technologies", "technology_choices", "technology", "tech_stack"}
    for fact in evidence.facts:
        field_name = str(getattr(fact, "field_name", "") or "").casefold()
        if field_name in field_names:
            values = _project_fact_values(fact)
            if values:
                return "、".join(values)
    markdown = _confirmed_project_markdown(evidence)
    match = re.search(r"^#{1,6}\s*技术栈(?:和接口)?\s*$", markdown, flags=re.MULTILINE)
    if match:
        section = re.split(r"^#{1,6}\s+", markdown[match.end() :], maxsplit=1, flags=re.MULTILINE)[0]
        values = [line.strip().lstrip("-*• ") for line in section.splitlines() if line.strip()]
        if values:
            return "、".join(values)
    return ""


def _confirmed_project_markdown(evidence) -> str:
    """Render every usable fact for one project as an isolated Markdown package."""
    facts = [fact for fact in evidence.facts if fact.is_usable]
    if len(facts) == 1:
        fact = facts[0]
        if str(getattr(fact, "field_name", "") or "").casefold() in {
            "content",
            "markdown",
            "project_markdown",
        }:
            text = str(fact.value or "").strip()
            if text:
                return text
    lines = [f"# {str(evidence.label or '').strip()}"]
    for fact in facts:
        field_name = str(getattr(fact, "field_name", "") or "fact").strip()
        value = fact.value
        rendered = (
            str(value).strip()
            if isinstance(value, str)
            else json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        )
        if rendered:
            lines.extend(("", f"## {field_name}", "", rendered))
    return "\n".join(lines).strip()


def _trim_resume_boundary(text: str, limit: int) -> str:
    """Trim text for one-line resume claims without cutting mid-phrase."""
    value = " ".join(str(text or "").split()).strip()
    if len(value) <= limit:
        return value.rstrip("；;。 ")
    clipped = value[:limit]
    boundaries = [clipped.rfind(mark) for mark in ("。", "；", ";", "，", ",")]
    boundary = max(boundaries)
    if boundary >= max(20, int(limit * 0.55)):
        return clipped[:boundary].rstrip("；;，,。 ")
    return clipped.rstrip("；;，,。 ")


def _project_resume_text(evidence, matched_facts) -> tuple[str, list]:
    """Build the strict project-block protocol before model editing."""
    usable = [fact for fact in evidence.facts if fact.is_usable]
    by_field: dict[str, list] = {}
    for fact in usable:
        by_field.setdefault(str(getattr(fact, "field_name", "") or "").casefold(), []).append(fact)

    summary_fact = next(
        (
            fact
            for field in ("goals", "summary", "background")
            for fact in by_field.get(field, [])
            if _project_fact_values(fact)
        ),
        None,
    )
    summary = _project_fact_values(summary_fact)[0] if summary_fact is not None else str(evidence.label)

    preferred_fields = {
        "core_flow": 0,
        "implemented_features": 1,
        "technology_choices": 2,
        "outcomes": 3,
        "architecture": 4,
        "responsibilities": 5,
    }
    ordered_facts = sorted(
        [*matched_facts, *usable],
        key=lambda fact: preferred_fields.get(str(getattr(fact, "field_name", "") or "").casefold(), 99),
    )
    points: list[str] = []
    support_facts: list = [summary_fact] if summary_fact is not None and summary_fact in matched_facts else []
    seen_fact_ids: set[str] = set()
    seen_points: set[str] = set()
    for fact in ordered_facts:
        if fact.fact_id in seen_fact_ids:
            continue
        seen_fact_ids.add(fact.fact_id)
        field_name = str(getattr(fact, "field_name", "") or "").casefold()
        if field_name in {"background", "goals", "summary", "core_modules", "technologies"}:
            continue
        for value in _project_fact_values(fact):
            value = re.sub(r"^(?:核心流程|技术选型|项目设计)\s*[:：]\s*", "", value).strip()
            if not value or value in seen_points:
                continue
            seen_points.add(value)
            points.append(value)
            if fact not in support_facts:
                support_facts.append(fact)
            if len(points) == 5:
                break
        if len(points) == 5:
            break

    summary = _trim_resume_boundary(summary, 95)
    point_budget = max(58, (540 - len(str(evidence.label)) - len(summary)) // 4)
    points = [_trim_resume_boundary(point, point_budget) for point in points]
    text = (
        f"项目名称：{evidence.label}；项目简述：{summary}；项目设计："
        + " ".join(f"{index}、{point}" for index, point in enumerate(points, start=1))
    )
    return text[:600], [fact for fact in support_facts if fact is not None]


def _legacy_resume_fact_section(fact) -> str:
    """Classify facts from the old combined project/skills resume import."""
    field_name = str(getattr(fact, "field_name", "") or "").strip().casefold()
    text = str(getattr(fact, "value", "") or "").strip()
    if field_name in {"technology", "skill", "skills", "competency"}:
        return "skills"
    if field_name in {"impact", "achievement", "result"}:
        return "projects"
    if re.match(r"^(?:[.；;\s]*)(?:技术栈|研究方向|研究⽅向|熟悉|熟练|掌握|精通)", text, re.IGNORECASE):
        return "skills"
    project_markers = (
        "项目",
        "smartcode",
        "系统",
        "平台",
        "框架",
        "完成",
        "实现",
        "搭建",
        "构建",
        "开发",
    )
    if any(marker in text.casefold() for marker in project_markers):
        return "projects"
    return "skills"


def _resume_candidate_items(
    store,
    jd,
    *,
    match_refs=(),
    evidence_refs=(),
    confidence_threshold: float = 0.72,
    include_unmatched_projects: bool = False,
    exclude_legacy_projects: bool = False,
    allowed_requirement_ids: Iterable[str] | None = None,
) -> list[dict]:
    """Convert persisted evidence matches into planner candidates.

    The HTTP layer never accepts evidence IDs or requirement IDs from the
    browser. They are derived from immutable store references after retrieval,
    so a client cannot smuggle an unsupported claim into a ResumePlan.
    """
    from smart_agent.career import EvidenceMatch, EvidenceType, PersonalEvidence, SupportLevel

    allowed_ids = (
        {str(item).strip() for item in allowed_requirement_ids if str(item).strip()}
        if allowed_requirement_ids is not None
        else None
    )
    requirement_order = tuple(
        ref.record_id
        for ref in jd.requirement_refs
        if allowed_ids is None or ref.record_id in allowed_ids
    )
    requirement_ids = set(requirement_order)
    requirement_texts: dict[str, str] = {}
    for ref in jd.requirement_refs:
        if allowed_ids is not None and ref.record_id not in allowed_ids:
            continue
        try:
            requirement = store.get(ref.record_id, ref.revision)
            requirement_texts[ref.record_id] = str(getattr(requirement, "text", "") or "").strip()
        except Exception:  # noqa: BLE001 - lightweight test stores may omit requirement records.
            requirement_texts[ref.record_id] = ""
    if not 0 <= float(confidence_threshold) <= 1:
        raise ValueError("careerResumeMatchConfidenceThreshold must be between 0 and 1")
    grouped: dict[str, set[str]] = {}
    matched_requirements: dict[str, set[str]] = {}
    fact_scores_by_requirement: dict[str, dict[tuple[str, str], float]] = {}
    fact_texts_by_key: dict[tuple[str, str], str] = {}
    scores: dict[str, float] = {}
    ranking_scores: dict[str, list[float]] = {}
    matches = (
        tuple(store.get(ref.record_id, ref.revision, expected_type=EvidenceMatch) for ref in match_refs)
        if match_refs
        else store.list_latest(EvidenceMatch)
    )
    for match in matches:
        requirement_id = match.requirement_ref.record_id
        if (
            requirement_id not in requirement_ids
            or match.support_level is SupportLevel.UNSUPPORTED
            or match.evidence_ref is None
        ):
            continue
        evidence_id = match.evidence_ref.record_id
        ranking_scores.setdefault(evidence_id, []).append(float(match.score))
        scores[evidence_id] = max(scores.get(evidence_id, 0.0), float(match.score))
        for fact_id in match.evidence_fact_ids:
            key = (evidence_id, fact_id)
            fact_scores_by_requirement.setdefault(requirement_id, {})[key] = max(
                fact_scores_by_requirement.setdefault(requirement_id, {}).get(key, 0.0),
                float(match.score),
            )
        if float(match.score) < float(confidence_threshold):
            continue

    evidence_records = (
        tuple(store.get(ref.record_id, ref.revision, expected_type=PersonalEvidence) for ref in evidence_refs)
        if evidence_refs
        else store.list_latest(PersonalEvidence)
    )


    evidence_by_id = {evidence.evidence_id: evidence for evidence in evidence_records}
    for evidence in evidence_records:
        for fact in evidence.facts:
            fact_texts_by_key[(evidence.evidence_id, fact.fact_id)] = str(fact.value or "").strip()

    selected_pairs_by_requirement: dict[str, tuple[str, str]] = {}
    missing_requirements: list[str] = []
    # The global grouping is retained for evidence filtering; project-specific
    # blocks are assembled later because each project has its own score matrix.
    for requirement_id in requirement_order:
        ranked_facts = sorted(
            (
                item
                for item in fact_scores_by_requirement.get(requirement_id, {}).items()
                if item[1] >= float(confidence_threshold)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if ranked_facts:
            selected_pairs_by_requirement[requirement_id] = ranked_facts[0][0]
        else:
            missing_requirements.append(requirement_id)

    for requirement_id, key in selected_pairs_by_requirement.items():
        evidence_id, fact_id = key
        grouped.setdefault(evidence_id, set()).add(fact_id)
        matched_requirements.setdefault(evidence_id, set()).add(requirement_id)

    fused_scores: dict[str, float] = {}
    for evidence_id, values in ranking_scores.items():
        ordered = sorted((max(0.0, min(1.0, float(value))) for value in values), reverse=True)
        if not ordered:
            continue
        fused_scores[evidence_id] = min(1.0, ordered[0] + 0.25 * sum(ordered[1:]))

    sections = {
        EvidenceType.EDUCATION: "education",
        EvidenceType.AWARD: "awards",
        EvidenceType.INTERNSHIP: "internships",
        EvidenceType.PROJECT: "projects",
        EvidenceType.RESEARCH: "research",
        EvidenceType.SKILLS: "skills",
    }
    candidates: list[dict] = []
    for evidence in evidence_records:
        section = sections.get(evidence.evidence_type, "projects")
        generic_project_label = evidence.evidence_type is EvidenceType.PROJECT and any(
            token in evidence.label for token in ("原始简历", "项目与技能", "用户确认")
        )
        project_jd_fact_blocks: list[dict] = []
        project_threshold_pairs: list[dict] = []
        project_gap_pairs: list[dict] = []
        project_threshold_scores: list[float] = []
        project_gap_scores: list[float] = []
        project_threshold_keys: set[tuple[str, str]] = set()
        project_threshold_requirement_ids: list[str] = []
        project_gap_keys: list[tuple[str, str]] = []
        if section == "projects" and not generic_project_label:
            usable_keys = [
                (evidence.evidence_id, fact.fact_id)
                for fact in evidence.facts
                if fact.is_usable
            ]
            missing_for_project: list[str] = []
            for requirement_id in requirement_order:
                ranked = sorted(
                    (
                        (key, fact_scores_by_requirement.get(requirement_id, {}).get(key, 0.0))
                        for key in usable_keys
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
                if ranked and ranked[0][1] >= float(confidence_threshold):
                    key, score = ranked[0]
                    project_threshold_keys.add(key)
                    project_threshold_requirement_ids.append(requirement_id)
                    project_threshold_scores.append(float(score))
                    project_threshold_pairs.append(
                        {"JD": requirement_texts.get(requirement_id, ""), "事实": fact_texts_by_key.get(key, "")}
                    )
                else:
                    missing_for_project.append(requirement_id)
            if missing_for_project and usable_keys:
                gap_ranked = sorted(
                    (
                        (
                            key,
                            sum(
                                fact_scores_by_requirement.get(requirement_id, {}).get(key, 0.0)
                                for requirement_id in missing_for_project
                            ) / len(missing_for_project),
                        )
                        for key in usable_keys
                        if key not in project_threshold_keys
                    ),
                    key=lambda item: (-item[1], item[0]),
                )[:3]
                for key, score in gap_ranked:
                    project_gap_keys.append(key)
                    project_gap_scores.append(float(score))
                    project_gap_pairs.append(
                        {"JD": "\n".join(requirement_texts.get(req, "") for req in missing_for_project if requirement_texts.get(req, "")),
                         "事实": fact_texts_by_key.get(key, "")}
                    )
                if project_gap_pairs:
                    project_jd_fact_blocks.append(
                        {
                            "block_type": "gap",
                            "JD": "\n".join(requirement_texts.get(req, "") for req in missing_for_project if requirement_texts.get(req, "")),
                            "事实": [pair["事实"] for pair in project_gap_pairs if pair["事实"]],
                        }
                    )
            project_jd_fact_blocks = [
                *[
                    {"block_type": "threshold", "JD": pair["JD"], "事实": [pair["事实"]]}
                    for pair in project_threshold_pairs
                    if pair["JD"] and pair["事实"]
                ],
                *project_jd_fact_blocks,
            ]
            fact_ids = {fact_id for evidence_id, fact_id in (*project_threshold_keys, *project_gap_keys) if evidence_id == evidence.evidence_id}
            if not project_threshold_keys and not include_unmatched_projects:
                fact_ids = set()
        else:
            fact_ids = grouped.get(evidence.evidence_id)
        if section in {"education", "research", "awards", "skills", "internships"}:
            # These sections do not use ordinary JD requirement matching.
            # Internship relevance is evaluated against the target role only.
            fact_ids = {fact.fact_id for fact in evidence.facts if fact.is_usable}
        if section in {"projects", "skills"} and not fact_ids:
            if not include_unmatched_projects:
                continue
            fact_ids = {fact.fact_id for fact in evidence.facts if fact.is_usable}
        facts = [
            fact
            for fact in evidence.facts
            if (not fact_ids or fact.fact_id in fact_ids) and fact.is_usable
        ]
        if not facts:
            continue
        if generic_project_label:
            grouped_facts = [
                (_legacy_resume_fact_section(fact), [fact])
                for fact in facts
            ]
        else:
            grouped_facts = [(section, facts)]
        for group_index, (candidate_section, candidate_facts) in enumerate(grouped_facts):
            if exclude_legacy_projects and generic_project_label and candidate_section == "projects":
                continue
            if candidate_section == "skills":
                details, candidate_facts = _skills_resume_text(evidence, candidate_facts)
            else:
                details = "；".join(
                    _resume_fact_text(candidate_section, fact)
                    for fact in candidate_facts
                )
            if re.search(r"测试\s*[，,]?\s*请删除", details, flags=re.IGNORECASE):
                # Imported resume placeholders are not candidate evidence and
                # must never become a generated claim.
                continue
            if not details.strip():
                continue
            requirement_fact_ids = [fact.fact_id for fact in candidate_facts]
            internship_meta = {}
            if candidate_section == "internships":
                internship_meta = _internship_fields(evidence)
                role_score = _internship_role_match_score(jd.role_title, internship_meta["role"])
                matched_role = role_score >= 0.45
                internship_meta["role_score"] = role_score
                internship_meta["matched_role"] = matched_role
                fields = (("company", "实习公司名称"), ("department", "项目部门"), ("role", "实习岗位"), ("summary", "实习简述"), ("content", "实习内容"), ("date", "实习日期"))
                details = "；".join(
                    f"{label}：{internship_meta[key]}"
                    for key, label in fields
                    if internship_meta[key]
                ) or details
            if candidate_section == "projects" and not generic_project_label:
                text, _project_support_facts = _project_resume_text(evidence, candidate_facts)
                # The project editor receives the complete confirmed project
                # package, so its ResumeItem must cite every usable project fact.
                candidate_facts = [fact for fact in evidence.facts if fact.is_usable]
            else:
                text = details
            if candidate_section in {"education", "research", "awards"}:
                meaningful = re.sub(r"(?:学校名|专业|绩点|主修课程|科研经历|获奖经历|内容)\s*[:：]", "", details)
                if not meaningful.strip(" ；;，,。.!！?？无暂无noneNone"):
                    continue
            source_block_id = f"{evidence.evidence_id}:{candidate_section}:{group_index}"
            requirement_ids_for_candidate = (
                list(project_threshold_requirement_ids)
                if candidate_section == "projects" and project_threshold_requirement_ids
                else []
            )
            jd_fact_blocks = project_jd_fact_blocks if candidate_section == "projects" else []
            jd_fact_pairs = [
                {"JD": block.get("JD", ""), "事实": fact}
                for block in jd_fact_blocks
                for fact in block.get("事实", [])
                if str(block.get("JD") or "").strip() and str(fact or "").strip()
            ]
            candidates.append(
                {
                    "section": candidate_section,
                    "proposed_text": text,
                    "evidence_ids": [evidence.evidence_id],
                    "evidence_fact_ids": [fact.fact_id for fact in candidate_facts],
                    "requirement_ids": requirement_ids_for_candidate,
                    "jd_fact_pairs": jd_fact_pairs,
                    "jd_fact_blocks": jd_fact_blocks,
                    "threshold_fact_count": len(project_threshold_keys) if candidate_section == "projects" else 0,
                    "threshold_fact_scores": list(project_threshold_scores) if candidate_section == "projects" else [],
                    "gap_fact_scores": list(project_gap_scores) if candidate_section == "projects" else [],
                    "project_match_average": (
                        sum([*project_threshold_scores, *project_gap_scores[:3]])
                        / max(1, len([*project_threshold_scores, *project_gap_scores[:3]]))
                        if candidate_section == "projects" else 0.0
                    ),
                    "project_name": str(evidence.label or "").strip(),
                    "project_description": str(
                        next(
                            (
                                value
                                for fact in evidence.facts
                                if str(getattr(fact, "field_name", "")).casefold() in {"goals", "summary", "background"}
                                for value in _project_fact_values(fact)
                                if value
                            ),
                            evidence.label,
                        )
                    ).strip(),
                    "technology_stack": _project_technology_stack(evidence)
                    if candidate_section == "projects" and not generic_project_label
                    else "",
                    "project_markdown": (
                        _confirmed_project_markdown(evidence)
                        if candidate_section == "projects" and not generic_project_label
                        else ""
                    ),
                    "requirement_evidence_fact_ids": (
                        [fact_id for _evidence_id, fact_id in project_threshold_keys]
                        if requirement_ids_for_candidate
                        else []
                    ),
                    "redundancy_group": source_block_id,
                    "length_cost": max(10, len(text)),
                    "selected_by_llm": False,
                    "match_confidence": round(
                        internship_meta.get(
                            "role_score",
                            fused_scores.get(evidence.evidence_id, scores.get(evidence.evidence_id, 0.0)),
                        ),
                        6,
                    ),
                    "evidence_label": evidence.label,
                    "evidence_type": evidence.evidence_type.value,
                    "legacy_split": bool(generic_project_label),
                    "source_block_id": source_block_id,
                    "block_type": candidate_section.rstrip("s"),
                    "features": {
                        "jd_relevance": min(
                            1.0,
                            scores.get(evidence.evidence_id, fused_scores.get(evidence.evidence_id, 0.0)),
                        ),
                        "evidence_strength": 1.0,
                        "title_and_brand_value": 0.8 if candidate_section in {"internships", "projects"} else 0.6,
                        "result_strength": 0.7,
                        "narrative_value": 0.8,
                    },
                }
            )
            if candidate_section == "internships":
                candidates[-1].update(
                    {
                        "matched_role": bool(internship_meta.get("matched_role", False)),
                        "role_match_score": float(internship_meta.get("role_score", 0.0)),
                    }
                )
                candidates[-1]["features"] = {
                    **candidates[-1]["features"],
                    "jd_relevance": float(internship_meta.get("role_score", 0.0)),
                    "result_strength": 0.8 if internship_meta.get("matched_role") else 0.45,
                }
    return candidates


def _candidate_text_key(item: dict) -> tuple[str, str]:
    text = str(item.get("proposed_text") or "")
    section = str(item.get("section") or "").strip().casefold()
    normalized = "\n".join(" ".join(line.split()) for line in text.splitlines())
    normalized = re.sub(r"\s+", " ", normalized).strip().casefold()
    return section, normalized


def _candidate_relevance(item: dict) -> float:
    features = item.get("features") if isinstance(item.get("features"), dict) else {}
    return max(
        float(item.get("match_confidence") or 0.0),
        float(features.get("jd_relevance") or 0.0),
    )


def _resume_customization_request(value: object) -> str:
    text = str(value or "").strip()
    if "\x00" in text or len(text) > 1200:
        raise ValueError("Resume customization request must be at most 1200 characters.")
    return text


def _resume_project_count_requirement(request: str) -> int | None:
    """Extract an explicit personal-project count from a user resume request."""
    text = _resume_customization_request(request)
    number = r"([0-9]{1,2}|[一二两三四五六七八九十]{1,3})"
    project_noun = r"(?:个人项目|项目经历|项目)"
    # A count of project details is not a count of resume projects. Keep the
    # exclusion close to the noun so wording such as “一个项目设计点” cannot
    # silently reduce an existing two-project resume to one project.
    project_detail_suffix = r"(?!\s*(?:的\s*)?(?:设计点?|要点|亮点|技术点|描述|内容|成果))"
    # Accept both natural Chinese orderings: “两个个人项目” and “个人项目要两个”.
    match = re.search(
        rf"{project_noun}\s*(?:数量|数目)?\s*"
        rf"(?:要求|要|需要|需|取|选|选择|放上|保留|至少|最多|为|是)?\s*[:：]?\s*{number}",
        text,
    )
    if match is None:
        match = re.search(
            rf"{number}\s*(?:个|项|段)?\s*(?:个人)?(?:项目经历|项目){project_detail_suffix}",
            text,
        )
    if match is None:
        return None
    raw_count = match.group(1)
    chinese_counts = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    if raw_count.isdecimal():
        count = int(raw_count)
    elif raw_count in chinese_counts:
        count = chinese_counts[raw_count]
    elif len(raw_count) == 2 and raw_count.startswith("十"):
        count = 10 + chinese_counts.get(raw_count[1], 0)
    elif len(raw_count) == 2 and raw_count.endswith("十"):
        count = chinese_counts.get(raw_count[0], 0) * 10
    else:
        count = 0
    if count < 1 or count > 10:
        raise ValueError("个人项目数量要求必须在 1 到 10 之间。")
    return count


def _resume_project_design_count_requirement(request: str) -> int | None:
    """Extract an exact project-design-point count from a resume request."""
    text = _resume_customization_request(request)
    number = r"([0-9]{1,2}|[一二两三四五六七八九十]{1,3})"
    design_noun = r"(?:项目设计点?|设计点)"
    exact_marker = r"(?:必须|固定|限定|要求|应当|应|设置为|指定为|一共|总共|共)"
    match = re.search(
        rf"{design_noun}\s*(?:数量|数目)?\s*"
        rf"(?:必须|固定为|限定为|要求为|应为|为|是)\s*{number}",
        text,
    )
    if match is None:
        match = re.search(
            rf"{exact_marker}\s*(?:是|为|给出|包含)?\s*{number}\s*"
            rf"(?:个|条|项)?\s*{design_noun}",
            text,
        )
    if match is None:
        # Treat "至少/不少于 N 个项目设计点" as the smallest acceptable
        # count. The generator's contract requires one concrete count, so N
        # is passed through as the requested minimum.
        match = re.search(
            rf"(?:至少|不少于)\s*(?:要|需|为)?\s*{number}\s*"
            rf"(?:个|条|项)?\s*{design_noun}",
            text,
        )
    if match is None:
        return None
    raw_count = match.group(1)
    chinese_counts = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
        "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }
    if raw_count.isdecimal():
        count = int(raw_count)
    elif raw_count in chinese_counts:
        count = chinese_counts[raw_count]
    elif len(raw_count) == 2 and raw_count.startswith("十"):
        count = 10 + chinese_counts.get(raw_count[1], 0)
    elif len(raw_count) == 2 and raw_count.endswith("十"):
        count = chinese_counts.get(raw_count[0], 0) * 10
    else:
        count = 0
    if count < 3 or count > 5:
        raise ValueError("项目设计点数量要求必须在 3 到 5 之间。")
    return count


def _resume_exclusive_project_subject(request: str) -> str | None:
    """Extract a project name from an exclusive project-selection request."""
    text = _resume_customization_request(request)
    match = re.search(
        r"(?:个人项目|项目经历|项目)?\s*"
        r"(?:只能选择|仅能选择|只可选择|仅可选择|只能保留|仅保留|只保留)\s*"
        r"([^，。；,;！？!?]+)",
        text,
    )
    if match is None:
        # Also accept the common form "一个个人项目并且必须是 XXX".  The
        # project-count parser handles the leading quantity separately; here
        # we only need the named project subject.
        match = re.search(
            r"(?:[0-9一二两三四五六七八九十]{1,3}\s*(?:个|项)?\s*)?"
            r"个人项目(?:经历)?\s*(?:(?:并且|且)\s*)?"
            r"(?:必须|应当|要求|指定)\s*(?:是|为)\s*"
            r"([^，。；,;！？!?]+)",
            text,
        )
    if match is None:
        return None
    subject = match.group(1).strip()
    # A natural request often combines the project count and the exclusive
    # project in one clause (for example, "一个个人项目并且必须是 RAG 项目"),
    # then continues with numbered requirements without punctuation. Strip
    # those control words before matching the human-facing project label.
    subject = re.sub(
        r"^(?:[0-9一二两三四五六七八九十]{1,3}\s*(?:个|项|条)?\s*)?"
        r"(?:个人)?项目(?:经历)?\s*(?:(?:并且|且)\s*)?"
        r"(?:(?:必须|应当|要求|指定)\s*)?(?:是|为)?\s*",
        "",
        subject,
        count=1,
    )
    # Numbered follow-up requirements may be separated only by whitespace.
    subject = re.split(r"\s+[0-9]{1,2}\s*[、.)]", subject, maxsplit=1)[0]
    subject = re.split(r"(?:作为|用于|放入|写入)", subject, maxsplit=1)[0].strip()
    subject = re.split(r"(?:并且|且)\s*(?:必须|应当|要求|至少|最多|添加|增加)", subject, maxsplit=1)[0].strip()
    subject = re.sub(r"(?:这个|该)?(?:个人)?项目经历?$", "", subject).strip()
    if not subject or re.fullmatch(r"[0-9一二两三四五六七八九十]+(?:个|项)?", subject):
        return None
    return subject


def _normalized_project_subject(value: object) -> str:
    """Normalize project labels for tolerant, industry-neutral matching."""
    normalized = re.sub(r"\s+", "", str(value or "").casefold())
    # Labels generated from project analysis may add generic descriptors such
    # as "垂直领域" even when the registered project is named "领域" only.
    # These words do not identify a project and should not block a user
    # abbreviation from matching the confirmed evidence.
    normalized = re.sub(r"(?:垂直)?领域", "", normalized)
    return re.sub(r"[^\w一-鿿]", "", normalized)


def _project_subject_variants(value: object) -> tuple[str, ...]:
    """Return exact and abbreviation-friendly project label variants."""
    normalized = _normalized_project_subject(value)
    if not normalized:
        return ()
    variants = [normalized]
    # Users commonly omit generic suffixes such as "问答系统" or "平台"
    # when referring to a project. Remove only generic nouns, preserving the
    # domain and technical tokens that identify the actual project.
    shortened = re.sub(r"(?:问答|系统|平台|项目|助手|应用)$", "", normalized)
    if shortened and shortened != normalized:
        variants.append(shortened)
    return tuple(dict.fromkeys(variants))


def _project_matches_subject(item: dict, requested_subject: str) -> bool:
    requested_variants = _project_subject_variants(requested_subject)
    if not requested_variants:
        return False
    subjects = [
        item.get("evidence_label"),
        item.get("project_name"),
        *_resume_candidate_subjects(item),
    ]
    candidate_variants = tuple(
        variant
        for value in subjects
        for variant in _project_subject_variants(value)
    )
    return any(
        requested == candidate or requested in candidate or candidate in requested
        for requested in requested_variants
        for candidate in candidate_variants
    )


def _strip_unsupported_numeric_requests(
    request: str,
    candidates: list[dict],
) -> tuple[str, tuple[str, ...]]:
    """Remove only numeric request clauses absent from the supplied evidence."""
    text = _resume_customization_request(request)
    if not text:
        return "", ()
    evidence_values: list[str] = []
    for item in candidates:
        evidence_values.extend(
            str(item.get(field) or "")
            for field in ("proposed_text", "project_name", "project_description")
        )
        for pair in item.get("jd_fact_pairs") or ():
            if isinstance(pair, dict):
                evidence_values.append(str(pair.get("事实") or ""))
            else:
                evidence_values.append(str(pair or ""))
        for block in item.get("jd_fact_blocks") or ():
            if not isinstance(block, dict):
                continue
            evidence_values.extend(str(fact or "") for fact in (block.get("事实") or ()))
    evidence_text = " ".join(evidence_values)
    clauses = re.split(r"([；;。.!！\n])", text)
    kept: list[str] = []
    ignored: list[str] = []
    pending = ""
    structural_terms = ("项目数量", "项目设计", "设计点", "项目经历")
    for part in clauses:
        if re.fullmatch(r"[；;。.!！\n]", part or ""):
            pending += part
            continue
        clause = str(part or "").strip()
        if not clause:
            pending += part
            continue
        numbers = re.findall(r"\d+(?:\.\d+)?\s*%", clause)
        unsupported = (
            numbers
            and not any(term in clause for term in structural_terms)
            and any(number.replace(" ", "") not in evidence_text.replace(" ", "") for number in numbers)
        )
        if unsupported:
            ignored.append(clause)
            pending = ""
            continue
        kept.append(pending + part)
        pending = ""
    if pending:
        kept.append(pending)
    sanitized = "".join(kept).strip(" ；;。.!！\n")
    return sanitized, tuple(ignored)


def _resume_candidate_subjects(item: dict) -> tuple[str, ...]:
    """Return human-facing subjects that a user can intentionally prioritize."""
    subjects = [str(item.get("evidence_label") or "").strip()]
    text = str(item.get("proposed_text") or "")
    for prefix in ("项目名称：", "实习公司名称："):
        if prefix not in text:
            continue
        subject = text.split(prefix, 1)[1].split("；", 1)[0].strip()
        if subject:
            subjects.append(subject)
    return tuple(dict.fromkeys(subject for subject in subjects if len(subject) >= 2))


def _apply_resume_customization_preference(candidates: list[dict], request: str) -> list[dict]:
    """Raise only explicitly named evidence blocks before normal JD ranking."""
    instruction = _resume_customization_request(request).casefold()
    if not instruction:
        return candidates
    positive_markers = ("突出", "优先", "偏向", "重点", "强调", "保留")
    negative_markers = ("弱化", "删除", "移除", "不突出", "不保留", "不要")
    updated: list[dict] = []
    for item in candidates:
        section = str(item.get("section") or "").casefold()
        if section not in {"projects", "internships"}:
            updated.append(item)
            continue
        score = 0.0
        matched_subject = ""
        for subject in _resume_candidate_subjects(item):
            subject_key = subject.casefold()
            position = instruction.find(subject_key)
            if position < 0:
                continue
            window = instruction[max(0, position - 12):position + len(subject_key) + 12]
            if any(marker in window for marker in negative_markers):
                score = min(score, -1.0)
                continue
            if any(marker in window for marker in positive_markers) or len(subject_key) >= 4:
                score = max(score, 1.0)
                matched_subject = subject
        if score == 0:
            updated.append(item)
            continue
        features = item.get("features") if isinstance(item.get("features"), dict) else {}
        updated.append(
            {
                **item,
                "features": {**features, "user_preference": score},
                "user_preference_score": score,
                "user_preference_subject": matched_subject,
            }
        )
    return updated


def _dedupe_resume_candidates(candidates: list[dict]) -> list[dict]:
    """Keep one candidate per exact rendered section/text block."""
    best: dict[tuple[str, str], dict] = {}
    for item in candidates:
        key = _candidate_text_key(item)
        if not key[1]:
            continue
        current = best.get(key)
        if current is None or (
            _candidate_relevance(item),
            float(item.get("features", {}).get("evidence_strength", 0.0)),
            -int(item.get("length_cost", 0)),
        ) > (
            _candidate_relevance(current),
            float(current.get("features", {}).get("evidence_strength", 0.0)),
            -int(current.get("length_cost", 0)),
        ):
            best[key] = item
    ordered: list[dict] = []
    emitted: set[tuple[str, str]] = set()
    for item in candidates:
        key = _candidate_text_key(item)
        if key in best and key not in emitted:
            ordered.append(best[key])
            emitted.add(key)
    return ordered


def _project_slot_count(projects: list[dict], remaining_budget: int, *, max_projects: int) -> int:
    if not projects or max_projects <= 0:
        return 0
    if remaining_budget <= 0:
        return 1
    costs = sorted(max(180, min(520, int(item.get("length_cost", 0) or 0))) for item in projects)
    typical_cost = costs[len(costs) // 2] if costs else 320
    return max(1, min(max_projects, max(1, remaining_budget // max(240, typical_cost))))


def _apply_resume_selection_policy(
    candidates: list[dict],
    *,
    content_budget: int,
    min_project_score: float = 0.12,
    max_projects: int = 3,
    required_project_count: int | None = None,
    exclusive_project_subject: str | None = None,
    user_requirement_item_ids: set[str] | tuple[str, ...] | list[str] | None = None,
) -> tuple[list[dict], list[str], dict]:
    """Apply section-level resume policy before LLM drafting and CP-SAT planning.

    Static sections are evidence display, not JD optimization. Internships keep
    at least one complete entry. Projects are first ranked by whole-JD fit, then
    capped by estimated remaining space so unrelated or stale projects cannot
    drift into the resume merely because they were present earlier.
    """
    if content_budget < 1:
        raise ValueError("content_budget must be at least 1")
    candidates = _dedupe_resume_candidates(candidates)
    user_required_ids = {str(item).strip() for item in (user_requirement_item_ids or ()) if str(item).strip()}
    for item in candidates:
        if str(item.get("item_id") or "") in user_required_ids:
            item["user_requirement_required"] = True
    static_sections = {"education", "research", "awards"}
    static = [item for item in candidates if str(item.get("section") or "").casefold() in static_sections]
    skills = [item for item in candidates if str(item.get("section") or "").casefold() == "skills"]
    internships = [item for item in candidates if str(item.get("section") or "").casefold() == "internships"]
    projects = [item for item in candidates if str(item.get("section") or "").casefold() == "projects"]
    if exclusive_project_subject:
        projects = [
            item
            for item in projects
            if _project_matches_subject(item, exclusive_project_subject)
        ]
        if not projects:
            raise ValueError(
                f"未找到用户指定的个人项目“{exclusive_project_subject}”，无法按要求生成简历。"
            )
    uses_project_matrix = any(
        bool(item.get("jd_fact_blocks"))
        or int(item.get("threshold_fact_count", 0) or 0) > 0
        for item in projects
    )
    if not uses_project_matrix and any("jd_fact_pairs" in item for item in projects):
        projects = [
            item
            for item in projects
            if item.get("jd_fact_pairs") or exclusive_project_subject
        ]

    required_ids: list[str] = []
    for item in static:
        item["required_for_resume"] = True
        if item.get("item_id"):
            required_ids.append(str(item["item_id"]))
    if skills:
        preferred_skill = max(
            skills,
            key=lambda item: (
                str(item.get("evidence_type") or "").casefold() == "skills",
                not bool(item.get("legacy_split")),
                "技能" in str(item.get("evidence_label") or ""),
                int(item.get("length_cost", 0) or 0),
                float(item.get("features", {}).get("evidence_strength", 0.0)),
            ),
        )
        preferred_skill["required_for_resume"] = True
        if preferred_skill.get("item_id"):
            required_ids.append(str(preferred_skill["item_id"]))
        static = [*static, preferred_skill]

    selected_internships: list[dict] = []
    if internships:
        preferred = max(
            internships,
            key=lambda item: (
                float(item.get("user_preference_score") or 0.0),
                bool(item.get("matched_role")),
                float(item.get("role_match_score", item.get("match_confidence", 0.0))),
                _candidate_relevance(item),
                -int(item.get("length_cost", 0)),
            ),
        )
        preferred["required_for_resume"] = True
        if preferred.get("item_id"):
            required_ids.append(str(preferred["item_id"]))
        selected_internships = [preferred]

    required_non_project_length = sum(
        int(item.get("length_cost", 0) or 0)
        for item in [*static, *selected_internships]
        if item.get("required_for_resume")
    )
    remaining_budget = int(content_budget) - required_non_project_length
    def project_sort_key(item: dict) -> tuple:
        if uses_project_matrix:
            return (
                bool(item.get("user_requirement_required")),
                int(item.get("threshold_fact_count", 0) or 0),
                float(item.get("project_match_average", 0.0) or 0.0),
                float(item.get("user_preference_score") or 0.0),
                _candidate_relevance(item),
                float(item.get("features", {}).get("evidence_strength", 0.0)),
                float(item.get("features", {}).get("result_strength", 0.0)),
                -int(item.get("length_cost", 0)),
            )
        return (
            bool(item.get("user_requirement_required")),
            float(item.get("user_preference_score") or 0.0),
            _candidate_relevance(item),
            float(item.get("features", {}).get("evidence_strength", 0.0)),
            float(item.get("features", {}).get("result_strength", 0.0)),
            -int(item.get("length_cost", 0)),
        )

    ranked_projects = sorted(
        projects if uses_project_matrix else [item for item in projects if _candidate_relevance(item) >= min_project_score],
        key=project_sort_key,
        reverse=True,
    )
    unique_ranked_projects: list[dict] = []
    seen_project_groups: set[str] = set()
    for item in ranked_projects:
        group = str(item.get("redundancy_group") or item.get("item_id") or "").strip()
        if group and group in seen_project_groups:
            continue
        unique_ranked_projects.append(item)
        if group:
            seen_project_groups.add(group)
    if required_project_count is not None:
        project_slots = min(required_project_count, len(unique_ranked_projects))
        selected_projects = unique_ranked_projects[:project_slots]
        for item in selected_projects:
            item["required_for_resume"] = True
            if item.get("item_id"):
                required_ids.append(str(item["item_id"]))
    else:
        if internships and len(unique_ranked_projects) >= 2:
            project_slots = min(max_projects, 2, len(unique_ranked_projects))
        else:
            project_slots = _project_slot_count(unique_ranked_projects, remaining_budget, max_projects=max_projects)
        selected_projects = unique_ranked_projects[:project_slots]
        for item in selected_projects:
            item["required_for_resume"] = True
            if item.get("item_id"):
                required_ids.append(str(item["item_id"]))

    selected_ids = {
        str(item.get("item_id") or "")
        for item in [*static, *selected_internships, *selected_projects]
        if item.get("item_id")
    }
    # Preserve policy order so the first project is the highest-ranked one;
    # static sections and the preferred internship remain ahead of projects.
    scoped = [*static, *selected_internships, *selected_projects]
    diagnostics = {
        "required_non_project_length": required_non_project_length,
        "remaining_project_budget": remaining_budget,
        "project_slots": project_slots,
        "required_project_count": required_project_count,
        "user_requirement_item_ids": sorted(user_required_ids),
        "exclusive_project_subject": exclusive_project_subject or "",
        "customization_matches": [
            {
                "item_id": item.get("item_id"),
                "section": item.get("section"),
                "subject": item.get("user_preference_subject"),
                "score": float(item.get("user_preference_score") or 0.0),
            }
            for item in candidates
            if float(item.get("user_preference_score") or 0.0) != 0.0
        ],
        "selected_project_item_ids": [item.get("item_id") for item in selected_projects],
        "project_scores": [
            {
                "item_id": item.get("item_id"),
                "section": item.get("section"),
                "score": round(float(item.get("project_match_average", _candidate_relevance(item)) or 0.0), 6),
                "threshold_fact_count": int(item.get("threshold_fact_count", 0) or 0),
                "title": str(item.get("proposed_text") or "").split("；", 1)[0],
            }
            for item in ranked_projects
        ],
    }
    return scoped, list(dict.fromkeys(required_ids)), diagnostics


def _llm_resume_draft(
    *,
    runtime,
    root: Path,
    jd_blocks: list[dict],
    candidates: list[dict],
    internship_candidates: list[dict] | None = None,
    required_item_ids: list[str] | tuple[str, ...] | None = None,
    template,
    customization_request: str = "",
    required_project_count: int | None = None,
    required_project_design_count: int | None = None,
    resume_items=None,
    evidence_policy: ResumeEvidencePolicy = FULL_RESUME_EVIDENCE_POLICY,
):
    from smart_agent.career import (
        CareerStore,
        LLMResumeDraftGenerator,
        ResumeDocumentBuilder,
        create_career_client,
    )
    from smart_agent.career.resume import SKILLS_CLAIM_MAX_LENGTH, project_claim_max_length

    try:
        resume_client = create_career_client(
            runtime,
            "resume_generator",
            audit_root=root / "audit",
            required=False,
        )
    except ValueError as exc:
        message = str(exc)
        if (
            "Career resume_generator requires model, base URL, and API key" not in message
            and "Career resume_generator model is not configured" not in message
        ):
            raise
        resume_client = None
    generator = LLMResumeDraftGenerator(resume_client) if resume_client is not None else None
    if generator is None:
        raise ValueError(
            "Career resume generation requires a configured resume_generator model; "
            "fixed candidate-text fallback is disabled."
        )
    ordered_candidates: list[dict] = []
    seen_candidate_ids: set[str] = set()
    for item in [*candidates, *(internship_candidates or [])]:
        item_id = str(item.get("item_id") or "").strip()
        if item_id and item_id not in seen_candidate_ids:
            ordered_candidates.append(item)
            seen_candidate_ids.add(item_id)
    resume_item_by_id = {
        str(item.item_id): item
        for item in (resume_items or ())
        if str(getattr(item, "item_id", "") or "").strip()
    }
    claim_builder = ResumeDocumentBuilder(
        CareerStore(root),
        enforce_project_evidence=evidence_policy.enforce_project_claim_validation,
    ) if resume_item_by_id else None
    project_blocks = [item for item in ordered_candidates if item.get("section") == "projects"]
    effective_project_count = (
        min(required_project_count, len(project_blocks))
        if required_project_count is not None
        else len(project_blocks)
    )
    project_claim_limit = project_claim_max_length(max(1, effective_project_count))
    claim_limits = {
        str(item.get("item_id") or ""): (
            project_claim_limit
            if item.get("section") == "projects"
            else SKILLS_CLAIM_MAX_LENGTH
            if item.get("section") == "skills"
            else 600
        )
        for item in ordered_candidates
        if item.get("item_id")
    }

    def validate_claim(item_id: str, text: str) -> None:
        if claim_builder is None:
            return
        resume_item = resume_item_by_id.get(str(item_id or ""))
        if resume_item is None:
            raise ValueError(f"Resume generator returned an unknown item: {item_id}")
        if (
            not evidence_policy.enforce_project_claim_validation
            and str(getattr(resume_item, "section", "")).casefold().endswith("projects")
        ):
            return
        claim_builder.validate_claim_text(
            text,
            resume_item,
            max_length=claim_limits.get(str(item_id), 600),
        )

    role_title = next(
        (
            str(block.get("text") or "").split("：", 1)[-1].strip()
            for block in jd_blocks
            if block.get("requirement_id") == "__role_title__"
        ),
        "",
    )
    job_description = "\n".join(
        str(block.get("text") or "").strip()
        for block in jd_blocks
        if str(block.get("text") or "").strip()
    )
    layout = template.layout_spec if template else {}
    project_jobs = [
        {
            "role_title": role_title,
            "job_description": job_description,
            "project_name": str(item.get("project_name") or item.get("evidence_label") or ""),
            "project_description": str(
                item.get("project_description") or item.get("proposed_text") or ""
            ),
            "technology_stack": str(item.get("technology_stack") or ""),
            "jd_fact_pairs": list(item.get("jd_fact_pairs") or []),
            "jd_fact_blocks": list(item.get("jd_fact_blocks") or []),
            "project_markdown": str(
                item.get("project_markdown") or item.get("proposed_text") or ""
            ),
            "template_layout": {**layout, "module": "projects"},
            "claim_max_length": project_claim_limit,
            "required_project_design_count": required_project_design_count,
            "user_instruction": customization_request,
            "claim_validator": (
                lambda text, item_id=str(item.get("item_id") or ""): validate_claim(item_id, text)
            )
            if claim_builder is not None
            and (
                evidence_policy.enforce_project_claim_validation
                or str(item.get("section") or "").casefold() != "projects"
            )
            else None,
            "enforce_project_evidence": evidence_policy.enforce_project_prompt,
        }
        for item in project_blocks
    ]
    project_results = generator.synthesize_projects(project_jobs) if project_jobs else []
    validation_feedback: list[str] = []
    project_claims: dict[str, str] = {}
    for item, synthesis in zip(project_blocks, project_results, strict=True):
        validation_feedback.extend(
            str(feedback).strip()
            for feedback in synthesis.pop("validation_feedback", [])
            if str(feedback).strip()
        )
        design_text = " ".join(
            f"{index}、{point}"
            for index, point in enumerate(synthesis["project_design"], start=1)
        )
        project_claims[str(item["item_id"])] = (
            f"项目名称：{synthesis['project_name']}；项目简述：{synthesis['project_description']}；"
            f"技术栈：{'、'.join(synthesis.get('technology_stack') or ['未提供'])}；"
            f"项目设计：{design_text}"
        )

    resume_blocks = [
        {
            "item_id": str(item["item_id"]),
            "section": str(item.get("section") or ""),
            "source_kind": (
                "project_draft" if item.get("section") == "projects" else "confirmed_fact"
            ),
            "format_version": "project_v2" if item.get("section") == "projects" else "",
            "text": (
                project_claims[str(item["item_id"])]
                if item.get("section") == "projects"
                else str(item.get("proposed_text") or "").strip()
            ),
        }
        for item in ordered_candidates
        if item.get("item_id")
    ]
    required = [str(item).strip() for item in (required_item_ids or ()) if str(item).strip()]
    if not required:
        required = [block["item_id"] for block in resume_blocks]
    assembled = generator.assemble_resume(
        jd_blocks=jd_blocks,
        resume_blocks=resume_blocks,
        required_item_ids=required,
        template_layout=layout,
        user_instruction=customization_request,
        claim_max_lengths=claim_limits,
        claim_validator=validate_claim if claim_builder is not None else None,
        enforce_project_evidence=evidence_policy.enforce_project_prompt,
    )
    validation_feedback.extend(
        str(feedback).strip()
        for feedback in assembled.pop("validation_feedback", [])
        if str(feedback).strip()
    )
    selected_ids = list(assembled["selected_item_ids"])
    claims = list(assembled["claims"])
    if required_project_count is not None:
        available_project_count = sum(
            1 for item in candidates if item.get("section") == "projects"
        )
        expected_project_count = min(required_project_count, available_project_count)
        project_selected = [
            item_id
            for item_id in selected_ids
            if any(
                str(item.get("item_id") or "") == item_id and item.get("section") == "projects"
                for item in candidates
            )
        ]
        if len(project_selected) != expected_project_count:
            raise ValueError(
                f"Resume generator did not produce the required {expected_project_count} personal projects"
            )
    return {
        "selected_item_ids": selected_ids,
        "claims": claims,
        "validation_feedback": validation_feedback,
    }


def _match_resume_run(
    run,
    *,
    root: Path,
    store,
    allowed_requirement_ids: Iterable[str] | None = None,
):
    """Match each requirement independently, reusing only compatible cache entries."""
    from smart_agent.career import (
        EvidenceMatch,
        JDRequirement,
        ProjectDescriptionDocument,
        RequirementMatchCacheService,
        ResumeRunService,
        record_from_dict,
        record_to_dict,
    )

    _current_store, _material_store, _candidate_store, manifests, _candidates, evidence_records, descriptions, catalog_version = (
        _career_material_snapshot()
    )
    allowed_ids = (
        {str(item).strip() for item in allowed_requirement_ids if str(item).strip()}
        if allowed_requirement_ids is not None
        else None
    )
    requirements = tuple(
        store.get(ref.record_id, ref.revision, expected_type=JDRequirement)
        for ref in run.requirement_refs
        if allowed_ids is None or ref.record_id in allowed_ids
    )
    cache_service = RequirementMatchCacheService(store)
    matched: list[EvidenceMatch] = []
    cache_hits: list[str] = []
    misses: list[JDRequirement] = []
    for requirement in requirements:
        cache = cache_service.get_reusable(
            requirement,
            manifests=manifests,
            material_catalog_version=catalog_version,
            evidence_records=evidence_records,
            retrieval_backend="",
            embedding_model="",
            config_version="fine-point-v3",
        )
        if cache is None:
            misses.append(requirement)
            continue
        matched.extend(cache_service.reuse(cache, requirement))
        cache_hits.append(requirement.requirement_id)

    retrieval = {
        "retrieval_mode": "requirement_cache" if not misses else "cache_plus_retrieval",
        "cache_hit_requirement_ids": cache_hits,
        "cache_miss_requirement_ids": [item.requirement_id for item in misses],
        "matches": [],
    }
    if misses:
        fresh = _execute_career_tool(
            "retrieve_career_evidence",
            {
                "requirement_refs": [
                    {"record_id": item.requirement_id, "revision": item.revision}
                    for item in misses
                ],
                "evidence_refs": [
                    {"record_id": ref.record_id, "revision": ref.revision}
                    for ref in run.evidence_snapshot_refs
                ],
            },
        )
        store = type(store)(root)
        cache_service = RequirementMatchCacheService(store)
        by_requirement: dict[str, list[EvidenceMatch]] = {}
        for payload in fresh.get("matches", []):
            match = record_from_dict(payload)
            by_requirement.setdefault(match.requirement_ref.record_id, []).append(match)
            matched.append(match)
        for requirement in misses:
            RequirementMatchCacheService(store).persist(
                requirement,
                tuple(by_requirement.get(requirement.requirement_id, ())),
                manifests=manifests,
                material_catalog_version=catalog_version,
                evidence_records=evidence_records,
                retrieval_backend=str(fresh.get("retrieval_mode") or "career-retriever"),
                embedding_model=str(fresh.get("embedding_provider") or ""),
                config_version="fine-point-v3",
            )
        retrieval["resume_source_candidates"] = fresh.get("resume_source_candidates", [])
    run = ResumeRunService(store).attach_matches(run.resume_run_id, tuple(matched))
    from smart_agent.career import ProjectDescriptionService, ProjectLibraryStore, ProjectSourceStatus

    project_library = ProjectLibraryStore(root / "project-library", workspace=ROOT)
    project_descriptions_service = ProjectDescriptionService(store)
    for project_source in project_library.list():
        refreshed_source = project_library.get(project_source.project_id, refresh=True)
        if refreshed_source.status is ProjectSourceStatus.NEEDS_REANALYSIS:
            project_descriptions_service.mark_source_changed(
                refreshed_source.project_id,
                current_project_version=refreshed_source.project_version,
            )
    descriptions = store.list_latest(ProjectDescriptionDocument)
    runtime = load_runtime_config(ROOT)
    retrieval["project_deep_scan_decisions"] = [
        decision.to_dict()
        for decision in _project_deep_dive_service().plan(
            requirements,
            matches=matched,
            projects=descriptions,
            budget=int(runtime.get("careerProjectDeepScanBudget", 3)),
        )
    ]
    retrieval["matches"] = [record_to_dict(match) for match in matched]
    retrieval["requirement_count"] = len(requirements)
    retrieval["gap_count"] = sum(match.evidence_ref is None for match in matched)
    retrieval["material_catalog_version"] = catalog_version
    return run, retrieval


def _persist_resume_template(store, template):
    from smart_agent.career import RecordNotFoundError, ResumeTemplate

    try:
        current = store.get(template.template_id, expected_type=ResumeTemplate)
    except RecordNotFoundError:
        return store.put(template)
    same_format = (
        current.name == template.name
        and current.source_kind == template.source_kind
        and current.original_name == template.original_name
        and current.layout_spec == template.layout_spec
        and current.content_access == template.content_access
        and current.resume_source_ref == template.resume_source_ref
        and [ref.content_hash for ref in current.source_refs] == [ref.content_hash for ref in template.source_refs]
    )
    if not same_format:
        # Bundled templates may gain new format metadata as the renderer evolves.
        # Keep the append-only store intact while refreshing only the derived layout.
        if (
            current.name == template.name
            and current.source_kind == template.source_kind
            and current.original_name == template.original_name
            and current.content_access == template.content_access
            and current.resume_source_ref == template.resume_source_ref
            and [ref.content_hash for ref in current.source_refs] == [ref.content_hash for ref in template.source_refs]
        ):
            return store.revise_record(
                template.template_id,
                actor="system:template-layout-migration",
                reason="Refresh format-only layout metadata",
                changes={"layout_spec": template.layout_spec},
                expected_revision=current.revision,
            )
        raise ValueError("A resume template with this identity already exists with different format data.")
    return current


def _list_resume_templates_lenient(store):
    from smart_agent.career import CareerStoreError, ResumeTemplate

    try:
        return tuple(store.list_latest(ResumeTemplate))
    except CareerStoreError:
        pass

    try:
        store._refresh_index()
        entries = sorted(store._index.get("records", {}).items())
    except Exception:
        return ()
    templates = []
    for identity, entry in entries:
        if entry.get("record_type") != "ResumeTemplate":
            continue
        try:
            templates.append(store.get(identity, expected_type=ResumeTemplate))
        except CareerStoreError:
            continue
    return tuple(templates)


_VISIBLE_RESUME_TEMPLATE_NAMES = ("模板1", "模板2", "模板3")
_VISIBLE_RESUME_TEMPLATE_NAME_SET = frozenset(_VISIBLE_RESUME_TEMPLATE_NAMES)


def _is_visible_resume_template(template) -> bool:
    """Return true for the three curated bundled templates shown in the UI."""

    return (
        template.source_kind == "template_upload"
        and str(template.name or "").strip() in _VISIBLE_RESUME_TEMPLATE_NAME_SET
        and Path(template.original_name).suffix.casefold() == ".pdf"
    )


def _visible_resume_templates(store):
    """Curated template choices: 模板1 / 模板2 / 模板3 only."""

    by_name = {}
    for template in _list_resume_templates_lenient(store):
        if not _is_visible_resume_template(template):
            continue
        name = str(template.name or "").strip()
        current = by_name.get(name)
        if current is None or float(template.updated_at or 0) > float(current.updated_at or 0):
            by_name[name] = template
    return tuple(
        by_name[name]
        for name in _VISIBLE_RESUME_TEMPLATE_NAMES
        if name in by_name
    )


def _ensure_workspace_resume_templates(store) -> None:
    """Register bundled PDF templates as format-only choices for the UI."""
    from smart_agent.career import create_uploaded_template

    template_dir = ROOT / "简历模板"
    if not template_dir.is_dir():
        return
    for path in sorted(template_dir.glob("*.pdf")):
        try:
            template = create_uploaded_template(
                name=path.stem,
                filename=path.name,
                content=path.read_bytes(),
            )
            _persist_resume_template(store, template)
        except (OSError, ValueError, RuntimeError):
            # A single invalid optional template must not block resume work. Do
            # not re-register "（内置）" aliases; the UI exposes only 模板1/2/3.
            continue


def _resolve_resume_template(store, material_store, template_id: str = ""):
    from smart_agent.career import RecordNotFoundError, ResumeTemplate

    _ensure_workspace_resume_templates(store)
    identity = str(template_id or "").strip()
    if identity:
        try:
            return store.get(identity, expected_type=ResumeTemplate)
        except RecordNotFoundError as exc:
            raise HTTPException(status_code=404, detail="The selected resume template no longer exists.") from exc

    visible = _visible_resume_templates(store)
    return visible[0] if visible else None


@app.get("/api/career/resume/templates")
async def career_resume_templates():
    from smart_agent.career import ResumeTemplate, record_to_dict

    _root, store, _material_store, _candidate_store = _career_stores()
    _ensure_workspace_resume_templates(store)
    templates = _visible_resume_templates(store)
    payload = []
    for item in templates:
        record = record_to_dict(item)
        if item.source_kind == "template_upload" and Path(item.original_name).suffix.casefold() == ".pdf":
            record["preview_url"] = f"/api/career/resume/templates/{item.template_id}/preview"
            record["preview_image_url"] = (
                f"/api/career/resume/templates/{item.template_id}/preview.png"
            )
        payload.append(record)
    return {"templates": payload}


@app.get("/api/career/resume/templates/{template_id}/preview")
async def career_resume_template_preview(template_id: str):
    from smart_agent.career import RecordNotFoundError, ResumeTemplate

    _root, store, _material_store, _candidate_store = _career_stores()
    _ensure_workspace_resume_templates(store)
    try:
        template = store.get(template_id, expected_type=ResumeTemplate)
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Resume template not found.") from exc
    template_root = (ROOT / "简历模板").resolve()
    path = (template_root / Path(template.original_name).name).resolve()
    if (
        template.source_kind != "template_upload"
        or not path.is_relative_to(template_root)
        or path.suffix.casefold() != ".pdf"
        or not path.is_file()
    ):
        raise HTTPException(status_code=404, detail="Template PDF preview is unavailable.")
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline", "Cache-Control": "no-store"},
    )


@app.get("/api/career/resume/templates/{template_id}/preview.png")
async def career_resume_template_preview_image(template_id: str):
    from smart_agent.career import RecordNotFoundError, ResumeTemplate

    _root, store, _material_store, _candidate_store = _career_stores()
    _ensure_workspace_resume_templates(store)
    try:
        template = store.get(template_id, expected_type=ResumeTemplate)
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Resume template not found.") from exc
    template_root = (ROOT / "简历模板").resolve()
    path = (template_root / Path(template.original_name).name).resolve()
    if (
        template.source_kind != "template_upload"
        or not path.is_relative_to(template_root)
        or path.suffix.casefold() != ".pdf"
        or not path.is_file()
    ):
        raise HTTPException(status_code=404, detail="Template PDF preview is unavailable.")
    return _pdf_page_response(path, 0)


@app.get("/api/career/resume/templates/{template_id}/preview-info")
async def career_resume_template_preview_info(template_id: str):
    from smart_agent.career import RecordNotFoundError, ResumeTemplate

    _root, store, _material_store, _candidate_store = _career_stores()
    _ensure_workspace_resume_templates(store)
    try:
        template = store.get(template_id, expected_type=ResumeTemplate)
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Resume template not found.") from exc
    template_root = (ROOT / "简历模板").resolve()
    path = (template_root / Path(template.original_name).name).resolve()
    if template.source_kind != "template_upload" or not path.is_relative_to(template_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Template PDF preview is unavailable.")
    try:
        from pypdf import PdfReader
        page_count = len(PdfReader(str(path)).pages)
    except (ImportError, OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Template preview metadata failed.") from exc
    encoded_id = urllib.parse.quote(str(template_id), safe="")
    return {"page_count": page_count, "page_urls": [f"/api/career/resume/templates/{encoded_id}/pages/{i + 1}.png" for i in range(page_count)]}


@app.get("/api/career/resume/templates/{template_id}/pages/{page_number}.png")
async def career_resume_template_page(template_id: str, page_number: int):
    from smart_agent.career import RecordNotFoundError, ResumeTemplate

    _root, store, _material_store, _candidate_store = _career_stores()
    _ensure_workspace_resume_templates(store)
    try:
        template = store.get(template_id, expected_type=ResumeTemplate)
    except RecordNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Resume template not found.") from exc
    template_root = (ROOT / "简历模板").resolve()
    path = (template_root / Path(template.original_name).name).resolve()
    if template.source_kind != "template_upload" or not path.is_relative_to(template_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Template PDF preview is unavailable.")
    return _pdf_page_response(path, page_number - 1)


@app.post("/api/career/resume/templates")
async def career_resume_template_upload(file: UploadFile, name: str = Form("")):
    from smart_agent.career import create_uploaded_template, record_to_dict

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Resume template exceeds 10 MB.")
    try:
        template = create_uploaded_template(
            name=name.strip() or Path(file.filename or "resume-template").stem,
            filename=file.filename or "resume-template",
            content=content,
        )
        _root, store, _material_store, _candidate_store = _career_stores()
        persisted = _persist_resume_template(store, template)
        return {"template": record_to_dict(persisted)}
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _career_resume_build(
    data: dict,
    *,
    evidence_policy: ResumeEvidencePolicy = FULL_RESUME_EVIDENCE_POLICY,
):
    """Run evidence retrieval, planning, and deterministic resume generation.

    This low-level endpoint produces a draft only. Browser delivery uses the
    dedicated endpoint below to run generation, verification, and rendering.
    """
    from smart_agent.career import JDDocument

    header = data.get("header")
    if not isinstance(header, dict) or not str(header.get("full_name") or "").strip():
        raise HTTPException(status_code=400, detail="header.full_name is required")
    try:
        customization_request = _resume_customization_request(
            data.get("customization_request") or data.get("message")
        )
        required_project_count = _resume_project_count_requirement(customization_request)
        required_project_design_count = _resume_project_design_count_requirement(customization_request)
        exclusive_project_subject = _resume_exclusive_project_subject(customization_request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    root, store, material_store, _candidate_store = _career_stores()
    from smart_agent.career import (
        JDRequirementCritic,
        PersonalEvidence,
        ResumeDocumentBuilder,
        ResumeRunService,
        build_jd_coverage_matrix,
        build_resume_jd_blocks,
        build_supported_customization_instruction,
        create_career_client,
        parse_resume_user_requirements,
        record_to_dict,
        remove_rejected_request_echo,
    )
    from smart_agent.career.resume import project_claim_max_length
    from smart_agent.career.resume_requirements import dedupe_resume_claim_fields

    run = _resolve_resume_run(store, str(data.get("resume_run_id") or ""))
    run = _refresh_resume_project_snapshot(root, store, run)
    _validate_resume_run_evidence_snapshot(store, run)
    try:
        template = _resolve_resume_template(store, material_store, str(data.get("template_id") or ""))
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    jd = store.get(run.jd_ref.record_id, run.jd_ref.revision, expected_type=JDDocument)
    requirement_ids = [ref.record_id for ref in run.requirement_refs]
    if not requirement_ids:
        raise HTTPException(status_code=409, detail="The selected JD has no normalized requirements.")
    # Filter stale requirement refs before retrieval as well as before
    # generation.  Otherwise an imported user-request row can influence
    # evidence ranking even if the final prompt later excludes it.
    preliminary_user_requirements = parse_resume_user_requirements(
        customization_request,
        allow_unsupported_project_requirements=not evidence_policy.enforce_project_user_requirement_filter,
    )
    preliminary_jd_blocks = build_resume_jd_blocks(
        jd,
        store,
        preliminary_user_requirements,
    )
    employer_requirement_ids = {
        str(block.get("requirement_id") or "")
        for block in preliminary_jd_blocks
        if block.get("kind") == "requirement"
    }
    if not employer_requirement_ids:
        raise HTTPException(status_code=409, detail="The selected JD has no usable employer requirements.")
    if not run.evidence_snapshot_refs:
        raise HTTPException(status_code=409, detail="Confirm personal facts before analyzing the JD.")
    run, retrieved = _match_resume_run(
        run,
        root=root,
        store=store,
        allowed_requirement_ids=employer_requirement_ids,
    )
    store = type(store)(root)
    _validate_resume_run_evidence_snapshot(store, run)
    _validate_resume_run_matches(store, run)
    runtime = load_runtime_config(ROOT)
    confidence_threshold = float(runtime.get("careerResumeMatchConfidenceThreshold", 0.72))
    has_confirmed_project_library_evidence = any(
        _is_project_library_evidence(
            store.get(ref.record_id, ref.revision, expected_type=PersonalEvidence)
        )
        for ref in run.evidence_snapshot_refs
    )
    candidates = _resume_candidate_items(
        store,
        jd,
        match_refs=run.match_refs,
        evidence_refs=run.evidence_snapshot_refs,
        confidence_threshold=confidence_threshold,
        include_unmatched_projects=True,
        exclude_legacy_projects=has_confirmed_project_library_evidence,
        allowed_requirement_ids=employer_requirement_ids,
    )
    if not candidates:
        raise HTTPException(status_code=409, detail="No confirmed evidence matches the selected JD.")
    # Apply ordering preferences only after classifying the request.  Feeding
    # raw customization text into ranking lets an unsupported business claim
    # influence project selection even though it must never influence content.
    preliminary_requirements = parse_resume_user_requirements(
        customization_request,
        candidates,
        allow_unsupported_project_requirements=not evidence_policy.enforce_project_user_requirement_filter,
    )
    supported_preference = build_supported_customization_instruction(preliminary_requirements)
    candidates = _apply_resume_customization_preference(candidates, supported_preference)
    from smart_agent.career import ResumeCandidateBuilder

    preview_items = ResumeCandidateBuilder(store).build(candidates)
    candidates = [
        {**candidate, "item_id": item.item_id}
        for candidate, item in zip(candidates, preview_items, strict=True)
    ]
    user_requirements = parse_resume_user_requirements(
        customization_request,
        candidates,
        allow_unsupported_project_requirements=not evidence_policy.enforce_project_user_requirement_filter,
    )
    supported_requirement_item_ids = {
        evidence_ref
        for requirement in user_requirements
        if requirement.decision in {"accepted", "partially_accepted"}
        for evidence_ref in requirement.evidence_refs
    }
    if not evidence_policy.enforce_project_user_requirement_filter:
        # Project-ablation requests may have no matched evidence refs. Still
        # pin the targeted project into selection so its request reaches the
        # per-project LLM instead of disappearing during ranking.
        for requirement in user_requirements:
            if requirement.support_status != "project_ablation" or not requirement.target_project:
                continue
            target = requirement.target_project.casefold()
            for item in candidates:
                if str(item.get("section") or "").casefold() != "projects":
                    continue
                candidate_text = " ".join(
                    str(item.get(field) or "")
                    for field in ("project_name", "evidence_label", "project_description", "proposed_text")
                ).casefold()
                if target in candidate_text:
                    item_id = str(item.get("item_id") or "").strip()
                    if item_id:
                        supported_requirement_item_ids.add(item_id)
    for item in candidates:
        if str(item.get("item_id") or "") in supported_requirement_item_ids:
            item["user_requirement_required"] = True
    content_budget = int(data.get("content_budget", 1200))
    try:
        candidates, required_resume_ids, selection_policy = _apply_resume_selection_policy(
            candidates,
            content_budget=content_budget,
            min_project_score=float(runtime.get("careerResumeProjectMinScore", 0.12)),
            max_projects=int(runtime.get("careerResumeMaxProjects", 3)),
            required_project_count=required_project_count,
            exclusive_project_subject=exclusive_project_subject,
            user_requirement_item_ids=supported_requirement_item_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    internship_candidates = [item for item in candidates if item.get("section") == "internships"]
    supported_instruction = build_supported_customization_instruction(user_requirements)
    if evidence_policy.enforce_project_user_requirement_filter:
        llm_customization_request, ignored_numeric_requests = _strip_unsupported_numeric_requests(
            supported_instruction,
            candidates,
        )
    else:
        # B2 keeps project-ablation requests in the supported instruction, while
        # rejected non-project requests remain filtered by the normal parser.
        # This preserves the non-project fact gate without hiding the intended
        # project safety ablation from any project LLM call.
        llm_customization_request, ignored_numeric_requests = supported_instruction, []
    rejected_user_requests = [
        item.text
        for item in user_requirements
        if item.decision == "rejected"
    ]

    jd_blocks = build_resume_jd_blocks(jd, store, user_requirements)
    if not any(block.get("kind") == "requirement" for block in jd_blocks):
        raise HTTPException(status_code=409, detail="The selected JD has no usable employer requirements.")
    stage_started = time.time()
    try:
        llm_draft = _llm_resume_draft(
            runtime=runtime,
            root=root,
            jd_blocks=jd_blocks,
            candidates=candidates,
            internship_candidates=internship_candidates,
            required_item_ids=required_resume_ids,
            template=template,
            customization_request=llm_customization_request,
            required_project_count=required_project_count,
            required_project_design_count=required_project_design_count,
            resume_items=preview_items,
            evidence_policy=evidence_policy,
        )
    except Exception as exc:
        _record_agent_stage(
            resume_run_id=run.resume_run_id,
            agent_type="resume_generator",
            agent_name="Resume Generator Agent",
            stage="draft",
            status="failed",
            started_at=stage_started,
            error=str(exc),
        )
        raise
    _record_agent_stage(
        resume_run_id=run.resume_run_id,
        agent_type="resume_generator",
        agent_name="Resume Generator Agent",
        stage="draft",
        status="completed",
        started_at=stage_started,
    )
    critic_stage_started = time.time()
    critic_result: dict[str, object]
    try:
        critic_client = None
        try:
            critic_client = create_career_client(
                runtime,
                "jd_critic",
                audit_root=root / "audit",
                required=False,
            )
        except ValueError:
            critic_client = None
        claim_limits = {
            str(item.get("item_id") or ""): (
                project_claim_max_length(
                    max(1, len([candidate for candidate in candidates if candidate.get("section") == "projects"]))
                )
                if item.get("section") == "projects"
                else 600
            )
            for item in candidates
            if item.get("item_id")
        }
        critic_store = type(store)(root)
        critic_builder = ResumeDocumentBuilder(
            critic_store,
            enforce_project_evidence=evidence_policy.enforce_project_claim_validation,
        )
        critic_items = {
            str(item.item_id): item
            for item in preview_items
            if str(getattr(item, "item_id", "") or "").strip()
        }

        def validate_critic_claim(item_id: str, text: str) -> None:
            item = critic_items.get(str(item_id))
            if item is None:
                raise ValueError(f"JD critic returned unknown item: {item_id}")
            if (
                not evidence_policy.enforce_project_claim_validation
                and str(getattr(item, "section", "")).casefold().endswith("projects")
            ):
                return
            critic_builder.validate_claim_text(
                text,
                item,
                max_length=claim_limits.get(str(item_id), 600),
            )

        selected_candidates = [
            item
            for item in candidates
            if str(item.get("item_id") or "") in set(llm_draft.get("selected_item_ids") or ())
        ]
        critic_result = JDRequirementCritic(critic_client).review(
            jd_blocks=jd_blocks,
            claims=llm_draft.get("claims", []),
            evidence_blocks=selected_candidates,
            user_requirements=user_requirements,
            claim_validator=validate_critic_claim,
        )
        revised_claims = critic_result.get("claims") if isinstance(critic_result, dict) else None
        if evidence_policy.apply_critic_revisions and isinstance(revised_claims, list) and revised_claims:
            llm_draft["claims"] = revised_claims
        if not evidence_policy.apply_critic_revisions and isinstance(critic_result, dict):
            critic_result = {
                **critic_result,
                "mode": "shadow",
                "applied_revisions": False,
            }
        _record_agent_stage(
            resume_run_id=run.resume_run_id,
            agent_type="jd_critic",
            agent_name="JD Critic Agent",
            stage="jd_coverage",
            status="completed",
            started_at=critic_stage_started,
        )
    except Exception as exc:
        _record_agent_stage(
            resume_run_id=run.resume_run_id,
            agent_type="jd_critic",
            agent_name="JD Critic Agent",
            stage="jd_coverage",
            status="failed",
            started_at=critic_stage_started,
            error=str(exc),
        )
        critic_result = {
            "critic": "jd_critic_error",
            "overall_pass": False,
            "requirement_reviews": [],
            "revisions": [],
            "claims": llm_draft.get("claims", []),
            "error": str(exc),
        }
        if not evidence_policy.apply_critic_revisions:
            critic_result["mode"] = "shadow"
            critic_result["applied_revisions"] = False
    candidate_by_id = {
        str(item.get("item_id") or ""): item
        for item in candidates
        if str(item.get("item_id") or "")
    }
    sanitized_claims: list[dict[str, object]] = []
    for claim in llm_draft.get("claims", []):
        if not isinstance(claim, dict):
            continue
        item_id = str(claim.get("item_id") or "")
        cleaned = remove_rejected_request_echo(
            str(claim.get("text") or ""),
            rejected_user_requests,
        )
        candidate = candidate_by_id.get(item_id, {})
        cleaned = dedupe_resume_claim_fields(
            cleaned,
            str(candidate.get("section") or claim.get("section") or ""),
        )
        if not cleaned:
            cleaned = str(
                candidate.get("project_name")
                or candidate.get("evidence_label")
                or ""
            ).strip()
        if cleaned:
            sanitized_claims.append({**claim, "text": cleaned})
    llm_draft["claims"] = sanitized_claims
    # Recompute coverage after the final echo/placeholder sanitization so the
    # audit describes exactly what is passed to the planner.
    coverage_matrix = build_jd_coverage_matrix(
        jd_blocks,
        llm_draft.get("claims", []),
        user_requirements,
    )
    if ignored_numeric_requests:
        llm_draft["validation_feedback"] = [
            *llm_draft.get("validation_feedback", []),
            *(f"Ignored unsupported user request: {item}" for item in ignored_numeric_requests),
        ]
    llm_selected = set(llm_draft["selected_item_ids"])
    candidates = [
        {**item, "selected_by_llm": True}
        for item in candidates
        if item.get("item_id") in llm_selected
    ]
    plan_stage_started = time.time()
    try:
        planned = _execute_career_tool(
            "plan_resume_content",
            {
                "jd_id": jd.jd_id,
                "content_budget": content_budget,
                "candidate_items": candidates,
                "required_item_ids": required_resume_ids,
            },
        )
    except Exception as exc:
        _record_agent_stage(
            resume_run_id=run.resume_run_id,
            agent_type="resume_generator",
            agent_name="Resume Generator Agent",
            stage="planning",
            status="failed",
            started_at=plan_stage_started,
            error=str(exc),
        )
        raise
    _record_agent_stage(
        resume_run_id=run.resume_run_id,
        agent_type="resume_generator",
        agent_name="Resume Generator Agent",
        stage="planning",
        status="completed",
        started_at=plan_stage_started,
    )
    store = type(store)(root)
    plan_record = store.get(planned["plan"]["plan_id"])
    run = ResumeRunService(store).attach_plan(run.resume_run_id, plan_record)
    selected = planned.get("selected_items") or planned.get("plan", {}).get("selected_items") or []
    if required_project_count is not None:
        selected_project_count = sum(
            1 for item in selected if str(item.get("section") or "").casefold() == "projects"
        )
        available_project_count = sum(
            1 for item in candidates if str(item.get("section") or "").casefold() == "projects"
        )
        expected_project_count = min(required_project_count, available_project_count)
        if selected_project_count != expected_project_count:
            raise HTTPException(
                status_code=502,
                detail=f"Resume planner did not preserve the required {expected_project_count} personal projects.",
            )
    claim_drafts = [
        {"item_id": item["item_id"], "text": item["proposed_text"]}
        for item in selected
        if item.get("item_id") and item.get("proposed_text")
    ]
    if not claim_drafts:
        raise HTTPException(status_code=409, detail="Resume planner selected no evidence-backed content.")
    llm_claims = {
        item["item_id"]: item["text"]
        for item in llm_draft["claims"]
    }
    claim_drafts = [
        {
            "item_id": item["item_id"],
            "text": llm_claims[item["item_id"]],
        }
        for item in selected
        if item.get("item_id") in llm_claims
    ]
    if len(claim_drafts) != len(selected):
        raise HTTPException(status_code=502, detail="Resume generator did not draft every selected section block.")
    llm_draft["claims"] = claim_drafts
    generation_stage_started = time.time()
    try:
        generated = _execute_career_tool(
            "generate_resume_document",
            {
                "plan_id": planned["plan"]["plan_id"],
                "header": header,
                "claim_drafts": claim_drafts,
                "template_id": template.template_id if template else "",
            },
            enable_model_runtime=True,
            runtime_overrides={
                "_resumeBuilder": ResumeDocumentBuilder(
                    type(store)(root),
                    enforce_project_evidence=evidence_policy.enforce_project_claim_validation,
                )
            },
        )
    except Exception as exc:
        _record_agent_stage(
            resume_run_id=run.resume_run_id,
            agent_type="resume_generator",
            agent_name="Resume Generator Agent",
            stage="document",
            status="failed",
            started_at=generation_stage_started,
            error=str(exc),
        )
        raise
    _record_agent_stage(
        resume_run_id=run.resume_run_id,
        agent_type="resume_generator",
        agent_name="Resume Generator Agent",
        stage="document",
        status="completed",
        started_at=generation_stage_started,
    )
    store = type(store)(root)
    document_record = store.get(generated["document"]["resume_document_id"])
    run = ResumeRunService(store).attach_document(run.resume_run_id, document_record)
    return {
        "resume_run": record_to_dict(run),
        "retrieval": retrieved,
        "plan": planned,
        "generation": generated,
        "template": record_to_dict(template) if template else None,
        "verification": "required",
        "match_confidence_threshold": confidence_threshold,
        "selection_policy": selection_policy,
        "resume_draft": llm_draft,
        "customization": {
            "request": customization_request,
            "required_project_count": required_project_count,
            "matches": selection_policy.get("customization_matches", []),
            "user_requirements": [item.to_dict() for item in user_requirements],
            "ignored_unsupported_requests": [
                *rejected_user_requests,
                *ignored_numeric_requests,
                *[
                    feedback
                    for feedback in llm_draft.get("validation_feedback", [])
                    if "failed evidence validation" in feedback.casefold()
                    or "numeric facts absent" in feedback.casefold()
                    or "ignored unsupported user request" in feedback.casefold()
                ],
            ],
        },
        "jd_coverage": coverage_matrix,
        "jd_critic": critic_result,
        "evidence_policy": {
            "project_prompt": "enforce" if evidence_policy.enforce_project_prompt else "ablation",
            "project_claim_validation": "enforce" if evidence_policy.enforce_project_claim_validation else "ablation",
            "project_user_requirement_filter": "enforce" if evidence_policy.enforce_project_user_requirement_filter else "ablation",
            "critic_revisions": "apply" if evidence_policy.apply_critic_revisions else "shadow",
            "verification": evidence_policy.verification_mode,
        },
        "next_actions": ["verify_resume_document", "render_resume_pdf"],
    }


@app.post("/api/career/resume/build")
async def career_resume_build(data: dict):
    """Run the production resume flow with the default evidence policy."""
    return await _career_resume_build(data, evidence_policy=FULL_RESUME_EVIDENCE_POLICY)


def _resume_customization_reply(
    request: str,
    *,
    selection_policy: dict | None,
    verification_status: str,
    rendered: bool,
    ignored_unsupported: bool = False,
) -> str:
    matches = (selection_policy or {}).get("customization_matches", [])
    emphasized = [str(item.get("subject") or "").strip() for item in matches if float(item.get("score") or 0) > 0]
    deprioritized = [str(item.get("subject") or "").strip() for item in matches if float(item.get("score") or 0) < 0]
    details: list[str] = []
    if emphasized:
        details.append("已优先考虑 " + "、".join(dict.fromkeys(item for item in emphasized if item)))
    if deprioritized:
        details.append("已降低 " + "、".join(dict.fromkeys(item for item in deprioritized if item)) + " 的优先级")
    if not details and request:
        details.append("已将你的定制要求交给受证据约束的内容选择和改写流程")
    if ignored_unsupported:
        details.append("已忽略缺少个人材料证据支持的要求，其余要求继续执行")
    outcome = "已完成独立核验并生成 PDF。" if rendered else (
        "简历已生成，但独立核验未通过，因此没有导出 PDF；请继续说明希望如何调整。"
        if verification_status != "passed"
        else "简历已通过核验，但 PDF 渲染未完成。"
    )
    return "；".join(details + ["只使用已确认材料，不会补写未提供的经历、指标或技术", outcome])


@app.post("/api/career/resume/deliver")
async def career_resume_deliver(data: dict):
    """Deliver one evidence-backed resume revision through all required stages."""
    from smart_agent.career import ResumeRunService, record_to_dict

    try:
        request = _resume_customization_request(
            data.get("customization_request") or data.get("message")
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    generated = await career_resume_build(
        {
            **data,
            "customization_request": request,
        }
    )
    run_id = generated["resume_run"]["resume_run_id"]
    verification = await career_resume_verify({"resume_run_id": run_id})
    verification_status = str(verification.get("report", {}).get("status") or "").strip().casefold()
    rendering = None
    if verification_status == "passed":
        rendering = await career_resume_render(
            {
                "resume_run_id": run_id,
                "max_pages": int(data.get("max_pages", 2)),
            }
        )
    reply = _resume_customization_reply(
        request,
        selection_policy=generated.get("selection_policy"),
        verification_status=verification_status,
        rendered=rendering is not None,
        ignored_unsupported=bool(
            generated.get("customization", {}).get("ignored_unsupported_requests")
        ),
    )
    root, store, _material_store, _candidate_store = _career_stores()
    run = _resolve_resume_run(store, run_id)
    if request:
        run = ResumeRunService(store).append_customization_turn(
            run.resume_run_id,
            user_message=request,
            assistant_message=reply,
            resume_document_id=generated["generation"]["document"]["resume_document_id"],
        )
    return {
        "status": "delivered" if rendering is not None else "needs_review",
        "agent": {
            "request": request,
            "reply": reply,
            "evidence_only": True,
        },
        "generation": generated["generation"],
        "verification": verification,
        "rendering": rendering,
        "resume_run": record_to_dict(run),
        "preview_url": rendering.get("preview_url") if rendering else "",
    }


@app.post("/api/career/requirements/match")
async def career_requirements_match(data: dict):
    from smart_agent.career import record_to_dict

    root, store, _material_store, _candidate_store = _career_stores()
    run = _resolve_resume_run(store, str(data.get("resume_run_id") or ""))
    run, retrieval = _match_resume_run(run, root=root, store=store)
    return {"resume_run": record_to_dict(run), "retrieval": retrieval}


@app.post("/api/career/resume/verify")
async def career_resume_verify(data: dict):
    from smart_agent.career import (
        DeterministicResumeEvidenceVerifier,
        ResumeRunService,
        VerificationReport,
        record_to_dict,
    )

    root, store, _material_store, _candidate_store = _career_stores()
    run = _resolve_resume_run(store, str(data.get("resume_run_id") or ""))
    if run.resume_document_ref is None:
        raise HTTPException(status_code=409, detail="The resume run has no generated document.")
    fallback_reason = ""
    stage_started = time.time()
    try:
        result = _execute_career_tool(
            "verify_resume_document",
            {"resume_document_id": run.resume_document_ref.record_id},
            enable_model_runtime=True,
        )
    except HTTPException as exc:
        fallback_reason = str(exc.detail or "")
        store = type(store)(root)
        fallback = DeterministicResumeEvidenceVerifier(store).verify(
            run.resume_document_ref.record_id,
            parent_session_id=run.resume_run_id,
        )
        result = fallback.to_dict()
        result["fallback_verification"] = True
        result["fallback_reason"] = fallback_reason
    _record_agent_stage(
        resume_run_id=run.resume_run_id,
        agent_type="resume_verification",
        agent_name="Resume Verification Agent",
        stage="verification",
        status="completed",
        started_at=stage_started,
        error=fallback_reason,
    )
    store = type(store)(root)
    report = store.get(result["report"]["verification_report_id"], expected_type=VerificationReport)
    run = ResumeRunService(store).attach_verification(run.resume_run_id, report)
    if fallback_reason:
        result["fallback_verification"] = True
        result["fallback_reason"] = fallback_reason
    return {**result, "resume_run": record_to_dict(run)}


@app.post("/api/career/resume/select")
async def career_resume_select(data: dict):
    """Select a generated resume version without deleting the alternatives."""
    from smart_agent.career import CareerStore, ResumeDocument, ResumeRunService, record_to_dict

    store = CareerStore(_career_root())
    run = _resolve_resume_run(store, str(data.get("resume_run_id") or ""))
    document_id = str(data.get("resume_document_id") or "").strip()
    if not document_id:
        raise HTTPException(status_code=400, detail="resume_document_id is required")
    try:
        document = store.get(document_id, expected_type=ResumeDocument)
        if document.jd_ref.record_id != run.jd_ref.record_id:
            raise ValueError("The selected resume belongs to another JD.")
        selected = ResumeRunService(store).attach_document(run.resume_run_id, document)
        return {"resume_run": record_to_dict(selected), "resume_document": record_to_dict(document)}
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/career/resume/versions/{resume_document_id}")
async def career_resume_version_delete(resume_document_id: str, data: dict | None = None):
    """Soft-delete one generated version while retaining audit history.

    Deleting the active version is allowed when another valid version exists;
    the run is switched first so interview and rendering references never
    point at an invalidated document.
    """
    from smart_agent.career import (
        ResumeDocument,
        ResumeDocumentStatus,
        ResumeRunService,
        ResumeRunStatus,
        record_to_dict,
    )

    root, store, _material_store, _candidate_store = _career_stores()
    document_id = str(resume_document_id or "").strip()
    payload = data or {}
    run_id = str(payload.get("resume_run_id") or "").strip()
    requested_document_id = str(payload.get("resume_document_id") or "").strip()
    if not document_id or not run_id:
        raise HTTPException(status_code=400, detail="resume_run_id and resume_document_id are required")
    if requested_document_id and requested_document_id != document_id:
        raise HTTPException(status_code=400, detail="The requested resume version does not match the URL.")
    run = _resolve_resume_run(store, run_id)
    try:
        document = store.get(document_id, expected_type=ResumeDocument)
        if document.jd_ref.record_id != run.jd_ref.record_id:
            raise ValueError("The selected resume belongs to another JD.")
        active_documents = [
            item
            for item in store.list_latest()
            if isinstance(item, ResumeDocument)
            and item.jd_ref.record_id == run.jd_ref.record_id
            and item.status is not ResumeDocumentStatus.INVALIDATED
        ]
        is_current = bool(run.resume_document_ref and run.resume_document_ref.record_id == document_id)
        replacement = None
        if is_current:
            replacements = [item for item in active_documents if item.resume_document_id != document_id]
            replacements.sort(
                key=lambda item: (item.updated_at, item.created_at, item.resume_document_id),
                reverse=True,
            )
            replacement = replacements[0] if replacements else None
            if replacement is not None:
                run = ResumeRunService(store).attach_document(run.resume_run_id, replacement)
            else:
                # A sole version may be deleted as requested. Keep the JD
                # session and audit history, but remove its active document
                # reference so the run returns to a regeneratable state.
                run = store.revise_record(
                    run.resume_run_id,
                    actor="career_resume_version_manager",
                    reason="Clear deleted sole resume version",
                    changes={
                        "resume_document_ref": None,
                        "verification_report_ref": None,
                        "rendered_artifacts": (),
                        "status": ResumeRunStatus.PLANNED if run.plan_ref is not None else ResumeRunStatus.MATCHED,
                    },
                )
        # Customization turns are shown from the run record. Remove turns
        # attached to the deleted version, and clear legacy turns that were
        # created before version IDs were persisted.
        remaining_messages = tuple(
            message
            for message in run.customization_messages
            if str(message.get("resume_document_id") or "").strip()
            not in {"", document_id}
        )
        if len(remaining_messages) != len(run.customization_messages):
            run = store.revise_record(
                run.resume_run_id,
                actor="career_resume_version_manager",
                reason="Remove customization conversation for deleted resume version",
                changes={"customization_messages": remaining_messages},
            )
        deleted = store.revise_record(
            document_id,
            actor="career_resume_version_manager",
            reason="User deleted an alternate resume version",
            changes={"status": "invalidated"},
        )
        return {
            "deleted": True,
            "resume_document": record_to_dict(deleted),
            "resume_run": record_to_dict(run),
            "replacement_document": record_to_dict(replacement) if replacement is not None else None,
        }
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/career/resume/render")
async def career_resume_render(data: dict):
    from smart_agent.career import ResumeRunService, record_to_dict

    root, store, _material_store, _candidate_store = _career_stores()
    run = _resolve_resume_run(store, str(data.get("resume_run_id") or ""))
    if run.resume_document_ref is None:
        raise HTTPException(status_code=409, detail="The resume run has no generated document.")
    stage_started = time.time()
    try:
        result = _execute_career_tool(
            "render_resume_pdf",
            {
                "resume_document_id": run.resume_document_ref.record_id,
                "output_dir": str(data.get("output_dir") or ".runtime/career/exports"),
                "max_pages": int(data.get("max_pages", 2)),
            },
        )
    except Exception as exc:
        _record_agent_stage(
            resume_run_id=run.resume_run_id,
            agent_type="resume_generator",
            agent_name="Resume Generator Agent",
            stage="render",
            status="failed",
            started_at=stage_started,
            error=str(exc),
        )
        raise
    _record_agent_stage(
        resume_run_id=run.resume_run_id,
        agent_type="resume_generator",
        agent_name="Resume Generator Agent",
        stage="render",
        status="completed",
        started_at=stage_started,
    )
    store = type(store)(root)
    run = ResumeRunService(store).attach_rendered_artifact(run.resume_run_id, result["pdf_path"])
    return {
        **result,
        "resume_run": record_to_dict(run),
        "preview_url": f"/api/career/resume/runs/{run.resume_run_id}/pdf",
    }


@app.get("/api/career/resume/runs/{resume_run_id}/pdf")
async def career_resume_pdf_preview(resume_run_id: str):
    _root, store, _material_store, _candidate_store = _career_stores()
    run = _resolve_resume_run(store, resume_run_id)
    if not run.rendered_artifacts:
        raise HTTPException(status_code=404, detail="The resume PDF has not been rendered.")
    export_root = (ROOT / ".runtime" / "career" / "exports").resolve()
    path = Path(run.rendered_artifacts[-1]).expanduser().resolve()
    if not path.is_relative_to(export_root) or path.suffix.casefold() != ".pdf" or not path.is_file():
        raise HTTPException(status_code=404, detail="The rendered resume PDF is unavailable.")
    _validate_rendered_pdf(path)
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": "inline", "Cache-Control": "no-store"},
    )


def _rendered_resume_pdf(resume_run_id: str) -> Path:
    _root, store, _material_store, _candidate_store = _career_stores()
    run = _resolve_resume_run(store, resume_run_id)
    if not run.rendered_artifacts:
        raise HTTPException(status_code=404, detail="The resume PDF has not been rendered.")
    export_root = (ROOT / ".runtime" / "career" / "exports").resolve()
    path = Path(run.rendered_artifacts[-1]).expanduser().resolve()
    if not path.is_relative_to(export_root) or path.suffix.casefold() != ".pdf" or not path.is_file():
        raise HTTPException(status_code=404, detail="The rendered resume PDF is unavailable.")
    return _validate_rendered_pdf(path)


def _validate_rendered_pdf(path: Path) -> Path:
    """Reject missing or corrupt PDF artifacts before download or preview."""

    try:
        from pypdf import PdfReader

        if len(PdfReader(str(path)).pages) < 1:
            raise ValueError("PDF has no pages")
    except Exception as exc:
        raise HTTPException(status_code=404, detail="The rendered resume PDF is unavailable.") from exc
    return path


def _rendered_resume_document_pdf(resume_document_id: str) -> Path:
    """Find the PDF artifact belonging to a specific resume version."""
    document_id = str(resume_document_id or "").strip()
    if not document_id:
        raise HTTPException(status_code=400, detail="resume_document_id is required.")
    export_root = (ROOT / ".runtime" / "career" / "exports").resolve()
    candidates = sorted(export_root.glob(f"{document_id}-r*.pdf"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise HTTPException(status_code=404, detail="The selected resume PDF has not been rendered.")
    path = candidates[0].resolve()
    if not path.is_relative_to(export_root) or not path.is_file():
        raise HTTPException(status_code=404, detail="The selected resume PDF is unavailable.")
    return _validate_rendered_pdf(path)


@app.get("/api/career/resume/documents/{resume_document_id}/pdf")
async def career_resume_document_pdf(resume_document_id: str):
    return FileResponse(
        _rendered_resume_document_pdf(resume_document_id),
        media_type="application/pdf",
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "private, max-age=31536000, immutable",
        },
    )


@app.get("/api/career/resume/documents/{resume_document_id}/preview-info")
async def career_resume_document_preview_info(resume_document_id: str):
    path = _rendered_resume_document_pdf(resume_document_id)
    try:
        from pypdf import PdfReader

        page_count = len(PdfReader(str(path)).pages)
    except (ImportError, OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="PDF preview metadata failed.") from exc
    encoded_id = urllib.parse.quote(str(resume_document_id), safe="")
    return {
        "page_count": page_count,
        "page_urls": [
            f"/api/career/resume/documents/{encoded_id}/pages/{index + 1}.png"
            for index in range(page_count)
        ],
        "pdf_url": f"/api/career/resume/documents/{encoded_id}/pdf",
    }


@app.get("/api/career/resume/documents/{resume_document_id}/pages/{page_number}.png")
async def career_resume_document_page(resume_document_id: str, page_number: int):
    return _pdf_page_response(_rendered_resume_document_pdf(resume_document_id), page_number - 1)


@lru_cache(maxsize=96)
def _static_pdf_preview_path(path: Path, page_index: int) -> Path | None:
    """Return a checked-in/local PNG preview when PDF rasterization is unavailable."""
    if page_index < 0:
        return None
    candidates = [
        path.with_name(f"{path.stem}-{page_index + 1}.png"),
    ]
    if page_index == 0:
        candidates.append(path.with_suffix(".png"))
    for candidate in candidates:
        try:
            if candidate.is_file() and candidate.resolve().parent == path.resolve().parent:
                return candidate
        except OSError:
            continue
    return None


@lru_cache(maxsize=96)
def _pdf_page_bytes(path_text: str, modified_ns: int, page_index: int) -> bytes:
    """Rasterize one immutable PDF page once per process."""
    static_preview = _static_pdf_preview_path(Path(path_text), page_index)
    if static_preview is not None:
        try:
            return static_preview.read_bytes()
        except OSError:
            pass
    try:
        import fitz

        with fitz.open(path_text) as document:
            if page_index < 0 or page_index >= document.page_count:
                raise HTTPException(status_code=404, detail="PDF page not found.")
            pixmap = document[page_index].get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
            return pixmap.tobytes("png")
    except HTTPException:
        raise
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="PDF preview rendering failed.") from exc


def _pdf_page_response(path: Path, page_index: int) -> Response:
    try:
        modified_ns = path.stat().st_mtime_ns
    except OSError as exc:
        raise HTTPException(status_code=404, detail="PDF preview source is unavailable.") from exc
    content = _pdf_page_bytes(str(path), modified_ns, page_index)
    return Response(content=content, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/career/resume/runs/{resume_run_id}/preview-info")
async def career_resume_pdf_preview_info(resume_run_id: str):
    path = _rendered_resume_pdf(resume_run_id)
    try:
        from pypdf import PdfReader

        page_count = len(PdfReader(str(path)).pages)
    except (ImportError, OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="PDF preview metadata failed.") from exc
    return {
        "page_count": page_count,
        "page_urls": [
            f"/api/career/resume/runs/{resume_run_id}/pages/{index + 1}.png"
            for index in range(page_count)
        ],
        "pdf_url": f"/api/career/resume/runs/{resume_run_id}/pdf",
    }


@app.get("/api/career/resume/runs/{resume_run_id}/pages/{page_number}.png")
async def career_resume_pdf_page(resume_run_id: str, page_number: int):
    return _pdf_page_response(_rendered_resume_pdf(resume_run_id), page_number - 1)


@app.get("/api/career/artifacts")
async def career_artifacts():
    """Return the persisted Career index plus latest typed records for UI recovery."""
    from smart_agent.career import CandidateFactStore, CareerStore, MaterialStore

    cache_key = _career_library_cache_key()
    now = time.monotonic()
    cached = _CAREER_ARTIFACTS_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < 10:
        return cached[1]
    root = _career_root()
    career_store = CareerStore(root)
    manifests = MaterialStore(root / "materials", workspace=str(ROOT)).list_manifests()
    candidates = CandidateFactStore(root / "candidate-facts").list()
    state = {
        "index": {
            "material_count": len(manifests),
            "candidate_fact_count": sum(1 for item in candidates if getattr(item.status, "value", item.status) == "candidate"),
        }
    }

    def _serialize_records() -> list[dict]:
        # The UI needs resume/JD state, not the full historical match ledger.
        # EvidenceMatch alone currently contains thousands of immutable rows
        # and made this endpoint return tens of megabytes on every login.
        record_types = (
            "JDDocument",
            "JDRequirement",
            "ResumeDocument",
            "VerificationReport",
            "ResumeRun",
        )
        records = []
        for record_type in record_types:
            for record in career_store.list_latest_payloads(record_type):
                try:
                    payload = dict(record)
                    # These immutable audit/evidence arrays are not needed by
                    # the browser recovery view and can dominate response size.
                    payload.pop("source_refs", None)
                    payload.pop("history", None)
                    records.append(payload)
                except (TypeError, ValueError):
                    continue
        return records

    records = _serialize_records()
    payload = {"workspace": state, "records": records}
    _CAREER_ARTIFACTS_CACHE[cache_key] = (time.monotonic(), payload)
    return payload


def _localize_interview_question_payload(question: dict | None, *, action: str = "ask") -> None:
    """Preserve recorded LLM output; presentation must not synthesize questions."""
def _localize_interview_report_question(item: dict) -> None:
    """Preserve historical question text instead of masking it with a template."""
def _localize_interview_session_payload(payload: dict) -> dict:
    _localize_interview_question_payload(payload.get("pending_question"))
    for turn in payload.get("turns") or []:
        if isinstance(turn, dict):
            _localize_interview_question_payload(turn.get("question"))
    report = payload.get("report")
    if isinstance(report, dict):
        for item in report.get("question_results") or []:
            _localize_interview_report_question(item)
    return payload


@app.get("/api/career/interviews")
async def career_interviews():
    """Return isolated InterviewSession state without exposing parent materials."""
    from smart_agent.career import InterviewSessionStore

    sessions = InterviewSessionStore(_career_root() / "interviews").list()
    payloads = []
    for session in sorted(sessions, key=lambda item: item.updated_at):
        payload = _repair_stale_interview_pending_for_payload(session)
        payloads.append(_localize_interview_session_payload(payload))
    return {"sessions": payloads}


@app.delete("/api/career/sessions/{session_type}/{session_id}")
async def career_session_delete(session_type: str, session_id: str):
    """Delete one UI session without removing reusable requirement matches."""
    selected_type = str(session_type or "").strip().lower()
    identity = str(session_id or "").strip()
    if selected_type not in {"resume", "interview"}:
        raise HTTPException(status_code=400, detail="session_type must be resume or interview")
    if not identity:
        raise HTTPException(status_code=400, detail="session_id is required")
    try:
        if selected_type == "resume":
            from smart_agent.career import (
                RequirementMatchCache,
                ResumeRunService,
                record_to_dict,
            )

            _root, store, _material_store, _candidate_store = _career_stores()
            preserved_cache_count = len(store.list_latest(RequirementMatchCache))
            deleted = ResumeRunService(store).delete(identity)
            return {
                "deleted": True,
                "session_type": selected_type,
                "session": record_to_dict(deleted),
                "preserved_requirement_cache_count": preserved_cache_count,
            }
        from smart_agent.career import InterviewSessionStore

        InterviewSessionStore(_career_root() / "interviews").delete(identity)
        return {
            "deleted": True,
            "session_type": selected_type,
            "session_id": identity,
            "preserved_requirement_cache_count": 0,
        }
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/career/interview-knowledge")
async def career_interview_knowledge():
    """List imported interview knowledge without exposing its full text."""
    return {"sources": _public_interview_knowledge_sources(_career_root())}


def _remove_internal_import_copy(path_value: str) -> bool:
    """Remove an upload staging copy, never a user-owned source file."""
    raw = str(path_value or "").strip()
    if not raw:
        return False
    target = Path(raw).expanduser().resolve()
    staging_roots = (
        (ROOT / ".tmp").resolve(),
        (_career_root() / "imports").resolve(),
    )
    if not any(target.is_relative_to(root) for root in staging_roots) or target in staging_roots:
        return False
    try:
        if target.is_file():
            target.unlink()
            return True
        if target.is_dir():
            shutil.rmtree(target)
            return True
    except FileNotFoundError:
        return False
    return False


@app.delete("/api/career/interview-knowledge/{source_id}")
async def career_interview_knowledge_delete(source_id: str):
    """Delete one imported interview source and every derived index entry."""
    from smart_agent.career import CareerGraphRAGIndex, InterviewKnowledgeLibrary

    try:
        root = _career_root()
        library = InterviewKnowledgeLibrary(root / "interview-knowledge")
        tenant_id = _career_tenant_id() or "legacy-web-user"
        source = library.get_source(source_id)
        if source.tenant_id != tenant_id:
            raise KeyError(f"Unknown interview knowledge source: {source_id}")
        graph_result = CareerGraphRAGIndex(root / "graph-rag").delete_corpus(source.corpus_id)
        deleted = library.delete_source(source.source_id, tenant_id=tenant_id)
        source_copy_removed = _remove_internal_import_copy(deleted.source_path)
        return {
            "deleted": True,
            "source": deleted.to_dict(),
            "source_copy_removed": source_copy_removed,
            "graph_rag": graph_result,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/career/interview-knowledge/delete-batch")
async def career_interview_knowledge_delete_batch(data: dict):
    """Delete selected interview sources and report every item independently."""
    raw_ids = data.get("source_ids") if isinstance(data, dict) else None
    if not isinstance(raw_ids, list):
        raise HTTPException(status_code=400, detail="source_ids must be a list")
    source_ids = tuple(dict.fromkeys(str(item or "").strip() for item in raw_ids if str(item or "").strip()))
    if not source_ids:
        raise HTTPException(status_code=400, detail="Select at least one interview knowledge source")
    results: list[dict] = []
    for source_id in source_ids:
        try:
            result = await career_interview_knowledge_delete(source_id)
        except HTTPException as exc:
            results.append({"source_id": source_id, "deleted": False, "status_code": exc.status_code, "error": str(exc.detail)})
        except Exception as exc:  # keep one failure from hiding other outcomes
            results.append({"source_id": source_id, "deleted": False, "status_code": 500, "error": f"{type(exc).__name__}: {exc}"})
        else:
            results.append({"source_id": source_id, "deleted": bool(result.get("deleted")), "result": result})
    deleted_count = sum(bool(item.get("deleted")) for item in results)
    failed_count = len(results) - deleted_count
    return {"deleted": failed_count == 0, "deleted_count": deleted_count, "failed_count": failed_count, "results": results}


_INTERVIEW_COACH_PROJECT_HINTS = (
    "项目",
    "源码",
    "代码",
    "仓库",
    "架构",
    "模块",
    "接口",
    "实现",
    "部署",
    "配置",
    "README",
    "readme",
    "个人项目",
    "我的项目",
    "我做的",
    "项目里",
    "系统里",
    "业务流程",
    "工程实现",
    "RAG",
    "rag",
    "Smart Agent",
    "smart agent",
    "smart-agent",
)
_INTERVIEW_COACH_KNOWLEDGE_HINTS = (
    "八股",
    "原理",
    "概念",
    "区别",
    "为什么",
    "怎么回答",
    "如何回答",
    "面试官",
    "Redis",
    "MySQL",
    "TCP",
    "HTTP",
    "线程",
    "进程",
    "锁",
    "索引",
    "事务",
    "Transformer",
    "LangChain",
    "LangGraph",
    "向量",
    "embedding",
    "召回",
    "重排",
)
_INTERVIEW_COACH_MIXED_HINTS = (
    "结合项目",
    "用我的项目",
    "在项目中",
    "项目中如何",
    "项目里如何",
    "项目怎么体现",
    "结合个人项目",
    "结合简历",
    "联系我的",
)
_READ_ONLY_WORKSPACE_PATTERN = re.compile(
    r"(?i)(?:(?:\b(?:read|inspect|list|show|look(?:\s+at)?|understand|examine|analy[sz]e|review|browse)\b|"
    r"读取|查看|看看|梳理|分析|介绍|说明|了解)\s*"
    r".{0,80}(?:\b(?:project|repo|repository|code|workspace|source)\b|smart\s*agent|"
    r"项目|仓库|代码|工作区|源码|入口|规范)|"
    r"(?:\b(?:project|repo|repository|code|workspace|source)\b|smart\s*agent|项目|仓库|代码|工作区|源码|入口|规范)\s*"
    r".{0,80}(?:\b(?:read|inspect|list|show|look(?:\s+at)?|understand|examine|analy[sz]e|review|browse)\b|"
    r"读取|查看|看看|梳理|分析|介绍|说明|了解))"
)
_MUTATION_REQUEST_PATTERN = re.compile(
    r"(?i)(?:\b(?:write|create|implement|modify|edit|fix|debug|refactor|run|test|delete|rename|move|commit)\b|修改|修复|创建|编写|删除|重命名|移动|提交)"
)


def _interview_coach_context(session_id: str) -> ContextManager:
    from smart_agent.web.workspace_context import read
    state = read(session_id)
    return ContextManager(model=str(load_runtime_config(ROOT).get("model") or "default"),
                          messages=state.get("coach_messages", []))


def _interview_coach_history(context: ContextManager) -> list[dict[str, str]]:
    return [
        {
            "role": str(item.get("role") or ""),
            "content": str(item.get("content") or "")[:4000],
        }
        for item in context.messages[-8:]
        if str(item.get("role") or "") in {"user", "assistant"}
    ]


def _classify_interview_coach_question(question: str) -> str:
    text = str(question or "").strip()
    if not text:
        raise ValueError("question is required")
    lowered = text.casefold()
    mixed = any(hint.casefold() in lowered for hint in _INTERVIEW_COACH_MIXED_HINTS)
    project = any(hint.casefold() in lowered for hint in _INTERVIEW_COACH_PROJECT_HINTS)
    knowledge = any(hint.casefold() in lowered for hint in _INTERVIEW_COACH_KNOWLEDGE_HINTS)
    if mixed or (project and knowledge):
        return "mixed"
    if project:
        return "project"
    if knowledge:
        return "interview_knowledge"
    return "mixed"


_MOJIBAKE_MARKERS = (
    "锛",
    "鏇",
    "浠",
    "绉",
    "鐨",
    "缁",
    "闂",
    "璧",
    "绔",
    "鍏",
    "瀹",
    "鎴",
    "杩",
    "�",
)


def _mojibake_score(value: str) -> int:
    return sum(value.count(marker) for marker in _MOJIBAKE_MARKERS)


def _repair_interview_text_encoding(value: object) -> str:
    text = str(value or "")
    if not text:
        return ""
    best = text
    best_score = _mojibake_score(text)
    if best_score == 0:
        return text
    for encoding in ("gb18030", "gbk"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
        score = _mojibake_score(candidate)
        if score < best_score:
            best, best_score = candidate, score
    return best


def _truncate_text(value: object, limit: int = 900) -> str:
    text = " ".join(_repair_interview_text_encoding(value).split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _interview_coach_project_source_payload(source) -> dict:
    payload = source.to_dict()
    return {
        "project_id": payload.get("project_id", ""),
        "display_name": payload.get("display_name", ""),
        "source_type": payload.get("source_type", ""),
        "status": payload.get("status", ""),
        "artifact_count": payload.get("artifact_count", 0),
        "revision": payload.get("revision", 1),
        "updated_at": payload.get("updated_at", 0),
    }


def _interview_coach_project_hit_payload(hit, index: int) -> dict:
    payload = hit.to_dict()
    return {
        "source_id": f"project_knowledge_{index}",
        "project_id": payload.get("project_id", ""),
        "material_id": payload.get("material_id", ""),
        "chunk_id": payload.get("chunk_id", ""),
        "content": _truncate_text(payload.get("content", ""), 1200),
        "locator": payload.get("locator") or {},
        "distance": payload.get("distance"),
        "lexical_score": payload.get("lexical_score"),
    }


_INTERVIEW_COACH_PROJECT_FILE_SCAN_LIMIT = 400
_INTERVIEW_COACH_PROJECT_FILE_READ_CHARS = 120_000
_INTERVIEW_COACH_PROJECT_FILE_SNIPPET_CHARS = 1_600


def _interview_coach_terms(question: str) -> tuple[str, ...]:
    text = str(question or "").casefold()
    terms: set[str] = set(re.findall(r"[a-z0-9][a-z0-9_.+-]*[a-z0-9]|[a-z0-9]+", text))
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.add(run)
        if len(run) <= 24:
            for size in (2, 3, 4):
                if len(run) >= size:
                    terms.update(run[index : index + size] for index in range(0, len(run) - size + 1))
    return tuple(sorted((term for term in terms if term.strip()), key=lambda item: (-len(item), item)))


def _interview_coach_project_file_score(
    *,
    question_terms: tuple[str, ...],
    source,
    artifact,
    text: str,
) -> float:
    if not question_terms:
        return 0.0
    haystack = " ".join(
        (
            str(getattr(source, "project_id", "")),
            str(getattr(source, "display_name", "")),
            str(getattr(artifact, "path", "")),
            text[:_INTERVIEW_COACH_PROJECT_FILE_READ_CHARS],
        )
    ).casefold()
    score = 0.0
    for term in question_terms:
        if term in haystack:
            score += max(1.0, min(4.0, len(term) / 2.0))
    return score / max(1, len(question_terms))


def _interview_coach_project_file_snippet(text: str, terms: tuple[str, ...]) -> str:
    cleaned = _repair_interview_text_encoding(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not cleaned:
        return ""
    lowered = cleaned.casefold()
    positions = [lowered.find(term) for term in terms if term and lowered.find(term) >= 0]
    if positions:
        center = min(positions)
        start = max(0, center - _INTERVIEW_COACH_PROJECT_FILE_SNIPPET_CHARS // 3)
        end = min(len(cleaned), start + _INTERVIEW_COACH_PROJECT_FILE_SNIPPET_CHARS)
        start = max(0, end - _INTERVIEW_COACH_PROJECT_FILE_SNIPPET_CHARS)
        prefix = "..." if start else ""
        suffix = "..." if end < len(cleaned) else ""
        return prefix + cleaned[start:end].strip() + suffix
    return cleaned[:_INTERVIEW_COACH_PROJECT_FILE_SNIPPET_CHARS] + (
        "..." if len(cleaned) > _INTERVIEW_COACH_PROJECT_FILE_SNIPPET_CHARS else ""
    )


def _interview_coach_project_artifact_rank(artifact, terms: tuple[str, ...]) -> tuple:
    path = str(getattr(artifact, "path", "") or "")
    path_lower = path.casefold()
    name = PurePosixPath(path_lower).name
    kind = str(getattr(getattr(artifact, "kind", ""), "value", getattr(artifact, "kind", "")))
    kind_rank = {"document": 0, "config": 1, "source": 2}.get(kind, 3)
    term_hits = sum(1 for term in terms if term and term in path_lower)
    readme_rank = 0 if name.startswith("readme") else 1
    depth = len(PurePosixPath(path).parts)
    size = int(getattr(artifact, "size", 0) or 0)
    return (-term_hits, readme_rank, kind_rank, depth, size, path_lower)


def _interview_coach_read_project_artifact(source, artifact) -> tuple[str, str]:
    from smart_agent.career import extract_project_document_text

    root = Path(getattr(source, "source_location", "")).expanduser().resolve()
    if root.is_file():
        target = root
    else:
        target = (root / str(getattr(artifact, "path", ""))).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("Project artifact path escapes source root") from exc
    kind = str(getattr(getattr(artifact, "kind", ""), "value", getattr(artifact, "kind", "")))
    if kind == "document":
        text, reader = extract_project_document_text(target)
        return text[:_INTERVIEW_COACH_PROJECT_FILE_READ_CHARS], reader
    return target.read_text(encoding="utf-8", errors="replace")[:_INTERVIEW_COACH_PROJECT_FILE_READ_CHARS], kind


def _interview_coach_project_file_payloads(
    question: str,
    project_sources: tuple | list,
    top_k: int,
) -> tuple[list[dict], list[str]]:
    from smart_agent.career import scan_project_artifacts

    terms = _interview_coach_terms(question)
    scored: list[tuple[float, tuple, dict]] = []
    fallbacks: list[tuple[tuple, dict]] = []
    warnings: list[str] = []
    for source in project_sources:
        project_id = str(getattr(source, "project_id", "") or "")
        try:
            artifacts = sorted(
                scan_project_artifacts(getattr(source, "source_location", "")),
                key=lambda item: _interview_coach_project_artifact_rank(item, terms),
            )[:_INTERVIEW_COACH_PROJECT_FILE_SCAN_LIMIT]
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"项目底层文件读取失败：{project_id or 'unknown'}: {exc}")
            continue
        for artifact in artifacts:
            rank = _interview_coach_project_artifact_rank(artifact, terms)
            try:
                text, reader = _interview_coach_read_project_artifact(source, artifact)
            except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
                warnings.append(f"项目文件片段读取失败：{project_id or 'unknown'}:{getattr(artifact, 'path', '')}: {exc}")
                continue
            if not text.strip():
                continue
            score = _interview_coach_project_file_score(
                question_terms=terms,
                source=source,
                artifact=artifact,
                text=text,
            )
            payload = {
                "project_id": project_id,
                "display_name": str(getattr(source, "display_name", "") or ""),
                "source_id": "",
                "path": str(getattr(artifact, "path", "") or ""),
                "kind": str(getattr(getattr(artifact, "kind", ""), "value", getattr(artifact, "kind", ""))),
                "reader": reader,
                "content": _interview_coach_project_file_snippet(text, terms),
                "lexical_score": round(score, 6),
            }
            if score > 0:
                scored.append((score, rank, payload))
            else:
                fallbacks.append((rank, payload))
    selected = [
        payload
        for _score, _rank, payload in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]
    ]
    if not selected:
        selected = [payload for _rank, payload in sorted(fallbacks, key=lambda item: item[0])[:top_k]]
    for index, payload in enumerate(selected, start=1):
        payload["source_id"] = f"project_file_{index}"
    return selected, warnings


_INTERVIEW_COACH_NUMBERED_POINT_RE = re.compile(r"(?m)^\s*[1-4](?:[.、)]|\s)")


def _ensure_interview_numbered_answer(answer: object) -> str:
    text = _repair_interview_text_encoding(answer).replace("\r\n", "\n").strip()
    if not text:
        return ""
    if len(_INTERVIEW_COACH_NUMBERED_POINT_RE.findall(text)) >= 4:
        return text
    paragraphs = [item.strip() for item in re.split(r"\n{2,}", text) if item.strip()]
    defaults = (
        "设计原因：说明这个方案解决了什么问题，以及为什么没有直接采用其他方案。",
        "项目实现：只引用已检索到的真实项目文件、代码和配置，不补写未经证明的细节。",
        "面试表达与边界：最后用简洁的话总结，并明确哪些是仓库事实、哪些是你本人可以确认的职责和结果。",
    )
    if len(paragraphs) >= 4:
        points = paragraphs[:4]
    else:
        points = paragraphs[:1]
        points.extend(defaults[: max(0, 4 - len(points))])
    return "\n\n".join(f"{index}. {point}" for index, point in enumerate(points, start=1))


def _interview_coach_rebuild_project_index(project_sources: tuple | list) -> int:
    from smart_agent.career import chunk_project_text, scan_project_artifacts

    store = _project_knowledge_store(_career_root() / "project-knowledge")
    project_ids = {str(source.project_id) for source in project_sources if str(getattr(source, "project_id", ""))}
    if project_ids:
        store.delete(project_ids=project_ids)
    chunks = []
    for source in project_sources:
        project_id = str(getattr(source, "project_id", "") or "")
        source_root = Path(getattr(source, "source_location", "")).expanduser().resolve()
        for artifact in scan_project_artifacts(source_root):
            try:
                text, _reader = _interview_coach_read_project_artifact(source, artifact)
            except (OSError, RuntimeError, UnicodeError, ValueError):
                continue
            if not text.strip():
                continue
            artifact_key = hashlib.sha256(
                f"{project_id}:{getattr(artifact, 'path', '')}".encode("utf-8")
            ).hexdigest()[:16]
            material_id = f"project-file-{project_id}-{artifact_key}"
            chunks.extend(
                chunk_project_text(
                    text,
                    project_id=project_id,
                    material_id=material_id,
                    source_id=material_id,
                )
            )
    return store.upsert(chunks)


def _interview_coach_knowledge_hit_payload(chunk, index: int) -> dict:
    payload = chunk.to_dict()
    return {
        "source_id": f"interview_knowledge_{index}",
        "corpus_id": payload.get("corpus_id", ""),
        "chunk_id": payload.get("chunk_id", ""),
        "topic": payload.get("topic", ""),
        "subdomain": payload.get("subdomain", ""),
        "difficulty": payload.get("difficulty", 3),
        "knowledge_type": payload.get("knowledge_type", ""),
        "text": _truncate_text(payload.get("text", ""), 1200),
        "locator": payload.get("locator") or {},
    }


def _interview_coach_collect_sources(question: str, question_type: str, top_k: int) -> tuple[dict, list[str]]:
    from smart_agent.career import InterviewKnowledgeStore, ProjectLibraryStore

    root = _career_root()
    warnings: list[str] = []
    project_sources = ()
    sources: dict[str, list[dict]] = {
        "interview_knowledge": [],
        "projects": [],
        "project_files": [],
        "project_sources": [],
    }
    if question_type in {"project", "mixed"}:
        try:
            project_sources = ProjectLibraryStore(root / "project-library", workspace=ROOT).list()
            sources["project_sources"] = [
                _interview_coach_project_source_payload(source)
                for source in project_sources
            ]
        except (OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"项目库源列表读取失败：{exc}")
        if project_sources:
            project_files, file_warnings = _interview_coach_project_file_payloads(question, project_sources, top_k)
            sources["project_files"] = project_files
            warnings.extend(file_warnings)
        try:
            valid_project_ids = {str(source.project_id).casefold() for source in project_sources}
            project_hits = (
                _project_knowledge_store(root / "project-knowledge").search(question, top_k=min(50, top_k * 3))
                if valid_project_ids
                else ()
            )
            filtered_hits = [
                hit for hit in project_hits
                if str(getattr(hit, "project_id", "") or "").casefold() in valid_project_ids
            ][:top_k]
            if project_hits and not filtered_hits:
                warnings.append("项目 RAG 命中不属于已导入项目库，已过滤；将按已登记项目文件重建。")
            if valid_project_ids and not filtered_hits:
                try:
                    indexed_count = _interview_coach_rebuild_project_index(project_sources)
                    warnings.append(f"项目 RAG 已按已导入项目文件重建（{indexed_count} 个片段）。")
                    rebuilt_hits = _project_knowledge_store(root / "project-knowledge").search(
                        question,
                        top_k=min(50, top_k * 3),
                    )
                    filtered_hits = [
                        hit for hit in rebuilt_hits
                        if str(getattr(hit, "project_id", "") or "").casefold() in valid_project_ids
                    ][:top_k]
                except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as exc:
                    warnings.append(f"项目 RAG 自动重建失败：{exc}")
            sources["projects"] = [
                _interview_coach_project_hit_payload(hit, index + 1)
                for index, hit in enumerate(filtered_hits)
            ]
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            warnings.append(f"项目 RAG 检索不可用：{exc}")
    if question_type in {"interview_knowledge", "mixed"}:
        try:
            knowledge_hits = InterviewKnowledgeStore(root / "interview-knowledge").search(question, top_k=top_k)
            sources["interview_knowledge"] = [
                _interview_coach_knowledge_hit_payload(chunk, index + 1)
                for index, chunk in enumerate(knowledge_hits)
            ]
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            warnings.append(f"八股知识库检索不可用：{exc}")
    return sources, warnings


def _interview_coach_numbered_fallback_answer(question: str, question_type: str, sources: dict) -> str:
    labels = {
        "project": "项目问题",
        "interview_knowledge": "原理问题",
        "mixed": "项目与原理结合问题",
    }
    knowledge = sources.get("interview_knowledge") or []
    projects = sources.get("projects") or []
    project_files = sources.get("project_files") or []
    project_sources = sources.get("project_sources") or []
    principle = (
        "先讲原理："
        + "；".join(_truncate_text(item.get("text", ""), 220) for item in knowledge[:2])
        if knowledge
        else "先讲原理：我会先把概念和判断依据讲清楚，再落到项目实现，不把检索片段直接当成结论。"
    )
    implementation_parts = [
        f"{item.get('display_name') or item.get('project_id')} 的 {item.get('path')}："
        f"{_truncate_text(item.get('content', ''), 260)}"
        for item in project_files[:3]
    ]
    implementation_parts.extend(_truncate_text(item.get("content", ""), 220) for item in projects[:2])
    if not implementation_parts and project_sources:
        implementation_parts.append(
            "已确认项目资料：" + "、".join(
                str(item.get("display_name") or item.get("project_id") or "").strip()
                for item in project_sources[:5]
            )
        )
    implementation = (
        "结合真实项目实现：" + "；".join(implementation_parts)
        if implementation_parts
        else "结合真实项目实现：当前检索资料不足，我不会替你补写未经确认的代码细节。"
    )
    return "\n\n".join(
        [
            f"我先按{labels.get(question_type, question_type)}给你拆成四点。",
            f"1. {principle}",
            f"2. {implementation}",
            "3. 你可以这样判断：先说明采用这个方案解决什么问题，再说明它和其他方案的取舍。",
            f"4. 面试表达与边界：{_repair_interview_text_encoding(question)}。只说仓库和个人材料能证明的职责、指标和结果。",
        ]
    )


def _interview_coach_direct_knowledge_fallback_answer(question: str, sources: dict) -> str:
    knowledge = sources.get("interview_knowledge") or []
    if knowledge:
        reference = "；".join(_truncate_text(item.get("text", ""), 260) for item in knowledge[:2])
        return (
            "简短回答："
            + reference
            + "\n\n面试里直接说重点即可，不需要套项目设计结构；如果追问细节，再展开原因和例子。"
        )
    return (
        f"简短回答：{_repair_interview_text_encoding(question)} 这个问题当前没有检索到足够的八股知识块，"
        "我不建议硬背编造答案。可以先补充对应知识资料后再问。"
    )


def _interview_coach_fallback_answer(question: str, question_type: str, sources: dict) -> str:
    if question_type == "interview_knowledge":
        return _interview_coach_direct_knowledge_fallback_answer(question, sources)
    return _interview_coach_numbered_fallback_answer(question, question_type, sources)


def _interview_coach_legacy_fallback_answer(question: str, question_type: str, sources: dict) -> str:
    labels = {
        "project": "项目问题",
        "interview_knowledge": "原理问题",
        "mixed": "项目与原理结合问题",
    }
    parts = [f"先给你一个面试老师式的回答框架：这是{labels.get(question_type, question_type)}。"]
    knowledge = sources.get("interview_knowledge") or []
    projects = sources.get("projects") or []
    project_files = sources.get("project_files") or []
    project_sources = sources.get("project_sources") or []
    if knowledge:
        parts.append("原理依据：" + "；".join(_truncate_text(item.get("text", ""), 180) for item in knowledge[:3]))
    if project_files:
        parts.append("你项目里的对应实现：" + "；".join(
            f"{item.get('display_name') or item.get('project_id')}:{item.get('path')} - "
            f"{_truncate_text(item.get('content', ''), 220)}"
            for item in project_files[:3]
        ))
    if projects:
        parts.append("项目检索片段：" + "；".join(_truncate_text(item.get("content", ""), 180) for item in projects[:3]))
    elif project_sources and question_type in {"project", "mixed"}:
        names = [str(item.get("display_name") or item.get("project_id") or "").strip() for item in project_sources[:5]]
        parts.append("已确认你的项目资料：" + "、".join(name for name in names if name))
    if len(parts) == 1:
        parts.append("目前资料还不足以支撑可靠回答，我不会替你编造项目细节。")
    parts.append(f"你的问题是：{_repair_interview_text_encoding(question)}")
    parts.append("面试表达时，先讲清原理，再落到项目中的真实实现；涉及个人职责或指标时，只说你能确认的部分。")
    return "\n\n".join(parts)


def _interview_coach_answer_with_model(
    question: str,
    question_type: str,
    sources: dict,
    context: ContextManager | None = None,
) -> tuple[str, list[str], list[str], list[str]]:
    from smart_agent.career import create_career_client

    context = context or ContextManager(model=str(load_runtime_config(ROOT).get("model") or "default"))

    client = create_career_client(
        load_runtime_config(ROOT),
        "interview_question",
        audit_root=_career_root() / "audit",
        required=False,
    )
    if client is None:
        return _interview_coach_fallback_answer(question, question_type, sources), [], ["面试辅导模型未配置，已返回检索摘要。"], []
    if question_type == "interview_knowledge":
        system_prompt = (
            "You are a patient Chinese interview teacher helping a student memorize foundation interview knowledge. "
            "Return JSON with keys answer, used_source_ids, and follow_up_suggestions. Use only supplied sources. "
            "For pure interview_knowledge questions, answer briefly and directly in natural Chinese. "
            "Give the key conclusion first, then one or two necessary explanation sentences. "
            "Do not force the project answer structure, do not mention project implementation unless the user asks to "
            "combine with a project, and do not output raw source IDs or encoding-garbled text. "
            "If the supplied knowledge is insufficient, say so clearly and do not invent facts."
        )
        answer_requirements = [
            "用中文简短回答，适合背诵和快速复述。",
            "直接给结论和必要解释，不要套原理、设计原因、项目实现、面试表达四点结构。",
            "不要强制 1、2、3、4 编号；除非自然需要，最多使用很短的要点。",
            "不要出现乱码、原始 source_id 或编造内容。",
        ]
        max_output_tokens = 900
    else:
        system_prompt = (
            "You are a patient, experienced Chinese interview teacher speaking naturally to a student. "
            "Return JSON with keys answer, used_source_ids, and follow_up_suggestions. Use only supplied sources. "
            "For project questions, prefer project_files because they are real files from the imported project "
            "library. Distinguish project_files from project RAG chunks and project_sources metadata. "
            "Answer like a human teacher, not a retrieval report: first explain the principle in plain Chinese, "
            "then connect it to the concrete implementation in the student's project, then give a concise "
            "interview-ready expression. Use natural transitions and complete sentences; do not output "
            "robotic labels, placeholder templates, raw source IDs, or encoding-garbled text. "
            "Organize the answer into exactly four clearly numbered points using 1., 2., 3., and 4.: "
            "principle, design reasoning, concrete project implementation, and interview wording with boundaries. "
            "Clearly separate repository-proven facts from the student's personal responsibility and metrics. "
            "If both foundation knowledge and project context are relevant, combine them. "
            "Never invent project details, metrics, files, or experience not present in sources."
        )
        answer_requirements = [
            "用中文回答。",
            "以面试老师的视角交流，像真人一样解释，不要只罗列检索片段。",
            "先解释原理，再对应项目真实实现，最后给出可直接用于面试的表达。",
            "必须给出 1、2、3、4 四个编号论点，分别覆盖原理、设计原因、项目实现、面试表达与边界。",
            "不要出现乱码、原始 source_id、机械化模板标题或空泛套话。",
            "如果资料不足，明确说明缺口，不编造。",
        ]
        max_output_tokens = 1800
    try:
        if context.should_auto_compact():
            context.compact_messages()
        context_history = _interview_coach_history(context)
        response = client.complete_json(
            system_prompt=system_prompt,
            user_payload={
                "question": question,
                "question_type": question_type,
                "sources": sources,
                "conversation_history": context_history,
                "answer_requirements": answer_requirements,
            },
            max_output_tokens=max_output_tokens,
        ).value
    except (OSError, RuntimeError, ValueError) as exc:
        return _interview_coach_fallback_answer(question, question_type, sources), [], [f"面试辅导模型调用失败，已返回检索摘要：{exc}"], []
    if question_type == "interview_knowledge":
        answer = _repair_interview_text_encoding(response.get("answer", "")).strip()
        if len(answer) > 2000:
            answer = answer[:2000] + "..."
    else:
        answer = _ensure_interview_numbered_answer(response.get("answer", ""))
        if len(answer) > 5000:
            answer = answer[:5000] + "..."
    if not answer:
        answer = _interview_coach_fallback_answer(question, question_type, sources)
    context.add_message({"role": "assistant", "content": answer})
    used = response.get("used_source_ids")
    suggestions = response.get("follow_up_suggestions")
    return (
        answer,
        [str(item) for item in used[:10]] if isinstance(used, list) else [],
        [],
        [str(item) for item in suggestions[:5]] if isinstance(suggestions, list) else [],
    )


@app.post("/api/career/interview-coach/ask")
async def career_interview_coach_ask(payload: dict):
    from smart_agent.career.workspace_context import ContextConflict
    from smart_agent.web.workspace_context import read, store
    question = " ".join(str(payload.get("question") or "").split())
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if len(question) > 12000:
        raise HTTPException(413, "Question exceeds 12000 characters")
    try:
        top_k = int(payload.get("top_k", 5))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="top_k must be an integer") from None
    top_k = max(1, min(top_k, 12))
    session_id = str(payload.get("session_id") or "").strip()
    state = read(session_id) if session_id else store().create("coach")
    session_id = state["session_id"]
    if state.get("status") != "active":
        raise HTTPException(409, "Session is not active")
    revision = payload.get("context_revision", state["context_revision"])
    if revision != state["context_revision"]:
        raise HTTPException(409, "Workspace context changed; reload before asking")
    context = ContextManager(model=str(load_runtime_config(ROOT).get("model") or "default"),
                             messages=state.get("coach_messages", []).copy())
    context.add_message({"role": "user", "content": question})
    try:
        question_type = _classify_interview_coach_question(question)
        sources, warnings = await asyncio.to_thread(_interview_coach_collect_sources, question, question_type, top_k)
        answer, used_source_ids, model_warnings, suggestions = await asyncio.to_thread(
            _interview_coach_answer_with_model, question, question_type, sources, context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    warnings.extend(model_warnings)
    # Store complete turns even when generation returns a labeled fallback.
    messages = (state.get("coach_messages", []) + [
        {"role": "user", "content": question}, {"role": "assistant", "content": answer},
    ])[-8:]
    try:
        state = store().commit(session_id, revision, "coach_turn_completed",
                               {"coach_messages": messages},
                               str(payload.get("request_id") or uuid.uuid4().hex))
    except ContextConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    return {
        "question": question,
        "session_id": session_id,
        "context_revision": state["context_revision"],
        "question_type": question_type,
        "answer": answer,
        "used_source_ids": used_source_ids,
        "follow_up_suggestions": suggestions,
        "sources": sources,
        "warnings": warnings,
    }


def _csv_values(value: str, *, default: str) -> list[str]:
    items = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return items or [default]


def _derive_interview_topic_from_role(role_title: str) -> str:
    text = str(role_title or "").strip()
    text = re.sub(r"^\s*(?:岗位|职位|目标岗位|目标职位|role|position)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[（(][^）)]*[）)]", "", text)
    text = re.split(r"[｜|/@,，;；]", text, maxsplit=1)[0].strip()
    text = re.sub(r"\s+", "", text)
    suffixes = (
        "实习生",
        "实习岗位",
        "实习",
        "高级工程师",
        "资深工程师",
        "工程师",
        "研究员",
        "专家",
        "开发岗",
        "算法岗",
        "岗位",
        "方向",
        "校招",
        "社招",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                changed = True
                break
    return text.strip()[:64]


def _interview_topic_values(topics: str, role_title: str) -> list[str]:
    value = str(topics or "").strip()
    fixed_defaults = {"backend", "python", "backend,python", "general"}
    if not value or value.casefold() in fixed_defaults:
        value = _derive_interview_topic_from_role(role_title)
    return _csv_values(value, default="general")


def _safe_upload_relative_name(filename: str, *, default: str = "interview-knowledge.txt") -> str:
    raw = str(filename or default).replace("\\", "/").strip() or default
    parts: list[str] = []
    for part in PurePosixPath(raw).parts:
        if part in {"", ".", "..", "/"} or part.endswith(":"):
            continue
        cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._ -]+", "-", part).strip(" .")
        if cleaned:
            parts.append(cleaned[:120])
    return "/".join(parts) or default


def _safe_staging_path(root: Path, relative_name: str) -> Path:
    root = root.resolve()
    target = root.joinpath(*PurePosixPath(relative_name).parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Upload path escapes the staging directory")
    return target


async def _write_upload_to_path(upload: UploadFile, target: Path, *, max_bytes: int) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with target.open("wb") as handle:
        while True:
            chunk = await upload.read(INTERVIEW_KNOWLEDGE_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise HTTPException(status_code=413, detail="Interview knowledge upload exceeds 5 GB")
            handle.write(chunk)
    if total <= 0:
        raise ValueError("Interview knowledge file is empty")
    return total


def _public_interview_knowledge_result(result: dict) -> dict:
    if isinstance(result.get("source"), dict):
        result["source"].pop("source_path", None)
    return result


def _compact_interview_knowledge_error(detail: object) -> str:
    if isinstance(detail, str):
        message = detail.strip()
        crashed = re.search(r"crashed:\s*(.*?)(?:\nTraceback|\Z)", message, flags=re.DOTALL)
        if crashed:
            message = crashed.group(1).strip()
        validation = re.search(r"Input validation error.*?:\s*(.*?)(?:\nInput was:|\Z)", message, flags=re.DOTALL)
        if validation:
            message = validation.group(1).strip()
        return message[:1000] or "Interview knowledge import failed"
    if isinstance(detail, dict):
        message = str(detail.get("message") or detail.get("error") or "").strip()
        if message:
            return message
    try:
        serialized = json.dumps(detail, ensure_ascii=False)
    except TypeError:
        serialized = str(detail)
    return serialized[:1000] or "Interview knowledge import failed"


def _interview_knowledge_failure(input_name: str, exc: BaseException) -> dict:
    if isinstance(exc, HTTPException):
        return {
            "input_name": input_name,
            "status_code": exc.status_code,
            "error": _compact_interview_knowledge_error(exc.detail),
            "detail": exc.detail,
        }
    return {
        "input_name": input_name,
        "error_type": exc.__class__.__name__,
        "error": _compact_interview_knowledge_error(str(exc)),
    }


def _interview_knowledge_batch_payload(
    *,
    batch_id: str,
    imported: list[dict],
    skipped: list[dict],
    failed: list[dict],
) -> dict:
    chunk_count = sum(int(item.get("corpus", {}).get("chunk_count", 0)) for item in imported)
    return {
        "batch_id": batch_id,
        "imported": imported,
        "skipped": skipped,
        "failed": failed,
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "chunk_count": chunk_count,
    }


def _validate_interview_knowledge_import_result(result: dict, *, input_name: str) -> None:
    corpus = result.get("corpus") if isinstance(result, dict) else None
    if not isinstance(corpus, dict):
        raise ValueError(f"Interview knowledge import for {input_name} did not return corpus metadata")
    chunk_count = int(corpus.get("chunk_count") or 0)
    if chunk_count <= 0:
        raise ValueError(f"Interview knowledge import for {input_name} produced no knowledge chunks")
    graph_rag = result.get("graph_rag")
    if not isinstance(graph_rag, dict):
        raise ValueError(f"Interview knowledge import for {input_name} did not return Graph RAG metadata")
    index_status = str(graph_rag.get("index_status") or "").strip()
    node_count = int(graph_rag.get("node_count") or 0)
    if index_status != "ready":
        raise ValueError(f"Graph RAG index for {input_name} is not ready: {index_status or 'unknown'}")
    if node_count != chunk_count:
        raise ValueError(
            f"Graph RAG index for {input_name} has {node_count} nodes but the corpus has {chunk_count} chunks"
        )


def _import_interview_knowledge_path(
    source: Path,
    *,
    display_name: str,
    domains: str,
    topics: str,
    role_title: str,
    subdomains: str,
    authority_level: int,
    difficulty: int,
    input_name: str,
    pdf_mode: str = "normal",
) -> dict:
    result = _execute_career_tool(
        "import_interview_knowledge_source",
        {
            "tenant_id": _career_tenant_id() or "legacy-web-user",
            "name": str(display_name or Path(input_name).stem or source.stem).strip(),
            "domains": _csv_values(domains, default="general"),
            "topics": _interview_topic_values(topics, role_title),
            "subdomains": [item.strip() for item in str(subdomains).split(",") if item.strip()],
            "authority_level": authority_level,
            "difficulty": difficulty,
            "source_path": str(source),
            "source_name": input_name,
            "pdf_mode": pdf_mode,
        },
        stage="interview",
    )
    result = _public_interview_knowledge_result(result)
    _validate_interview_knowledge_import_result(result, input_name=input_name)
    result["input_name"] = input_name
    return result


def _import_interview_knowledge_zip(
    archive_path: Path,
    *,
    staging: Path,
    domains: str,
    topics: str,
    role_title: str,
    subdomains: str,
    authority_level: int,
    difficulty: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    imported: list[dict] = []
    skipped: list[dict] = []
    failed: list[dict] = []
    extracted_root = staging / f"zip-{archive_path.stem}-{uuid.uuid4().hex}"
    extracted_root.mkdir(parents=True, exist_ok=True)
    total_uncompressed = 0
    with zipfile.ZipFile(archive_path) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) > INTERVIEW_KNOWLEDGE_MAX_MEMBERS:
            raise ValueError("Interview knowledge archive contains too many files")
        for info in infos:
            relative_name = _safe_upload_relative_name(info.filename)
            suffix = Path(relative_name).suffix.lower()
            if suffix not in INTERVIEW_KNOWLEDGE_SUFFIXES:
                skipped.append({"input_name": info.filename, "reason": "unsupported_file_type"})
                continue
            total_uncompressed += int(info.file_size)
            if total_uncompressed > INTERVIEW_KNOWLEDGE_UPLOAD_MAX_BYTES:
                raise HTTPException(status_code=413, detail="Interview knowledge archive expands beyond 5 GB")
            target = _safe_staging_path(extracted_root, relative_name)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(info) as source_handle, target.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle, length=INTERVIEW_KNOWLEDGE_CHUNK_BYTES)
                imported.append(
                    _import_interview_knowledge_path(
                        target,
                        display_name=Path(relative_name).with_suffix("").as_posix(),
                        domains=domains,
                        topics=topics,
                        role_title=role_title,
                        subdomains=subdomains,
                        authority_level=authority_level,
                        difficulty=difficulty,
                        input_name=info.filename,
                    )
                )
            except zipfile.BadZipFile as exc:
                failed.append(_interview_knowledge_failure(info.filename, exc))
            except HTTPException as exc:
                failed.append(_interview_knowledge_failure(info.filename, exc))
            except (OSError, ValueError, RuntimeError) as exc:
                failed.append(_interview_knowledge_failure(info.filename, exc))
    return imported, skipped, failed


@app.post("/api/career/interview-knowledge")
async def career_interview_knowledge_upload(
    file: UploadFile,
    name: str = Form(""),
    domains: str = Form("general"),
    topics: str = Form(""),
    role_title: str = Form(""),
    subdomains: str = Form(""),
    authority_level: int = Form(3),
    difficulty: int = Form(3),
    pdf_mode: str = Form("normal"),
):
    """Import a user-owned interview note into the isolated knowledge library."""
    # Direct callers (including lightweight tests) may omit FastAPI form defaults.
    if not isinstance(pdf_mode, str):
        pdf_mode = "normal"
    if not isinstance(domains, str):
        domains = "general"
    if not isinstance(topics, str):
        topics = ""
    if not isinstance(role_title, str):
        role_title = ""
    if not isinstance(subdomains, str):
        subdomains = ""
    filename = file.filename or "interview-knowledge.txt"
    suffix = Path(filename).suffix.lower() or ".txt"
    if suffix not in INTERVIEW_KNOWLEDGE_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Interview knowledge supports TXT, Markdown, PDF, DOCX, IPYNB, PNG, JPG/JPEG, BMP, WEBP, "
                "and TIFF files"
            ),
        )
    staging = ROOT / ".tmp" / "career-interview-knowledge"
    staging.mkdir(parents=True, exist_ok=True)
    source = staging / f"{uuid.uuid4().hex}{suffix}"
    try:
        await _write_upload_to_path(file, source, max_bytes=INTERVIEW_KNOWLEDGE_UPLOAD_MAX_BYTES)
        return _import_interview_knowledge_path(
            source,
            display_name=str(name or Path(filename).stem).strip(),
            domains=domains,
            topics=topics,
            role_title=role_title,
            subdomains=subdomains,
            authority_level=authority_level,
            difficulty=difficulty,
            input_name=filename,
            pdf_mode=pdf_mode,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/career/interview-knowledge/batch")
async def career_interview_knowledge_batch_upload(
    files: list[UploadFile] = File(...),
    name: str = Form(""),
    domains: str = Form("general"),
    topics: str = Form(""),
    role_title: str = Form(""),
    subdomains: str = Form(""),
    authority_level: int = Form(3),
    difficulty: int = Form(3),
    pdf_modes: str = Form("{}"),
):
    """Import many interview knowledge files from multi-file, folder, or zip uploads."""
    if not files:
        raise HTTPException(status_code=400, detail="Select at least one interview knowledge file")
    batch_id = f"batch-{uuid.uuid4().hex}"
    staging = ROOT / ".tmp" / "career-interview-knowledge" / batch_id
    staging.mkdir(parents=True, exist_ok=True)
    imported: list[dict] = []
    skipped: list[dict] = []
    failed: list[dict] = []
    uploaded_bytes = 0
    explicit_name = str(name or "").strip()
    pdf_modes_value = pdf_modes if isinstance(pdf_modes, str) else "{}"
    try:
        raw_pdf_modes = json.loads(pdf_modes_value or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="pdf_modes must be a JSON object") from exc
    if not isinstance(raw_pdf_modes, dict):
        raise HTTPException(status_code=400, detail="pdf_modes must be a JSON object")
    normalized_pdf_modes: dict[str, str] = {}
    for key, value in raw_pdf_modes.items():
        mode = str(value or "normal").strip().casefold()
        if mode not in {"normal", "qa"}:
            raise HTTPException(status_code=400, detail=f"Invalid PDF mode for {key}: {mode}")
        normalized_pdf_modes[str(key)] = mode
    for upload in files:
        input_name = upload.filename or "interview-knowledge.txt"
        relative_name = _safe_upload_relative_name(input_name)
        suffix = Path(relative_name).suffix.lower()
        if suffix not in {*INTERVIEW_KNOWLEDGE_SUFFIXES, ".zip"}:
            skipped.append({"input_name": input_name, "reason": "unsupported_file_type"})
            continue
        target = _safe_staging_path(staging, f"{uuid.uuid4().hex}/{relative_name}")
        try:
            uploaded_bytes += await _write_upload_to_path(
                upload,
                target,
                max_bytes=max(1, INTERVIEW_KNOWLEDGE_UPLOAD_MAX_BYTES - uploaded_bytes),
            )
            if suffix == ".zip":
                zip_imported, zip_skipped, zip_failed = _import_interview_knowledge_zip(
                    target,
                    staging=staging,
                    domains=domains,
                    topics=topics,
                    role_title=role_title,
                    subdomains=subdomains,
                    authority_level=authority_level,
                    difficulty=difficulty,
                )
                imported.extend(zip_imported)
                skipped.extend(zip_skipped)
                failed.extend(zip_failed)
                continue
            display_name = explicit_name if explicit_name and len(files) == 1 else Path(relative_name).with_suffix("").as_posix()
            imported.append(
                _import_interview_knowledge_path(
                    target,
                    display_name=display_name,
                    domains=domains,
                    topics=topics,
                    role_title=role_title,
                    subdomains=subdomains,
                    authority_level=authority_level,
                    difficulty=difficulty,
                    input_name=input_name,
                    pdf_mode=normalized_pdf_modes.get(input_name, "normal") if suffix == ".pdf" else "normal",
                )
            )
        except zipfile.BadZipFile as exc:
            failed.append(_interview_knowledge_failure(input_name, exc))
        except HTTPException as exc:
            failed.append(_interview_knowledge_failure(input_name, exc))
        except (OSError, ValueError, RuntimeError) as exc:
            failed.append(_interview_knowledge_failure(input_name, exc))
    if not imported and failed:
        detail = _interview_knowledge_batch_payload(
            batch_id=batch_id,
            imported=imported,
            skipped=skipped,
            failed=failed,
        )
        detail["message"] = f"Interview knowledge batch import failed for {len(failed)} file(s)"
        raise HTTPException(status_code=400, detail=detail)
    if not imported:
        detail = _interview_knowledge_batch_payload(
            batch_id=batch_id,
            imported=imported,
            skipped=skipped,
            failed=failed,
        )
        detail["message"] = "No supported interview knowledge files were found"
        raise HTTPException(status_code=400, detail=detail)
    return _interview_knowledge_batch_payload(
        batch_id=batch_id,
        imported=imported,
        skipped=skipped,
        failed=failed,
    )


@app.post("/api/career/interviews/start")
async def career_interview_start(data: dict):
    """Create an isolated interview session and return its first question."""
    from smart_agent.career.workspace_context import ContextConflict
    from smart_agent.web.workspace_context import read as read_context
    from smart_agent.web.workspace_context import store as context_store
    workspace = None
    if data.get("workspace_session_id"):
        workspace = read_context(str(data["workspace_session_id"]))
        if workspace["context_revision"] != data.get("context_revision"):
            raise HTTPException(409, "Workspace context changed; reload before starting")
        data = {**data, "resume_run_id": workspace.get("active_run_id", ""),
                "input_bundle_id": workspace.get("input_bundle_id", "")}
        if workspace.get("input_bundle_id"):
            bundle = read_context(workspace["input_bundle_id"])
            if not bundle.get("confirmed"):
                raise HTTPException(409, "Confirm interview inputs first")
            from smart_agent.career import InterviewKnowledgeLibrary
            tenant = _career_tenant_id() or "legacy-web-user"
            corpora = InterviewKnowledgeLibrary(_career_root() / "interview-knowledge").metadata_summary(tenant_id=tenant)["corpus_ids"]
            prefix = "input:" + bundle["session_id"] + ":"
            session = await asyncio.to_thread(_execute_career_tool, "create_interview_session", {
                "mode": data.get("mode", "realistic_simulation"), "tenant_id": tenant,
                "jd_id": prefix + "jd", "resume_document_id": prefix + "resume",
                "resume_document_revision": 1, "corpus_ids": list(corpora),
                "max_turns": 8, "duration_minutes": 30,
            }, stage="interview", enable_model_runtime=True)
            try:
                context_store().commit(workspace["session_id"], workspace["context_revision"],
                                       "interview_started", {"active_interview_id": session["interview_id"]},
                                       str(data.get("request_id") or uuid.uuid4().hex))
            except ContextConflict as exc:
                from smart_agent.career import InterviewSessionStore
                InterviewSessionStore(_career_root() / "interviews").delete(session["interview_id"])
                raise HTTPException(409, str(exc)) from exc
            question = await asyncio.to_thread(_execute_next_interview_question_with_trace,
                                              session["interview_id"], source="interview_start")
            return {"session": session, "question": question}
    from smart_agent.career import (
        CareerStore,
        InterviewKnowledgeLibrary,
        InterviewMode,
        JDDocument,
        ResumeDocument,
        ResumeDocumentStatus,
        ResumeRunService,
    )

    _root, store, _material_store, _candidate_store = _career_stores()
    try:
        mode = InterviewMode(data.get("mode", InterviewMode.REALISTIC_SIMULATION.value))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Unsupported interview mode") from exc
    resume_run_id = str(data.get("resume_run_id") or "").strip()
    requested_jd_id = str(data.get("jd_id") or "").strip()
    requested_resume_id = str(data.get("resume_document_id") or "").strip()
    raw_resume_revision = data.get("resume_document_revision")
    requested_resume_revision = None
    if raw_resume_revision not in (None, ""):
        try:
            requested_resume_revision = int(raw_resume_revision)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="resume_document_revision must be an integer") from exc
        if requested_resume_revision < 1:
            raise HTTPException(status_code=400, detail="resume_document_revision must be at least 1")
    if not resume_run_id:
        raise HTTPException(
            status_code=409,
            detail="Select a current JD session before starting an interview.",
        )
    run = _resolve_resume_run(store, resume_run_id)
    if requested_jd_id and requested_jd_id != run.jd_ref.record_id:
        raise HTTPException(status_code=400, detail="Requested JD does not belong to the current session.")
    if requested_resume_id and (
        run.resume_document_ref is None
        or requested_resume_id != run.resume_document_ref.record_id
    ):
        raise HTTPException(status_code=400, detail="Requested resume does not belong to the current session.")
    if requested_resume_revision is not None and (
        run.resume_document_ref is None
        or requested_resume_revision != run.resume_document_ref.revision
    ):
        raise HTTPException(status_code=400, detail="Requested resume revision does not match the current version.")
    jd = store.get(run.jd_ref.record_id, run.jd_ref.revision, expected_type=JDDocument)
    resume = None
    if run.resume_document_ref is not None:
        # The document is the source of truth for verification. Older runs
        # could be downgraded to GENERATED when a verified version was
        # selected, even though the selected ResumeDocument stayed verified.
        if requested_resume_revision is None:
            # Older clients sent only an ID; preserve their latest-verified
            # behavior while newer clients pin the selected revision below.
            latest_resume = store.get(
                run.resume_document_ref.record_id,
                expected_type=ResumeDocument,
            )
            candidate_resume = latest_resume
        else:
            candidate_resume = store.get(
                run.resume_document_ref.record_id,
                run.resume_document_ref.revision,
                expected_type=ResumeDocument,
            )
            if candidate_resume.status is not ResumeDocumentStatus.VERIFIED:
                # Compatibility for runs persisted before verification
                # advanced the bound ResumeDocument reference to its new
                # revision.
                latest_resume = store.get(
                    run.resume_document_ref.record_id,
                    expected_type=ResumeDocument,
                )
                if latest_resume.status is ResumeDocumentStatus.VERIFIED:
                    candidate_resume = latest_resume
                    if requested_resume_revision is not None and (
                        latest_resume.revision != requested_resume_revision
                    ):
                        # Repair legacy runs whose verification advanced the
                        # document revision before the run reference was
                        # updated. This is the same logical selected version,
                        # not a switch to another generated resume.
                        run = ResumeRunService(store).attach_document(
                            run.resume_run_id,
                            latest_resume,
                        )
                        requested_resume_revision = latest_resume.revision
        if candidate_resume.status is ResumeDocumentStatus.VERIFIED:
            resume = candidate_resume
        elif requested_resume_id:
            raise HTTPException(status_code=409, detail="Current session resume has not passed verification.")
    elif requested_resume_id:
        raise HTTPException(status_code=409, detail="Current session has no generated resume. Generate and verify it first.")
    if mode is InterviewMode.REALISTIC_SIMULATION and resume is None:
        raise HTTPException(
            status_code=409,
            detail="Realistic interview requires a resume that has passed verification. Generate and verify the current resume first.",
        )
    knowledge_library = InterviewKnowledgeLibrary(_career_root() / "interview-knowledge")
    corpus_ids = tuple(dict.fromkeys(str(item).strip() for item in (data.get("corpus_ids") or ()) if str(item).strip()))
    tenant_id = _career_tenant_id() or "legacy-web-user"
    if not corpus_ids:
        corpus_ids = tuple(knowledge_library.metadata_summary(tenant_id=tenant_id)["corpus_ids"])
    if not corpus_ids:
        raise HTTPException(status_code=409, detail="Import at least one interview knowledge source first.")
    session = _execute_career_tool(
        "create_interview_session",
        {
            "mode": mode.value,
            "tenant_id": tenant_id,
            "resume_run_id": resume_run_id or None,
            "jd_id": jd.jd_id if jd else None,
            "resume_document_id": resume.resume_document_id if resume else None,
            "resume_document_revision": (
                resume.revision
                if resume is not None and requested_resume_revision is not None
                else None
            ),
            "corpus_ids": list(corpus_ids),
            "random_seed": int(data.get("random_seed", 0)),
            "max_turns": int(data.get("max_turns", 8)),
            "duration_minutes": int(data.get("duration_minutes", 30)),
        },
        stage="interview",
        enable_model_runtime=True,
    )
    if workspace is not None:
        try:
            context_store().commit(workspace["session_id"], workspace["context_revision"],
                                   "interview_started", {"active_interview_id": session["interview_id"]},
                                   str(data.get("request_id") or uuid.uuid4().hex))
        except ContextConflict as exc:
            from smart_agent.career import InterviewSessionStore
            InterviewSessionStore(_career_root() / "interviews").delete(session["interview_id"])
            raise HTTPException(409, str(exc)) from exc
    question = _execute_next_interview_question_with_trace(
        session["interview_id"],
        source="interview_start",
    )
    return {"session": session, "question": question}


@app.get("/api/career/interviews/{interview_id}")
async def career_interview_get(interview_id: str):
    return _localize_interview_session_payload(_execute_career_tool(
        "get_interview_session",
        {"interview_id": interview_id},
        stage="interview",
    ))


@app.get("/api/career/interviews/{interview_id}/audit")
async def career_interview_audit(interview_id: str):
    """Return durable per-round interviewer, answer, and RAG audit data."""
    from smart_agent.career import InterviewSessionStore

    try:
        state = InterviewSessionStore(_career_root() / "interviews").get(interview_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Interview session not found: {interview_id}") from exc
    return _interview_audit_payload(state)


@app.post("/api/career/interviews/{interview_id}/answer")
async def career_interview_answer(interview_id: str, data: dict):
    answer = str(data.get("answer") or "").strip()
    if not answer:
        raise HTTPException(status_code=400, detail="answer is required")
    return _execute_career_tool(
        "submit_interview_answer",
        {"interview_id": interview_id, "answer": answer},
        stage="interview",
        enable_model_runtime=True,
    )


@app.post("/api/career/interviews/{interview_id}/answer/process")
async def career_interview_answer_process(interview_id: str, data: dict):
    answer = str(data.get("answer") or "").strip()
    if not answer:
        raise HTTPException(status_code=400, detail="answer is required")
    return _execute_career_tool(
        "process_interview_answer",
        {
            "interview_id": interview_id,
            "answer": answer,
            "question_id": str(data.get("question_id") or "").strip(),
        },
        stage="interview",
        enable_model_runtime=True,
    )


@app.post("/api/career/interviews/{interview_id}/question/depth")
async def career_interview_depth_question(interview_id: str, data: dict):
    refs = data.get("candidate_knowledge_refs", [])
    if not isinstance(refs, list):
        raise HTTPException(status_code=400, detail="candidate_knowledge_refs must be an array")
    return _execute_career_tool(
        "generate_interview_depth_question",
        {"interview_id": interview_id, "candidate_knowledge_refs": refs},
        stage="interview",
        enable_model_runtime=True,
    )


@app.post("/api/career/interviews/{interview_id}/question/breadth")
async def career_interview_breadth_question(interview_id: str, data: dict | None = None):
    data = data or {}
    refs = data.get("candidate_knowledge_refs", [])
    if not isinstance(refs, list):
        raise HTTPException(status_code=400, detail="candidate_knowledge_refs must be an array")
    return _execute_career_tool(
        "generate_interview_breadth_question",
        {"interview_id": interview_id, "candidate_knowledge_refs": refs},
        stage="interview",
        enable_model_runtime=True,
    )


@app.post("/api/career/interviews/{interview_id}/answer/retry")
async def career_interview_answer_retry(interview_id: str, data: dict):
    answer = str(data.get("answer") or "").strip()
    if not answer:
        raise HTTPException(status_code=400, detail="answer is required")
    retry_event_count = 0
    try:
        retry_event_count = len(_interview_runtime_direct().session_store.get(interview_id).retry_events)
    except (KeyError, ValueError):
        pass
    try:
        payload = _execute_career_tool(
            "retry_interview_answer",
            {
                "interview_id": interview_id,
                "answer": answer,
                "expected_revision": data.get("expected_revision"),
                "max_attempts": data.get("max_attempts", 2),
            },
            stage="interview",
            enable_model_runtime=True,
        )
    except HTTPException:
        # The runtime has already persisted the failed retry attempt when the
        # exception is transient. Validation and authorization failures never
        # create a retry event.
        try:
            state = _interview_runtime_direct().session_store.get(interview_id)
            if len(state.retry_events) > retry_event_count:
                latest = state.retry_events[-1]
                agent_event_hub.publish(
                    "interview_answer_retry_failed",
                    agent="Interviewer Agent",
                    session_id=interview_id,
                    status=latest.get("status", "failed"),
                    elapsed_ms=latest.get("elapsed_ms", 0),
                    input_summary=f"retry answer; question_id={latest.get('question_id', '')}",
                    output_summary=f"attempt={latest.get('attempt', '')}; error={latest.get('error', '')}",
                )
        except (KeyError, ValueError):
            pass
        raise
    retry_events = [item for item in payload.get("retry_events", []) if isinstance(item, dict)]
    latest = retry_events[-1] if retry_events else {}
    agent_event_hub.publish(
        "interview_answer_retry_finished",
        agent="Interviewer Agent",
        session_id=interview_id,
        status=latest.get("status", "succeeded"),
        elapsed_ms=latest.get("elapsed_ms", 0),
        input_summary=f"retry answer; question_id={latest.get('question_id', '')}",
        output_summary=f"attempt={latest.get('attempt', '')}; status={latest.get('status', 'succeeded')}",
    )
    return payload


@app.post("/api/career/interviews/{interview_id}/question")
async def career_interview_question(interview_id: str):
    return _execute_next_interview_question_with_trace(interview_id, source="manual_next_question")


@app.post("/api/career/interviews/{interview_id}/voice/start")
async def career_interview_voice_start(interview_id: str, data: dict):
    """Start a direct voice stream for an active Interviewer session."""
    stream_id = str(data.get("stream_id") or f"voice-{uuid.uuid4().hex[:16]}").strip()
    if not stream_id:
        raise HTTPException(status_code=400, detail="stream_id is required")
    try:
        interview_runtime = _interview_runtime_direct()
        voice_runtime = _voice_session_runtime()
        asr, tts = _voice_adapters()
        question = _direct_next_interview_question_with_trace(
            interview_runtime,
            interview_id,
            source="voice_start",
        )
        playback = voice_runtime.start(
            interview_id=interview_id,
            stream_id=stream_id,
            prompt_text=question.text,
            asr=asr,
            tts=tts,
            submit_final=lambda text: interview_runtime.submit_answer(interview_id, text),
            media_type=str(data.get("media_type") or "audio/webm"),
        )
        return {
            "interview_id": interview_id,
            "stream_id": stream_id,
            "asr": asr.name,
            "tts": tts.name,
            "question": _interview_question_payload_direct(interview_id, question),
            "playback": playback.to_dict(),
        }
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/career/interviews/{interview_id}/voice/chunk")
async def career_interview_voice_chunk(interview_id: str, data: dict):
    """Accept one audio chunk; partial ASR never enters the Interview Session."""
    stream_id = str(data.get("stream_id") or "").strip()
    audio_base64 = str(data.get("audio_base64") or "").strip()
    if not stream_id:
        raise HTTPException(status_code=400, detail="stream_id is required")
    if not audio_base64:
        raise HTTPException(status_code=400, detail="audio_base64 is required")
    try:
        audio = base64.b64decode(audio_base64, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="audio_base64 is invalid") from exc
    try:
        voice_runtime = _voice_session_runtime()
        interview_runtime = _interview_runtime_direct()
        result = voice_runtime.push_audio_detailed(
            interview_id=interview_id,
            stream_id=stream_id,
            audio=audio,
            save_recording=bool(data.get("save_recording", False)),
            media_type=str(data.get("media_type") or "audio/webm"),
            consent=bool(data.get("consent", False)),
            silence_ms=int(data.get("silence_ms", 0) or 0),
            elapsed_ms=int(data.get("elapsed_ms", 0) or 0),
            auto_finalize=bool(data.get("auto_finalize", True)),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (PermissionError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    answer_payload = (
        _interview_answer_result_payload_direct(interview_runtime, result.answer_result)
        if result.answer_result is not None
        else None
    )
    next_question = None
    playback = None
    if result.answer_result is not None and result.answer_result.state.status.value != "completed":
        question = _direct_next_interview_question_with_trace(
            interview_runtime,
            interview_id,
            source="voice_auto_next",
        )
        next_question = _interview_question_payload_direct(interview_id, question)
        playback = voice_runtime.speak(
            interview_id=interview_id,
            stream_id=stream_id,
            text=question.text,
        ).to_dict()
    elif result.answer_result is not None:
        voice_runtime.close(interview_id=interview_id, stream_id=stream_id)
    return {
        "interview_id": interview_id,
        "stream_id": stream_id,
        "events": [event.to_dict() for event in result.events],
        "accepted_final": result.accepted_final,
        "recordings": [recording.to_dict() for recording in result.recordings],
        "turn_detection": result.turn_detection.to_dict(),
        "final_reason": result.final_reason,
        "answer_result": answer_payload,
        "next_question": next_question,
        "playback": playback,
    }


@app.post("/api/career/interviews/{interview_id}/voice/stop")
async def career_interview_voice_stop(interview_id: str, data: dict):
    """Finalize ASR; only the final transcript is submitted as one answer."""
    stream_id = str(data.get("stream_id") or "").strip()
    if not stream_id:
        raise HTTPException(status_code=400, detail="stream_id is required")
    interview_runtime = _interview_runtime_direct()
    voice_runtime = _voice_session_runtime()
    try:
        event, submitted, answer_result = voice_runtime.stop(
            interview_id=interview_id,
            stream_id=stream_id,
            reason="manual_stop",
        )
        answer_payload = (
            _interview_answer_result_payload_direct(interview_runtime, answer_result)
            if answer_result is not None
            else None
        )
        next_question = None
        playback = None
        if answer_result is not None and answer_result.state.status.value != "completed":
            question = _direct_next_interview_question_with_trace(
                interview_runtime,
                interview_id,
                source="voice_stop_next",
            )
            next_question = _interview_question_payload_direct(interview_id, question)
            playback = voice_runtime.speak(
                interview_id=interview_id,
                stream_id=stream_id,
                text=question.text,
            ).to_dict()
        else:
            voice_runtime.close(interview_id=interview_id, stream_id=stream_id)
        return {
            "interview_id": interview_id,
            "stream_id": stream_id,
            "event": event.to_dict() if event is not None else None,
            "submitted": submitted,
            "final_reason": "manual_stop" if submitted else "",
            "answer_result": answer_payload,
            "next_question": next_question,
            "playback": playback,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/career/interviews/{interview_id}/voice/interrupt")
async def career_interview_voice_interrupt(interview_id: str, data: dict):
    """Cancel current ASR stream and unfinished TTS playback without touching the text session."""
    stream_id = str(data.get("stream_id") or "").strip()
    if not stream_id:
        raise HTTPException(status_code=400, detail="stream_id is required")
    try:
        playback_id = _voice_session_runtime().interrupt(
            interview_id=interview_id,
            stream_id=stream_id,
            playback_id=str(data.get("playback_id") or "").strip() or None,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "interview_id": interview_id,
        "stream_id": stream_id,
        "cancelled": True,
        "playback_id": playback_id,
    }


@app.websocket("/ws/career/interviews/{interview_id}/voice")
async def websocket_career_interview_voice(websocket: WebSocket, interview_id: str):
    """Realtime voice relay for Interviewer Agent hot path."""
    await websocket.accept()
    stream_id = ""
    tokens = None
    try:
        user = await asyncio.to_thread(_auth_service().user_from_token, _websocket_token(websocket))
        if user is None:
            await websocket.send_json({"type": "error", "content": "请先登录"})
            return
        tokens = _set_request_user(user)
        while True:
            message = await websocket.receive_json()
            message_type = str(message.get("type") or "").strip().lower()
            if message_type == "start":
                started = await career_interview_voice_start(interview_id, message)
                stream_id = started["stream_id"]
                await websocket.send_json({"type": "started", **started})
            elif message_type == "audio":
                if not stream_id:
                    await websocket.send_json({"type": "error", "content": "voice stream is not started"})
                    continue
                chunk = await career_interview_voice_chunk(
                    interview_id,
                    {
                        **message,
                        "stream_id": str(message.get("stream_id") or stream_id),
                    },
                )
                await websocket.send_json({"type": "speech_events", **chunk})
                answer_result = chunk.get("answer_result") or {}
                if answer_result.get("status") == "completed":
                    stream_id = ""
            elif message_type == "stop":
                if not stream_id:
                    await websocket.send_json({"type": "error", "content": "voice stream is not started"})
                    continue
                stopped = await career_interview_voice_stop(
                    interview_id,
                    {
                        **message,
                        "stream_id": str(message.get("stream_id") or stream_id),
                    },
                )
                await websocket.send_json({"type": "stopped", **stopped})
                answer_result = stopped.get("answer_result") or {}
                if answer_result.get("status") == "completed":
                    stream_id = ""
            elif message_type == "interrupt":
                if not stream_id:
                    await websocket.send_json({"type": "error", "content": "voice stream is not started"})
                    continue
                interrupted = await career_interview_voice_interrupt(
                    interview_id,
                    {
                        **message,
                        "stream_id": str(message.get("stream_id") or stream_id),
                    },
                )
                stream_id = ""
                await websocket.send_json({"type": "interrupted", **interrupted})
            elif message_type == "close":
                break
            else:
                await websocket.send_json({"type": "error", "content": "Unsupported voice message type"})
    except WebSocketDisconnect:
        pass
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "content": str(exc.detail)})
    except Exception as exc:
        await websocket.send_json({"type": "error", "content": str(exc)})
    finally:
        if stream_id:
            try:
                _voice_session_runtime().close(interview_id=interview_id, stream_id=stream_id)
            except Exception:
                pass
        if tokens is not None:
            _reset_request_user(tokens)
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/api/career/materials/structured")
async def career_structured_material(data: dict):
    """Save manually entered personal experience as a traceable material."""
    from smart_agent.career import (
        CandidateFactStore,
        CareerStore,
        HeuristicCareerFactExtractor,
        MaterialCategory,
        MaterialStore,
        PersonalMaterialWorkflow,
    )

    try:
        selected_category = MaterialCategory(str(data.get("category") or "").strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Unsupported Career material category") from exc
    if selected_category in {MaterialCategory.RESUME, MaterialCategory.PROJECT}:
        raise HTTPException(status_code=400, detail="Resume and project materials use their dedicated upload flow")

    field_specs = {
        MaterialCategory.INTERNSHIP: (
            ("company", "实习公司名称"),
            ("department", "项目部门"),
            ("role", "实习岗位"),
            ("summary", "实习简述"),
            ("content", "实习内容"),
            ("date", "实习日期"),
        ),
        MaterialCategory.EDUCATION: (
            ("school", "学校名"),
            ("major", "专业"),
            ("degree", "学位层次"),
            ("start_date", "开始时间"),
            ("end_date", "结束时间"),
            ("gpa", "绩点"),
            ("courses", "主修课程"),
        ),
        MaterialCategory.RESEARCH: (("content", "科研经历"),),
        MaterialCategory.AWARD: (("content", "获奖经历"),),
        MaterialCategory.SKILLS: (("content", "技能/其他"),),
    }
    specs = field_specs[selected_category]
    fields = data.get("fields")
    if not isinstance(fields, dict):
        raise HTTPException(status_code=400, detail="fields must be an object")
    values = {key: str(fields.get(key) or "").strip() for key, _ in specs}
    if selected_category is MaterialCategory.SKILLS and data.get("groups") is not None:
        groups = data.get("groups")
        if not isinstance(groups, list):
            raise HTTPException(status_code=400, detail="groups must be an array")
        normalized_groups: list[tuple[str, str]] = []
        for group in groups:
            if not isinstance(group, dict):
                raise HTTPException(status_code=400, detail="Each skill group must be an object")
            title = str(group.get("title") or "").strip()
            content = str(group.get("content") or "").strip()
            if not title or not content:
                raise HTTPException(status_code=400, detail="Each skill group requires a title and content")
            normalized_groups.append((title, content))
        if not normalized_groups:
            raise HTTPException(status_code=400, detail="At least one skill group is required")
        values["content"] = "\n\n".join(f"{title}：\n{content}" for title, content in normalized_groups)
    if selected_category is MaterialCategory.EDUCATION:
        legacy_date = str(fields.get("date") or fields.get("education_date") or "").strip()
        if legacy_date and not values["start_date"] and not values["end_date"]:
            values["start_date"] = legacy_date
    if selected_category is MaterialCategory.INTERNSHIP:
        start_date = str(fields.get("start_date") or "").strip()
        end_date = str(fields.get("end_date") or "").strip()
        if (start_date or end_date) and not values["date"]:
            values["date"] = " 至 ".join(item for item in (start_date, end_date) if item)
        legacy_outcome = str(fields.get("outcome") or "").strip()
        if legacy_outcome:
            values["content"] = values["content"] or legacy_outcome
            values["summary"] = values["summary"] or legacy_outcome
            values["date"] = values["date"] or "未提供"
    required = {
        MaterialCategory.INTERNSHIP: ("company", "role", "summary", "date"),
        MaterialCategory.EDUCATION: ("school", "major"),
        MaterialCategory.RESEARCH: ("content",),
        MaterialCategory.AWARD: ("content",),
        MaterialCategory.SKILLS: ("content",),
    }[selected_category]
    missing = [label for key, label in specs if key in required and not values[key]]
    if missing:
        raise HTTPException(status_code=400, detail=f"请填写：{'、'.join(missing)}")
    label = str(data.get("label") or "").strip()
    if not label:
        if selected_category is MaterialCategory.INTERNSHIP:
            label = f"{values['company']} · {values['role']}"
        elif selected_category is MaterialCategory.EDUCATION:
            label = f"{values['school']} · {values['major']}"
        elif selected_category is MaterialCategory.RESEARCH:
            label = "科研经历"
        elif selected_category is MaterialCategory.SKILLS:
            label = "技能/其他"
        else:
            label = "获奖经历"
    if selected_category is MaterialCategory.SKILLS:
        text = values["content"]
    else:
        text = "\n".join(f"{field_label}：{values[key]}" for key, field_label in specs if values[key])
    replaced_id = str(data.get("replace_material_id") or "").strip()
    try:
        root = _career_root()
        career_store = CareerStore(root)
        material_store = MaterialStore(root / "materials", workspace=str(ROOT))
        candidate_store = CandidateFactStore(root / "candidate-facts")
        if replaced_id:
            previous = next(
                (item for item in material_store.list_manifests() if item.material_id == replaced_id),
                None,
            )
            if previous is None or previous.category is not selected_category:
                raise HTTPException(status_code=400, detail="The material to replace does not match this category")
        result = PersonalMaterialWorkflow(
            material_store,
            candidate_store,
            career_store=career_store,
            fact_extractor=HeuristicCareerFactExtractor(),
        ).organize(category=selected_category, label=label, text=text)
        if replaced_id and replaced_id != result.manifest.material_id:
            material_store.delete(
                replaced_id,
                career_store=career_store,
                project_store=None,
                reason="Replaced by a newer structured experience",
            )
            candidate_store.retire_material(
                replaced_id,
                reason="Replaced by a newer structured experience",
            )
        return {
            "material": result.import_result.to_dict(include_extracted_text=False),
            "candidate_facts": [fact.to_dict() for fact in result.candidate_facts],
            "structured_path": result.structured_path,
            "workflow_status": result.status,
        }
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/career/materials")
async def career_material_upload(
    file: UploadFile,
    category: str = Form("project"),
    label: str = Form(""),
    project_id: str = Form(""),
):
    """Import a classified Career source and stage extracted facts."""
    from smart_agent.career import (
        CandidateFactStore,
        CareerStore,
        HeuristicCareerFactExtractor,
        MaterialCategory,
        MaterialStore,
        PersonalMaterialWorkflow,
    )

    try:
        selected_category = MaterialCategory(category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Unsupported Career material category") from exc
    if selected_category is MaterialCategory.PROJECT:
        raise HTTPException(
            status_code=400,
            detail="Project materials must be registered through the project library as a ZIP or workspace path.",
        )
    if selected_category is not MaterialCategory.RESUME:
        raise HTTPException(
            status_code=400,
            detail="Internship, education, research, and award experiences must be entered in the structured form.",
        )
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Material file is empty")
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Material upload exceeds 20 MB")
    staging = ROOT / ".tmp" / "career-materials"
    staging.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "material.txt").suffix.lower() or ".txt"
    source = staging / f"{uuid.uuid4().hex}{suffix}"
    source.write_bytes(content)
    try:
        root = _career_root()
        career_store = CareerStore(root)
        material_store = MaterialStore(root / "materials", workspace=str(ROOT))
        candidate_store = CandidateFactStore(root / "candidate-facts")
        workflow = PersonalMaterialWorkflow(
            material_store,
            candidate_store,
            career_store=career_store,
            fact_extractor=HeuristicCareerFactExtractor(),
        )
        result = workflow.organize(
            category=selected_category,
            label=label.strip() or Path(file.filename or "material").stem,
            source_path=source,
            project_id=project_id,
        )
        split_materials = _split_resume_into_materials(
            root=root,
            career_store=career_store,
            material_store=material_store,
            candidate_store=candidate_store,
            label=label.strip() or Path(file.filename or "material").stem,
            resume_source=result.import_result.record,
        )
        return {
            "material": result.import_result.to_dict(include_extracted_text=False),
            "candidate_facts": [fact.to_dict() for fact in result.candidate_facts],
            "structured_path": result.structured_path,
            "workflow_status": result.status,
            "split_materials": split_materials,
        }
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            source.unlink()
        except OSError:
            pass


def _extract_project_archive(archive_path: Path, destination: Path) -> None:
    from smart_agent.career import is_ignored_project_path

    if archive_path.stat().st_size > PROJECT_ARCHIVE_MAX_BYTES:
        raise ValueError("Project archive exceeds 5 GB")
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > PROJECT_ARCHIVE_MAX_MEMBERS:
            raise ValueError(f"Project archive contains more than {PROJECT_ARCHIVE_MAX_MEMBERS} entries")
        total_size = 0
        destination_root = destination.resolve()
        for member in members:
            name = str(member.filename or "").replace("\\", "/")
            if not name or name.endswith("/"):
                continue
            if is_ignored_project_path(name):
                continue
            target = (destination / name).resolve()
            if not target.is_relative_to(destination_root):
                raise ValueError("Project archive contains an unsafe path")
            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError("Project archive contains a symbolic link")
            if member.file_size > PROJECT_ARCHIVE_MAX_BYTES:
                raise ValueError("A project file exceeds 5 GB")
            total_size += member.file_size
            if total_size > PROJECT_ARCHIVE_MAX_BYTES:
                raise ValueError("Expanded project exceeds 5 GB")
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with archive.open(member) as source, target.open("wb") as output:
                while chunk := source.read(PROJECT_ARCHIVE_CHUNK_BYTES):
                    written += len(chunk)
                    if written > member.file_size or total_size - member.file_size + written > PROJECT_ARCHIVE_MAX_BYTES:
                        raise ValueError("Expanded project exceeds 5 GB")
                    output.write(chunk)


def _project_import_id(value: str, content_hash: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-.")
    return normalized[:100] or f"project-{content_hash[:16]}"


@app.post("/api/career/projects/import")
async def career_project_import(
    file: UploadFile | None = None,
    source_url: str = Form(""),
    label: str = Form(""),
    project_id: str = Form(""),
):
    """Compatibility endpoint: register a ZIP as a project source without RAG indexing."""
    from smart_agent.career import ProjectSourceType

    if file is None and not str(source_url or "").strip():
        raise HTTPException(status_code=400, detail="Provide a project ZIP file or source_url.")
    import shutil

    if file is not None and Path(file.filename or "").suffix.lower() != ".zip":
        raise HTTPException(
            status_code=400,
            detail="项目经历入口只接受项目 ZIP 包；PDF、Markdown、DOCX、TXT 等八股资料请切换到“八股资料”入口导入。",
        )

    staging_root = ROOT / ".tmp" / "career-project-imports" / uuid.uuid4().hex
    managed_root = _career_root() / "project-library" / "content" / uuid.uuid4().hex
    staging_root.mkdir(parents=True, exist_ok=False)
    managed_root.mkdir(parents=True, exist_ok=False)
    archive_path = staging_root / "source.zip"
    source_dir = managed_root / "source"
    registered = False
    try:
        archive_hash = hashlib.sha256()
        archive_size = 0
        if file is not None:
            with archive_path.open("wb") as output:
                while chunk := await file.read(PROJECT_ARCHIVE_CHUNK_BYTES):
                    archive_size += len(chunk)
                    if archive_size > PROJECT_ARCHIVE_MAX_BYTES:
                        raise HTTPException(status_code=413, detail="Project archive exceeds 5 GB")
                    archive_hash.update(chunk)
                    output.write(chunk)
            if not archive_size:
                raise HTTPException(status_code=400, detail="Project archive is empty")
        else:
            parsed = urllib.parse.urlparse(str(source_url).strip())
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise HTTPException(status_code=400, detail="source_url must be an http(s) archive URL")
            request = urllib.request.Request(
                str(source_url).strip(),
                headers={"User-Agent": "Smart-Agent-ProjectImporter/2.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                content_length = int(response.headers.get("Content-Length") or 0)
                if content_length > PROJECT_ARCHIVE_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Remote project archive exceeds 5 GB")
                with archive_path.open("wb") as output:
                    while chunk := response.read(PROJECT_ARCHIVE_CHUNK_BYTES):
                        archive_size += len(chunk)
                        if archive_size > PROJECT_ARCHIVE_MAX_BYTES:
                            raise HTTPException(status_code=413, detail="Remote project archive exceeds 5 GB")
                        archive_hash.update(chunk)
                        output.write(chunk)
                if not archive_size:
                    raise HTTPException(status_code=400, detail="Remote project archive is empty")
        requested_project_id = str(project_id or "").strip()
        if not requested_project_id and file is not None:
            requested_project_id = Path(file.filename or "").stem
        if not requested_project_id:
            requested_project_id = Path(urllib.parse.urlparse(str(source_url)).path).stem
        selected_project_id = _project_import_id(requested_project_id, archive_hash.hexdigest())
        _extract_project_archive(archive_path, source_dir)
        project_store, _analysis_store = _project_library_stores()
        source = project_store.register(
            project_id=selected_project_id,
            display_name=str(label or Path(file.filename if file is not None else selected_project_id).stem).strip(),
            source_type=ProjectSourceType.ZIP,
            source_location=source_dir,
            managed_copy=True,
        )
        if source.status.value == "needs_reanalysis":
            from smart_agent.career import CareerStore, ProjectDescriptionService

            ProjectDescriptionService(CareerStore(_career_root())).mark_source_changed(
                source.project_id,
                current_project_version=source.project_version,
            )
        registered = True
        return {
            "project_source": source.to_dict(),
            "source": "zip" if file is not None else "remote_archive",
            "rag_indexed": False,
            "analysis_status": "pending",
            "analysis_scheduled": False,
            "next_action": f"/api/career/project-library/{source.project_id}/analyze",
        }
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, urllib.error.URLError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        if not registered:
            shutil.rmtree(managed_root, ignore_errors=True)


@app.post("/api/career/project-library/register")
async def career_project_library_register(data: dict):
    from smart_agent.career import ProjectSourceType

    source_path = str(data.get("source_path") or "").strip()
    if not source_path:
        raise HTTPException(status_code=400, detail="source_path is required")
    try:
        source = _tenant_owned_path(_workspace_path(source_path))
        project_store, _analysis_store = _project_library_stores()
        record = project_store.register(
            project_id=str(data.get("project_id") or source.name),
            display_name=str(data.get("display_name") or source.name),
            source_type=ProjectSourceType(str(data.get("source_type") or ProjectSourceType.LOCAL_PATH.value)),
            source_location=source,
            managed_copy=False,
        )
        if record.status.value == "needs_reanalysis":
            from smart_agent.career import CareerStore, ProjectDescriptionService

            ProjectDescriptionService(CareerStore(_career_root())).mark_source_changed(
                record.project_id,
                current_project_version=record.project_version,
            )
        return {"project_source": record.to_dict()}
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/career/project-library")
async def career_project_library_list():
    cache_key = _career_library_cache_key()
    cached = _career_library_cache_get(_CAREER_PROJECT_LIBRARY_CACHE, cache_key)
    if cached is not None:
        return cached
    from smart_agent.career import CareerStore, ProjectDescriptionDocument, record_to_dict

    project_store, analysis_store = _project_library_stores()
    descriptions = {
        item.project_id: record_to_dict(item)
        for item in CareerStore(_career_root()).list_latest(ProjectDescriptionDocument)
    }
    projects = []
    for source in project_store.list():
        analysis = analysis_store.latest(source.project_id)
        runs = _project_experience_service().run_store.list(project_id=source.project_id)
        latest_run = runs[-1] if runs else None
        analysis_status = _project_analysis_task_status(source.project_id)
        if not analysis_status:
            analysis_status = latest_run.status if latest_run is not None else (
                "completed" if analysis is not None else "pending"
            )
        projects.append(
            {
                "project_source": source.to_dict(),
                "latest_analysis": analysis.to_dict() if analysis is not None else None,
                "project_description": descriptions.get(source.project_id),
                "latest_analysis_run": latest_run.to_dict() if latest_run is not None else None,
                "analysis_status": analysis_status,
            }
        )
    payload = {"projects": projects, "count": len(projects)}
    _CAREER_PROJECT_LIBRARY_CACHE[cache_key] = (time.monotonic(), payload)
    return payload


@app.get("/api/career/project-library/{project_id}")
async def career_project_library_get(project_id: str):
    try:
        from smart_agent.career import CareerStore, ProjectDescriptionDocument, record_to_dict

        project_store, analysis_store = _project_library_stores()
        source = project_store.get(project_id, refresh=True)
        analysis = analysis_store.latest(source.project_id)
        runs = _project_experience_service().run_store.list(project_id=source.project_id)
        latest_run = runs[-1] if runs else None
        analysis_status = _project_analysis_task_status(source.project_id)
        if not analysis_status:
            analysis_status = latest_run.status if latest_run is not None else (
                "completed" if analysis is not None else "pending"
            )
        description = next(
            (
                item
                for item in CareerStore(_career_root()).list_latest(ProjectDescriptionDocument)
                if item.project_id.casefold() == source.project_id.casefold()
            ),
            None,
        )
        return {
            "project_source": source.to_dict(),
            "latest_analysis": analysis.to_dict() if analysis is not None else None,
            "latest_analysis_run": latest_run.to_dict() if latest_run is not None else None,
            "analysis_status": analysis_status,
            "project_description": record_to_dict(description) if description is not None else None,
        }
    except (KeyError, FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/career/experiences/projects")
async def career_project_experiences_list():
    from smart_agent.career import record_to_dict

    service = _project_experience_service()
    project_store, analysis_store = _project_library_stores()
    sources = {item.project_id: item for item in project_store.list()}
    experiences = []
    for document in service.list_experiences():
        source = sources.get(document.project_id)
        analysis = analysis_store.latest(document.project_id)
        runs = service.run_store.list(project_id=document.project_id)
        experiences.append(
            {
                "project_experience": record_to_dict(document),
                "project_source": source.to_dict() if source else None,
                "latest_analysis": analysis.to_dict() if analysis else None,
                "latest_run": runs[-1].to_dict() if runs else None,
            }
        )
    return {"projects": experiences, "count": len(experiences)}


@app.get("/api/career/experiences/projects/{project_id}")
async def career_project_experience_get(project_id: str):
    from smart_agent.career import record_to_dict

    try:
        service = _project_experience_service()
        document = service.get_experience(project_id)
        source = service.project_store.get(project_id, refresh=True)
        analysis = service.analysis_store.latest(project_id)
        runs = service.run_store.list(project_id=project_id)
        return {
            "project_experience": record_to_dict(document),
            "project_source": source.to_dict(),
            "latest_analysis": analysis.to_dict() if analysis else None,
            "runs": [item.to_dict() for item in runs],
        }
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.patch("/api/career/experiences/projects/{project_id}")
async def career_project_experience_update(project_id: str, data: dict):
    from smart_agent.career import record_to_dict

    if "expected_revision" not in data:
        raise HTTPException(status_code=400, detail="expected_revision is required")
    try:
        updated = _project_experience_service().update_draft(
            project_id,
            changes=dict(data.get("changes") or {}),
            expected_revision=int(data["expected_revision"]),
        )
        return {"project_experience": record_to_dict(updated)}
    except (KeyError, OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/career/experiences/projects/{project_id}/confirm")
async def career_project_experience_confirm(project_id: str, data: dict):
    from smart_agent.career import record_to_dict

    if "expected_revision" not in data:
        raise HTTPException(status_code=400, detail="expected_revision is required")
    try:
        confirmed = _project_experience_service().confirm(
            project_id,
            changes=dict(data.get("changes") or {}),
            expected_revision=int(data["expected_revision"]),
        )
        return {"project_experience": record_to_dict(confirmed)}
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/career/experiences/projects/{project_id}/reanalyze")
async def career_project_experience_reanalyze(project_id: str, data: dict | None = None):
    return await career_project_library_analyze(project_id, {**dict(data or {}), "refresh": True, "intent": "overview"})


@app.get("/api/career/project-analysis/metrics")
async def career_project_analysis_metrics():
    return _project_experience_service().run_store.metrics()


def _persist_project_understanding(
    *,
    source,
    understanding,
    analysis_run_id: str,
    intent: str,
    generator: str,
    model: str = "",
    elapsed_ms: float = 0.0,
    tool_call_count: int = 0,
    token_count: int = 0,
    token_count_source: str = "",
    agent_trace: tuple[dict[str, object], ...] | list[dict[str, object]] = (),
):
    return _project_experience_service().persist_understanding(
        project_id=source.project_id,
        understanding=understanding,
        analysis_run_id=analysis_run_id,
        intent=intent,
        generator=generator,
        model=model,
        trigger="http_api",
        elapsed_ms=elapsed_ms,
        tool_call_count=tool_call_count,
        token_count=token_count,
        token_count_source=token_count_source,
        agent_trace=agent_trace,
    ).to_dict()


@app.post("/api/career/project-library/{project_id}/analyze")
async def career_project_library_analyze(project_id: str, data: dict | None = None):
    from smart_agent.career import HeuristicProjectAnalyzer, ProjectUnderstanding

    payload = dict(data or {})
    mode = str(payload.get("mode") or "agent").strip().casefold()
    intent = str(payload.get("intent") or "overview").strip()
    refresh = bool(payload.get("refresh", False))
    if bool(payload.get("background", False)):
        try:
            project_store, _analysis_store = _project_library_stores()
            source = project_store.get(project_id, refresh=True)
        except (KeyError, FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        scheduled = await _schedule_project_analysis(source.project_id)
        return {
            "project_source": source.to_dict(),
            "analysis_status": "running",
            "analysis_scheduled": scheduled,
            "next_action": f"/api/career/project-library/{source.project_id}/analyze",
        }
    started = time.monotonic()
    source = None
    generator = "project-analyst-agent-v1" if mode == "agent" else "heuristic-project-analyzer-v1"
    model = ""
    tool_call_count = 0
    token_count = 0
    token_count_source = ""
    agent_trace: tuple[dict[str, object], ...] = ()
    try:
        project_store, analysis_store = _project_library_stores()
        source = project_store.get(project_id, refresh=True)
        cached = None if refresh else analysis_store.latest(source.project_id, intent=intent)
        cache_matches_mode = (
            mode == "offline"
            or str(cached.generator or "").strip() == "project-analyst-agent-v1"
        ) if cached is not None else False
        if cached is not None and cached.project_version == source.project_version and cache_matches_mode:
            description = None
            if intent == "overview":
                from smart_agent.career import CareerStore, ProjectDescriptionDocument, record_to_dict

                description = next(
                    (
                        item
                        for item in CareerStore(_career_root()).list_latest(ProjectDescriptionDocument)
                            if item.project_id.casefold() == source.project_id.casefold()
                    ),
                    None,
                )
            cache_run = _project_experience_service().record_cache_hit(
                source=source,
                analysis=cached,
                trigger="http_api",
            )
            return {
                "analysis": cached.to_dict(),
                "analysis_run": cache_run.to_dict(),
                "project_description": record_to_dict(description) if description is not None else None,
                "cached": True,
            }
        if mode == "offline":
            understanding = HeuristicProjectAnalyzer().analyze(source)
            analysis_run_id = f"analysis-{uuid.uuid4().hex[:16]}"
            generator = HeuristicProjectAnalyzer.name
            model = "offline"
        elif mode == "agent":
            result = await asyncio.to_thread(
                _execute_career_tool,
                "career_task",
                {
                    "agent_type": "project_analyst",
                    "description": f"Analyze registered project {source.project_id}",
                    "artifacts": [
                        {
                            "artifact_type": "project_analysis_request",
                            "content": {"project_id": source.project_id, "intent": intent},
                        }
                    ],
                },
                stage="materials",
                enable_model_runtime=True,
            )
            output = result.get("output_artifact") or {}
            content = output.get("content") or {}
            returned_version = str(content.get("project_version") or "").strip()
            if returned_version != source.project_version:
                raise ValueError("Project Analyst result version does not match the requested project version")
            understanding = ProjectUnderstanding.from_dict(content.get("understanding") or {})
            analysis_run_id = str(result.get("task", {}).get("task_id") or f"analysis-{uuid.uuid4().hex[:16]}")
            # A provider outage may be converted by career_task into an
            # explicitly marked local fallback. Keep that provenance in the
            # persisted analysis record instead of presenting it as LLM output.
            analysis_source = str(content.get("analysis_source") or "").strip().casefold()
            if analysis_source == "heuristic_fallback":
                generator = "heuristic-project-analyzer-v1:fallback"
                model = "offline"
            else:
                generator = "project-analyst-agent-v1"
                model = str(load_runtime_config(ROOT).get("model") or "")
            tool_call_count = int(result.get("tool_call_count") or 0)
            token_count = int(result.get("token_count") or 0)
            token_count_source = str(result.get("token_count_source") or "")
            agent_trace = tuple(
                dict(item) for item in (result.get("tool_trace") or ()) if isinstance(item, dict)
            )
        else:
            raise ValueError("mode must be agent or offline")
        current_source = project_store.get(project_id, refresh=True)
        if current_source.project_version != source.project_version:
            raise ValueError("Project source changed during analysis; discard the result and analyze again")
        return {
            **_persist_project_understanding(
                source=source,
                understanding=understanding,
                analysis_run_id=analysis_run_id,
                intent=intent,
                generator=generator,
                model=model,
                elapsed_ms=(time.monotonic() - started) * 1000,
                tool_call_count=tool_call_count,
                token_count=token_count,
                token_count_source=token_count_source,
                agent_trace=agent_trace,
            ),
            "cached": False,
        }
    except HTTPException as exc:
        if source is not None:
            _project_experience_service().record_failure(
                source=source,
                intent=intent,
                trigger="http_api",
                generator=generator,
                model=model,
                elapsed_ms=(time.monotonic() - started) * 1000,
                error=exc.detail or exc,
                tool_call_count=tool_call_count,
                token_count=token_count,
                token_count_source=token_count_source,
            )
        raise
    except (KeyError, FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        if source is not None:
            _project_experience_service().record_failure(
                source=source,
                intent=intent,
                trigger="http_api",
                generator=generator,
                model=model,
                elapsed_ms=(time.monotonic() - started) * 1000,
                error=exc,
                tool_call_count=tool_call_count,
                token_count=token_count,
                token_count_source=token_count_source,
            )
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/career/project-library/{project_id}/deep-dive")
async def career_project_library_deep_dive(project_id: str, data: dict):
    decision_id = str(data.get("decision_id") or "").strip()
    if not decision_id:
        raise HTTPException(status_code=400, detail="decision_id is required")

    async def analyze(selected_project_id: str, intent: str) -> dict:
        return await career_project_library_analyze(
            selected_project_id,
            {**data, "intent": intent, "refresh": bool(data.get("refresh", False))},
        )

    try:
        decision, result = await _project_deep_dive_service().execute_async(
            decision_id,
            project_id=project_id,
            analyzer=analyze,
        )
        return {**result, "deep_dive_decision": decision.to_dict()}
    except HTTPException:
        raise
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/career/project-deep-dives")
async def career_project_deep_dives_list():
    decisions = _project_deep_dive_service().decision_store.list()
    return {"decisions": [item.to_dict() for item in decisions], "count": len(decisions)}


@app.delete("/api/career/project-library/{project_id}")
async def career_project_library_delete(project_id: str, expected_revision: int | None = None):
    from smart_agent.career import (
        CandidateFactStore,
        CareerStore,
        MaterialStore,
        PersonalEvidence,
        ProjectAnalysisRunStore,
        ProjectDeepDiveStore,
        ProjectDescriptionDocument,
    )

    def matches_project(ref, selected: str) -> bool:
        selected_key = selected.casefold()
        locator = dict(getattr(ref, "locator", {}) or {})
        for key in ("project_id", "project", "project_source_id"):
            if str(locator.get(key) or "").strip().casefold() == selected_key:
                return True
        source_id = str(getattr(ref, "source_id", "") or "").strip()
        return source_id.casefold() in {
            selected_key,
            f"project:{selected_key}",
            f"project-source:{selected_key}",
        }

    def invalidate_project_evidence(store, selected: str) -> int:
        changed = 0
        for evidence in store.list_latest(PersonalEvidence):
            if evidence.evidence_type.value != "project":
                continue
            if not any(matches_project(ref, selected) for ref in evidence.source_refs):
                continue
            for fact in evidence.facts:
                if fact.status.value == "invalidated":
                    continue
                store.invalidate_fact(
                    evidence.evidence_id,
                    fact.fact_id,
                    actor="project_library_deleter",
                    reason="Project source was deleted by user",
                )
                changed += 1
        return changed

    try:
        root = _career_root()
        project_store, analysis_store = _project_library_stores()
        current = project_store.get(project_id)
        if expected_revision is not None and current.revision != int(expected_revision):
            raise ValueError(
                f"Expected {current.project_id}@{expected_revision}, found revision {current.revision}."
            )
        selected_project_id = current.project_id
        career_store = CareerStore(root)
        material_store = MaterialStore(root / "materials", workspace=str(ROOT))
        candidate_store = CandidateFactStore(root / "candidate-facts")
        project_index = _project_knowledge_store(root / "project-knowledge")
        related_materials = tuple(
            item
            for item in material_store.list_manifests(include_deleted=False)
            if item.category.value == "project"
            and (
                item.project_id.casefold() == selected_project_id.casefold()
                or item.material_id.casefold() == selected_project_id.casefold()
            )
        )
        invalidated_facts = invalidate_project_evidence(career_store, selected_project_id)
        invalidated_descriptions = 0
        for description in career_store.list_latest(ProjectDescriptionDocument):
            if description.project_id.casefold() != selected_project_id.casefold():
                continue
            if description.status.value == "invalidated":
                continue
            career_store.revise_record(
                description.project_description_id,
                actor="project_library_deleter",
                reason="Project source was deleted by user",
                changes={"status": "invalidated"},
            )
            invalidated_descriptions += 1
        removed_materials = 0
        retired_candidate_facts = 0
        for material in related_materials:
            material_store.delete(
                material.material_id,
                career_store=career_store,
                project_store=project_index,
                reason="Project source was deleted by user",
            )
            retired_candidate_facts += len(
                candidate_store.retire_material(
                    material.material_id,
                    reason="Project source was deleted by user",
                )
            )
            removed_materials += 1
        source = project_store.delete(project_id, expected_revision=expected_revision)
        removed_analyses = analysis_store.purge(source.project_id)
        removed_runs = ProjectAnalysisRunStore(root / "project-library" / "runs").purge(source.project_id)
        removed_deep_dives = ProjectDeepDiveStore(root / "project-library" / "deep-dives").purge(source.project_id)
        removed_project_chunks = project_index.delete(project_ids={source.project_id})
        return {
            "project_source": source.to_dict(),
            "deleted": True,
            "source_files_removed": bool(source.managed_copy),
            "analysis_count_removed": removed_analyses,
            "run_count_removed": removed_runs,
            "deep_dive_count_removed": removed_deep_dives,
            "project_material_count_removed": removed_materials,
            "project_chunk_count_removed": removed_project_chunks,
            "invalidated_project_fact_count": invalidated_facts,
            "invalidated_project_description_count": invalidated_descriptions,
            "retired_candidate_fact_count": retired_candidate_facts,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/career/project-library/delete-batch")
async def career_project_library_delete_batch(data: dict):
    """Delete selected project sources and report every item independently."""
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="projects or project_ids must be a list")
    raw_projects = data.get("projects")
    raw_ids = data.get("project_ids")
    if isinstance(raw_projects, list):
        raw_items = raw_projects
    elif isinstance(raw_ids, list):
        raw_items = raw_ids
    else:
        raise HTTPException(status_code=400, detail="projects or project_ids must be a list")

    items: list[tuple[str, object]] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        if isinstance(raw_item, dict):
            project_id = str(raw_item.get("project_id") or raw_item.get("id") or "").strip()
            expected_revision = raw_item.get("expected_revision", raw_item.get("revision"))
        else:
            project_id = str(raw_item or "").strip()
            expected_revision = None
        if not project_id or project_id in seen:
            continue
        seen.add(project_id)
        items.append((project_id, expected_revision))
    if not items:
        raise HTTPException(status_code=400, detail="Select at least one project")

    results: list[dict] = []
    for project_id, raw_revision in items:
        try:
            expected_revision = None if raw_revision in (None, "") else int(raw_revision)
            result = await career_project_library_delete(project_id, expected_revision=expected_revision)
        except HTTPException as exc:
            results.append({"project_id": project_id, "deleted": False, "status_code": exc.status_code, "error": str(exc.detail)})
        except (TypeError, ValueError) as exc:
            results.append({"project_id": project_id, "deleted": False, "status_code": 400, "error": str(exc)})
        except Exception as exc:  # keep one failure from hiding other outcomes
            results.append({"project_id": project_id, "deleted": False, "status_code": 500, "error": f"{type(exc).__name__}: {exc}"})
        else:
            results.append({"project_id": project_id, "deleted": bool(result.get("deleted")), "result": result})
    deleted_count = sum(bool(item.get("deleted")) for item in results)
    failed_count = len(results) - deleted_count
    return {"deleted": failed_count == 0, "deleted_count": deleted_count, "failed_count": failed_count, "results": results}


@app.patch("/api/career/project-library/{project_id}")
async def career_project_library_update(project_id: str, data: dict):
    from smart_agent.career import CareerStore, ProjectDescriptionService

    if "expected_revision" not in data:
        raise HTTPException(status_code=400, detail="expected_revision is required")
    try:
        project_store, _analysis_store = _project_library_stores()
        current = project_store.get(project_id)
        updated = project_store.update(
            project_id,
            display_name=data.get("display_name"),
            source_location=data.get("source_path") or None,
            expected_revision=int(data["expected_revision"]),
        )
        if updated.status.value == "needs_reanalysis":
            ProjectDescriptionService(CareerStore(_career_root())).mark_source_changed(
                updated.project_id,
                current_project_version=updated.project_version,
            )
        return {"project_source": updated.to_dict(), "previous_revision": current.revision}
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/career/materials/{material_id}")
async def career_material_get(material_id: str):
    from smart_agent.career import MaterialStore

    try:
        material_store = MaterialStore(_career_root() / "materials", workspace=str(ROOT))
        manifest = next(
            (item for item in material_store.list_manifests() if item.material_id == str(material_id).strip()),
            None,
        )
        if manifest is None:
            raise KeyError(f"Unknown material_id: {material_id}")
        return {"material": manifest.to_dict(), "text": material_store.read_text(manifest.material_id)}
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/career/materials/{material_id}")
async def career_material_delete(material_id: str):
    from smart_agent.career import CandidateFactStore, CareerStore, MaterialStore, ResumeSource

    try:
        root = _career_root()
        career_store = CareerStore(root)
        material_store = MaterialStore(root / "materials", workspace=str(ROOT))
        candidate_store = CandidateFactStore(root / "candidate-facts")
        all_manifests = material_store.list_manifests(include_deleted=True)
        selected = next((item for item in all_manifests if item.material_id == str(material_id).strip()), None)
        if selected is None:
            raise KeyError(f"Unknown material_id: {material_id}")
        if not selected.active:
            return {
                "material": selected.to_dict(),
                "deleted_material_ids": [selected.material_id],
                "deleted": True,
                "already_deleted": True,
                "retired_candidate_fact_count": 0,
            }
        manifests = tuple(item for item in all_manifests if item.active)
        project_index = _project_knowledge_store(root / "project-knowledge")
        related_ids = [selected.material_id]
        # A raw resume creates section materials pinned to the same
        # ResumeSource revision. Deleting the source must remove those derived
        # sections as well, otherwise their candidate facts remain visible.
        resume_sources = career_store.list_latest(ResumeSource)
        resume_ids = {
            source.resume_source_id
            for source in resume_sources
            if any(ref.source_id == selected.material_id for ref in source.source_refs)
        }
        derived_material_ids: set[str] = set()
        if resume_ids:
            for resume in resume_sources:
                if resume.resume_source_id not in resume_ids:
                    continue
                for section in resume.sections:
                    if not section.text:
                        continue
                    identity_payload = (
                        f"{resume.resume_source_id}@{resume.revision}:"
                        f"{section.section_id}:{hashlib.sha256(section.text.encode('utf-8')).hexdigest()}"
                    )
                    derived_material_ids.add(
                        f"material-{hashlib.sha256(identity_payload.encode('utf-8')).hexdigest()[:20]}"
                    )
        if resume_ids:
            for candidate in manifests:
                if candidate.material_id == selected.material_id:
                    continue
                facts = candidate_store.list(material_id=candidate.material_id)
                if any(
                    str((ref.locator or {}).get("resume_source", {}).get("record_id") or "") in resume_ids
                    for fact in facts
                    for ref in fact.source_refs
                ):
                    related_ids.append(candidate.material_id)
        related_ids.extend(
            item.material_id
            for item in manifests
            if item.material_id in derived_material_ids
        )
        deleted_materials = []
        retired_count = 0
        for identity in dict.fromkeys(related_ids):
            deleted_material = material_store.delete(
                identity,
                career_store=career_store,
                project_store=project_index,
            )
            retired = candidate_store.retire_material(identity, reason="Source material was deleted")
            retired_count += len(retired)
            deleted_materials.append(deleted_material)
        return {
            "material": deleted_materials[0].to_dict(),
            "deleted_material_ids": [item.material_id for item in deleted_materials],
            "deleted": True,
            "retired_candidate_fact_count": retired_count,
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/career/materials/delete-batch")
async def career_material_delete_batch(data: dict):
    """Delete selected personal materials and report every item independently."""
    raw_ids = data.get("material_ids") if isinstance(data, dict) else None
    if not isinstance(raw_ids, list):
        raise HTTPException(status_code=400, detail="material_ids must be a list")
    material_ids = tuple(dict.fromkeys(str(item or "").strip() for item in raw_ids if str(item or "").strip()))
    if not material_ids:
        raise HTTPException(status_code=400, detail="Select at least one material")
    results: list[dict] = []
    for material_id in material_ids:
        try:
            result = await career_material_delete(material_id)
        except HTTPException as exc:
            results.append({"material_id": material_id, "deleted": False, "status_code": exc.status_code, "error": str(exc.detail)})
        except Exception as exc:  # keep one failure from hiding other outcomes
            results.append({"material_id": material_id, "deleted": False, "status_code": 500, "error": f"{type(exc).__name__}: {exc}"})
        else:
            results.append({"material_id": material_id, "deleted": bool(result.get("deleted")), "result": result})
    deleted_count = sum(bool(item.get("deleted")) for item in results)
    failed_count = len(results) - deleted_count
    return {"deleted": failed_count == 0, "deleted_count": deleted_count, "failed_count": failed_count, "results": results}


@app.post("/api/career/projects/{project_id}/description/generate")
async def career_project_description_generate(project_id: str):
    result = await career_project_library_analyze(
        project_id,
        {"intent": "overview", "refresh": True, "mode": "agent"},
    )
    result["deprecated_endpoint"] = True
    return result


@app.post("/api/career/projects/{project_id}/description/confirm")
async def career_project_description_confirm(project_id: str, data: dict):
    from smart_agent.career import record_to_dict

    try:
        service = _project_experience_service()
        document = service.get_experience(project_id)
        confirmed = service.confirm(
            project_id,
            changes=data.get("changes") or {},
            expected_revision=int(data.get("expected_revision") or document.revision),
        )
        return {"project_description": record_to_dict(confirmed), "deprecated_endpoint": True}
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/files")
async def list_files(project_id: str, path: str = "."):
    """列出项目文件。"""
    target = _tenant_project_path(project_id, path)
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="path is not a directory")
    if not target.exists():
        return {"error": f"Path not found: {path}"}

    items = []
    for entry in sorted(target.iterdir()):
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        items.append({
            "name": entry.name,
            "path": str(entry.relative_to(target)),
            "type": "dir" if entry.is_dir() else "file",
            "size": entry.stat().st_size if entry.is_file() else 0,
        })
    return {"project_id": project_id, "items": items, "current": path, "parent": str(Path(path).parent) if path != "." else None}


@app.get("/api/file")
async def read_file(project_id: str, path: str):
    """读取文件内容。"""
    target = _tenant_project_path(project_id, path)
    if not target.exists() or target.is_dir():
        return {"error": "File not found"}
    size = target.stat().st_size
    if size > MAX_EDITABLE_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large for the browser editor ({size:,} bytes).",
        )
    try:
        raw = target.read_bytes()
        content = target.read_text(encoding="utf-8", errors="replace")
        return {
            "project_id": project_id,
            "path": path,
            "content": content,
            "size": size,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/save")
async def save_file(data: dict):
    """保存文件内容。"""
    project_id = str(data.get("project_id") or "").strip()
    path = data.get("path", "")
    content = data.get("content", "")
    if not path:
        return {"ok": False, "error": "path required"}
    try:
        target = _tenant_project_path(project_id, str(path))
        if target.exists() and target.is_dir():
            raise HTTPException(status_code=400, detail="cannot write a directory")
        expected_sha256 = str(data.get("expected_sha256") or "").strip().lower()
        if expected_sha256 and target.exists():
            current_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
            if current_sha256 != expected_sha256:
                raise HTTPException(status_code=409, detail="file changed; refresh before saving")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "project_id": project_id, "path": path, "sha256": hashlib.sha256(str(content).encode("utf-8")).hexdigest()}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/upload")
async def upload_file(project_id: str = Form(...), file: UploadFile = File(...)):
    """上传文件到项目目录。"""
    try:
        filename = Path(file.filename or "").name
        if not filename or filename in {".", ".."}:
            raise HTTPException(status_code=400, detail="filename is required")
        if Path(filename).suffix.casefold() not in _LEGACY_UPLOAD_SUFFIXES:
            raise HTTPException(status_code=415, detail="file type is not allowed")
        content = await _read_limited_upload(file, limit=MAX_LEGACY_UPLOAD_BYTES)
        target = _tenant_project_path(project_id, filename)
        if target.exists():
            raise HTTPException(status_code=409, detail="file already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return {"status": "ok", "project_id": project_id, "path": filename, "size": len(content)}
    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/career/ocr")
async def career_ocr(file: UploadFile):
    """Extract image/text content as a traceable candidate material."""
    from smart_agent.career import OCRArtifactStore, OCRService, PytesseractOCRAdapter, TextOCRAdapter

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="OCR upload exceeds 20 MB")
    suffix = Path(file.filename or "material.txt").suffix.lower()
    if suffix not in {".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=415, detail="OCR accepts TXT, Markdown, PNG, JPEG, or WebP")
    temp_root = ROOT / ".tmp" / "career-ocr"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_path = temp_root / f"{uuid.uuid4().hex}{suffix}"
    temp_path.write_bytes(content)
    try:
        provider = os.environ.get("SMART_AGENT_OCR_PROVIDER", "text").strip().lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            if provider != "pytesseract":
                raise HTTPException(
                    status_code=503,
                    detail="Image OCR is disabled until SMART_AGENT_OCR_PROVIDER=pytesseract is configured.",
                )
            adapter = PytesseractOCRAdapter(language=os.environ.get("SMART_AGENT_OCR_LANGUAGE", "eng"))
        else:
            adapter = TextOCRAdapter()
        document = OCRService(adapter).extract(str(temp_path))
        persisted = OCRArtifactStore(_career_root() / "ocr").save(document, content, suffix=suffix)
        return {"candidate": persisted.to_candidate_payload(), "artifact": persisted.to_dict()}
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


@app.post("/api/career/application/plan")
async def career_application_plan(data: dict):
    """Plan safe fields; this endpoint never submits a job application."""
    from smart_agent.career import ApplicationMetrics, ApplicationWorkflow

    try:
        url = str(data.get("url") or "")
        resume_document_id = str(data.get("resume_document_id") or "")
        fields = tuple(data.get("fields") or ())
        form, audit = ApplicationWorkflow().prepare(
            url=url,
            resume_document_id=resume_document_id,
            fields=fields,
        )
        return {
            "form": {
                "form_id": form.form_id,
                "url": form.url,
                "resume_document_id": form.resume_document_id,
                "verified_resume": form.verified_resume,
                "can_submit": form.can_submit,
                "fields": [field.to_dict() for field in form.fields],
            },
            "audit": [entry.to_dict() for entry in audit],
            "metrics": ApplicationMetrics.from_form(form).to_dict(),
            "submission": "manual_confirmation_required",
        }
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/report")
async def get_report():
    """返回运行报告。"""
    try:
        from smart_agent.report import build_session_report, format_session_report
        report = build_session_report()
        return report
    except Exception as e:
        return {"error": str(e)}


# ── WebSocket Chat ──

def _original_user_message(payload: dict[str, Any]) -> str:
    """Return the raw user utterance, excluding agent-added prompt context."""
    if not isinstance(payload, dict):
        return ""
    explicit = str(
        payload.get("original_message")
        or payload.get("user_message")
        or payload.get("message")
        or ""
    ).strip()
    if payload.get("original_message") or payload.get("user_message"):
        return explicit

    # Compatibility with clients released before ``original_message`` was
    # added. Code Agent wrapped prompts end with a dedicated request marker.
    marker = re.search(r"(?:^|\n)\s*(?:本轮请求|Request)\s*[:：]\s*(.+?)\s*$", explicit, re.DOTALL)
    return marker.group(1).strip() if marker else explicit


def _apply_memory_operations_to_messages(
    messages: list[dict[str, Any]],
    *,
    fact_ledger: FactLedger,
    session_id: str,
    source_turn_id: str,
) -> list[dict[str, Any]]:
    """Persist validated model proposals and hide the protocol block."""
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "")
        operations = parse_memory_operations(content)
        for operation in operations:
            fact_ledger.apply_memory_operation(
                operation, session_id=session_id,
                source_turn_id=source_turn_id, actor_type="llm",
            )
        if operations:
            message["content"] = strip_memory_operations(content)
    return messages


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        payload = json.loads(data)
        # Keep the user's actual utterance separate from the model prompt.
        # Code Agent adds project/history context to ``message``; intent
        # routing and fact persistence must never inspect that wrapper.
        user_input = str(payload.get("message") or "")
        original_user_input = _original_user_message(payload)
        mode = payload.get("mode", "off")
        requested_model = str(payload.get("model", "") or "").strip()
        api_key = payload.get("api_key", "")
        base_url = payload.get("base_url", "")
        reasoning_effort = str(payload.get("reasoning_effort") or "").strip().lower()
        code_agent_prompt = str(payload.get("code_agent_prompt") or "").strip()
        auth_token = str(payload.get("auth_token") or payload.get("token") or _websocket_token(websocket) or "").strip()
        websocket_user = (
            await asyncio.to_thread(_auth_service().user_from_token, auth_token)
            if auth_token
            else None
        )

        if not original_user_input:
            await websocket.send_json({"type": "error", "content": "消息不能为空"})
            await websocket.close()
            return

        # 从前端传入的配置覆写环境变量
        os.environ["SMART_AGENT_REVIEW_MODE"] = mode
        os.environ["SMART_AGENT_SHOW_GUIDE"] = "0"

        # 预检只负责选择是否需要工具；记忆提取统一由模型的结构化协议完成。
        try:
            preflight_runtime = (
                _tenant_runtime(websocket_user.tenant_id)
                if websocket_user is not None
                else load_runtime_config(ROOT)
            )
        except Exception:
            preflight_runtime = {}
        preflight = classify_with_optional_model(
            original_user_input,
            runtime=preflight_runtime,
        )
        await websocket.send_json({
            "type": "status",
            "content": "正在初始化模型...",
        })

        loop = asyncio.get_running_loop()
        cancel_requested = threading.Event()

        def send_from_worker(message: dict) -> None:
            # The browser may close the socket while the worker is unwinding.
            # Treat that as cancellation instead of creating a second failure.
            try:
                future = asyncio.run_coroutine_threadsafe(websocket.send_json(message), loop)
                future.result(timeout=10)
            except (RuntimeError, TimeoutError, WebSocketDisconnect):
                cancel_requested.set()

        def run_agent() -> list[dict]:
            tenant_runtime = (
                _tenant_runtime(websocket_user.tenant_id)
                if websocket_user is not None
                else load_runtime_config(ROOT)
            )
            fallback_model_name = str(tenant_runtime.get("model") or "").strip()
            runtime = _apply_frontend_model_override(
                tenant_runtime,
                model=requested_model,
                base_url=base_url,
                api_key=api_key,
            )
            model_name = str(runtime.get("model", "")).strip()
            active_domain = route_domain(
                original_user_input,
                configured=payload.get("domain") or runtime.get("domain"),
            )
            if preflight.intent is PreflightIntent.PROJECT_INFO:
                # A source-grounded project explanation belongs to the Code
                # workspace even when the browser was left on Career mode.
                active_domain = AgentDomain.CODE
            session_id = str(payload.get("session_id") or f"web-{uuid.uuid4().hex[:12]}")
            turn_id = str(payload.get("turn_id") or f"turn-{uuid.uuid4().hex[:12]}")
            agent_name = "Career Agent" if active_domain.value == "career" else f"{active_domain.value.title()} Agent"
            tokens = None

            def publish_agent_event(event: dict) -> None:
                event_type = str(event.get("event_type") or "agent_progress")
                fields = {
                    "agent_type": str(event.get("agent_type") or active_domain.value),
                    "agent_name": str(event.get("agent_name") or agent_name),
                    "session_id": str(event.get("session_id") or session_id),
                    "parent_session_id": str(event.get("parent_session_id") or ""),
                    "child_session_id": str(event.get("child_session_id") or ""),
                    **{key: value for key, value in event.items() if key != "event_type"},
                }
                agent_event_hub.publish(event_type, **fields)

            try:
                if active_domain.value in {"career", "code"}:
                    if websocket_user is None:
                        raise ValueError("请先登录")
                    tokens = _set_request_user(websocket_user)

                runtime = {
                    **runtime,
                    "activeDomain": active_domain.value,
                    "sessionId": session_id,
                    "agentName": agent_name,
                    "careerTenantId": _career_tenant_id() or "legacy-web-user",
                    "careerStorePath": str(_career_root()),
                    "turnId": turn_id,
                    "originalUserInput": original_user_input,
                    "_event_sink": publish_agent_event,
                    "preflight": {
                        "intent": preflight.intent.value,
                        "complexity": preflight.complexity,
                        "needs_tools": preflight.needs_tools,
                        "confidence": preflight.confidence,
                        "reason": preflight.reason,
                    },
                }
                fact_ledger = FactLedger(
                    _career_root() / "fact-ledger",
                    tenant_id=_career_tenant_id() or "legacy-web-user",
                )
                runtime["factLedgerPath"] = str(fact_ledger.path)
                fact_context = fact_ledger.render_context(session_id=session_id)
                if reasoning_effort:
                    runtime["reasoningEffort"] = reasoning_effort
                project_root = ROOT
                simple_request = preflight.intent is PreflightIntent.GENERAL
                inspect_request = preflight.intent is PreflightIntent.PROJECT_INFO
                workspace_inspection = (
                    active_domain.value == "code"
                    and _is_read_only_workspace_request(original_user_input)
                )
                if inspect_request:
                    runtime["readOnlyToolsOnly"] = True
                if workspace_inspection:
                    inspect_request = True
                    runtime["readOnlyToolsOnly"] = True
                if active_domain.value == "code" and not simple_request:
                    project_id = str(payload.get("project_id") or "").strip()
                    if not project_id:
                        # The repository hosting Smart Agent is itself a valid
                        # Code workspace.  Do not make access depend on the
                        # heuristic preflight classifier recognizing a
                        # read-only phrasing; callers inspecting or changing
                        # this demo repository may omit project_id.
                        project_root = Path(ROOT).resolve()
                        runtime.update({"projectRoot": str(project_root), "workspaceFallback": True})
                    else:
                        from smart_agent.career import ProjectLibraryStore, ProjectSourceStatus
                        workspace = Path(ROOT).resolve()
                        source = ProjectLibraryStore(
                            _career_root() / "project-library", workspace=workspace
                        ).get(project_id, refresh=True)
                        if source.status in {ProjectSourceStatus.UNAVAILABLE, ProjectSourceStatus.INVALIDATED}:
                            raise ValueError(f"Registered project is unavailable: {source.status.value}")
                        project_root = Path(source.source_location).resolve()
                        if not project_root.is_dir():
                            project_root = project_root.parent
                        runtime.update({"projectId": source.project_id, "projectRoot": str(project_root)})
                        runtime["optimisticLockRequired"] = True
                domain_system_prompt = get_domain_module(active_domain).system_prompt
                domain_system_prompt = (
                    f"{domain_system_prompt}\n\n"
                    "Long-term user fact protocol: every explicit, first-person, stable personal fact in the user's "
                    "message MUST be represented as a machine-readable memory operation, even when the user does not "
                    "say 'remember'. This includes name, education/major, occupation, stable interests, preferences, "
                    "and other durable profile facts. Append exactly one machine-readable "
                    "<memory_ops>[...]</memory_ops> block with operation, content, fact_key, confidence, and "
                    "requires_confirmation. Use requires_confirmation=false only for explicit, stable, high-confidence "
                    "facts; use true for uncertain, sensitive, or consequential claims. The runtime stores valid operations and hides "
                    "the block. Never infer the user's name or identity from the model provider, API account, relay "
                    "station, authentication display name, or any other external account metadata. If the ledger has "
                    "no matching fact, say that the user's name is unknown rather than guessing."
                )
                if inspect_request:
                    domain_system_prompt = (
                        f"{domain_system_prompt}\n\n"
                        "This is a project-information request. You MUST inspect the selected project's "
                        "source with the available read-only tools before answering. Ground every claim in "
                        "files or tool results; do not guess or describe an uninspected implementation. "
                        "Use the minimum number of read-only calls, then provide a concise evidence-based summary."
                    )
                if fact_context:
                    domain_system_prompt = f"{domain_system_prompt}\n\n{fact_context}"
                domain_system_prompt = (
                    f"{domain_system_prompt}\n\n"
                    "Identity precedence: the User Fact Ledger is the only source for the user's personal identity. "
                    "Authentication display names, API provider accounts, relay/proxy identities, and model metadata "
                    "are not the user. If no ledger fact establishes a name, state plainly that the user's name is unknown."
                )
                if active_domain.value == "code" and code_agent_prompt:
                    domain_system_prompt = (
                        f"{domain_system_prompt}\n\n"
                        f"# 前端 Code Agent 附加提示\n{code_agent_prompt}"
                    )
                selected_path = str(payload.get("active_path") or "").strip()
                domain_system_prompt = (
                    f"{domain_system_prompt}\n\n"
                    "Code Agent workspace rules: the current working directory is already the selected "
                    "project root. Use '.' or paths relative to that directory; never prefix paths with "
                    "'source', the project display name, or an outer archive directory."
                    + (f" The user currently has this file open: {selected_path}." if selected_path else "")
                )
                publish_agent_event({
                    "event_type": "agent_started",
                    "agent_type": active_domain.value,
                    "agent_name": agent_name,
                    "session_id": session_id,
                    "status": "running",
                })
                if simple_request or not _should_use_agent_loop(preflight, original_user_input):
                    tools = ToolRegistry([])
                    model = create_model_adapter(model=model_name, tools=tools, runtime=runtime)
                    messages = [
                        {"role": "system", "content": domain_system_prompt},
                        # The wrapped ``message`` contains rendered history and
                        # runtime labels for the Code UI.  Simple requests must
                        # be answered from the current utterance only; feeding
                        # the wrapper back to the model lets stale assistant
                        # output masquerade as the current answer.
                        {"role": "user", "content": original_user_input},
                    ]
                    try:
                        step = model.next(
                            messages,
                            on_stream_chunk=(
                                None if active_domain.value == "code"
                                else lambda chunk: send_from_worker({"type": "delta", "content": chunk})
                            ),
                        )
                    except RuntimeError as exc:
                        if not (
                            _is_unsupported_model_error(exc)
                            and fallback_model_name
                            and fallback_model_name != model_name
                        ):
                            raise
                        runtime["model"] = fallback_model_name
                        model = create_model_adapter(model=fallback_model_name, tools=tools, runtime=runtime)
                        send_from_worker({
                            "type": "status",
                            "content": f"Model {model_name} unavailable; switched to {fallback_model_name}",
                        })
                        step = model.next(
                            messages,
                            on_stream_chunk=(
                                None if active_domain.value == "code"
                                else lambda chunk: send_from_worker({"type": "delta", "content": chunk})
                            ),
                        )
                    simple_messages = [{"role": "assistant", "content": str(step.content or "")}]
                    _apply_memory_operations_to_messages(
                        simple_messages, fact_ledger=fact_ledger,
                        session_id=session_id, source_turn_id=turn_id,
                    )
                    publish_agent_event({
                        "event_type": "agent_finished",
                        "agent_type": active_domain.value,
                        "agent_name": agent_name,
                        "session_id": session_id,
                        "status": "completed",
                        "output_summary": simple_messages[0]["content"][:500],
                    })
                    return [{"role": "assistant", "content": _sanitize_user_facing_text(simple_messages[0]["content"])}]

                tools = create_default_tool_registry(str(project_root), runtime=runtime)
                model = create_model_adapter(model=model_name, tools=tools, runtime=runtime)
                try:
                    requested_max_steps = int(payload.get("max_steps") or 0)
                except (TypeError, ValueError):
                    requested_max_steps = 0
                agent_max_steps = requested_max_steps if requested_max_steps > 0 else (
                    6 if inspect_request else (24 if active_domain.value == "code" else 8)
                )
                agent_max_steps = max(1, min(agent_max_steps, 120))
                result_messages = _run_agent_turn_with_model_fallback(
                    model=model, tools=tools,
                    fallback_model_name=fallback_model_name,
                    current_model_name=model_name,
                    # ``message`` is a UI rendering that may contain stale
                    # history and display labels.  The current request is the
                    # only user message; project context lives in the system
                    # prompt/runtime.
                    messages=[{"role": "user", "content": original_user_input}],
                    cwd=str(project_root), max_steps=agent_max_steps,
                    runtime=runtime,
                    system_prompt=domain_system_prompt,
                    should_cancel=cancel_requested.is_set,
                    on_tool_start=lambda tool_name, tool_input: send_from_worker(
                        {"type": "status", "content": f"工具执行中：{tool_name}"}
                    ),
                    on_tool_result=lambda tool_name, output, is_error: send_from_worker(
                        {"type": "status", "content": f"工具{'失败' if is_error else '完成'}：{tool_name}"}
                    ),
                    on_assistant_stream_chunk=(
                        None if active_domain.value == "code"
                        else lambda chunk: send_from_worker({"type": "delta", "content": chunk})
                    ),
                    on_progress_message=lambda message: (
                        send_from_worker({"type": "status", "content": visible})
                        if (visible := _user_visible_progress_message(message))
                        else None
                    ),
                )
                _apply_memory_operations_to_messages(
                    result_messages, fact_ledger=fact_ledger,
                    session_id=session_id, source_turn_id=turn_id,
                )
                return result_messages
            finally:
                if tokens is not None:
                    _reset_request_user(tokens)

        if True:
            await websocket.send_json({"type": "status", "content": "模型已就绪，开始处理..."})

        agent_future = loop.run_in_executor(None, run_agent)
        receive_task = asyncio.create_task(websocket.receive_text())
        while True:
            done, _pending = await asyncio.wait(
                {agent_future, receive_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if agent_future in done:
                receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await receive_task
                result = agent_future.result()
                break
            if receive_task in done:
                try:
                    control_raw = receive_task.result()
                except WebSocketDisconnect:
                    cancel_requested.set()
                    result = await agent_future
                    break
                try:
                    control = json.loads(control_raw)
                except json.JSONDecodeError:
                    control = {}
                control_type = str(control.get("type") or "").strip().lower()
                if control_type in {"pause", "cancel", "stop"}:
                    cancel_requested.set()
                    await websocket.send_json({
                        "type": "paused",
                        "content": "Pause requested. Waiting for the current step to stop.",
                    })
                receive_task = asyncio.create_task(websocket.receive_text())

        # 提取助手回复
        response = ""
        tool_calls = []
        for m in result:
            role = m.get("role", "")
            if role == "assistant" and m.get("content"):
                response = _sanitize_user_facing_text(m["content"])
            elif role == "assistant_tool_call":
                tool_calls.append({
                    "tool": m.get("toolName", ""),
                    "input": m.get("input", {}),
                })

        # 检查生成的文件
        output_files = []
        import glob as py_glob
        for f in py_glob.glob(str(ROOT / "*.py")):
            fname = Path(f).name
            if fname not in ("pyproject.toml",) and not fname.startswith("smart_agent") and not fname.startswith("benchmark"):
                output_files.append(fname)

        await websocket.send_json({
            "type": "result",
            "content": response or "（无回复）",
            "tool_calls": tool_calls[:5],
            "files": output_files[:10],
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except (RuntimeError, WebSocketDisconnect):
            # The client can disconnect while the background turn is awaited.
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── 前端页面 ──

def _frontend_index_response() -> FileResponse:
    built_index = FRONTEND_DIST / "index.html"
    if not built_index.is_file():
        raise HTTPException(
            status_code=503,
            detail="frontend build missing; run npm run build in frontend/app",
        )
    return FileResponse(built_index, headers={"Cache-Control": "no-cache"})


@app.get("/admin")
async def admin_page():
    return _frontend_index_response()


@app.get("/")
async def index():
    return _frontend_index_response()


def main() -> None:
    print("  Smart Agent: http://localhost:8341")
    uvicorn.run(app, host="0.0.0.0", port=8341)


if __name__ == "__main__":
    main()
