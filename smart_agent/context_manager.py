"""LLM 对话的上下文窗口管理。

跟踪 token 使用情况，估算上下文窗口消耗，并提供自动压缩功能，
以防止长对话中的上下文溢出。
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable

from smart_agent.config import SMART_AGENT_DIR
from smart_agent.prompt_pipeline import SYSTEM_PROMPT_DYNAMIC_BOUNDARY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default context window sizes (tokens)
DEFAULT_CONTEXT_WINDOWS = {
    # Anthropic
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-haiku-3-20240307": 100_000,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
    # OpenRouter popular models
    "openrouter/auto": 200_000,
    "anthropic/claude-sonnet-4": 200_000,
    "anthropic/claude-opus-4": 200_000,
    "openai/gpt-4o": 128_000,
    "openai/gpt-4o-mini": 128_000,
    "google/gemini-2.5-pro": 1_000_000,
    "google/gemini-2.5-flash": 1_000_000,
    "meta-llama/llama-4-maverick": 1_000_000,
    "deepseek/deepseek-r1": 128_000,
    "deepseek/deepseek-chat": 128_000,
    "qwen/qwen3-235b-a22b": 128_000,
    "minimax/minimax-m1": 1_000_000,
    "default": 128_000,  # Fallback
}


# ---------------------------------------------------------------------------
# Model context window resolution (port of TS getModelContextWindow).
# Case-insensitive substring matching with an output reserve, so model ids like
# "claude-opus-4-6", "CLAUDE-...", or "anthropic/claude-3-5-sonnet-latest"
# resolve correctly instead of falling through to the 128k default.
# ---------------------------------------------------------------------------


class ModelContextWindow:
    """模型上下文窗口配置。

    存储模型的上下文窗口总大小、为输出保留的 token 数量，
    以及实际可用于输入的有效大小（总大小减去输出保留量）。
    """
    __slots__ = ("context_window", "output_reserve", "effective_input")

    def __init__(self, context_window: int, output_reserve: int) -> None:
        """初始化模型上下文窗口配置。

        计算有效输入大小（上下文窗口减去输出保留量）。

        参数:
            context_window: 上下文窗口总大小（token 数）。
            output_reserve: 为输出保留的 token 数。
        """
        self.context_window = context_window
        self.output_reserve = output_reserve
        self.effective_input = context_window - output_reserve


_UNKNOWN_MODEL_CONTEXT = (128_000, 8_000)

_MODEL_CONTEXT_RULES: list[tuple[list[str], int, int]] = [
    (["claude-opus-4-6", "claude opus 4.6", "opus-4-6"], 200_000, 16_000),
    (["claude-sonnet-4-6", "claude sonnet 4.6", "sonnet-4-6"], 200_000, 16_000),
    (["claude-haiku-4-5", "claude haiku 4.5", "haiku-4-5"], 200_000, 16_000),
    (["claude-opus-4-1", "claude opus 4.1", "opus-4-1", "claude-opus-4", "claude opus 4", "opus-4"], 200_000, 16_000),
    (["claude-sonnet-4", "claude sonnet 4", "sonnet-4"], 200_000, 16_000),
    (["claude-3-7-sonnet", "claude 3.7 sonnet", "3-7-sonnet"], 200_000, 8_192),
    (["claude-3-5-sonnet", "claude 3.5 sonnet", "3-5-sonnet", "claude-3-sonnet"], 200_000, 8_192),
    (["claude-3-5-haiku", "claude 3.5 haiku", "3-5-haiku"], 200_000, 8_192),
    (["claude-3-opus", "claude 3 opus"], 200_000, 4_096),
    (["claude-3-haiku", "claude 3 haiku"], 200_000, 4_096),
    (["gpt-5-codex", "gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5"], 128_000, 16_000),
    (["o4-mini", "o3", "o1-pro", "o1"], 200_000, 16_000),
    (["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1"], 1_047_576, 16_000),
    (["gpt-4o-mini", "gpt-4o"], 128_000, 16_384),
    (["gpt-4"], 128_000, 8_192),
    (["gemini-2.5-pro", "gemini 2.5 pro"], 1_048_576, 16_000),
    (["gemini-2.5-flash-lite", "gemini 2.5 flash-lite"], 1_048_576, 16_000),
    (["gemini-2.5-flash", "gemini 2.5 flash"], 1_048_576, 16_000),
    (["deepseek-reasoner"], 128_000, 16_000),
    (["deepseek-chat"], 128_000, 4_000),
]


def get_model_context_window(model: str) -> ModelContextWindow:
    """将模型 ID 解析为其上下文窗口和输出保留量。

    不区分大小写的子串匹配（移植自 TypeScript 版的 getModelContextWindow）。
    对于未知模型，回退使用 128K 上下文窗口和 8K 输出保留量。

    参数:
        model: 模型标识字符串，如 "claude-sonnet-4-20250514"。

    返回:
        包含上下文窗口大小、输出保留量和有效输入大小的 ModelContextWindow 实例。
    """
    normalized = (model or "").strip().lower()
    for patterns, context_window, output_reserve in _MODEL_CONTEXT_RULES:
        if any(pattern in normalized for pattern in patterns):
            return ModelContextWindow(context_window, output_reserve)
    context_window, output_reserve = _UNKNOWN_MODEL_CONTEXT
    return ModelContextWindow(context_window, output_reserve)

# Start a new context generation before the provider's hard limit. Cached
# conversation prefixes are invalidated once per generation instead of being
# rewritten repeatedly near the limit.
AUTOCOMPACT_THRESHOLD = 0.85
COMPACTION_TARGET = 0.55

# L1 deterministically projects oversized textual tool results.  The original
# request snapshot is archived before this projection is committed.
LARGE_TOOL_RESULT_CHARS = 8_192
LARGE_TOOL_RESULT_HEAD_CHARS = 4_096
LARGE_TOOL_RESULT_TAIL_CHARS = 1_024
LARGE_TOOL_RESULT_MARKER = "\n\n[... tool result middle pruned ...]\n\n"

# Working Turn semantic compression is deliberately harder to enter than
# History compression.  It is configurable per ContextManager instance.
DEFAULT_WORKING_TURN_THRESHOLD = 16_000

# Estimated tokens per character (rough average for English/Code)
CHARS_PER_TOKEN = 4.0

# Minimum messages to keep after compaction
MIN_MESSAGES_TO_KEEP = 10

# System prompt is always kept (counts as 1 message)
SYSTEM_PROMPT_RESERVED = 1


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# 预编译的正则表达式用于快速 CJK 字符检测
_CJK_PATTERN = re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]')

# LRU 缓存：token 估算被频繁调用（每条消息、每次上下文检查），
# 相同文本的 token 数是确定性的，缓存可避免重复计算。
_token_cache: dict[str, int] = {}
_TOKEN_CACHE_MAX = 1024


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数量，支持中英文混合。

    英文和代码按约 4 字符/token 估算，中文、日文和韩文等 CJK 字符按约 1.5 字符/token 估算，
    混合文本使用启发式加权计算。使用预编译的正则表达式替代逐字符检查，性能提升 10-50 倍。
    内置 LRU 缓存机制，避免对相同文本的重复计算。

    参数:
        text: 待估算的文本字符串。

    返回:
        估算得到的 token 数量，至少为 1（空文本返回 0）。
    """
    if not text:
        return 0

    # 缓存查找（短文本优先缓存）
    cache_key = (
        text
        if len(text) < 256
        else hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    cached = _token_cache.get(cache_key)
    if cached is not None:
        return cached

    # 使用正则表达式快速统计 CJK 字符数量
    cjk_count = len(_CJK_PATTERN.findall(text))

    # CJK 字符约 1.5 字符/token，英文约 4 字符/token
    ascii_chars = len(text) - cjk_count

    result = max(1, int(cjk_count / 1.5 + ascii_chars / 4.0))

    # 缓存结果（防止无限增长）
    if len(_token_cache) < _TOKEN_CACHE_MAX:
        _token_cache[cache_key] = result

    return result


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """估算单条消息的 token 数量。

    根据消息的角色（system/user/assistant/tool_call/tool_result 等）添加不同的角色开销 token，
    并对消息内容及工具调用输入进行 token 估算。对于没有内容和工具输入的 message 返回 0。

    参数:
        message: 消息字典，包含 role、content、input 等字段。

    返回:
        该消息的总 token 估算值。
    """
    # Match TS semantics: a message with no content and no tool input contributes
    # zero tokens (TS estimateMessageTokens is ceil(0/ratio) == 0). We keep the
    # CJK-aware estimate_tokens() for non-empty content — it is more accurate for
    # Chinese/Japanese text than TS's plain char/3.5 ratio.
    content = message.get("content", "")
    has_input = bool(message.get("input"))
    content_str = content if isinstance(content, str) else ""
    if not content_str and not has_input:
        return 0

    tokens = 0

    # Role overhead
    role = message.get("role", "")
    if role == "system":
        tokens += 3  # System prompt overhead
    elif role == "user":
        tokens += 4  # User message overhead
    elif role == "assistant":
        tokens += 3  # Assistant overhead
    elif role == "assistant_tool_call":
        tokens += 7  # Tool call overhead
    elif role == "tool_result":
        tokens += 6  # Tool result overhead
    elif role == "assistant_progress":
        tokens += 3

    # Content tokens
    content = message.get("content", "")
    if isinstance(content, str):
        tokens += estimate_tokens(content)

    # Tool call input/output
    if "input" in message:
        input_str = json.dumps(message["input"]) if isinstance(message["input"], dict) else str(message["input"])
        tokens += estimate_tokens(input_str)

    return tokens


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """估算多条消息的总 token 数量。

    对消息列表中的每条消息逐条调用 estimate_message_tokens 并求和。

    参数:
        messages: 消息字典列表。

    返回:
        所有消息的总 token 估算值。
    """
    return sum(estimate_message_tokens(msg) for msg in messages)


# ---------------------------------------------------------------------------
# Provider-usage-aware token accounting (port of TS tokenCountWithEstimation
# and computeContextStats). NOTE: Python does not currently record provider
# usage on individual messages, so the provider_usage path falls back to
# estimate_only until that wiring is added; the estimate_only + warning-level
# math is still correct and useful.
# ---------------------------------------------------------------------------


def _message_provider_usage(message: dict[str, Any]) -> dict[str, Any] | None:
    """获取消息中的 provider 用量数据。

    仅对 assistant 角色（含 assistant_progress 和 assistant_tool_call）的消息提取 providerUsage
    或 provider_usage 字段，并跳过标记为 stale 的用量数据。

    参数:
        message: 消息字典。

    返回:
        用量字典（如存在且未过期），否则返回 None。
    """
    role = message.get("role", "")
    if role not in ("assistant", "assistant_progress", "assistant_tool_call"):
        return None
    usage = message.get("providerUsage") or message.get("provider_usage")
    if usage and not message.get("usageStale") and not message.get("usage_stale"):
        return usage
    return None


def _stale_usage_reason(messages: list[dict[str, Any]]) -> str | None:
    """查找消息列表中第一个标记为 stale 的 provider 用量及其原因。

    遍历消息列表，查找 assistant 角色消息中标记为 usageStale 或 usage_stale 的条目，
    并返回其过期原因。

    参数:
        messages: 消息字典列表。

    返回:
        过期原因字符串（如存在），否则返回 None。
    """
    for message in messages:
        if (
            message.get("role") in ("assistant", "assistant_progress", "assistant_tool_call")
            and (message.get("providerUsage") or message.get("provider_usage"))
            and (message.get("usageStale") or message.get("usage_stale"))
        ):
            return message.get("usageStaleReason") or message.get("usage_stale_reason") or "provider usage was marked stale"
    return None


def token_count_with_estimation(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """基于 provider 用量感知的 token 计数（移植自 TypeScript 版 tokenCountWithEstimation）。

    从消息列表末尾向前查找最新的非过期 provider 用量数据，将其与后续消息的估算值相加。
    如果没有可用的 provider 用量数据，则完全依赖估算。

    参数:
        messages: 消息字典列表。

    返回:
        包含 total_tokens、provider_usage_tokens、estimated_tokens、source、
        is_exact、usage_boundary、stale、reason 等字段的字典。
    """
    for i in range(len(messages) - 1, -1, -1):
        usage = _message_provider_usage(messages[i])
        if not usage:
            continue
        total_usage = int(usage.get("totalTokens", usage.get("total_tokens", 0)) or 0)
        estimated = estimate_messages_tokens(messages[i + 1:])
        boundary_id = messages[i].get("toolUseId") or messages[i].get("tool_use_id")
        return {
            "total_tokens": total_usage + estimated,
            "provider_usage_tokens": total_usage,
            "estimated_tokens": estimated,
            "source": "provider_usage_plus_estimate" if estimated > 0 else "provider_usage",
            "is_exact": estimated == 0,
            "usage_boundary": {"message_index": i, "message_id": boundary_id},
            "stale": False,
            "reason": None,
        }

    reason = _stale_usage_reason(messages)
    estimated = estimate_messages_tokens(messages)
    return {
        "total_tokens": estimated,
        "provider_usage_tokens": 0,
        "estimated_tokens": estimated,
        "source": "estimate_only",
        "is_exact": False,
        "stale": bool(reason),
        "reason": reason or "no provider usage available",
    }


def _stable_message_hash(message: dict[str, Any]) -> str:
    """Return a process-independent cache key for one message."""
    encoded = json.dumps(
        message,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TokenBreakdown:
    """One auditable request measurement tied to a Surface revision."""

    surface_revision: int
    system_tokens: int
    tools_tokens: int
    dynamic_tokens: int
    history_tokens: int
    working_turn_tokens: int
    output_reserve: int
    estimated_input_tokens: int
    request_total_tokens: int
    provider_usage_tokens: int = 0
    token_source: str = "estimate_only"


@dataclass
class TokenMeter:
    """Measure every request component with stable content-hash caching."""

    _message_cache: dict[str, int] = field(default_factory=dict, repr=False)

    def message_tokens(self, message: dict[str, Any]) -> int:
        key = _stable_message_hash(message)
        cached = self._message_cache.get(key)
        if cached is not None:
            return cached
        tokens = estimate_message_tokens(message)
        if len(self._message_cache) >= _TOKEN_CACHE_MAX:
            self._message_cache.clear()
        self._message_cache[key] = tokens
        return tokens

    def measure(
        self,
        messages: list[dict[str, Any]],
        *,
        surface_revision: int,
        tools_tokens: int,
        dynamic_tokens: int,
        output_reserve: int,
    ) -> TokenBreakdown:
        last_user_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            len(messages),
        )
        system_tokens = 0
        message_dynamic_tokens = 0
        history_tokens = 0
        working_turn_tokens = 0
        for index, message in enumerate(messages):
            tokens = self.message_tokens(message)
            if message.get("role") == "system":
                if message.get("_token_category") == "dynamic":
                    message_dynamic_tokens += tokens
                else:
                    content = message.get("content")
                    if isinstance(content, str) and SYSTEM_PROMPT_DYNAMIC_BOUNDARY in content:
                        static, _boundary, dynamic = content.partition(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
                        static_message = {**message, "content": static + SYSTEM_PROMPT_DYNAMIC_BOUNDARY}
                        system_tokens += self.message_tokens(static_message)
                        message_dynamic_tokens += estimate_tokens(dynamic)
                    else:
                        system_tokens += tokens
            elif index >= last_user_index:
                working_turn_tokens += tokens
            else:
                history_tokens += tokens

        dynamic_total = max(0, int(dynamic_tokens)) + message_dynamic_tokens
        input_total = (
            system_tokens
            + max(0, int(tools_tokens))
            + dynamic_total
            + history_tokens
            + working_turn_tokens
        )
        accounting = token_count_with_estimation(messages)
        return TokenBreakdown(
            surface_revision=max(0, int(surface_revision)),
            system_tokens=system_tokens,
            tools_tokens=max(0, int(tools_tokens)),
            dynamic_tokens=dynamic_total,
            history_tokens=history_tokens,
            working_turn_tokens=working_turn_tokens,
            output_reserve=max(0, int(output_reserve)),
            estimated_input_tokens=input_total,
            request_total_tokens=input_total + max(0, int(output_reserve)),
            provider_usage_tokens=int(accounting["provider_usage_tokens"]),
            token_source=str(accounting["source"]),
        )


def compute_context_stats(messages: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """计算上下文统计信息，包含利用率和告警级别（移植自 TS 版 computeContextStats）。

    根据模型获取上下文窗口，结合 provider 用量或估算值计算上下文利用率，
    并按阈值划分告警级别：95% 以上为 blocked，85% 以上为 critical，50% 以上为 warning。

    参数:
        messages: 消息字典列表。
        model: 模型标识字符串。

    返回:
        包含 estimated_tokens、total_tokens、provider_usage_tokens、context_window、
        effective_input、utilization、warning_level、accounting 等字段的字典。
    """
    window = get_model_context_window(model)
    accounting = token_count_with_estimation(messages)
    effective = window.effective_input or 1
    utilization = min(1.0, accounting["total_tokens"] / effective)
    if utilization >= 0.95:
        warning_level = "blocked"
    elif utilization >= 0.85:
        warning_level = "critical"
    elif utilization >= 0.50:
        warning_level = "warning"
    else:
        warning_level = "normal"
    return {
        "estimated_tokens": accounting["estimated_tokens"],
        "total_tokens": accounting["total_tokens"],
        "provider_usage_tokens": accounting["provider_usage_tokens"],
        "context_window": window.context_window,
        "effective_input": window.effective_input,
        "utilization": utilization,
        "warning_level": warning_level,
        "accounting": accounting,
    }



@dataclass
class _ExtractedInfo:
    """从被移除的消息中提取的结构化信息，用于生成摘要。

    在上下文压缩过程中，从被丢弃的消息中提取各类信息，包括用户意图、
    文件路径、工具调用结果、助手结论、代码片段和关键决策等。
    """
    user_intents: list[str] = field(default_factory=list)
    file_paths: set[str] = field(default_factory=set)
    key_tool_results: list[str] = field(default_factory=list)
    assistant_conclusions: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)
    code_snippets: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)

    def merge(self, other: "_ExtractedInfo") -> "_ExtractedInfo":
        """Merge checkpoint state without depending on a domain schema."""
        self.user_intents.extend(item for item in other.user_intents if item not in self.user_intents)
        self.file_paths.update(other.file_paths)
        self.key_tool_results.extend(
            item for item in other.key_tool_results if item not in self.key_tool_results
        )
        self.assistant_conclusions.extend(
            item for item in other.assistant_conclusions if item not in self.assistant_conclusions
        )
        self.tool_names.extend(other.tool_names)
        self.code_snippets.extend(item for item in other.code_snippets if item not in self.code_snippets)
        self.decisions.extend(item for item in other.decisions if item not in self.decisions)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_intents": self.user_intents,
            "file_paths": sorted(self.file_paths),
            "key_tool_results": self.key_tool_results,
            "assistant_conclusions": self.assistant_conclusions,
            "tool_names": self.tool_names,
            "code_snippets": self.code_snippets,
            "decisions": self.decisions,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "_ExtractedInfo":
        value = value or {}
        return cls(
            user_intents=list(value.get("user_intents", [])),
            file_paths=set(value.get("file_paths", [])),
            key_tool_results=list(value.get("key_tool_results", [])),
            assistant_conclusions=list(value.get("assistant_conclusions", [])),
            tool_names=list(value.get("tool_names", [])),
            code_snippets=list(value.get("code_snippets", [])),
            decisions=list(value.get("decisions", [])),
        )


# Tool categories for classification
_EDIT_TOOLS = frozenset({"edit_file", "write_file", "modify_file", "patch_file", "multi_edit"})
_READ_TOOLS = frozenset({"read_file", "list_files", "grep_files", "file_tree"})
_SEARCH_TOOLS = frozenset({"grep_files", "find_symbols", "find_references", "web_search", "web_fetch"})
_COMMAND_TOOLS = frozenset({"run_command", "execute_command", "bash"})

# Regex for extracting code-like content and decisions
_CODE_FENCE_RE = re.compile(r'```[\w]*\n(.{20,300}?)```', re.DOTALL)
_DECISION_KEYWORDS = re.compile(
    r'(?:decided|decision|chose|chosen|will use|using|switching to|'
    r'implemented|fixed|resolved|refactored|migrated|upgraded|'
    r'recommend|should|must|need to|going to|plan to|'
    r'approach:|strategy:|solution:|conclusion:)',
    re.IGNORECASE,
)


def _extract_from_messages(messages: list[dict[str, Any]]) -> _ExtractedInfo:
    """从消息中提取结构化信息，用于分层摘要生成。

    这是核心提取步骤，从各类型消息中提取不同类别的信息，包括用户意图、
    助手决策、代码片段、工具调用、文件路径及错误结果等。提取的内容将
    供给预算感知的摘要构建器，优先包含最重要的信息。

    参数:
        messages: 待提取的消息字典列表。

    返回:
        包含 user_intents、file_paths、key_tool_results、assistant_conclusions、
        tool_names、code_snippets、decisions 等字段的 _ExtractedInfo 实例。
    """
    info = _ExtractedInfo()

    for msg in messages:
        if msg.get("_context_checkpoint"):
            info.merge(_ExtractedInfo.from_dict(msg.get("_checkpoint_state")))
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user" and content.strip():
            # Extract user intent — keep more context for short queries,
            # truncate long paste-heavy messages
            preview = content.strip().replace("\n", " ")
            # For short queries (<200 chars), keep them fully
            # For long ones, keep first 200 chars
            if len(preview) > 200:
                preview = preview[:200] + "..."
            info.user_intents.append(preview)

        elif role == "assistant" and content.strip():
            text = content.strip()

            # Extract decisions/conclusions
            sentences = text.replace("\n", " ").split(". ")
            for sentence in sentences:
                if _DECISION_KEYWORDS.search(sentence):
                    decision = sentence.strip()[:180]
                    if decision and decision not in info.decisions:
                        info.decisions.append(decision)

            # Extract code snippets from assistant responses
            for match in _CODE_FENCE_RE.finditer(text):
                snippet = match.group(1).strip()
                if len(snippet) >= 20 and len(info.code_snippets) < 5:
                    info.code_snippets.append(snippet[:300])

            # General conclusion preview
            preview = text[:200].replace("\n", " ")
            info.assistant_conclusions.append(preview)

        elif role == "assistant_tool_call":
            tool_name = msg.get("toolName", "unknown")
            info.tool_names.append(tool_name)

            # Extract file paths from edit/write tools
            if tool_name in _EDIT_TOOLS:
                inp = msg.get("input", {})
                path = inp.get("path") or inp.get("filePath", "")
                if path:
                    info.file_paths.add(path)

            # Extract searched patterns from grep/search tools
            if tool_name in _SEARCH_TOOLS:
                inp = msg.get("input", {})
                pattern = inp.get("pattern") or inp.get("query", "")
                if pattern:
                    info.file_paths.add(f"search:{pattern[:80]}")

            # Extract command names from run_command
            if tool_name in _COMMAND_TOOLS:
                inp = msg.get("input", {})
                cmd = inp.get("command", "")
                if cmd:
                    cmd_name = cmd.split()[0] if cmd.split() else ""
                    if cmd_name:
                        info.key_tool_results.append(f"ran: {cmd_name}")

        elif role == "tool_result":
            tool_name = msg.get("toolName", "")
            is_error = msg.get("isError", False)

            # Preserve error results (highest priority tool info)
            if is_error:
                error_preview = content.strip()[:150].replace("\n", " ")
                info.key_tool_results.append(f"ERROR({tool_name}): {error_preview}")

            # Preserve edit confirmations with file paths
            elif tool_name in _EDIT_TOOLS and content.strip():
                success_preview = content.strip()[:100].replace("\n", " ")
                info.key_tool_results.append(f"{tool_name} ok: {success_preview}")

            # Extract file paths from read_file results
            elif tool_name in _READ_TOOLS and content.strip():
                # Check if content references a file path
                first_line = content.strip().split("\n")[0][:100]
                if "/" in first_line or "\\" in first_line:
                    info.file_paths.add(first_line.strip())

    return info


def _build_layered_summary(info: _ExtractedInfo, max_summary_tokens: int = 2000) -> str:
    """从提取的信息构建预算感知的分层摘要。

    各层级按重要性排序，每层分配不同的 token 预算：
    - 第 1 层：用户意图（35% 预算）——用户的需求
    - 第 2 层：决策和文件路径（20% 预算）——关键选择
    - 第 3 层：关键工具结果（15% 预算）——错误和重要输出
    - 第 4 层：助手结论（15% 预算）——达到的结果
    - 第 5 层：代码片段（10% 预算）——重要的代码模式
    - 第 6 层：工具使用摘要（5% 预算）——紧凑的活动日志

    参数:
        info: 从消息中提取的结构化信息。
        max_summary_tokens: 摘要的最大 token 预算，默认 2000。

    返回:
        格式化后的分层摘要字符串。
    """
    lines: list[str] = []

    # Budget allocations per layer (as fraction of total)
    layer_budgets = [0.35, 0.20, 0.15, 0.15, 0.10, 0.05]

    def _remaining_budget() -> int:
        """计算当前剩余的 token 预算。"""
        return max(0, max_summary_tokens - estimate_tokens("\n".join(lines)))

    # Layer 1: User intents (highest priority)
    if info.user_intents:
        budget = int(max_summary_tokens * layer_budgets[0])
        lines.append("## User requests:")
        for intent in info.user_intents[:12]:
            if estimate_tokens("\n".join(lines)) > budget:
                lines.append(f"  ... and {len(info.user_intents) - info.user_intents.index(intent)} more")
                break
            lines.append(f"- {intent}")

    # Layer 2: Decisions and file paths
    has_decisions = bool(info.decisions)
    has_files = bool(info.file_paths)
    if has_decisions or has_files:
        budget = int(max_summary_tokens * (layer_budgets[0] + layer_budgets[1]))

        if info.decisions:
            lines.append("## Key decisions:")
            for dec in info.decisions[:8]:
                if estimate_tokens("\n".join(lines)) > budget:
                    break
                lines.append(f"- {dec}")

        if info.file_paths:
            # Separate real paths from search patterns
            real_paths = sorted(p for p in info.file_paths if not p.startswith("search:"))
            search_patterns = sorted(p[8:] for p in info.file_paths if p.startswith("search:"))

            path_line = f"## Files: {', '.join(real_paths[:20])}"
            if len(real_paths) > 20:
                path_line += f" (+{len(real_paths)-20} more)"
            if search_patterns:
                path_line += f"\n## Searched: {', '.join(search_patterns[:5])}"

            if estimate_tokens("\n".join(lines) + path_line) <= budget:
                lines.append(path_line)

    # Layer 3: Key tool results (errors + edits)
    if info.key_tool_results:
        budget = int(max_summary_tokens * sum(layer_budgets[:3]))
        lines.append("## Key results:")
        for result in info.key_tool_results[:15]:
            if estimate_tokens("\n".join(lines)) > budget:
                break
            lines.append(f"- {result}")

    # Layer 4: Assistant conclusions
    if info.assistant_conclusions:
        budget = int(max_summary_tokens * sum(layer_budgets[:4]))
        lines.append("## Conclusions:")
        for conc in info.assistant_conclusions[:8]:
            if estimate_tokens("\n".join(lines)) > budget:
                break
            lines.append(f"- {conc}")

    # Layer 5: Code snippets (most selective)
    if info.code_snippets:
        budget = int(max_summary_tokens * sum(layer_budgets[:5]))
        lines.append("## Code patterns:")
        for snippet in info.code_snippets[:3]:
            snippet_line = f"```\n{snippet}\n```"
            if estimate_tokens("\n".join(lines) + snippet_line) > budget:
                break
            lines.append(snippet_line)

    # Layer 6: Tool usage summary (most compact)
    if info.tool_names:
        from collections import Counter
        tool_counts = Counter(info.tool_names)
        tool_summary = ", ".join(
            f"{name}×{count}" if count > 1 else name
            for name, count in tool_counts.most_common()
        )
        lines.append(f"## Tools: {tool_summary}")

    return "\n".join(lines)


def _summarize_removed_messages(messages: list[dict[str, Any]], max_summary_tokens: int = 2000) -> str:
    """为上下文保留构建被移除消息的浓缩摘要。

    采用两阶段方法：
    1. 提取：从所有类型消息中拉取结构化信息。
    2. 构建：按 token 预算分配组装各层摘要。

    确保最重要的信息（用户意图、关键决策）始终被包含，
    而次要细节（工具名称、代码片段）填充剩余预算。

    参数:
        messages: 被移除的消息字典列表。
        max_summary_tokens: 摘要的最大 token 预算，默认 2000。

    返回:
        浓缩摘要字符串，空列表时返回空字符串。
    """
    if not messages:
        return ""

    info = _extract_from_messages(messages)
    return _build_layered_summary(info, max_summary_tokens)


# ---------------------------------------------------------------------------
# Context tracking
# ---------------------------------------------------------------------------

@dataclass
class ContextStats:
    """当前上下文窗口的统计数据。

    记录上下文的总 token 使用量、上下文窗口大小、使用率百分比、
    消息数量、系统 token、对话 token、工具调用次数，
    以及是否接近限制或需要压缩的状态标记。
    """
    total_tokens: int = 0
    context_window: int = 0
    usage_percentage: float = 0.0
    messages_count: int = 0
    system_tokens: int = 0
    tools_tokens: int = 0
    dynamic_tokens: int = 0
    conversation_tokens: int = 0
    history_tokens: int = 0
    working_turn_tokens: int = 0
    output_reserve: int = 0
    request_total_tokens: int = 0
    estimated_tokens: int = 0
    provider_usage_tokens: int = 0
    token_source: str = "estimate_only"
    surface_revision: int = 0
    tool_calls_count: int = 0
    is_near_limit: bool = False
    should_compact: bool = False


class CompressionLevel(IntEnum):
    """Highest compression stage applied during one compaction cycle."""

    NONE = 0
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4


class CompactionMode(str, Enum):
    """Scope selected from one pre-compaction token snapshot."""

    NONE = "none"
    HISTORY = "history"
    HISTORY_AND_WORKING = "history_and_working"


@dataclass(frozen=True)
class CompactionPlan:
    """A fixed decision produced from one pre-compaction snapshot."""

    surface_revision: int
    mode: CompactionMode
    trigger: str
    before_tokens: int
    history_tokens: int
    working_turn_tokens: int
    total_threshold: int
    working_turn_threshold: int
    target_tokens: int


@dataclass(frozen=True)
class WorkingTurnCheckpoint:
    """Audit metadata for one Working Turn semantic checkpoint."""

    checkpoint_id: str
    turn_id: str
    generation: int
    goal: str
    key_decisions: tuple[str, ...]
    errors_and_fixes: tuple[str, ...]
    current_work: str
    next_step: str = ""
    unresolved: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "turn_id": self.turn_id,
            "generation": self.generation,
            "goal": self.goal,
            "key_decisions": list(self.key_decisions),
            "errors_and_fixes": list(self.errors_and_fixes),
            "current_work": self.current_work,
            "next_step": self.next_step,
            "unresolved": list(self.unresolved),
        }


@dataclass
class UserTurn:
    """One user message and every message before the next user message."""

    turn_id: str
    messages: list[dict[str, Any]]
    compression_level: CompressionLevel = CompressionLevel.NONE


@dataclass
class L3Lock:
    """Immutable summary of one contiguous batch of terminal user turns."""

    lock_id: str
    first_turn_id: str
    last_turn_id: str
    turn_count: int
    summary: str
    protected_record_ids: list[str] = field(default_factory=list)

    def to_message(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.summary,
            "_l3_lock": True,
            "_lock_id": self.lock_id,
            "_first_turn_id": self.first_turn_id,
            "_last_turn_id": self.last_turn_id,
            "_covered_turns": self.turn_count,
            "_protected_record_ids": list(self.protected_record_ids),
            "_compression_level": int(CompressionLevel.L3),
        }


@dataclass
class ProtectedRecord:
    """Content that survives lock eviction and is injected independently."""

    record_id: str
    content: str
    source_turn_id: str = ""
    active: bool = True

    def to_message(self) -> dict[str, Any]:
        return {
            "role": "system",
            "content": self.content,
            "_protected_record": True,
            "_protected_record_id": self.record_id,
            "_source_turn_id": self.source_turn_id,
        }


@dataclass
class ContextManager:
    """管理上下文窗口跟踪和自动压缩。 【为什么需要】防止长对话中因上下文窗口溢出导致
    LLM 调用失败，通过 token 估算、使用率监控和渐进式
    消息压缩，确保对话始终适配模型上下文窗口限制。

    ╔══ 完整执行流程 ══╗
    ║  核心职责: Token 估算                                 ║
    ║  ├─ estimate_tokens(): CJK 感知的启发式估算           ║
    ║  ├─ estimate_message_tokens(): 角色开销 + 内容估算     ║
    ║  ├─ token_count_with_estimation(): 优先使用             ║
    ║  │  provider 返回的精确用量，回退到纯估算               ║
    ║  └─ LRU 缓存避免重复计算                               ║
    ║                                                         ║
    ║  核心职责: 上下文窗口管理                               ║
    ║  ├─ get_model_context_window(): 模型 → (窗口, 输出保留) ║
    ║  ├─ compute_context_stats(): 计算利用率 + 告警级别      ║
    ║  │     [0-50%)  normal  |  [50-85%)  warning            ║
    ║  │     [85-95%) critical |  [95%+]   blocked            ║
    ║  └─ get_stats(): 返回 ContextStats 结构体               ║
    ║                                                         ║
    ║  get_stats() 返回:                                      ║
    ║  ├─ total_tokens — 系统 token + 对话 token 总和         ║
    ║  ├─ context_window — 模型上下文窗口大小                  ║
    ║  ├─ usage_percentage — 使用率百分比 (total/窗口)        ║
    ║  ├─ messages_count — 消息总数                           ║
    ║  ├─ system_tokens — 系统提示词 token 小计               ║
    ║  ├─ conversation_tokens — 对话消息 token 小计           ║
    ║  ├─ tool_calls_count — assistant_tool_call 调用次数     ║
    ║  ├─ is_near_limit — usage ≥ 80% 接近限制标记            ║
    ║  └─ should_compact — 有效输入用量达到高水位              ║
    ║                                                         ║
    ║  should_auto_compact() 判断逻辑:                        ║
    ║  ├─ 从模型窗口扣除输出预留，得到有效输入预算              ║
    ║  ├─ 达到 85% 高水位时创建新的上下文代际                  ║
    ║  ├─ 旧前缀转换为通用结构化检查点                         ║
    ║  └─ 检查点后按 Token 保留最近完整消息与工具对             ║
    ╚══════════════════════════════════════════════════╝
    """
    model: str = "default"
    context_window: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    compaction_history: list[dict[str, Any]] = field(default_factory=list)
    generation: int = 0
    protected_records: list[ProtectedRecord] = field(default_factory=list)
    archive: list[dict[str, Any]] = field(default_factory=list)
    working_turn_threshold: int = DEFAULT_WORKING_TURN_THRESHOLD
    summary_provider: Callable[[str, list[dict[str, Any]]], Any] | None = field(
        default=None, repr=False, compare=False
    )
    _bound_session: Any = field(default=None, repr=False, compare=False)
    tool_schema_tokens: int = 0
    dynamic_context_tokens: int = 0
    surface_revision: int = 0
    active_turn_id: str | None = field(default=None, repr=False, compare=False)
    _token_meter: TokenMeter = field(default_factory=TokenMeter, repr=False, compare=False)

    # Retained for compatibility with older context_state.json files.
    _compaction_level: int = field(default_factory=lambda: 0)  # 0=无压缩, 1=轻微, 2=中等, 3=深度

    def __post_init__(self):
        """初始化后处理：如果未设置上下文窗口大小，则根据模型自动获取。"""
        if self.context_window == 0:
            self.context_window = get_model_context_window(self.model).context_window

    def update_model(self, model: str) -> None:
        """更新模型并调整上下文窗口大小。

        当会话切换模型时调用，同步更新上下文窗口配置。

        参数:
            model: 新的模型标识字符串。
        """
        self.model = model
        self.context_window = get_model_context_window(model).context_window

    def add_message(self, message: dict[str, Any]) -> None:
        """添加一条消息并更新跟踪状态。

        将消息追加到消息列表，并立即缓存其 token 估算值，
        避免在后续 get_stats 调用中重复计算。

        参数:
            message: 待添加的消息字典。
        """
        self.messages.append(message)
        self._token_meter.message_tokens(message)

    def configure_request(
        self,
        *,
        tool_definitions: list[dict[str, Any]] | None = None,
        dynamic_context: list[str] | None = None,
    ) -> None:
        """Set non-Surface request components used by the next model call."""
        definitions = tool_definitions or []
        self.tool_schema_tokens = (
            estimate_tokens(
                json.dumps(definitions, ensure_ascii=False, sort_keys=True, default=str)
            )
            if definitions
            else 0
        )
        self.dynamic_context_tokens = sum(
            estimate_tokens(str(item)) for item in (dynamic_context or []) if str(item)
        )

    def get_stats(self) -> ContextStats:
        """计算当前上下文统计信息。

        优先使用缓存的 token 计数（通过 add_message 添加的消息可实现均摊 O(1) 查询）。
        按角色统计系统 token 和对话 token，并计算使用率及是否接近限制。

        返回:
            包含总 token、上下文窗口、使用率百分比、消息数等字段的 ContextStats 实例。
        """
        if self._bound_session is not None:
            self.surface_revision = int(
                getattr(self._bound_session, "surface_revision", self.surface_revision)
            )
        output_reserve = self.context_window - self._effective_input_budget()
        breakdown = self._token_meter.measure(
            self.messages,
            surface_revision=self.surface_revision,
            tools_tokens=self.tool_schema_tokens,
            dynamic_tokens=self.dynamic_context_tokens,
            output_reserve=output_reserve,
        )
        tool_calls = 0
        for msg in self.messages:
            if msg.get("role") == "assistant_tool_call":
                tool_calls += 1
        total_tokens = breakdown.estimated_input_tokens
        usage_pct = (total_tokens / self.context_window * 100) if self.context_window > 0 else 0

        effective_input = self._effective_input_budget()
        is_near_limit = total_tokens >= int(effective_input * 0.80)
        should_compact = total_tokens >= int(effective_input * AUTOCOMPACT_THRESHOLD)

        return ContextStats(
            total_tokens=total_tokens,
            context_window=self.context_window,
            usage_percentage=usage_pct,
            messages_count=len(self.messages),
            system_tokens=breakdown.system_tokens,
            tools_tokens=breakdown.tools_tokens,
            dynamic_tokens=breakdown.dynamic_tokens,
            conversation_tokens=breakdown.history_tokens + breakdown.working_turn_tokens,
            history_tokens=breakdown.history_tokens,
            working_turn_tokens=breakdown.working_turn_tokens,
            output_reserve=breakdown.output_reserve,
            request_total_tokens=breakdown.request_total_tokens,
            estimated_tokens=breakdown.estimated_input_tokens,
            provider_usage_tokens=breakdown.provider_usage_tokens,
            token_source=breakdown.token_source,
            surface_revision=breakdown.surface_revision,
            tool_calls_count=tool_calls,
            is_near_limit=is_near_limit,
            should_compact=should_compact,
        )

    def should_auto_compact(self) -> bool:
        """Return whether the current generation crossed its high-water mark."""
        stats = self.get_stats()
        return stats.total_tokens >= int(self._effective_input_budget() * AUTOCOMPACT_THRESHOLD)

    def plan_compaction(self, stats: ContextStats | None = None, *, force: bool = False) -> CompactionPlan:
        """Create one immutable plan; later stages may not change its scope."""
        snapshot = stats or self.get_stats()
        total_threshold = int(self._effective_input_budget() * AUTOCOMPACT_THRESHOLD)
        working_threshold = max(1, int(self.working_turn_threshold))
        if not force and snapshot.total_tokens < total_threshold:
            mode = CompactionMode.NONE
        elif snapshot.working_turn_tokens < working_threshold:
            mode = CompactionMode.HISTORY
        else:
            mode = CompactionMode.HISTORY_AND_WORKING
        return CompactionPlan(
            surface_revision=snapshot.surface_revision,
            mode=mode,
            trigger="manual" if force else "auto",
            before_tokens=snapshot.total_tokens,
            history_tokens=snapshot.history_tokens,
            working_turn_tokens=snapshot.working_turn_tokens,
            total_threshold=total_threshold,
            working_turn_threshold=working_threshold,
            target_tokens=int(self._effective_input_budget() * COMPACTION_TARGET),
        )

    @staticmethod
    def _group_messages(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """按 user 消息边界分组，每组对应一次完整交互。"""
        if not messages:
            return []
        groups = []
        cur = []
        for m in messages:
            if m.get("role") == "user" and cur:
                groups.append(cur)
                cur = []
            cur.append(m)
        if cur:
            groups.append(cur)
        return groups

    @staticmethod
    def _atomic_segments(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Group tool calls with their results so a tail cut stays valid."""
        segments: list[list[dict[str, Any]]] = []
        index = 0
        while index < len(messages):
            current = messages[index]
            if (
                current.get("role") == "assistant_tool_call"
                and index + 1 < len(messages)
                and messages[index + 1].get("role") == "tool_result"
            ):
                segments.append([current, messages[index + 1]])
                index += 2
                continue
            segments.append([current])
            index += 1
        return segments

    def _effective_input_budget(self) -> int:
        """Return an input budget that leaves room for the next response."""
        configured_reserve = get_model_context_window(self.model).output_reserve
        reserve = min(configured_reserve, max(1, self.context_window // 5))
        return max(1, self.context_window - reserve)

    def bind_session(self, session: Any) -> None:
        """Bind archive and compression state to a SessionData-like object."""
        self._bound_session = session
        replay_surface = getattr(session, "replay_surface", None)
        if callable(replay_surface):
            self.messages = replay_surface()
        elif getattr(session, "messages", None):
            self.messages = copy.deepcopy(session.messages)
        self.surface_revision = int(getattr(session, "surface_revision", 0))
        state = getattr(session, "context_state", {}) or {}
        self.generation = int(state.get("generation", self.generation))
        self.working_turn_threshold = int(
            state.get("working_turn_threshold", self.working_turn_threshold)
        )
        self.compaction_history = list(
            state.get("compaction_history", self.compaction_history)
        )
        records = state.get("protected_records", [])
        if records:
            self.protected_records = [
                item if isinstance(item, ProtectedRecord) else ProtectedRecord(**item)
                for item in records
            ]
        self.archive = getattr(session, "context_archive", self.archive)

    def add_protected_record(
        self, record_id: str, content: str, *, source_turn_id: str = ""
    ) -> None:
        """Add or replace an independently retained context record."""
        replacement = ProtectedRecord(record_id, content, source_turn_id)
        for index, record in enumerate(self.protected_records):
            if record.record_id == record_id:
                self.protected_records[index] = replacement
                return
        self.protected_records.append(replacement)

    @staticmethod
    def _message_level(message: dict[str, Any]) -> CompressionLevel:
        try:
            return CompressionLevel(int(message.get("_compression_level", 0)))
        except (TypeError, ValueError):
            return CompressionLevel.NONE

    def _partition_context(
        self,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[UserTurn],
    ]:
        """Separate stable prefix, protected records, locks, and user turns."""
        system_messages: list[dict[str, Any]] = []
        embedded_protected: list[dict[str, Any]] = []
        locks: list[dict[str, Any]] = []
        orphan_messages: list[dict[str, Any]] = []
        turns: list[UserTurn] = []
        current: list[dict[str, Any]] = []

        for message in self.messages:
            if message.get("_protected_record"):
                embedded_protected.append(message)
                continue
            if message.get("role") == "system":
                system_messages.append(message)
                continue
            if message.get("_l3_lock") or message.get("_context_checkpoint"):
                locks.append(message)
                continue
            if message.get("role") == "user":
                if current:
                    turns.append(self._make_turn(current, len(turns)))
                current = [message]
                continue
            if current:
                current.append(message)
            else:
                orphan_messages.append(message)

        if current:
            turns.append(self._make_turn(current, len(turns)))
        if orphan_messages and turns:
            turns[0] = self._make_turn(orphan_messages + turns[0].messages, 0)
        elif orphan_messages:
            turns.append(self._make_turn(orphan_messages, 0))
        return system_messages, embedded_protected, locks, turns

    def _make_turn(self, messages: list[dict[str, Any]], index: int) -> UserTurn:
        user = next((m for m in messages if m.get("role") == "user"), {})
        turn_id = str(user.get("_turn_id") or f"g{self.generation}-t{index + 1}")
        level = max((self._message_level(m) for m in messages), default=CompressionLevel.NONE)
        return UserTurn(turn_id, list(messages), level)

    def _protected_messages(
        self, embedded: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        by_id = {
            str(message.get("_protected_record_id")): message
            for message in embedded
            if message.get("_protected_record_id")
        }
        for record in self.protected_records:
            if record.active:
                by_id[record.record_id] = record.to_message()
        return list(by_id.values())

    @staticmethod
    def _compose(
        system_messages: list[dict[str, Any]],
        protected_messages: list[dict[str, Any]],
        locks: list[dict[str, Any]],
        turns: list[UserTurn],
    ) -> list[dict[str, Any]]:
        return (
            list(system_messages)
            + list(protected_messages)
            + list(locks)
            + [message for turn in turns for message in turn.messages]
        )

    @staticmethod
    def _is_forced(message: dict[str, Any]) -> bool:
        return bool(
            message.get("_force_preserve")
            or message.get("_protected")
            or message.get("_protected_record_id")
        )

    @staticmethod
    def _mark_message(
        message: dict[str, Any], turn_id: str, level: CompressionLevel
    ) -> dict[str, Any]:
        marked = dict(message)
        marked["_turn_id"] = turn_id
        marked["_compression_level"] = int(level)
        return marked

    def _tool_pairs(
        self, messages: list[dict[str, Any]]
    ) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for index in range(len(messages) - 1):
            call = messages[index]
            result = messages[index + 1]
            if call.get("role") != "assistant_tool_call" or result.get("role") != "tool_result":
                continue
            call_id = call.get("toolUseId") or call.get("tool_call_id")
            result_id = result.get("toolUseId") or result.get("tool_call_id")
            if not call_id or not result_id or call_id == result_id:
                pairs.append((index, index + 1))
        return pairs

    def _compact_turn_l1(self, turn: UserTurn) -> UserTurn:
        """Prune only textual tool results larger than 8192 Unicode characters."""
        output: list[dict[str, Any]] = []
        changed = False
        for message in turn.messages:
            content = message.get("content")
            if (
                message.get("role") != "tool_result"
                or not isinstance(content, str)
                or len(content) <= LARGE_TOOL_RESULT_CHARS
                or message.get("_large_tool_result_pruned")
            ):
                output.append(message)
                continue

            compact = self._mark_message(message, turn.turn_id, CompressionLevel.L1)
            compact["content"] = (
                content[:LARGE_TOOL_RESULT_HEAD_CHARS]
                + LARGE_TOOL_RESULT_MARKER
                + content[-LARGE_TOOL_RESULT_TAIL_CHARS:]
            )
            compact["_large_tool_result_pruned"] = True
            compact["_tool_result_original_chars"] = len(content)
            compact["_tool_result_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
            output.append(compact)
            changed = True

        if not changed:
            return turn
        return UserTurn(turn.turn_id, output, max(turn.compression_level, CompressionLevel.L1))

    def _compact_turn_l2(self, turn: UserTurn) -> UserTurn:
        """Keep user input, the two newest complete tool pairs, and forced records."""
        pairs = self._tool_pairs(turn.messages)
        keep_indexes = {index for pair in pairs[-2:] for index in pair}
        output = [
            self._mark_message(message, turn.turn_id, CompressionLevel.L2)
            for index, message in enumerate(turn.messages)
            if message.get("role") == "user"
            or index in keep_indexes
            or self._is_forced(message)
        ]
        return UserTurn(turn.turn_id, output, CompressionLevel.L2)

    def _working_turn_protected_indexes(self, turn: UserTurn) -> set[int]:
        """Return Working Turn records that semantic L4 must keep verbatim."""
        pairs = self._tool_pairs(turn.messages)
        paired_indexes = {index for pair in pairs for index in pair}
        protected = {
            index
            for index, message in enumerate(turn.messages)
            if message.get("role") == "user"
            or self._is_forced(message)
            or (
                message.get("role") in ("assistant_tool_call", "tool_result")
                and index not in paired_indexes
            )
        }
        active_progress = next(
            (
                index
                for index in range(len(turn.messages) - 1, -1, -1)
                if turn.messages[index].get("role") == "assistant_progress"
            ),
            None,
        )
        if active_progress is not None:
            protected.add(active_progress)
        return protected

    @staticmethod
    def _brief(value: Any, limit: int = 240) -> str:
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, default=str)
        return " ".join(value.split())[:limit]

    def _fallback_l3_summary(self, turns: list[UserTurn]) -> str:
        blocks = ["[L3 locked history]"]
        for turn in turns:
            user = next((m for m in turn.messages if m.get("role") == "user"), {})
            final = next(
                (m for m in reversed(turn.messages) if m.get("role") == "assistant"),
                {},
            )
            calls = [
                f"{m.get('toolName', 'unknown')}({self._brief(m.get('input', {}), 120)})"
                for m in turn.messages
                if m.get("role") == "assistant_tool_call"
            ]
            results = [
                self._brief(m.get("content", ""), 160)
                for m in turn.messages
                if m.get("role") == "tool_result"
            ]
            forced = [
                self._brief(m.get("content", ""), 200)
                for m in turn.messages
                if self._is_forced(m)
            ]
            blocks.extend(
                [
                    f"Turn {turn.turn_id}:",
                    f"- User: {self._brief(user.get('content', ''))}",
                    f"- Tool Call: {'; '.join(calls) or 'none'}",
                    f"- Tool Result: {'; '.join(results) or 'none'}",
                    f"- Assistant: {self._brief(final.get('content', '')) or 'none'}",
                ]
            )
            if forced:
                blocks.append(f"- Required: {'; '.join(forced)}")
        return "\n".join(blocks)

    def _summarize_l3(self, turns: list[UserTurn]) -> str:
        payload = [message for turn in turns for message in turn.messages]
        if self.summary_provider is not None:
            try:
                result = self.summary_provider("l3_lock", copy.deepcopy(payload))
                if isinstance(result, str) and result.strip():
                    return result.strip()
            except Exception:
                pass
        return self._fallback_l3_summary(turns)

    def _fallback_l4_summary(self, turn: UserTurn) -> UserTurn:
        output: list[dict[str, Any]] = []
        protected_indexes = self._working_turn_protected_indexes(turn)
        checkpoint = self._working_turn_checkpoint(turn)
        for index, message in enumerate(turn.messages):
            compact = self._mark_message(message, turn.turn_id, CompressionLevel.L4)
            if index not in protected_indexes and "content" in compact:
                compact["content"] = self._brief(compact.get("content", ""), 240)
            if (
                index not in protected_indexes
                and compact.get("role") == "assistant_tool_call"
                and "input" in compact
            ):
                compact["input"] = {"summary": self._brief(compact["input"], 160)}
            compact["_current_turn_summary"] = True
            compact["_checkpoint_id"] = checkpoint.checkpoint_id
            if index == 0:
                compact["_working_turn_checkpoint"] = checkpoint.to_dict()
            compact["_archive_generation"] = self.generation + 1
            output.append(compact)
        return UserTurn(turn.turn_id, output, CompressionLevel.L4)

    def _working_turn_checkpoint(self, turn: UserTurn) -> WorkingTurnCheckpoint:
        user_messages = [
            str(message.get("content") or "")
            for message in turn.messages
            if message.get("role") == "user"
        ]
        decisions = tuple(
            self._brief(message.get("content", ""), 240)
            for message in turn.messages
            if message.get("role") == "assistant" and message.get("content")
        )
        errors = tuple(
            self._brief(message.get("content", ""), 240)
            for message in turn.messages
            if message.get("role") == "tool_result" and message.get("isError")
        )
        progress = [
            str(message.get("content") or "")
            for message in turn.messages
            if message.get("role") == "assistant_progress"
        ]
        return WorkingTurnCheckpoint(
            checkpoint_id=f"working-g{self.generation + 1}-{turn.turn_id}",
            turn_id=turn.turn_id,
            generation=self.generation + 1,
            goal=user_messages[0] if user_messages else "",
            key_decisions=decisions,
            errors_and_fixes=errors,
            current_work=progress[-1] if progress else "",
        )

    def _restore_l4_turn(self, turn: UserTurn) -> UserTurn:
        """Restore an L4 summary from archive before producing a new summary."""
        summarized = [
            message for message in turn.messages if message.get("_current_turn_summary")
        ]
        if not summarized:
            return turn
        archive_generation = summarized[0].get("_archive_generation")
        archive_entry = next(
            (
                entry
                for entry in reversed(self.archive)
                if entry.get("generation") == archive_generation
            ),
            None,
        )
        if not archive_entry:
            return turn
        archived_turn = archive_entry.get("current_turn_original")
        if isinstance(archived_turn, list) and archived_turn:
            original = [copy.deepcopy(message) for message in archived_turn]
            appended = [
                copy.deepcopy(message)
                for message in turn.messages
                if not message.get("_current_turn_summary")
            ]
            return UserTurn(turn.turn_id, original + appended, CompressionLevel.NONE)
        archived_messages = archive_entry.get("messages", [])
        last_user_index = next(
            (
                index
                for index in range(len(archived_messages) - 1, -1, -1)
                if archived_messages[index].get("role") == "user"
            ),
            -1,
        )
        if last_user_index < 0:
            return turn
        original = [
            copy.deepcopy(message)
            for message in archived_messages[last_user_index:]
            if message.get("role") != "system"
            and not message.get("_l3_lock")
            and not message.get("_protected_record")
        ]
        appended = [
            copy.deepcopy(message)
            for message in turn.messages
            if not message.get("_current_turn_summary")
        ]
        return UserTurn(turn.turn_id, original + appended, CompressionLevel.NONE)

    def _summarize_l4(self, turn: UserTurn) -> UserTurn:
        if self.summary_provider is not None:
            try:
                result = self.summary_provider("l4_current_turn", copy.deepcopy(turn.messages))
                if isinstance(result, list) and len(result) == len(turn.messages):
                    valid = all(
                        isinstance(new, dict)
                        and new.get("role") == old.get("role")
                        and (
                            old.get("role") not in ("assistant_tool_call", "tool_result")
                            or (new.get("toolUseId") or new.get("tool_call_id"))
                            == (old.get("toolUseId") or old.get("tool_call_id"))
                        )
                        for old, new in zip(turn.messages, result)
                    )
                    if valid:
                        protected_indexes = self._working_turn_protected_indexes(turn)
                        checkpoint = self._working_turn_checkpoint(turn)
                        return UserTurn(
                            turn.turn_id,
                            [
                                {
                                    **(turn.messages[index] if index in protected_indexes else message),
                                    "_turn_id": turn.turn_id,
                                    "_compression_level": int(CompressionLevel.L4),
                                    "_current_turn_summary": True,
                                    "_checkpoint_id": checkpoint.checkpoint_id,
                                    **(
                                        {"_working_turn_checkpoint": checkpoint.to_dict()}
                                        if index == 0
                                        else {}
                                    ),
                                    "_archive_generation": self.generation + 1,
                                }
                                for index, message in enumerate(result)
                            ],
                            CompressionLevel.L4,
                        )
            except Exception:
                pass
        return self._fallback_l4_summary(turn)

    def _sync_bound_session(self, replacement_metadata: dict[str, Any] | None = None) -> None:
        if self._bound_session is None:
            return
        sync_surface = getattr(self._bound_session, "sync_surface", None)
        if callable(sync_surface):
            sync_surface(
                self.messages,
                turn_id=self.active_turn_id,
                source="context_manager",
                reason="context compaction replacement",
                replacement_metadata=replacement_metadata,
            )
            self.surface_revision = int(
                getattr(self._bound_session, "surface_revision", self.surface_revision)
            )
        else:
            self._bound_session.messages = self.messages
        self._bound_session.context_archive = self.archive
        self._bound_session.context_state = {
            "generation": self.generation,
            "working_turn_threshold": self.working_turn_threshold,
            "compaction_history": list(self.compaction_history),
            "protected_records": [record.__dict__.copy() for record in self.protected_records],
        }
        self._bound_session._requires_full_save = True

    def compact_messages(self, *, force: bool = False) -> list[dict[str, Any]]:
        """Run one L1-L4 cycle from a single History/Working token snapshot.

        The total high-water mark starts automatic compaction.  The independent
        Working Turn threshold only grants L4 permission; it never starts a
        cycle by itself.
        """
        stats = self.get_stats()
        plan = self.plan_compaction(stats, force=force)
        if plan.mode is CompactionMode.NONE:
            return self.messages

        target_tokens = plan.target_tokens
        working_threshold = plan.working_turn_threshold
        mode = plan.mode
        original = copy.deepcopy(self.messages)
        original_estimated_tokens = estimate_messages_tokens(original)
        archive_entry: dict[str, Any] = {
            "archive_id": f"cycle-{self.generation + 1}-{len(self.archive) + 1}",
            "generation": self.generation + 1,
            "generation_before": self.generation,
            "timestamp": time.time(),
            "status": "pending",
            "trigger": plan.trigger,
            "before_tokens": stats.total_tokens,
            "target_tokens": target_tokens,
            "history_tokens": stats.history_tokens,
            "working_turn_tokens": stats.working_turn_tokens,
            "working_turn_threshold": working_threshold,
            "compaction_mode": mode.value,
            "surface_revision": plan.surface_revision,
            "messages": original,
            "current_turn_original": None,
        }
        self.archive.append(archive_entry)
        self._sync_bound_session()
        system_messages, embedded_protected, locks, turns = self._partition_context()
        protected_messages = self._protected_messages(embedded_protected)
        if not locks and not turns:
            archive_entry["status"] = "no_change"
            self._sync_bound_session()
            return self.messages
        stages: list[str] = []
        highest = CompressionLevel.NONE
        current_turn_original: list[dict[str, Any]] | None = None
        working_turn_compacted = False

        # The last explicit turn is the mutable Working Turn.  All preceding
        # turns are closed History Turns.
        history_turns = turns[:-1] if turns else []
        working_turns = turns[-1:] if turns else []

        # L1: deterministic large-result projection only.  It is safe for both
        # scopes because it does not summarize task semantics or break pairs.
        projected_history = [self._compact_turn_l1(turn) for turn in history_turns]
        projected_working = [self._compact_turn_l1(turn) for turn in working_turns]
        if projected_history != history_turns or projected_working != working_turns:
            stages.append("L1")
            highest = CompressionLevel.L1
        history_turns = projected_history
        working_turns = projected_working
        turns = history_turns + working_turns
        compacted = self._compose(system_messages, protected_messages, locks, turns)

        # L2: History only.  Each turn keeps its user boundary, protected
        # records, and two newest complete tool call/result pairs.
        if estimate_messages_tokens(compacted) > target_tokens and history_turns:
            l2_history = [self._compact_turn_l2(turn) for turn in history_turns]
            if l2_history != history_turns:
                stages.append("L2")
                highest = max(highest, CompressionLevel.L2)
            history_turns = l2_history
            turns = history_turns + working_turns
            compacted = self._compose(system_messages, protected_messages, locks, turns)

        # L3: summarize every terminal unlocked turn as one immutable batch lock.
        if estimate_messages_tokens(compacted) > target_tokens and history_turns:
            terminal_turns = [self._restore_l4_turn(turn) for turn in history_turns]
            protected_ids = sorted(
                {
                    str(message.get("_protected_record_id"))
                    for turn in terminal_turns
                    for message in turn.messages
                    if message.get("_protected_record_id")
                }
            )
            lock = L3Lock(
                lock_id=f"lock-g{self.generation + 1}",
                first_turn_id=terminal_turns[0].turn_id,
                last_turn_id=terminal_turns[-1].turn_id,
                turn_count=len(terminal_turns),
                summary=self._summarize_l3(terminal_turns),
                protected_record_ids=protected_ids,
            )
            locks = list(locks) + [lock.to_message()]
            history_turns = []
            turns = working_turns
            stages.append("L3")
            highest = max(highest, CompressionLevel.L3)
            compacted = self._compose(system_messages, protected_messages, locks, turns)

        # L4 is reached only if L1-L3 did not meet the target.  History lock
        # eviction still runs at L4, while Working Turn semantic compression
        # additionally requires the independent threshold selected above.
        reached_l4 = estimate_messages_tokens(compacted) > target_tokens
        if reached_l4:
            l4_changed = False
            if mode is CompactionMode.HISTORY_AND_WORKING and working_turns:
                restored_current = self._restore_l4_turn(working_turns[-1])
                current_turn_original = copy.deepcopy(restored_current.messages)
                working_turns[-1] = self._summarize_l4(restored_current)
                working_turn_compacted = True
                l4_changed = True
            if locks:
                retained_locks = locks[math.ceil(len(locks) * 0.5):]
                l4_changed = l4_changed or retained_locks != locks
                locks = retained_locks
            turns = history_turns + working_turns
            if l4_changed:
                stages.append("L4")
                highest = max(highest, CompressionLevel.L4)
            compacted = self._compose(system_messages, protected_messages, locks, turns)

        if compacted == self.messages:
            archive_entry["status"] = "no_change"
            self._sync_bound_session()
            return self.messages

        after_tokens = estimate_messages_tokens(compacted)
        if after_tokens >= original_estimated_tokens:
            archive_entry["status"] = "rejected_not_smaller"
            archive_entry["candidate_tokens"] = after_tokens
            self._sync_bound_session()
            return self.messages

        self.generation += 1
        archive_entry["generation"] = self.generation
        archive_entry["status"] = "committed"
        archive_entry["current_turn_original"] = current_turn_original
        self.compaction_history.append(
            {
                "timestamp": time.time(),
                "before_tokens": stats.total_tokens,
                "after_tokens": after_tokens,
                "messages_removed": len(original) - len(compacted),
                "target_tokens": target_tokens,
                "trigger_ratio": AUTOCOMPACT_THRESHOLD,
                "target_ratio": COMPACTION_TARGET,
                "highest_level": highest.name,
                "stages": stages,
                "generation": self.generation,
                "target_reached": after_tokens <= target_tokens,
                "history_tokens": stats.history_tokens,
                "working_turn_tokens": stats.working_turn_tokens,
                "working_turn_threshold": working_threshold,
                "compaction_mode": mode.value,
                "reached_l4": reached_l4,
                "working_turn_compacted": working_turn_compacted,
            }
        )
        self._compaction_level = int(highest)
        self.messages = compacted
        self._token_meter = TokenMeter()
        self._sync_bound_session(
            replacement_metadata={
                "generation": self.generation,
                "model": self.model,
                "before_tokens": stats.total_tokens,
                "after_tokens": after_tokens,
                "target_tokens": target_tokens,
                "trigger": plan.trigger,
                "compaction_mode": mode.value,
                "stages": list(stages),
                "surface_revision": plan.surface_revision,
            }
        )
        return compacted

    @staticmethod
    def _compress_tool_pair(call_msg: dict[str, Any], result_msg: dict[str, Any]) -> str:
        """将 tool_call 和 tool_result 消息对压缩为紧凑的内联摘要。

        各类工具的特定压缩策略：
        - 编辑工具：保留文件路径和成功/失败状态
        - 只读工具：仅记录读取的文件（内容可重新读取）
        - 搜索工具：保留搜索模式和结果数量
        - 命令工具：保留命令名称和退出状态
        - 错误结果：保留错误消息（对调试至关重要）

        参数:
            call_msg: 工具调用消息字典。
            result_msg: 工具结果消息字典。

        返回:
            压缩后的内联摘要字符串，如 "[Edited /path/to/file: ok]"。
        """
        tool_name = call_msg.get("toolName", "unknown")
        inp = call_msg.get("input", {})
        result_content = result_msg.get("content", "")
        is_error = result_msg.get("isError", False)

        # Error results: preserve the error message
        if is_error:
            error_text = result_content.strip()[:200].replace("\n", " ")
            return f"[Tool {tool_name} ERROR: {error_text}]"

        # Tool-specific compression
        if tool_name in _EDIT_TOOLS:
            path = inp.get("path") or inp.get("filePath", "unknown")
            # Preserve key edit details
            if tool_name == "multi_edit":
                edits = inp.get("edits", [])
                return f"[Edited {path}: {len(edits)} changes applied]"
            return f"[Edited {path}: ok]"

        if tool_name in _READ_TOOLS:
            path = inp.get("path") or inp.get("filePath", "")
            if path:
                # Note: content can be re-read, so just record that it was read
                line_count = result_content.count("\n") + 1
                return f"[Read {path}: {line_count} lines]"
            return f"[{tool_name}: completed]"

        if tool_name in _SEARCH_TOOLS:
            pattern = inp.get("pattern") or inp.get("query", "")
            # Count matches from result
            match_lines = [line for line in result_content.split("\n") if line.strip() and not line.startswith("#")]
            return f"[Searched '{pattern[:50]}': {len(match_lines)} results]"

        if tool_name in _COMMAND_TOOLS:
            cmd = inp.get("command", "")
            cmd_name = cmd.split()[0] if cmd.split() else "command"
            # Check for success indicators
            exit_info = ""
            if "exit code" in result_content.lower():
                for line in result_content.split("\n"):
                    if "exit code" in line.lower():
                        exit_info = f" ({line.strip()[:50]})"
                        break
            return f"[Ran {cmd_name}{exit_info}]"

        # Generic compression: tool name + brief result
        brief = result_content.strip()[:100].replace("\n", " ")
        if brief:
            return f"[{tool_name}: {brief}]"
        return f"[{tool_name}: completed]"

    def _finalize_compaction(
        self,
        system_messages: list[dict[str, Any]],
        original_other: list[dict[str, Any]],
        filtered: list[dict[str, Any]],
        stats: ContextStats,
        target_tokens: int,
        dropped_messages: list[dict[str, Any]] | None = None,
        summary_budget: int = 2000,
    ) -> list[dict[str, Any]]:
        """构建压缩后的消息列表：保留的系统消息 + 保留的对话消息 + 末尾摘要。

        不插入 compaction marker 到系统 prompt 区域，不修改任何消息内容，
        仅从开头删除整组消息并在末尾追加一条 assistant 摘要。

        参数:
            system_messages: 系统提示消息列表（始终保留）。
            original_other: 原始非系统消息列表（仅用于计算统计）。
            filtered: 压缩后保留的消息列表。
            stats: 压缩前的上下文统计信息。
            target_tokens: 压缩目标 token 数。
            dropped_messages: 可选的已删除消息列表。为 None 时通过 diff 计算。

        返回:
            最终的消息列表（系统消息 + 保留消息 + 末尾摘要）。
        """
        # 确定被删除的消息
        if dropped_messages is not None:
            removed_messages = dropped_messages
        else:
            removed_set = set(id(m) for m in filtered)
            removed_messages = [m for m in original_other if id(m) not in removed_set]

        removed_count = len(removed_messages)

        # 构建分层摘要
        checkpoint_state = _extract_from_messages(removed_messages)
        summary_text = _build_layered_summary(checkpoint_state, summary_budget)

        # A generation switch intentionally invalidates the dynamic prefix
        # once. The stable system prefix remains first and recent messages keep
        # their chronological position after the checkpoint.
        compacted = list(system_messages)
        if summary_text:
            compacted.append({
                "role": "assistant",
                "content": (
                    f"[Context checkpoint for generation {self.generation + 1}]\n"
                    f"{summary_text}"
                ),
                "_context_checkpoint": True,
                "_checkpoint_state": checkpoint_state.to_dict(),
                "_context_generation": self.generation + 1,
            })
        compacted.extend(filtered)

        # Record compaction
        self.compaction_history.append({
            "timestamp": time.time(),
            "before_tokens": stats.total_tokens,
            "after_tokens": estimate_messages_tokens(compacted),
            "messages_removed": removed_count,
            "compaction_level": self._compaction_level,
            "generation": self.generation + 1,
        })

        self._compaction_level = min(self._compaction_level + 1, 3)
        self.generation += 1
        self.messages = compacted
        self._token_meter = TokenMeter()
        return compacted

    def get_context_summary(self) -> str:
        """获取人类可读的上下文使用摘要。

        格式化的单行摘要，包含使用率、总 token、上下文窗口大小、
        消息数量和工具调用次数，并使用状态指示符（✓/⚠/🔴）。

        返回:
            格式化的上下文摘要字符串，如 "Context: ✓ 45% (45,000/100,000 tokens, 12 msgs, 3 tools)"。
        """
        stats = self.get_stats()

        if stats.messages_count == 0:
            return "Context: empty"

        status = "✓"
        if stats.is_near_limit:
            status = "⚠"
        if stats.should_compact:
            status = "🔴"

        return (
            f"Context: {status} {stats.usage_percentage:.0f}% "
            f"({stats.total_tokens:,}/{stats.context_window:,} tokens, "
            f"{stats.messages_count} msgs, {stats.tool_calls_count} tools)"
        )

    def format_context_details(self) -> str:
        """获取详细的上下文信息，用于 /context 命令展示。

        包含模型名称、上下文窗口大小、token 使用详情、使用率百分比、
        消息数量和工具调用次数。在接近容量时显示告警，并列出最近
        3 次压缩历史记录。

        返回:
            格式化的多行详情字符串。
        """
        stats = self.get_stats()

        lines = [
            "Context Window Usage",
            "=" * 50,
            f"Model: {self.model}",
            f"Context window: {stats.context_window:,} tokens",
            "",
            f"Total tokens: {stats.total_tokens:,}",
            f"System / tools / dynamic: {stats.system_tokens:,} / {stats.tools_tokens:,} / {stats.dynamic_tokens:,}",
            f"History tokens: {stats.history_tokens:,}",
            f"Working Turn tokens: {stats.working_turn_tokens:,} / {self.working_turn_threshold:,}",
            f"Output reserve: {stats.output_reserve:,}",
            f"Request total: {stats.request_total_tokens:,} (Surface r{stats.surface_revision})",
            f"Usage: {stats.usage_percentage:.1f}%",
            f"Messages: {stats.messages_count}",
            f"Tool calls: {stats.tool_calls_count}",
            "",
        ]

        if stats.should_compact:
            lines.append("⚠️  WARNING: Context is near capacity!")
            lines.append("Auto-compaction will trigger soon.")
            lines.append("")

        if self.compaction_history:
            lines.append("Compaction History:")
            for comp in self.compaction_history[-3:]:  # Last 3
                ts = time.strftime("%H:%M:%S", time.localtime(comp["timestamp"]))
                lines.append(
                    f"  {ts}: {comp['messages_removed']} messages removed, "
                    f"{comp['before_tokens']:,} → {comp['after_tokens']:,} tokens"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_context_state(manager: ContextManager) -> None:
    """将上下文管理器状态保存到磁盘。

    将模型名称、上下文窗口大小、消息列表和压缩历史等状态序列化为 JSON，
    写入 SMART_AGENT_DIR 下的 context_state.json 文件。仅保留最近 10 条压缩历史记录。

    参数:
        manager: 要保存状态的 ContextManager 实例。
    """
    if manager._bound_session is not None:
        manager._sync_bound_session()
        return
    raise ValueError("ContextManager must be bound to a session before persistence")

def load_context_state() -> ContextManager | None:
    """从磁盘加载上下文管理器状态。

    读取 SMART_AGENT_DIR 下的 context_state.json 文件，反序列化恢复 ContextManager 实例。
    如果文件不存在或 JSON 解析失败，返回 None。

    返回:
        成功时返回恢复的 ContextManager 实例，失败时返回 None。
    """
    state_path = SMART_AGENT_DIR / "context_state.json"
    if not state_path.exists():
        return None

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manager = ContextManager(
            model=state.get("model", "default"),
            context_window=state.get("context_window", 0),
            messages=state.get("messages", []),
            compaction_history=state.get("compaction_history", []),
            generation=state.get("generation", 0),
            working_turn_threshold=state.get(
                "working_turn_threshold", DEFAULT_WORKING_TURN_THRESHOLD
            ),
            protected_records=[
                ProtectedRecord(**item)
                for item in state.get("protected_records", [])
                if isinstance(item, dict)
            ],
        )
        # Restore compaction level if saved
        if "_compaction_level" in state:
            manager._compaction_level = state["_compaction_level"]
        return manager
    except (json.JSONDecodeError, KeyError):
        return None


def clear_context_state() -> None:
    """清除已保存的上下文状态。

    删除磁盘上的 context_state.json 文件。如果文件不存在，则不执行任何操作。
    """
    state_path = SMART_AGENT_DIR / "context_state.json"
    if state_path.exists():
        state_path.unlink()
