"""Persist and project oversized tool results before model exposure."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolResultPersisted:
    original_size: int
    persisted_path: Path
    preview_text: str
    tool_name: str
    content_hash: str = ""
    timestamp: float = field(default_factory=time.time)


class ToolResultBudgetManager:
    """Store complete large text results and expose a deterministic preview."""

    DEFAULT_BUDGET_PER_MESSAGE = 8_192
    PERSIST_THRESHOLD = 8_192
    PREVIEW_HEAD_CHARS = 4_096
    PREVIEW_TAIL_CHARS = 1_024
    PRUNED_MARKER = "\n\n[... tool result middle pruned ...]\n\n"

    def __init__(
        self,
        workspace: str | Path | None = None,
        budget_per_message: int = DEFAULT_BUDGET_PER_MESSAGE,
        persist_threshold: int = PERSIST_THRESHOLD,
    ) -> None:
        self._workspace = Path(workspace) if workspace else Path.cwd()
        self._budget = budget_per_message
        self._persist_threshold = persist_threshold
        self._results_dir = self._workspace / ".runtime/tool-results"
        self._persisted: dict[str, ToolResultPersisted] = {}

    def check_and_replace(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        modified = list(messages)
        bytes_saved = 0
        for index, message in enumerate(modified):
            if message.get("role") != "tool_result":
                continue
            tool_name = message.get("toolName", "unknown")
            projected, persisted_path, saved = self.project_result(
                message.get("content"),
                tool_name=tool_name,
                result_key=str(index),
            )
            modified[index] = {**message, "content": projected}
            if persisted_path is not None:
                modified[index]["_persisted_path"] = persisted_path
            bytes_saved += saved
        return modified, bytes_saved

    def project_result(
        self,
        content: Any,
        *,
        tool_name: str,
        result_key: str,
    ) -> tuple[Any, str | None, int]:
        if not isinstance(content, str) or len(content) <= self._persist_threshold:
            return content, None, 0
        preview = self._generate_preview(content)
        if len(preview) >= len(content):
            return content, None, 0
        persisted = self._persist_content(content, tool_name, result_key)
        self._persisted[f"{result_key}-{tool_name}"] = persisted
        return preview, str(persisted.persisted_path), len(content) - len(preview)

    def _persist_content(
        self,
        content: str,
        tool_name: str,
        result_key: str,
    ) -> ToolResultPersisted:
        self._results_dir.mkdir(parents=True, exist_ok=True)
        safe_key = hashlib.sha256(result_key.encode("utf-8")).hexdigest()[:12]
        safe_tool_name = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in tool_name
        )
        path = self._results_dir / f"{safe_tool_name}_{safe_key}_{int(time.time() * 1000)}.txt"
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        metadata = {
            "tool_name": tool_name,
            "result_key": result_key,
            "original_size": len(content),
            "sha256": content_hash,
            "timestamp": time.time(),
        }
        file_descriptor, temporary_path = tempfile.mkstemp(
            dir=str(self._results_dir),
            prefix=".tool_result_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(metadata, ensure_ascii=False) + "\n---CONTENT---\n")
                handle.write(content)
            os.replace(temporary_path, path)
        except BaseException:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise
        return ToolResultPersisted(
            original_size=len(content),
            persisted_path=path,
            preview_text="",
            tool_name=tool_name,
            content_hash=content_hash,
        )

    @classmethod
    def _generate_preview(cls, content: str) -> str:
        return (
            content[: cls.PREVIEW_HEAD_CHARS]
            + cls.PRUNED_MARKER
            + content[-cls.PREVIEW_TAIL_CHARS :]
        )

    def get_persisted_count(self) -> int:
        return len(self._persisted)

    def get_total_saved_bytes(self) -> int:
        return sum(result.original_size for result in self._persisted.values())
