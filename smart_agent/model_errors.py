"""Structured model-provider errors consumed by the Agent runtime."""

from __future__ import annotations

from typing import Any


class ContextWindowExceeded(RuntimeError):
    """The provider rejected a request because its context was too large."""

    code = "CONTEXT_WINDOW_EXCEEDED"


_CONTEXT_ERROR_CODES = {
    "context_length_exceeded",
    "context_window_exceeded",
    "max_tokens_exceeded",
    "prompt_too_long",
    "request_too_large",
}


def raise_for_context_window_error(payload: Any, status: int) -> None:
    """Normalize provider-specific structured error codes at the Adapter boundary."""
    if not isinstance(payload, dict):
        return
    error = payload.get("error")
    if not isinstance(error, dict):
        error = payload
    code = str(error.get("code") or error.get("type") or "").strip().lower()
    if code in _CONTEXT_ERROR_CODES or (status == 413 and code != ""):
        message = str(error.get("message") or ContextWindowExceeded.code)
        raise ContextWindowExceeded(message)
