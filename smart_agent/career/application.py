"""Safety-first contracts for semi-automated job application forms."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

APPLICATION_SCHEMA_VERSION = "1.0"


class ApplicationFieldState(str, Enum):
    AUTO_FILLED = "auto_filled"
    NEEDS_CONFIRMATION = "needs_confirmation"
    MISSING = "missing"
    SENSITIVE = "sensitive"


class ApplicationAction(str, Enum):
    NAVIGATE = "navigate"
    INSPECT = "inspect"
    FILL = "fill"
    UPLOAD = "upload"
    LEAVE_BLANK = "leave_blank"
    REQUEST_CONFIRMATION = "request_confirmation"
    SUBMIT = "submit"


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required.")
    return text


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ApplicationField:
    field_id: str
    label: str
    value: str = ""
    state: ApplicationFieldState = ApplicationFieldState.MISSING
    source_claim_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    sensitive: bool = False
    locator: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "field_id", _required_text(self.field_id, "field_id"))
        object.__setattr__(self, "label", _required_text(self.label, "label"))
        object.__setattr__(self, "value", str(self.value or "").strip())
        object.__setattr__(self, "state", ApplicationFieldState(self.state))
        object.__setattr__(self, "source_claim_ids", tuple(dict.fromkeys(str(item).strip() for item in self.source_claim_ids if str(item).strip())))
        object.__setattr__(self, "locator", dict(self.locator))
        confidence = float(self.confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        if self.sensitive:
            object.__setattr__(self, "state", ApplicationFieldState.SENSITIVE)
        if self.state is ApplicationFieldState.AUTO_FILLED and not self.value:
            raise ValueError("Auto-filled fields require a value")
        if self.state is ApplicationFieldState.MISSING and self.value:
            raise ValueError("Missing fields cannot contain a value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "label": self.label,
            "value": self.value,
            "state": self.state.value,
            "source_claim_ids": list(self.source_claim_ids),
            "confidence": self.confidence,
            "sensitive": self.sensitive,
            "locator": self.locator,
        }


@dataclass(frozen=True, slots=True)
class ApplicationForm:
    form_id: str
    url: str
    fields: tuple[ApplicationField, ...]
    resume_document_id: str
    verified_resume: bool = False
    domain_allowlist: tuple[str, ...] = ()
    schema_version: str = APPLICATION_SCHEMA_VERSION
    content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "form_id", _required_text(self.form_id, "form_id"))
        object.__setattr__(self, "resume_document_id", _required_text(self.resume_document_id, "resume_document_id"))
        parsed = urlparse(_required_text(self.url, "url"))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Application URL must be an absolute HTTP(S) URL")
        object.__setattr__(self, "url", self.url.strip())
        object.__setattr__(self, "fields", tuple(self.fields))
        if len({field.field_id for field in self.fields}) != len(self.fields):
            raise ValueError("Application field IDs must be unique")
        domains = tuple(dict.fromkeys(str(item).strip().lower() for item in self.domain_allowlist if str(item).strip()))
        object.__setattr__(self, "domain_allowlist", domains)
        if domains and parsed.hostname and parsed.hostname.lower() not in domains:
            raise ValueError("Application URL is outside the configured domain allowlist")
        if self.schema_version != APPLICATION_SCHEMA_VERSION:
            raise ValueError("Unsupported ApplicationForm schema version")
        identity = {
            "form_id": self.form_id,
            "url": self.url,
            "fields": [field.to_dict() for field in self.fields],
            "resume_document_id": self.resume_document_id,
            "verified_resume": self.verified_resume,
            "domain_allowlist": self.domain_allowlist,
            "schema_version": self.schema_version,
        }
        expected = _hash(identity)
        if self.content_hash and self.content_hash != expected:
            raise ValueError("ApplicationForm content_hash does not match its content")
        object.__setattr__(self, "content_hash", expected)

    @property
    def can_submit(self) -> bool:
        return self.verified_resume and not any(
            field.state
            in {
                ApplicationFieldState.NEEDS_CONFIRMATION,
                ApplicationFieldState.MISSING,
                ApplicationFieldState.SENSITIVE,
            }
            for field in self.fields
        )


class ApplicationBrowser(Protocol):
    """Browser adapter boundary; CAPTCHA and submission remain user-controlled."""

    name: str

    def inspect(self, url: str) -> tuple[dict[str, Any], ...]: ...

    def fill(self, field_id: str, value: str) -> None: ...

    def upload(self, field_id: str, path: str) -> None: ...

    def close(self) -> None: ...


class ApplicationPlanner(Protocol):
    name: str

    def plan(self, *, url: str, resume_document_id: str, fields: tuple[dict[str, Any], ...]) -> ApplicationForm: ...


@dataclass(frozen=True, slots=True)
class ApplicationAuditEntry:
    field_id: str
    action: ApplicationAction
    state: ApplicationFieldState
    reason: str
    source_claim_ids: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "action": self.action.value,
            "state": self.state.value,
            "reason": self.reason,
            "source_claim_ids": list(self.source_claim_ids),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ApplicationMetrics:
    """Offline-evaluable metrics for the safe field-planning boundary."""

    total_fields: int
    auto_filled_fields: int
    auto_fill_rate: float
    direct_evidence_accuracy: float
    manual_modification_count: int = 0
    elapsed_ms: float = 0.0

    @classmethod
    def from_form(
        cls,
        form: ApplicationForm,
        *,
        manual_modification_count: int = 0,
        elapsed_ms: float = 0.0,
    ) -> "ApplicationMetrics":
        total = len(form.fields)
        auto_fields = tuple(field for field in form.fields if field.state is ApplicationFieldState.AUTO_FILLED)
        supported = sum(1 for field in auto_fields if field.source_claim_ids)
        return cls(
            total_fields=total,
            auto_filled_fields=len(auto_fields),
            auto_fill_rate=len(auto_fields) / max(1, total),
            direct_evidence_accuracy=supported / max(1, len(auto_fields)),
            manual_modification_count=max(0, int(manual_modification_count)),
            elapsed_ms=max(0.0, float(elapsed_ms)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_fields": self.total_fields,
            "auto_filled_fields": self.auto_filled_fields,
            "auto_fill_rate": self.auto_fill_rate,
            "direct_evidence_accuracy": self.direct_evidence_accuracy,
            "manual_modification_count": self.manual_modification_count,
            "elapsed_ms": self.elapsed_ms,
        }


class EvidenceApplicationPlanner:
    """Map only directly evidenced resume claims to form fields."""

    name = "evidence-gated-planner"

    def plan(self, *, url: str, resume_document_id: str, fields: tuple[dict[str, Any], ...]) -> ApplicationForm:
        planned: list[ApplicationField] = []
        for raw in fields:
            label = _required_text(raw.get("label"), "field label")
            field_id = str(raw.get("field_id") or normalize_field_id(label))
            claim_ids = tuple(str(item).strip() for item in raw.get("source_claim_ids", ()) if str(item).strip())
            value = str(raw.get("value") or "").strip()
            sensitive = bool(raw.get("sensitive", False))
            if sensitive:
                state = ApplicationFieldState.SENSITIVE
                value = ""
                reason = "Sensitive fields require direct user input."
            elif value and claim_ids and bool(raw.get("verified", True)):
                state = ApplicationFieldState.AUTO_FILLED
                reason = "Directly supported by the verified resume claim set."
            elif value:
                state = ApplicationFieldState.NEEDS_CONFIRMATION
                value = ""
                reason = "A value exists but its evidence is missing or ambiguous."
            else:
                state = ApplicationFieldState.MISSING
                reason = "No direct evidence; leave blank for the user."
            planned.append(
                ApplicationField(
                    field_id=field_id,
                    label=label,
                    value=value,
                    state=state,
                    source_claim_ids=claim_ids,
                    confidence=float(raw.get("confidence", 1.0 if state is ApplicationFieldState.AUTO_FILLED else 0.0)),
                    sensitive=sensitive,
                    locator=dict(raw.get("locator") or {}),
                )
            )
        return ApplicationForm(
            form_id=f"form-{_hash([url, resume_document_id, [field.to_dict() for field in planned]])[:16]}",
            url=url,
            fields=tuple(planned),
            resume_document_id=resume_document_id,
            verified_resume=bool(fields and all(bool(raw.get("resume_verified", True)) for raw in fields)),
        )


class ApplicationWorkflow:
    """Prepare and fill safe fields; submission is intentionally absent."""

    def __init__(self, planner: ApplicationPlanner | None = None) -> None:
        self.planner = planner or EvidenceApplicationPlanner()

    def prepare(
        self,
        *,
        url: str,
        resume_document_id: str,
        fields: tuple[dict[str, Any], ...],
    ) -> tuple[ApplicationForm, tuple[ApplicationAuditEntry, ...]]:
        form = self.planner.plan(url=url, resume_document_id=resume_document_id, fields=fields)
        audit = tuple(
            ApplicationAuditEntry(
                field_id=field.field_id,
                action=ApplicationAction.FILL if field.state is ApplicationFieldState.AUTO_FILLED else ApplicationAction.LEAVE_BLANK,
                state=field.state,
                reason=(
                    "direct evidence"
                    if field.state is ApplicationFieldState.AUTO_FILLED
                    else "missing, sensitive, or needs confirmation"
                ),
                source_claim_ids=field.source_claim_ids,
            )
            for field in form.fields
        )
        return form, audit

    def fill_auto_fields(self, form: ApplicationForm, browser: ApplicationBrowser) -> tuple[ApplicationAuditEntry, ...]:
        audit: list[ApplicationAuditEntry] = []
        for application_field in form.fields:
            if application_field.state is ApplicationFieldState.AUTO_FILLED:
                browser.fill(application_field.field_id, application_field.value)
                audit.append(
                    ApplicationAuditEntry(
                        field_id=application_field.field_id,
                        action=ApplicationAction.FILL,
                        state=application_field.state,
                        reason="filled from verified claim",
                        source_claim_ids=application_field.source_claim_ids,
                    )
                )
            else:
                audit.append(
                    ApplicationAuditEntry(
                        field_id=application_field.field_id,
                        action=ApplicationAction.LEAVE_BLANK,
                        state=application_field.state,
                        reason="not directly evidenced; user action required",
                        source_claim_ids=application_field.source_claim_ids,
                    )
                )
        return tuple(audit)


class PlaywrightApplicationBrowser:
    """Optional Playwright adapter with exact-domain navigation boundaries."""

    name = "playwright"

    def __init__(
        self,
        *,
        allowed_domains: tuple[str, ...],
        headless: bool = True,
        allow_insecure_localhost: bool = False,
    ) -> None:
        domains = tuple(dict.fromkeys(str(item).strip().lower() for item in allowed_domains if str(item).strip()))
        if not domains:
            raise ValueError("At least one allowed domain is required")
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Install the optional web dependencies to use Playwright.") from exc
        self._allowed_domains = domains
        self._allow_insecure_localhost = bool(allow_insecure_localhost)
        self._playwright = sync_playwright().start()
        try:
            launch_kwargs: dict[str, Any] = {"headless": headless}
            candidates: list[Path] = []
            for pattern in (
                "AppData/Local/ms-playwright/chromium-*/chrome-*/chrome.exe",
                ".cache/ms-playwright/chromium-*/chrome-linux/chrome",
                ".cache/ms-playwright/chromium-*/chrome-linux/headless_shell",
            ):
                candidates.extend(Path.home().glob(pattern))
            if candidates:
                launch_kwargs["executable_path"] = str(sorted(candidates)[-1])
            self._browser = self._playwright.chromium.launch(**launch_kwargs)
            self._page = self._browser.new_page()
        except Exception:
            self._playwright.stop()
            raise
        self._audit: list[ApplicationAuditEntry] = []
        self._snapshot_sha256 = ""

    @property
    def audit_log(self) -> tuple[ApplicationAuditEntry, ...]:
        """Return an audit-only trace; it is not supplied to the model context."""
        return tuple(self._audit)

    @property
    def page_snapshot_sha256(self) -> str:
        return self._snapshot_sha256

    def _record(self, *, field_id: str, action: ApplicationAction, reason: str) -> None:
        self._audit.append(
            ApplicationAuditEntry(
                field_id=field_id,
                action=action,
                state=ApplicationFieldState.MISSING,
                reason=reason,
            )
        )

    def _assert_allowed(self, url: str) -> None:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        secure = parsed.scheme == "https"
        local_test = (
            self._allow_insecure_localhost
            and parsed.scheme == "http"
            and hostname in {"localhost", "127.0.0.1"}
        )
        if not (secure or local_test) or hostname not in self._allowed_domains:
            raise PermissionError("Application navigation is outside the HTTPS domain allowlist")

    def inspect(self, url: str) -> tuple[dict[str, Any], ...]:
        self._assert_allowed(url)
        self._page.goto(url, wait_until="domcontentloaded")
        snapshot = self._page.content()
        self._snapshot_sha256 = hashlib.sha256(snapshot.encode("utf-8")).hexdigest()
        self._record(
            field_id="__page__",
            action=ApplicationAction.NAVIGATE,
            reason=f"Navigated within the allowlist; snapshot={self._snapshot_sha256}.",
        )
        fields: list[dict[str, Any]] = []
        for locator in self._page.locator("input, textarea, select").all():
            field_id = locator.get_attribute("id") or locator.get_attribute("name")
            if not field_id:
                continue
            fields.append(
                {
                    "field_id": field_id,
                    "label": locator.get_attribute("aria-label") or locator.get_attribute("name") or field_id,
                    "type": locator.get_attribute("type") or "text",
                    "required": bool(locator.get_attribute("required")),
                    "locator": {"id": field_id},
                }
            )
        self._record(field_id="__form__", action=ApplicationAction.INSPECT, reason=f"Inspected {len(fields)} form fields.")
        return tuple(fields)

    def fill(self, field_id: str, value: str) -> None:
        if not str(field_id or "").strip() or not str(value or "").strip():
            raise ValueError("field_id and value are required")
        locator = self._page.locator(f"#{field_id}")
        if locator.count() == 0:
            locator = self._page.locator(f"[name='{field_id}']")
        if locator.count() == 0:
            raise KeyError(f"Application field not found: {field_id}")
        locator.first.fill(value)
        self._record(field_id=field_id, action=ApplicationAction.FILL, reason="Filled from a verified claim.")

    def upload(self, field_id: str, path: str) -> None:
        if not str(field_id or "").strip() or not str(path or "").strip():
            raise ValueError("field_id and path are required")
        locator = self._page.locator(f"#{field_id}")
        if locator.count() == 0:
            locator = self._page.locator(f"[name='{field_id}']")
        if locator.count() == 0:
            raise KeyError(f"Application upload field not found: {field_id}")
        locator.first.set_input_files(path)
        self._record(field_id=field_id, action=ApplicationAction.UPLOAD, reason="Uploaded after user-approved field planning.")

    def close(self) -> None:
        try:
            self._browser.close()
        finally:
            self._playwright.stop()


def normalize_field_id(label: str, *, fallback: str = "field") -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(label or "").casefold()).strip("_")
    return normalized or fallback
