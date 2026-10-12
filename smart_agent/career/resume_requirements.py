"""Structured user requirements and evidence-safe JD coverage checks.

The resume generator used to receive one free-form customization string.  This
module keeps the user-facing wording while turning each numbered request into a
small auditable record.  The deterministic critic is deliberately conservative:
it can only reuse text already present in selected evidence blocks and never
creates a new claim, metric, tool, or responsibility.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Iterable

_NUMBERED_MARKER = re.compile(r"(?m)(?:^|\s)(\d{1,3})\s*[、.．)）:：]\s*")
_USER_REQUEST_HEADING = re.compile(
    r"^\s*#{1,6}\s*(?:用户需求|用户要求|user\s+requests?)\s*:?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
_JD_METADATA_LINE = re.compile(
    r"^\s*(?:公司(?:名字|名称)?|company|职位|岗位|position|role)\s*[:：]",
    re.IGNORECASE,
)
_UNSUPPORTED_MARKERS = re.compile(
    r"(?:线上|真实业务|生产环境|a\s*/\s*b|ab实验|成功率|提升(?:至|到|数据|百分点)?|"
    r"客户|用户规模|百万级|qps|slo|营收|收入|上线|回滚|芯片测试通过率|跨用户长期)",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9+#./-]{1,}|\d+(?:\.\d+)?%?|[\u4e00-\u9fff]{2,}", re.IGNORECASE)
_INTERNSHIP_LABEL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("company", ("实习公司名称", "公司")),
    ("department", ("项目部门",)),
    ("role", ("实习岗位", "岗位")),
    ("summary", ("实习简述",)),
    ("content", ("实习内容", "内容", "职责")),
    ("date", ("实习日期", "日期", "时间")),
)

_FACETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("任务理解", ("任务理解", "需求理解", "目标进入")),
    ("规划", ("规划", "planning", "plan-execute", "重规划")),
    ("工具调用", ("工具调用", "多工具", "tool use", "toolresult", "工具编排")),
    ("结果验证", ("结果验证", "验证", "核验", "verification", "评测")),
    ("Agent Runtime", ("agent runtime", "runtime", "运行时")),
    ("Root Supervisor", ("root supervisor", "根监督")),
    ("Domain/Stage", ("domain/stage", "domain", "stage")),
    ("多 Agent 协作", ("多 agent", "multi-agent", "child agent", "多智能体")),
    ("上下文与记忆", ("context", "上下文", "memory", "记忆", "checkpoint")),
    ("RAG 混合检索", ("混合检索", "bge-m3", "weighted rrf", "crossencoder", "双路召回")),
    ("文档解析与 OCR", ("文档解析", "ocr", "paddleocr", "ppstructure", "复杂版面")),
    ("来源追踪", ("来源", "source", "page", "chunk", "元数据")),
)


@dataclass(frozen=True, slots=True)
class ResumeUserRequirement:
    """One user request with an explicit evidence decision."""

    request_id: str
    text: str
    support_status: str
    decision: str
    must_cover_facets: tuple[str, ...] = ()
    target_project: str = ""
    evidence_refs: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "text": self.text,
            "support_status": self.support_status,
            "decision": self.decision,
            "must_cover_facets": list(self.must_cover_facets),
            "target_project": self.target_project,
            "evidence_refs": list(self.evidence_refs),
            "reason": self.reason,
        }


def _tokens(value: str) -> set[str]:
    raw = str(value or "").casefold()
    result: set[str] = set()
    for token in _TOKEN_PATTERN.findall(raw):
        if len(token) > 2 and re.fullmatch(r"[\u4e00-\u9fff]+", token):
            result.update(token[index : index + 2] for index in range(len(token) - 1))
        else:
            result.add(token)
    return result


def _normalize_requirement_phrase(value: str) -> str:
    """Normalize a requirement for stable duplicate/section comparisons."""

    text = str(value or "").strip()
    text = re.sub(r"^\s*\d{1,3}\s*[、.．)）:：]\s*", "", text)
    text = re.sub(r"[；;。.!！]+$", "", text)
    return re.sub(r"\s+", "", text).casefold()


def _user_request_section_start(raw_text: str) -> int | None:
    match = _USER_REQUEST_HEADING.search(str(raw_text or ""))
    return match.start() if match else None


def _record_source_starts(record: Any) -> tuple[int, ...]:
    refs = getattr(record, "source_refs", ())
    if isinstance(record, dict):
        refs = record.get("source_refs") or record.get("source_refs", ())
    starts: list[int] = []
    for ref in refs or ():
        locator = getattr(ref, "locator", None)
        if isinstance(ref, dict):
            locator = ref.get("locator")
        if not isinstance(locator, dict):
            continue
        value = locator.get("start")
        if isinstance(value, int) and value >= 0:
            starts.append(value)
    return tuple(starts)


def _record_texts(record: Any) -> tuple[str, ...]:
    values = (
        getattr(record, "text", "") if not isinstance(record, dict) else record.get("text", ""),
        getattr(record, "original_text", "")
        if not isinstance(record, dict)
        else record.get("original_text", ""),
    )
    return tuple(str(value or "").strip() for value in values if str(value or "").strip())


def _is_jd_metadata_record(record: Any) -> bool:
    return any(_JD_METADATA_LINE.match(text) for text in _record_texts(record))


def _is_user_request_record(
    record: Any,
    *,
    raw_text: str,
    user_requirements: Iterable[ResumeUserRequirement] = (),
) -> bool:
    """Detect user-request records, including stale imported records.

    Source offsets are authoritative when present.  Text matching and source
    lookup are conservative fallbacks for older records that predate source
    provenance or were imported from a flat JSON payload.
    """

    section_start = _user_request_section_start(raw_text)
    if section_start is not None:
        starts = _record_source_starts(record)
        if any(start >= section_start for start in starts):
            return True
    user_phrases = {
        _normalize_requirement_phrase(item.text)
        for item in user_requirements
        if isinstance(item, ResumeUserRequirement) and item.text
    }
    texts = _record_texts(record)
    if any(_normalize_requirement_phrase(text) in user_phrases for text in texts):
        return True
    if section_start is None:
        return False
    for text in texts:
        cursor = 0
        found_before = False
        found_after = False
        while True:
            position = raw_text.find(text, cursor)
            if position < 0:
                break
            if position >= section_start:
                found_after = True
            else:
                found_before = True
            cursor = position + max(1, len(text))
        if found_after and not found_before:
            return True
    return False


def build_resume_jd_blocks(
    jd: Any,
    store: Any,
    user_requirements: Iterable[ResumeUserRequirement] = (),
) -> list[dict[str, Any]]:
    """Build typed JD blocks while excluding embedded user customizations.

    ``text`` remains available for backward-compatible model clients, while
    ``source``/``kind`` make the boundary explicit for new clients and audit
    consumers.  The function also protects runs that load a JD normalized by
    an older parser which accidentally persisted user-request rows.
    """

    raw_text = str(getattr(jd, "raw_text", "") or "")
    blocks: list[dict[str, Any]] = []
    role_title = str(getattr(jd, "role_title", "") or "").strip()
    if role_title:
        blocks.append(
            {
                "requirement_id": "__role_title__",
                "text": f"目标岗位名称：{role_title}",
                "source": "jd_metadata",
                "kind": "role_title",
            }
        )
    seen: set[str] = set()
    for ref in getattr(jd, "requirement_refs", ()) or ():
        record_id = str(getattr(ref, "record_id", "") or "").strip()
        revision = getattr(ref, "revision", None)
        if not record_id:
            continue
        try:
            record = store.get(record_id, revision)
        except Exception:  # noqa: BLE001 - stale optional records are skipped.
            continue
        if not bool(getattr(record, "active", True)):
            continue
        text = str(getattr(record, "text", "") or "").strip()
        if not text or _is_jd_metadata_record(record) or _is_user_request_record(
            record,
            raw_text=raw_text,
            user_requirements=user_requirements,
        ):
            continue
        normalized = _normalize_requirement_phrase(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        category = getattr(getattr(record, "category", None), "value", None) or str(
            getattr(record, "category", "") or ""
        )
        blocks.append(
            {
                "requirement_id": record_id,
                "text": text,
                "source": "jd",
                "kind": "requirement",
                "category": category,
                "required": bool(getattr(record, "required", True)),
            }
        )
    return blocks


def _flatten_candidate_text(item: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("evidence_label", "project_name", "project_description", "proposed_text"):
        value = item.get(key)
        if value:
            values.append(str(value))
    # Project matching stores a JD/fact pair.  Only the fact side is safe to
    # reuse; copying the JD side can echo an unsupported user instruction.
    for key in ("jd_fact_pairs", "jd_fact_blocks"):
        for value in item.get(key) or ():
            if isinstance(value, dict):
                fact = value.get("事实")
                if isinstance(fact, list):
                    values.extend(str(entry) for entry in fact if str(entry).strip())
                elif fact and not _UNSUPPORTED_MARKERS.search(str(fact)):
                    values.append(str(fact))
    return " ".join(values)


def remove_rejected_request_echo(text: str, rejected_requests: Iterable[str]) -> str:
    """Remove verbatim unsupported-request echoes and known placeholders."""

    cleaned = str(text or "")
    for request in rejected_requests:
        value = str(request or "").strip()
        if value:
            cleaned = cleaned.replace(value, "")
    cleaned = re.sub(r"测试\s*[，,]?\s*请删除", "", cleaned)
    cleaned = re.sub(r"[；;。.!！]\s*(?=[；;。.!！])", "", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ；;。.!！")


def _claim_fragment_signature(value: str) -> str:
    """Normalize a structured field value for duplicate comparison only."""

    return re.sub(r"[^a-z0-9一-鿿]+", "", str(value or "").casefold())


def _claim_fragments_near_duplicate(left: str, right: str) -> bool:
    left_signature = _claim_fragment_signature(left)
    right_signature = _claim_fragment_signature(right)
    if not left_signature or not right_signature:
        return False
    if left_signature in right_signature or right_signature in left_signature:
        return True
    if SequenceMatcher(None, left_signature, right_signature).ratio() >= 0.82:
        return True
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    overlap = len(left_tokens & right_tokens)
    return bool(overlap and overlap / max(1, min(len(left_tokens), len(right_tokens))) >= 0.72)


def dedupe_resume_claim_fields(text: str, section: str) -> str:
    """Collapse repeated structured internship labels without inventing facts.

    The model may echo one source fact twice, sometimes with a lightly
    paraphrased second value.  Only text already present in the claim is used:
    near-duplicates keep the longer fragment, while distinct fragments are
    joined under the first occurrence of the same field label.
    """

    if str(section or "").casefold() not in {"internship", "internships"}:
        return str(text or "")
    value = str(text or "").replace("\r", " ").replace("\n", " ")
    labels = tuple(label for _key, aliases in _INTERNSHIP_LABEL_GROUPS for label in aliases)
    escaped = "|".join(re.escape(label) for label in labels)
    matches = list(re.finditer(rf"(?<![\w一-鿿])(?P<label>{escaped})\s*[:：]", value))
    if not matches:
        return str(text or "")

    alias_to_key = {
        alias: key
        for key, aliases in _INTERNSHIP_LABEL_GROUPS
        for alias in aliases
    }
    fields: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        field_value = value[match.end() : end].strip(" ;；，,\t\n")
        if field_value:
            label = match.group("label")
            fields.append((alias_to_key[label], label, field_value))
    if not fields:
        return str(text or "")

    grouped: dict[str, tuple[str, list[str]]] = {}
    order: list[str] = []
    for key, label, field_value in fields:
        if key not in grouped:
            grouped[key] = (label, [field_value])
            order.append(key)
            continue
        first_label, values = grouped[key]
        merged = False
        for position, existing in enumerate(values):
            if _claim_fragments_near_duplicate(existing, field_value):
                if len(field_value) > len(existing):
                    values[position] = field_value
                merged = True
                break
        if not merged:
            values.append(field_value)
        grouped[key] = (first_label, values)

    prefix = value[: matches[0].start()].strip(" ;；，,\t\n")
    output = [prefix] if prefix else []
    for key in order:
        label, values = grouped[key]
        output.append(f"{label}：{'；'.join(values)}")
    return "；".join(item for item in output if item).strip(" ；;。.!！")


def _project_target(text: str) -> str:
    value = str(text or "").casefold()
    if any(token in value for token in ("smart agent", "agent runtime", "root supervisor", "domain/stage")):
        return "Smart Agent"
    if any(token in value for token in ("服装", "bge-m3", "crossencoder", "weighted rrf", "rag")):
        return "服装领域 RAG"
    return ""


def _facets(text: str) -> tuple[str, ...]:
    value = str(text or "").casefold()
    return tuple(label for label, markers in _FACETS if any(marker.casefold() in value for marker in markers))


def _split_numbered(text: str) -> list[tuple[int, str]]:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    matches = list(_NUMBERED_MARKER.finditer(value))
    if not matches:
        return [(1, value)] if value else []
    result: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        clause = value[match.end() : end].strip(" \t\n；;。.!！")
        if clause:
            result.append((int(match.group(1)), clause))
    return result


def parse_resume_user_requirements(
    request: str,
    candidates: Iterable[dict[str, Any]] = (),
    *,
    allow_unsupported_project_requirements: bool = False,
) -> tuple[ResumeUserRequirement, ...]:
    """Parse and classify numbered user requests against candidate evidence.

    A request containing production/business claims is rejected even if it has
    a few overlapping words with the evidence. The benchmark-only
    ``allow_unsupported_project_requirements`` escape hatch applies solely to
    project-targeted requests; non-project facts remain protected.
    """

    candidate_rows = [item for item in candidates if isinstance(item, dict)]
    rows: list[ResumeUserRequirement] = []
    for ordinal, text in _split_numbered(request):
        target = _project_target(text)
        facets = _facets(text)
        request_tokens = _tokens(text)
        matched: list[tuple[float, dict[str, Any]]] = []
        for item in candidate_rows:
            candidate_text = _flatten_candidate_text(item)
            candidate_tokens = _tokens(candidate_text)
            if target:
                item_target = _project_target(candidate_text)
                if item_target and item_target != target:
                    continue
            overlap = len(request_tokens & candidate_tokens)
            facet_overlap = sum(
                1
                for facet in facets
                if any(marker.casefold() in candidate_text.casefold() for label, markers in _FACETS if label == facet for marker in markers)
            )
            if overlap or facet_overlap:
                matched.append((float(overlap + facet_overlap * 3), item))
        matched.sort(key=lambda row: row[0], reverse=True)
        matched_refs = tuple(
            str(item.get("item_id") or item.get("evidence_id") or "").strip()
            for _, item in matched[:4]
            if str(item.get("item_id") or item.get("evidence_id") or "").strip()
        )
        unsupported_match = bool(_UNSUPPORTED_MARKERS.search(text))
        if unsupported_match and allow_unsupported_project_requirements and target:
            status, decision = "project_ablation", "partially_accepted"
            reason = "项目事实安全消融：该项目要求旁路进入项目生成，不参与非项目事实约束。"
            evidence_refs = matched_refs
        elif unsupported_match:
            status, decision = "unsupported", "rejected"
            reason = "该要求包含候选人材料未确认的线上、业务指标或生产事实，已拒绝写入简历。"
            evidence_refs = ()
        elif matched_refs:
            status, decision = "supported", "accepted"
            reason = "已在确认的个人材料中找到可引用证据，并转为必须覆盖面。"
            evidence_refs = matched_refs
        else:
            status, decision = "unsupported", "rejected"
            reason = "未在当前确认的个人材料中找到足够证据，已拒绝写入简历。"
            evidence_refs = ()
        rows.append(
            ResumeUserRequirement(
                request_id=f"user-{ordinal}",
                text=text,
                support_status=status,
                decision=decision,
                must_cover_facets=facets,
                target_project=target,
                evidence_refs=evidence_refs,
                reason=reason,
            )
        )
    return tuple(rows)


def build_supported_customization_instruction(
    requirements: Iterable[ResumeUserRequirement],
) -> str:
    """Create a generator instruction containing supported requests only."""

    lines: list[str] = []
    ablation_lines: list[str] = []
    for item in requirements:
        if item.decision not in {"accepted", "partially_accepted"}:
            continue
        facets = "、".join(item.must_cover_facets)
        suffix = f"；必须覆盖：{facets}" if facets else ""
        target = "；目标项目：" + item.target_project if item.target_project else ""
        row = f"{item.request_id}：{item.text}{target}{suffix}"
        (ablation_lines if item.support_status == "project_ablation" else lines).append(row)
    if not lines and not ablation_lines:
        return ""
    parts: list[str] = []
    if lines:
        parts.append(
            "仅执行以下已由个人材料支持的定制要求，并将其作为简历内容的硬性覆盖约束；"
            "不得执行未列出的用户要求，也不得新增材料之外的事实：\n" + "\n".join(lines)
        )
    if ablation_lines:
        parts.append(
            "以下项目型要求仅在项目事实安全消融组旁路传入项目生成；不得用于修改教育、实习、科研、获奖或技能事实：\n"
            + "\n".join(ablation_lines)
        )
    return "\n\n".join(parts)


def build_jd_coverage_matrix(
    jd_blocks: Iterable[dict[str, Any]],
    claims: Iterable[dict[str, Any]],
    requirements: Iterable[ResumeUserRequirement] = (),
) -> dict[str, Any]:
    """Return a traceable requirement-to-claim coverage matrix."""

    claim_rows = [item for item in claims if isinstance(item, dict)]
    normalized_claims = {
        str(item.get("item_id") or ""): str(item.get("text") or "")
        for item in claim_rows
        if str(item.get("item_id") or "")
    }
    rows: list[dict[str, Any]] = []
    for block in jd_blocks:
        if not isinstance(block, dict):
            continue
        source = str(block.get("source") or "jd").casefold()
        if source != "jd" or str(block.get("requirement_id") or "") == "__role_title__":
            continue
        requirement_id = str(block.get("requirement_id") or "")
        text = str(block.get("text") or "").strip()
        target = _tokens(text)
        scored = []
        for claim_id, claim_text in normalized_claims.items():
            claim_tokens = _tokens(claim_text)
            overlap = len(target & claim_tokens) / len(target) if target else 0.0
            if overlap > 0:
                scored.append((overlap, claim_id))
        scored.sort(reverse=True)
        best = scored[0][0] if scored else 0.0
        rows.append(
            {
                "requirement_id": requirement_id,
                "requirement": text,
                "claim_ids": [claim_id for score, claim_id in scored if score >= max(0.12, best * 0.7)][:4],
                "coverage": round(best, 4),
                "covered": bool(best >= 0.2),
            }
        )
    user_rows = []
    for item in requirements:
        if not isinstance(item, ResumeUserRequirement):
            continue
        if item.decision not in {"accepted", "partially_accepted"}:
            user_rows.append({**item.to_dict(), "covered": False, "claim_ids": []})
            continue
        target_tokens = _tokens(" ".join((*item.must_cover_facets, item.text)))
        scored = []
        for claim_id, claim_text in normalized_claims.items():
            overlap = len(target_tokens & _tokens(claim_text)) / len(target_tokens) if target_tokens else 0.0
            if overlap > 0:
                scored.append((overlap, claim_id))
        scored.sort(reverse=True)
        best = scored[0][0] if scored else 0.0
        user_rows.append(
            {
                **item.to_dict(),
                "covered": bool(best >= 0.2),
                "coverage": round(best, 4),
                "claim_ids": [claim_id for _, claim_id in scored[:4]],
            }
        )
    return {"jd_requirements": rows, "user_requirements": user_rows}


def _evidence_fragment(text: str, target_tokens: set[str]) -> str:
    parts = [part.strip() for part in re.split(r"[\n。；;]", str(text or "")) if part.strip()]
    scored = sorted(
        ((len(target_tokens & _tokens(part)), part) for part in parts),
        key=lambda row: (row[0], -len(row[1])),
        reverse=True,
    )
    for score, part in scored:
        if score > 0:
            return " ".join(part.split())[:180]
    return ""


class JDRequirementCritic:
    """Independent, evidence-only critic with deterministic safe revision."""

    def __init__(self, client: Any | None = None) -> None:
        self.client = client

    def review(
        self,
        *,
        jd_blocks: Iterable[dict[str, Any]],
        claims: Iterable[dict[str, Any]],
        evidence_blocks: Iterable[dict[str, Any]],
        user_requirements: Iterable[ResumeUserRequirement],
        claim_validator: Callable[[str, str], None] | None = None,
    ) -> dict[str, Any]:
        jd_list = [
            item
            for item in jd_blocks
            if isinstance(item, dict)
            and str(item.get("source") or "jd").casefold() == "jd"
            and str(item.get("kind") or "requirement").casefold() != "role_title"
        ]
        claim_list = [item for item in claims if isinstance(item, dict)]
        evidence_list = [item for item in evidence_blocks if isinstance(item, dict)]
        user_list = [item for item in user_requirements if isinstance(item, ResumeUserRequirement)]
        if self.client is not None:
            try:
                return self._llm_review(
                    jd_list,
                    claim_list,
                    evidence_list,
                    user_list,
                    claim_validator,
                )
            except Exception:
                # A critic outage must not discard an otherwise valid resume;
                # deterministic review remains auditable and evidence-bound.
                pass
        return self._deterministic_review(
            jd_list,
            claim_list,
            evidence_list,
            user_list,
            claim_validator,
        )

    def _deterministic_review(
        self,
        jd_blocks: list[dict[str, Any]],
        claims: list[dict[str, Any]],
        evidence_blocks: list[dict[str, Any]],
        user_requirements: list[ResumeUserRequirement],
        claim_validator: Callable[[str, str], None] | None,
    ) -> dict[str, Any]:
        claim_text = {str(item.get("item_id") or ""): str(item.get("text") or "") for item in claims}
        evidence_by_id = {str(item.get("item_id") or ""): _flatten_candidate_text(item) for item in evidence_blocks}
        reviews: list[dict[str, Any]] = []
        revisions: dict[str, str] = {}

        def propose(target_text: str, preferred_ids: list[str]) -> None:
            target_tokens = _tokens(target_text)
            for claim_id in preferred_ids or list(claim_text):
                fragment = _evidence_fragment(evidence_by_id.get(claim_id, ""), target_tokens)
                base = revisions.get(claim_id, claim_text.get(claim_id, ""))
                if not fragment or fragment.casefold() in base.casefold():
                    continue
                revised = f"{base}；{fragment}".strip("；")
                if claim_validator is not None:
                    try:
                        claim_validator(claim_id, revised)
                    except Exception:
                        continue
                revisions[claim_id] = revised
                break

        for block in jd_blocks:
            requirement_id = str(block.get("requirement_id") or "")
            if (
                not requirement_id
                or requirement_id == "__role_title__"
                or str(block.get("source") or "jd").casefold() != "jd"
                or str(block.get("kind") or "requirement").casefold() == "role_title"
            ):
                continue
            text = str(block.get("text") or "")
            target = _tokens(text)
            scored = sorted(
                ((len(target & _tokens(value)), claim_id) for claim_id, value in claim_text.items()),
                reverse=True,
            )
            best = scored[0][0] if scored else 0
            covered = bool(target and best / len(target) >= 0.2)
            preferred = [claim_id for _, claim_id in scored[:3]]
            if not covered:
                propose(text, preferred)
            reviews.append(
                {
                    "requirement_id": requirement_id,
                    "covered": covered,
                    "gap": "" if covered else "当前 claims 与 JD 关键词覆盖不足",
                    "claim_ids": preferred,
                    "revision": revisions.get(preferred[0], "") if preferred else "",
                }
            )
        for item in user_requirements:
            if item.decision not in {"accepted", "partially_accepted"}:
                continue
            target = " ".join((*item.must_cover_facets, item.text))
            target_tokens = _tokens(target)
            scored = sorted(
                ((len(target_tokens & _tokens(value)), claim_id) for claim_id, value in claim_text.items()),
                reverse=True,
            )
            best = scored[0][0] if scored else 0
            covered = bool(target_tokens and best / len(target_tokens) >= 0.2)
            preferred = [claim_id for _, claim_id in scored[:3]]
            if not covered:
                preferred = [ref for ref in item.evidence_refs if ref in claim_text] or preferred
                propose(target, preferred)
            reviews.append(
                {
                    "requirement_id": item.request_id,
                    "covered": covered,
                    "gap": "" if covered else "支持的用户要求尚未充分体现在 claims 中",
                    "claim_ids": preferred,
                    "revision": revisions.get(preferred[0], "") if preferred else "",
                }
            )
        revised_claims = [
            {"item_id": claim_id, "text": revisions.get(claim_id, text)}
            for claim_id, text in claim_text.items()
        ]
        return {
            "critic": "deterministic_jd_critic_v1",
            "overall_pass": all(item["covered"] or item["revision"] for item in reviews),
            "requirement_reviews": reviews,
            "revisions": [{"claim_id": key, "text": value} for key, value in revisions.items()],
            "claims": revised_claims,
            "fallback": True,
        }

    def _llm_review(
        self,
        jd_blocks: list[dict[str, Any]],
        claims: list[dict[str, Any]],
        evidence_blocks: list[dict[str, Any]],
        user_requirements: list[ResumeUserRequirement],
        claim_validator: Callable[[str, str], None] | None,
    ) -> dict[str, Any]:
        response = self.client.complete_json(
            system_prompt=(
                "You are an independent JD critic for a resume. Check every JD requirement and supported user "
                "request against only current claims and cited evidence. Return JSON with overall_pass, "
                "requirement_reviews and revisions. A revision may edit an existing claim only; never add a claim, "
                "number, tool, employer, duty, or outcome. If evidence is insufficient, leave revision empty."
            ),
            user_payload={
                "jd_blocks": jd_blocks,
                "claims": claims,
                "evidence_blocks": evidence_blocks,
                "user_requirements": [item.to_dict() for item in user_requirements],
            },
        ).value
        if not isinstance(response, dict):
            raise ValueError("JD critic response must be an object")
        known = {str(item.get("item_id") or "") for item in claims}
        claim_by_id = {
            str(item.get("item_id") or ""): str(item.get("text") or "")
            for item in claims
        }
        evidence_tokens = {
            claim_id: _tokens(_flatten_candidate_text(item))
            for claim_id, item in (
                (str(item.get("item_id") or ""), item)
                for item in evidence_blocks
                if str(item.get("item_id") or "")
            )
        }
        revisions: dict[str, str] = {}
        for item in response.get("revisions") or ():
            if not isinstance(item, dict):
                continue
            claim_id = str(item.get("claim_id") or "").strip()
            text = " ".join(str(item.get("text") or "").split())
            if claim_id not in known or not text:
                continue
            novel = _tokens(text) - _tokens(claim_by_id.get(claim_id, ""))
            unsupported_novel = novel - evidence_tokens.get(claim_id, set())
            if unsupported_novel and len(unsupported_novel) > max(2, len(novel) // 3):
                continue
            if claim_validator is not None:
                try:
                    claim_validator(claim_id, text)
                except Exception:
                    continue
            revisions[claim_id] = text
        normalized_reviews = [item for item in response.get("requirement_reviews") or () if isinstance(item, dict)]
        return {
            "critic": "llm_jd_critic",
            "overall_pass": bool(response.get("overall_pass", not normalized_reviews)),
            "requirement_reviews": normalized_reviews,
            "revisions": [{"claim_id": key, "text": value} for key, value in revisions.items()],
            "claims": [
                {"item_id": claim_id, "text": revisions.get(claim_id, text)}
                for claim_id, text in ((str(item.get("item_id") or ""), str(item.get("text") or "")) for item in claims)
            ],
            "fallback": False,
        }
