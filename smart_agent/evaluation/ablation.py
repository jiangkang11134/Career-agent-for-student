"""Deterministic ablation harness for the competition report.

The default protocol is intentionally offline. It measures algorithmic and
orchestration choices on a versioned fixture; it does not claim to measure a
remote model or a production API. Remote model adapters can implement the
same record format later without changing aggregation or provenance rules.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ABLATION_FAMILIES = ("planning", "verification", "interview", "context", "retrieval")
PROTOCOL_VERSION = "ablation-protocol-v1"


@dataclass(frozen=True, slots=True)
class AblationConfig:
    dataset_version: str
    seed: int = 17
    prompt_version: str = "offline-baseline-v1"
    model_id: str = "deterministic-offline"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AblationRecord:
    family: str
    variant: str
    case_id: str
    metrics: dict[str, float]
    details: dict[str, Any]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AblationReport:
    protocol_version: str
    config: AblationConfig
    environment: dict[str, str]
    records: tuple[AblationRecord, ...]
    aggregate: dict[str, dict[str, dict[str, float]]]
    cross_cutting: dict[str, float]
    failures: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "config": self.config.to_dict(),
            "environment": self.environment,
            "records": [record.to_dict() for record in self.records],
            "aggregate": self.aggregate,
            "cross_cutting": self.cross_cutting,
            "failures": list(self.failures),
        }

    def write(self, output_dir: str | os.PathLike[str]) -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        paths = {
            "report": root / "report.json",
            "raw": root / "raw-results.json",
            "aggregate": root / "aggregate.json",
            "environment": root / "environment.json",
            "failures": root / "failures.json",
        }
        _write_json(paths["report"], payload)
        _write_json(paths["raw"], {"records": payload["records"]})
        _write_json(paths["aggregate"], {"families": payload["aggregate"], "cross_cutting": payload["cross_cutting"]})
        _write_json(paths["environment"], {"config": payload["config"], "environment": payload["environment"]})
        _write_json(paths["failures"], {"failures": payload["failures"]})
        return paths


def load_dataset(path: str | os.PathLike[str]) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("version"), str):
        raise ValueError("Ablation dataset must be an object with a string version.")
    for family in ABLATION_FAMILIES:
        if not isinstance(value.get(family), list) or not value[family]:
            raise ValueError(f"Ablation dataset family {family!r} must be a non-empty list.")
    for section in ("safety", "runtime"):
        if not isinstance(value.get(section), list) or not value[section]:
            raise ValueError(f"Ablation dataset section {section!r} must be a non-empty list.")
    return value


def run_ablation(
    dataset_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
    *,
    seed: int = 17,
    prompt_version: str = "offline-baseline-v1",
    model_id: str = "deterministic-offline",
) -> AblationReport:
    dataset = load_dataset(dataset_path)
    config = AblationConfig(
        dataset_version=str(dataset["version"]),
        seed=int(seed),
        prompt_version=prompt_version,
        model_id=model_id,
    )
    rng = random.Random(config.seed)
    records: list[AblationRecord] = []
    failures: list[dict[str, str]] = []
    for family in ABLATION_FAMILIES:
        for case in dataset[family]:
            case_id = str(case.get("case_id") or "unknown")
            for variant in _variants(family):
                started = time.perf_counter()
                try:
                    metrics, details = _run_variant(family, variant, case, rng)
                    records.append(
                        AblationRecord(
                            family=family,
                            variant=variant,
                            case_id=case_id,
                            metrics={key: round(float(value), 6) for key, value in metrics.items()},
                            details=details,
                            elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append({"family": family, "variant": variant, "case_id": case_id, "error": str(exc)})
    aggregate = _aggregate(records)
    cross_cutting = _cross_cutting_metrics(dataset)
    report = AblationReport(
        protocol_version=PROTOCOL_VERSION,
        config=config,
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
            "dataset_sha256": _sha256(Path(dataset_path)),
        },
        records=tuple(records),
        aggregate=aggregate,
        cross_cutting=cross_cutting,
        failures=tuple(failures),
    )
    if output_dir is not None:
        report.write(output_dir)
    return report


def _variants(family: str) -> tuple[str, ...]:
    return {
        "planning": ("llm", "similarity", "cp_sat"),
        "verification": ("self_check", "independent", "independent_repair"),
        "interview": ("fixed_question", "jd_only", "jd_resume", "jd_resume_rag", "adaptive"),
        "context": ("none", "truncate", "four_level"),
        "retrieval": ("bm25", "vector", "hybrid", "hybrid_reranker"),
    }[family]


def _run_variant(family: str, variant: str, case: dict[str, Any], rng: random.Random) -> tuple[dict[str, float], dict[str, Any]]:
    if family == "planning":
        return _planning(variant, case)
    if family == "verification":
        return _verification(variant, case)
    if family == "interview":
        return _interview(variant, case, rng)
    if family == "context":
        return _context(variant, case)
    return _retrieval(variant, case)


def _planning(variant: str, case: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    items = list(case["items"])
    requirements = set(case["requirements"])
    budget = int(case["budget"])
    if variant == "llm":
        ordered = items
    elif variant == "similarity":
        ordered = sorted(items, key=lambda item: (-float(item.get("similarity", 0)), str(item["id"])))
    else:
        ordered = sorted(
            items,
            key=lambda item: (
                -(
                    float(item.get("similarity", 0))
                    + float(item.get("evidence", 0))
                    + float(item.get("title_value", 0))
                    + float(item.get("result", 0))
                ),
                str(item["id"]),
            ),
        )
    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    used_length = 0
    for item in ordered:
        group = str(item.get("group", item["id"]))
        length = int(item.get("length", 1))
        if variant == "cp_sat" and group in used_groups:
            continue
        if used_length + length > budget:
            continue
        selected.append(item)
        used_groups.add(group)
        used_length += length
    covered = {req for item in selected for req in item.get("requirements", [])}
    supported = sum(1 for item in selected if item.get("evidence", 0) >= 0.5)
    duplicate_count = len(selected) - len(used_groups)
    metrics = {
        "jd_coverage_rate": len(covered & requirements) / max(1, len(requirements)),
        "evidence_support_rate": supported / max(1, len(selected)),
        "length_violation_rate": float(max(0, used_length - budget) > 0),
        "duplicate_rate": duplicate_count / max(1, len(selected)),
        "utility": sum(float(item.get("utility", item.get("similarity", 0))) for item in selected),
    }
    return metrics, {"selected_ids": [item["id"] for item in selected], "length": used_length}


def _verification(variant: str, case: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    claims = list(case["claims"])
    predictions: list[bool] = []
    for claim in claims:
        overlap = _token_set(claim["text"]) & _token_set(claim.get("evidence", ""))
        lexical = len(overlap) / max(1, len(_token_set(claim["text"])))
        prediction = lexical >= (0.25 if variant == "self_check" else 0.4)
        if variant == "independent_repair" and lexical < 0.4:
            prediction = False
        predictions.append(prediction)
    truth = [bool(claim["supported"]) for claim in claims]
    tp = sum(p and t for p, t in zip(predictions, truth, strict=True))
    fp = sum(p and not t for p, t in zip(predictions, truth, strict=True))
    fn = sum((not p) and t for p, t in zip(predictions, truth, strict=True))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-9, precision + recall),
        "unsupported_claim_rate": fp / max(1, len(claims)),
        "repair_rate": (sum(not p for p in predictions) / max(1, len(claims))) if variant == "independent_repair" else 0.0,
    }
    return metrics, {"predicted_supported": predictions, "truth_supported": truth}


def _interview(variant: str, case: dict[str, Any], rng: random.Random) -> tuple[dict[str, float], dict[str, Any]]:
    topics = list(case["topics"])
    answer_quality = {str(key): float(value) for key, value in case["answer_quality"].items()}
    if variant == "fixed_question":
        asked = topics[: int(case.get("turns", len(topics)))]
    elif variant == "jd_only":
        asked = [topic for topic in topics if topic in case.get("jd_topics", [])]
    elif variant == "jd_resume":
        asked = [topic for topic in topics if topic in set(case.get("jd_topics", [])) | set(case.get("resume_topics", []))]
    elif variant == "jd_resume_rag":
        asked = [topic for topic in topics if topic in set(case.get("jd_topics", [])) | set(case.get("resume_topics", [])) | set(case.get("rag_topics", []))]
    else:
        remaining = set(topics)
        asked = []
        while remaining and len(asked) < int(case.get("turns", len(topics))):
            topic = min(remaining, key=lambda value: (answer_quality.get(value, 0.5), value))
            asked.append(topic)
            remaining.remove(topic)
            if answer_quality.get(topic, 0.5) < 0.55 and len(asked) < int(case.get("turns", len(topics))):
                asked.append(topic)
    weak_topics = {topic for topic, quality in answer_quality.items() if quality < 0.6}
    detected = len(weak_topics & set(asked)) / max(1, len(weak_topics))
    jd_topics = set(case.get("jd_topics", []))
    relevant = len(jd_topics & set(asked)) / max(1, len(jd_topics))
    followup_count = max(0, len(asked) - len(set(asked)))
    # Stability means that the policy stays within the information scope
    # available to the interviewer.  Keep this deterministic and auditable.
    policy_targets = (
        set(case.get("jd_topics", []))
        | set(case.get("resume_topics", []))
        | set(case.get("rag_topics", []))
    )
    off_target = [topic for topic in asked if topic not in policy_targets]
    policy_stability = 1.0 - (len(off_target) / max(1, len(asked)))
    metrics = {
        "jd_topic_coverage": relevant,
        "weak_point_detection_rate": detected,
        "followup_association_rate": followup_count / max(1, len(asked)),
        "duplicate_question_rate": followup_count / max(1, len(asked)),
        "knowledge_source_traceability": 1.0 if variant in {"jd_resume_rag", "adaptive"} else 0.0,
        "policy_stability": policy_stability,
    }
    return metrics, {
        "asked_topics": asked,
        "weak_topics": sorted(weak_topics),
        "off_target_topics": off_target,
    }


def _context(variant: str, case: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    messages = list(case["messages"])
    budget = int(case["budget"])
    token = lambda message: max(1, len(str(message.get("content", ""))) // 4)
    before = sum(token(message) for message in messages)
    if variant == "none":
        retained = messages
    elif variant == "truncate":
        retained = []
        total = 0
        for message in reversed(messages):
            size = token(message)
            if total + size > budget:
                break
            retained.append(message)
            total += size
        retained.reverse()
    else:
        head = messages[: max(1, len(messages) // 3)]
        tail = messages[-2:]
        summary = {"role": "system", "content": "summary:" + " | ".join(str(item.get("content", ""))[:48] for item in head)}
        retained = [summary, *tail]
        while sum(token(message) for message in retained) > budget and len(retained) > 2:
            retained.pop(1)
    after = sum(token(message) for message in retained)
    forced = set(case.get("forced_terms", []))
    text = " ".join(str(message.get("content", "")) for message in retained)
    return {
        "token_ratio": after / max(1, before),
        "goal_retained": float(str(case.get("goal", "")) in text),
        "forced_content_retained": sum(term in text for term in forced) / max(1, len(forced)),
        "cache_prefix_ratio": float(variant in {"none", "truncate"}),
    }, {"retained_roles": [message.get("role") for message in retained], "tokens_before": before, "tokens_after": after}


def _retrieval(variant: str, case: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    query = _token_set(case["query"])
    documents = list(case["documents"])
    relevant = set(case["relevant_ids"])
    scores: list[tuple[str, float]] = []
    query_vector = _vector(query)
    for document in documents:
        tokens = _token_set(document["text"])
        lexical = len(query & tokens) / max(1, len(query))
        semantic = _cosine(query_vector, _vector(tokens))
        if variant == "bm25":
            score = lexical
        elif variant == "vector":
            score = semantic
        else:
            score = 0.5 * lexical + 0.5 * semantic
            if variant == "hybrid_reranker":
                score += 0.1 * float(document.get("authority", 0.0)) + 0.1 * float(query <= tokens)
        scores.append((str(document["id"]), score))
    ranked = [identity for identity, _ in sorted(scores, key=lambda row: (-row[1], row[0]))]
    top_k = min(int(case.get("top_k", 3)), len(ranked))
    top = ranked[:top_k]
    hits = [identity for identity in top if identity in relevant]
    first_rank = next((index + 1 for index, identity in enumerate(ranked) if identity in relevant), None)
    return {
        "recall_at_k": len(hits) / max(1, len(relevant)),
        "mrr": 1 / first_rank if first_rank else 0.0,
        "false_recall_rate": len([identity for identity in top if identity not in relevant]) / max(1, top_k),
        "traceability_rate": 1.0,
    }, {"ranked_ids": ranked, "top_k_ids": top}


def _token_set(value: Any) -> set[str]:
    return {token.casefold() for token in str(value or "").replace("/", " ").replace(",", " ").split() if len(token) > 1}


def _vector(tokens: set[str]) -> list[float]:
    values = [0.0] * 32
    for token in sorted(tokens):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index in range(4):
            slot = digest[index] % len(values)
            values[slot] += 1.0 if digest[index + 4] % 2 else -1.0
    return values


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(value * value for value in left) * sum(value * value for value in right))
    return max(0.0, numerator / denominator) if denominator else 0.0


def _aggregate(records: list[AblationRecord]) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[str, dict[str, list[AblationRecord]]] = {}
    for record in records:
        grouped.setdefault(record.family, {}).setdefault(record.variant, []).append(record)
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for family, variants in grouped.items():
        aggregate[family] = {}
        for variant, items in variants.items():
            keys = sorted({key for item in items for key in item.metrics})
            aggregate[family][variant] = {
                key: round(statistics.mean(item.metrics[key] for item in items if key in item.metrics), 6)
                for key in keys
            }
    return aggregate


def _cross_cutting_metrics(dataset: dict[str, Any]) -> dict[str, float]:
    safety = list(dataset.get("safety", []))
    runtime = list(dataset.get("runtime", []))
    if not safety or not runtime:
        raise ValueError("Ablation dataset must include non-empty safety and runtime sections.")
    private_total = sum(len(case.get("private_terms", [])) for case in safety)
    private_leaks = sum(int(case.get("leaked_private_terms", 0)) for case in safety)
    injection_total = len(safety)
    injection_successes = sum(int(bool(case.get("prompt_injection_succeeded", False))) for case in safety)
    form_total = sum(int(case.get("form_field_count", 0)) for case in safety)
    form_hallucinations = sum(int(case.get("unsupported_form_fields_filled", 0)) for case in safety)
    runtime_count = max(1, len(runtime))
    return {
        "personal_information_overreach_rate": private_leaks / max(1, private_total),
        "prompt_injection_success_rate": injection_successes / max(1, injection_total),
        "unsupported_application_field_rate": form_hallucinations / max(1, form_total),
        "latency_ms": statistics.mean(float(case.get("latency_ms", 0)) for case in runtime),
        "tokens": statistics.mean(float(case.get("tokens", 0)) for case in runtime),
        "api_cost": statistics.mean(float(case.get("api_cost", 0)) for case in runtime),
        "failure_rate": sum(int(bool(case.get("failed", False))) for case in runtime) / runtime_count,
        "recovery_success_rate": sum(int(bool(case.get("recovered", False))) for case in runtime) / runtime_count,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
