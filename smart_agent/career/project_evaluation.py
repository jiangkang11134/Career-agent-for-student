"""Read-only automatic evaluation for Project Analyst runs.

This module deliberately measures observable constraints and evidence quality.
It does not claim semantic Precision/Recall/F1 without a human gold set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from smart_agent.career.project_library import (
    ProjectAnalysisRecord,
    ProjectSource,
    extract_project_document_text,
    is_ignored_project_path,
    scan_project_artifacts,
)
from smart_agent.career.project_service import ProjectAnalysisRun


def _norm_path(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./").casefold()


def _norm_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _trace_path(item: dict[str, Any]) -> str:
    """Support both persisted trace formats (top-level and nested input.path)."""
    return str(item.get("path") or dict(item.get("input") or {}).get("path") or "").strip()


def _readable_analysis_records(root: Path) -> tuple[ProjectAnalysisRecord, ...]:
    if not root.exists():
        return ()
    records: list[ProjectAnalysisRecord] = []
    for path in root.glob("**/*.json"):
        try:
            records.append(ProjectAnalysisRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return tuple(records)


def _readable_sources(index_path: Path) -> tuple[ProjectSource, ...]:
    """Read the registry directly so evaluation never creates or updates stores."""
    if not index_path.exists():
        return ()
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return tuple(ProjectSource.from_dict(item) for item in payload.get("sources", ()))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ()


def _readable_runs(root: Path) -> tuple[ProjectAnalysisRun, ...]:
    if not root.exists():
        return ()
    runs: list[ProjectAnalysisRun] = []
    for path in root.glob("*.json"):
        try:
            runs.append(ProjectAnalysisRun.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return tuple(sorted(runs, key=lambda item: (item.created_at, item.run_id)))


def _analysis_for_run(
    run: ProjectAnalysisRun,
    records: Iterable[ProjectAnalysisRecord],
) -> ProjectAnalysisRecord | None:
    candidates = [
        item
        for item in records
        if item.project_id == run.project_id
        and item.project_version == run.project_version
        and item.intent == run.intent
    ]
    exact = [item for item in candidates if item.analysis_run_id == run.run_id]
    return max(exact or candidates, key=lambda item: item.created_at, default=None)


def _source_ref_metrics(source: ProjectSource, analysis: ProjectAnalysisRecord | None) -> dict[str, int]:
    if analysis is None:
        return {"total": 0, "valid_path": 0, "valid_line": 0, "supported_excerpt": 0}
    root = Path(source.source_location)
    artifacts = {item.path: item for item in scan_project_artifacts(root)}
    total = valid_path = valid_line = supported = 0
    text_cache: dict[str, str] = {}
    for raw_ref in analysis.understanding.source_refs:
        ref = dict(raw_ref)
        total += 1
        relative = _norm_path(ref.get("path"))
        artifact_path = next((path for path in artifacts if _norm_path(path) == relative), None)
        if artifact_path is None:
            continue
        valid_path += 1
        target = root / artifact_path
        try:
            text = text_cache.get(artifact_path)
            if text is None:
                if target.suffix.casefold() in {
                    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs",
                    ".c", ".cc", ".cpp", ".h", ".hpp", ".cs", ".php", ".rb", ".swift", ".scala",
                    ".sql", ".sh", ".ps1", ".bat", ".html", ".css", ".scss", ".vue", ".svelte",
                    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg",
                }:
                    text = target.read_text(encoding="utf-8", errors="replace")
                else:
                    text, _parser = extract_project_document_text(target)
                text_cache[artifact_path] = text
            lines = text.splitlines() or [text]
            line = int(ref.get("line") or 0)
            if 1 <= line <= len(lines):
                valid_line += 1
            excerpt = _norm_text(ref.get("excerpt"))
            haystack = _norm_text(text)
            if excerpt and excerpt in haystack:
                supported += 1
        except (OSError, TypeError, ValueError, UnicodeError):
            continue
    return {
        "total": total,
        "valid_path": valid_path,
        "valid_line": valid_line,
        "supported_excerpt": supported,
    }


def _run_metrics(
    run: ProjectAnalysisRun,
    source: ProjectSource | None,
    analysis: ProjectAnalysisRecord | None,
) -> dict[str, Any]:
    trace = [dict(item) for item in run.agent_trace if isinstance(item, dict)]
    successful_reads = [
        item
        for item in trace
        if item.get("ok") is True and item.get("tool_name") in {"read_file", "read_project_document"}
    ]
    all_paths = [_norm_path(_trace_path(item)) for item in trace if _trace_path(item)]
    expected_readme = ""
    artifact_paths: set[str] = set()
    if source is not None:
        try:
            artifacts = scan_project_artifacts(source.source_location)
            artifact_paths = {_norm_path(item.path) for item in artifacts}
            expected_readme = next(
                (
                    _norm_path(item.path)
                    for item in artifacts
                    if Path(item.path).name.casefold().startswith("readme")
                ),
                "",
            )
        except (OSError, ValueError):
            pass
    read_paths = {_norm_path(_trace_path(item)) for item in successful_reads}
    core_modules = {
        _norm_path(item)
        for item in (analysis.understanding.core_modules if analysis is not None else ())
    }
    refs = _source_ref_metrics(source, analysis) if source is not None else {
        "total": 0,
        "valid_path": 0,
        "valid_line": 0,
        "supported_excerpt": 0,
    }
    forbidden = sum(is_ignored_project_path(path) for path in all_paths if path)
    first_read = _norm_path(_trace_path(successful_reads[0])) if successful_reads else ""
    return {
        "run_id": run.run_id,
        "project_id": run.project_id,
        "status": run.status,
        "cache_hit": run.cache_hit,
        "readme_first": bool(expected_readme and first_read == expected_readme),
        "core_module_count": len(core_modules),
        "core_module_read_count": len(core_modules & read_paths),
        "core_module_read_rate": _ratio(len(core_modules & read_paths), len(core_modules)),
        "read_file_count": len(successful_reads),
        "trace_available": bool(trace),
        "forbidden_path_count": forbidden,
        "all_read_paths_are_registered": all(path in artifact_paths for path in all_paths) if all_paths else True,
        "citation_count": refs["total"],
        "citation_valid_path_count": refs["valid_path"],
        "citation_valid_line_count": refs["valid_line"],
        "citation_excerpt_support_count": refs["supported_excerpt"],
        "citation_valid_path_rate": _ratio(refs["valid_path"], refs["total"]),
        "citation_valid_line_rate": _ratio(refs["valid_line"], refs["total"]),
        "citation_excerpt_support_rate": _ratio(refs["supported_excerpt"], refs["total"]),
        "complete": run.status in {"completed", "cached"} and bool(analysis),
        "elapsed_ms": round(run.elapsed_ms, 3),
        "tool_call_count": run.tool_call_count,
        "token_count": run.token_count,
    }


def evaluate_project_analysis(
    career_root: str | Path,
    *,
    project_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Aggregate automatic Project Analyst metrics without writing any data."""

    root = Path(career_root).expanduser().resolve()
    project_root = root / "project-library"
    sources = {source.project_id: source for source in _readable_sources(project_root / "sources.json")}
    selected = {str(item).strip() for item in project_ids if str(item).strip()}
    runs = tuple(
        run for run in _readable_runs(project_root / "runs")
        if not selected or run.project_id in selected
    )
    analyses = _readable_analysis_records(project_root / "analyses")
    per_run = [
        _run_metrics(run, sources.get(run.project_id), _analysis_for_run(run, analyses))
        for run in runs
    ]
    completed = sum(bool(item["complete"]) for item in per_run)
    uncached = [item for item in per_run if not item["cache_hit"]]
    traced = [item for item in per_run if item["trace_available"]]
    total_citations = sum(item["citation_count"] for item in per_run)
    valid_paths = sum(item["citation_valid_path_count"] for item in per_run)
    valid_lines = sum(item["citation_valid_line_count"] for item in per_run)
    supported = sum(item["citation_excerpt_support_count"] for item in per_run)
    total_modules = sum(item["core_module_count"] for item in traced)
    read_modules = sum(item["core_module_read_count"] for item in traced)
    return {
        "schema_version": "project-analyst-auto-eval-v1",
        "scope": "project_analyst_only",
        "gold_set": "none",
        "semantic_f1_available": False,
        "run_count": len(per_run),
        "project_count": len({item["project_id"] for item in per_run}),
        "completion_rate": _ratio(completed, len(per_run)),
        "trace_available_rate": _ratio(len(traced), len(per_run)),
        "trace_missing_count": len(per_run) - len(traced),
        "readme_first_rate": _ratio(sum(item["readme_first"] for item in traced), len(traced)),
        "core_module_read_rate": _ratio(read_modules, total_modules),
        "citation_valid_path_rate": _ratio(round(valid_paths), total_citations),
        "citation_valid_line_rate": _ratio(round(valid_lines), total_citations),
        "citation_excerpt_support_rate": _ratio(round(supported), total_citations),
        "forbidden_path_rate": _ratio(sum(item["forbidden_path_count"] for item in traced), sum(item["read_file_count"] for item in traced)),
        "registered_path_rate": _ratio(sum(item["all_read_paths_are_registered"] for item in traced), len(traced)),
        "cache_hit_rate": _ratio(sum(item["cache_hit"] for item in per_run), len(per_run)),
        "average_elapsed_ms_uncached": round(sum(item["elapsed_ms"] for item in uncached) / len(uncached), 3) if uncached else 0.0,
        "total_tool_call_count": sum(item["tool_call_count"] for item in per_run),
        "total_token_count": sum(item["token_count"] for item in per_run),
        "runs": per_run,
    }


__all__ = ["evaluate_project_analysis"]
