"""Evidence-based resume review for the Career Agent domain."""

from __future__ import annotations

import re
from typing import Any

KNOWN_TECH_SKILLS = (
    "Python", "Java", "Go", "C++", "JavaScript", "TypeScript", "SQL", "PyTorch",
    "TensorFlow", "LangChain", "LangGraph", "RAG", "Agent", "Docker", "Kubernetes",
    "FastAPI", "Django", "Flask", "Redis", "MySQL", "PostgreSQL", "Git",
)


def extract_tech_skills(text: str) -> list[str]:
    return [
        skill
        for skill in KNOWN_TECH_SKILLS
        if re.search(rf"(?i)(?<!\w){re.escape(skill)}(?!\w)", text or "")
    ]


def _key(value: str) -> str:
    return re.sub(r"\s+", "", value or "").casefold()


def _metrics(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)", text or ""))


def review_resume(
    *,
    jd: str,
    resume_projects: list[dict[str, Any]],
    project_summaries: list[dict[str, Any]],
    known_skills: list[str] | None = None,
) -> dict[str, Any]:
    """Review structured resume projects against user evidence and a JD."""
    evidence_by_name = {
        _key(str(project.get("name", ""))): project
        for project in project_summaries
        if str(project.get("name", "")).strip()
    }
    mastered = {_key(skill) for skill in (known_skills or [])}
    findings: list[dict[str, str]] = []
    represented_skills: set[str] = set()

    for project in resume_projects:
        name = str(project.get("name", "")).strip()
        evidence = evidence_by_name.get(_key(name))
        if evidence is None:
            findings.append({
                "severity": "error",
                "rule": "unknown-project",
                "project": name,
                "message": "Resume project does not exist in the imported project summaries.",
            })
            continue

        evidence_text = " ".join(
            [str(evidence.get("summary", "")), *map(str, evidence.get("skills", []))]
        )
        claimed_skills = {
            *map(str, project.get("skills", [])),
            *extract_tech_skills(" ".join(map(str, project.get("bullets", [])))),
        }
        evidence_skills = {_key(skill) for skill in extract_tech_skills(evidence_text)}
        evidence_skills.update(_key(str(skill)) for skill in evidence.get("skills", []))
        represented_skills.update(_key(skill) for skill in claimed_skills)

        for skill in sorted(claimed_skills):
            if _key(skill) not in evidence_skills and _key(skill) not in mastered:
                findings.append({
                    "severity": "error",
                    "rule": "unsupported-skill",
                    "project": name,
                    "message": f"Claimed skill '{skill}' is not supported by project evidence or the user's skill profile.",
                })

        evidence_metrics = _metrics(evidence_text)
        claimed_metrics = _metrics(" ".join(map(str, project.get("bullets", []))))
        for metric in sorted(claimed_metrics - evidence_metrics):
            findings.append({
                "severity": "error",
                "rule": "unsupported-metric",
                "project": name,
                "message": f"Metric '{metric}' is not present in the project summary evidence.",
            })

    jd_skills = extract_tech_skills(jd)
    matched = [skill for skill in jd_skills if _key(skill) in represented_skills or _key(skill) in mastered]
    missing = [skill for skill in jd_skills if skill not in matched]
    coverage = 1.0 if not jd_skills else len(matched) / len(jd_skills)
    if missing:
        findings.append({
            "severity": "warning",
            "rule": "jd-skill-gap",
            "project": "",
            "message": "JD skills not covered by verified resume evidence: " + ", ".join(missing),
        })

    fact_consistent = not any(item["severity"] == "error" for item in findings)
    return {
        "fact_consistent": fact_consistent,
        "jd_match": {
            "coverage": round(coverage, 3),
            "matched_skills": matched,
            "missing_skills": missing,
        },
        "findings": findings,
        "verdict": "pass" if fact_consistent and coverage >= 0.6 else "revise",
    }
