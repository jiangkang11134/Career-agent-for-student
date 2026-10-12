"""JSON Schema export for Career data contracts."""

from __future__ import annotations

import types
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

from smart_agent.career import models

_RECORD_TYPES = (
    models.ResumeSource,
    models.ResumeTemplate,
    models.EducationEvidence,
    models.AwardEvidence,
    models.InternshipEvidence,
    models.ProjectEvidence,
    models.ResearchEvidence,
    models.ProjectDescriptionDocument,
    models.JDDocument,
    models.JDRequirement,
    models.EvidenceMatch,
    models.ResumeItem,
    models.ResumePlan,
    models.ResumeClaim,
    models.ResumeDocument,
    models.VerificationReport,
    models.ResumeRun,
    models.RequirementMatchCache,
)

_NESTED_TYPES = (
    models.VersionRef,
    models.SourceReference,
    models.ResumeSectionBlock,
    models.ChangeEvent,
    models.EvidenceFact,
    models.ResumeHeader,
)

_ENUM_TYPES = (
    models.SourceType,
    models.EvidenceType,
    models.FactStatus,
    models.ChangeAction,
    models.RequirementCategory,
    models.SupportLevel,
    models.ClaimStatus,
    models.ResumeDocumentStatus,
    models.VerificationStatus,
    models.ProjectDescriptionStatus,
    models.ResumeRunStatus,
    models.RequirementMatchCacheStatus,
)


def _type_schema(annotation: Any) -> dict[str, Any]:
    if annotation is Any:
        return {}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is type(None):
        return {"type": "null"}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        return {"anyOf": [_type_schema(option) for option in args]}
    if origin in (tuple, list):
        item_type = args[0] if args else Any
        return {"type": "array", "items": _type_schema(item_type)}
    if origin is dict:
        value_type = args[1] if len(args) == 2 else Any
        return {"type": "object", "additionalProperties": _type_schema(value_type)}
    if isinstance(annotation, type) and issubclass(annotation, (Enum,)):
        return {"$ref": f"#/$defs/{annotation.__name__}"}
    if isinstance(annotation, type) and is_dataclass(annotation):
        return {"$ref": f"#/$defs/{annotation.__name__}"}
    return {}


def _dataclass_schema(cls: type, *, record: bool) -> dict[str, Any]:
    hints = get_type_hints(cls)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for item in fields(cls):
        properties[item.name] = _type_schema(hints.get(item.name, Any))
        if item.default is MISSING and item.default_factory is MISSING:
            required.append(item.name)
    if record:
        properties["record_type"] = {"const": cls.__name__}
        required.append("record_type")
        if issubclass(cls, (models.ResumeItem, models.ResumeClaim)):
            properties["evidence_ids"] = {
                "type": "array",
                "items": {"type": "string"},
                "readOnly": True,
            }
            properties["requirement_ids"] = {
                "type": "array",
                "items": {"type": "string"},
                "readOnly": True,
            }
            required.extend(("evidence_ids", "requirement_ids"))
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def career_record_schema() -> dict[str, Any]:
    """Return the stable Draft 2020-12 schema for every persisted Career record."""

    definitions: dict[str, Any] = {
        enum_type.__name__: {
            "type": "string",
            "enum": [member.value for member in enum_type],
        }
        for enum_type in _ENUM_TYPES
    }
    definitions.update(
        {
            nested_type.__name__: _dataclass_schema(nested_type, record=False)
            for nested_type in _NESTED_TYPES
        }
    )
    definitions.update(
        {
            record_type.__name__: _dataclass_schema(record_type, record=True)
            for record_type in _RECORD_TYPES
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://smart-agent.local/schema/career/{models.CAREER_SCHEMA_VERSION}",
        "title": "Smart Agent Career Record",
        "oneOf": [
            {"$ref": f"#/$defs/{record_type.__name__}"}
            for record_type in _RECORD_TYPES
        ],
        "$defs": definitions,
    }


def schema_for(record_type: type[models.VersionedCareerRecord]) -> dict[str, Any]:
    """Return a single-record schema while retaining all referenced definitions."""

    if record_type not in _RECORD_TYPES:
        raise ValueError(f"Unsupported Career record type: {record_type!r}")
    schema = career_record_schema()
    schema["title"] = record_type.__name__
    schema.pop("oneOf")
    schema["$ref"] = f"#/$defs/{record_type.__name__}"
    return schema
