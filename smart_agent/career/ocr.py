"""Provider-neutral OCR contracts with traceable correction boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

OCR_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class OCRRegion:
    page: int
    text: str
    confidence: float
    polygon: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.page < 1 or not str(self.text or "").strip():
            raise ValueError("OCRRegion page and text are required")
        if not 0 <= float(self.confidence) <= 1:
            raise ValueError("OCR confidence must be between 0 and 1")
        object.__setattr__(self, "text", str(self.text).strip())
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "polygon", tuple(tuple(map(float, point)) for point in self.polygon))


@dataclass(frozen=True, slots=True)
class OCRDocument:
    ocr_id: str
    provider: str
    model: str
    original_path: str
    original_sha256: str
    regions: tuple[OCRRegion, ...]
    corrected_text: str = ""
    corrections: tuple[dict[str, Any], ...] = ()
    elapsed_ms: float = 0.0
    input_bytes: int = 0
    estimated_cost: float = 0.0
    schema_version: str = OCR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not str(self.ocr_id or "").strip() or not str(self.provider or "").strip():
            raise ValueError("OCR identity is required")
        if self.schema_version != OCR_SCHEMA_VERSION:
            raise ValueError("Unsupported OCR schema version")
        if len(self.original_sha256) != 64 or not re.fullmatch(r"[0-9a-f]{64}", self.original_sha256):
            raise ValueError("original_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "corrections", tuple(dict(item) for item in self.corrections))
        for name in ("elapsed_ms", "estimated_cost"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        if int(self.input_bytes) < 0:
            raise ValueError("input_bytes cannot be negative")
        object.__setattr__(self, "input_bytes", int(self.input_bytes))
        if not self.corrected_text:
            object.__setattr__(self, "corrected_text", "\n".join(region.text for region in self.regions))

    @property
    def average_confidence(self) -> float:
        return sum(region.confidence for region in self.regions) / max(1, len(self.regions))

    @property
    def correction_count(self) -> int:
        return len(self.corrections)

    def to_candidate_payload(self) -> dict[str, Any]:
        """Return candidate material data; this is never a confirmed fact."""
        return {
            "ocr_id": self.ocr_id,
            "provider": self.provider,
            "model": self.model,
            "original_path": self.original_path,
            "original_sha256": self.original_sha256,
            "text": self.corrected_text,
            "regions": [
                {"page": region.page, "text": region.text, "confidence": region.confidence}
                for region in self.regions
            ],
            "metrics": {
                "average_confidence": self.average_confidence,
                "correction_count": self.correction_count,
                "elapsed_ms": self.elapsed_ms,
                "input_bytes": self.input_bytes,
                "estimated_cost": self.estimated_cost,
            },
            "status": "candidate",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "ocr_id": self.ocr_id,
            "provider": self.provider,
            "model": self.model,
            "original_path": self.original_path,
            "original_sha256": self.original_sha256,
            "regions": [
                {
                    "page": region.page,
                    "text": region.text,
                    "confidence": region.confidence,
                    "polygon": [list(point) for point in region.polygon],
                }
                for region in self.regions
            ],
            "corrected_text": self.corrected_text,
            "corrections": [dict(item) for item in self.corrections],
            "elapsed_ms": self.elapsed_ms,
            "input_bytes": self.input_bytes,
            "estimated_cost": self.estimated_cost,
            "schema_version": self.schema_version,
        }


class OCRAdapter(Protocol):
    name: str
    model: str

    def extract(self, path: str) -> tuple[OCRRegion, ...]: ...


class TextOCRAdapter:
    """Deterministic adapter for tests and text sidecars."""

    name = "text-sidecar"
    model = "plain-text-v1"

    def extract(self, path: str) -> tuple[OCRRegion, ...]:
        text = open(path, encoding="utf-8").read()
        return tuple(
            OCRRegion(page=1, text=line, confidence=1.0)
            for line in text.splitlines()
            if line.strip()
        )


class PytesseractOCRAdapter:
    """Optional local image adapter; dependency is imported only when used."""

    name = "pytesseract"
    model = "tesseract-local"

    def __init__(self, *, language: str = "eng") -> None:
        self.language = language

    def extract(self, path: str) -> tuple[OCRRegion, ...]:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install the optional OCR dependencies to use image OCR.") from exc
        image = Image.open(path)
        data = pytesseract.image_to_data(image, lang=self.language, output_type=pytesseract.Output.DICT)
        regions: list[OCRRegion] = []
        for index, raw_text in enumerate(data.get("text", [])):
            text = str(raw_text or "").strip()
            if not text:
                continue
            confidence = max(0.0, min(1.0, float(data["conf"][index]) / 100))
            regions.append(OCRRegion(page=1, text=text, confidence=confidence))
        return tuple(regions)


class OCRService:
    def __init__(self, adapter: OCRAdapter) -> None:
        self.adapter = adapter

    def extract(self, path: str) -> OCRDocument:
        original = os.path.abspath(path)
        with open(original, "rb") as handle:
            content = handle.read()
        started = time.perf_counter()
        digest = hashlib.sha256(content).hexdigest()
        regions = self.adapter.extract(original)
        return OCRDocument(
            ocr_id=f"ocr-{uuid.uuid4().hex[:16]}",
            provider=self.adapter.name,
            model=self.adapter.model,
            original_path=original,
            original_sha256=digest,
            regions=regions,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            input_bytes=len(content),
            estimated_cost=float(getattr(self.adapter, "estimated_cost", 0.0)),
        )

    @staticmethod
    def correct(document: OCRDocument, corrections: list[dict[str, Any]]) -> OCRDocument:
        text = document.corrected_text
        applied: list[dict[str, Any]] = []
        for correction in corrections:
            old = str(correction.get("from") or "")
            new = str(correction.get("to") or "")
            if not old or not new or old not in text:
                raise ValueError("OCR correction must replace an existing non-empty token")
            text = text.replace(old, new, 1)
            applied.append({"from": old, "to": new, "reason": str(correction.get("reason") or "user")})
        return OCRDocument(
            ocr_id=document.ocr_id,
            provider=document.provider,
            model=document.model,
            original_path=document.original_path,
            original_sha256=document.original_sha256,
            regions=document.regions,
            corrected_text=text,
            corrections=(*document.corrections, *applied),
            elapsed_ms=document.elapsed_ms,
            input_bytes=document.input_bytes,
            estimated_cost=document.estimated_cost,
        )


class OCRArtifactStore:
    """Persist source bytes and OCR metadata without promoting content to facts."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = os.path.abspath(os.fspath(root))
        self.raw_root = os.path.join(self.root, "raw")
        self.document_root = os.path.join(self.root, "documents")
        os.makedirs(self.raw_root, exist_ok=True)
        os.makedirs(self.document_root, exist_ok=True)

    def save(self, document: OCRDocument, content: bytes, *, suffix: str = ".bin") -> OCRDocument:
        if not content:
            raise ValueError("OCR source content is required")
        safe_suffix = str(suffix or ".bin").lower()
        if not safe_suffix.startswith(".") or len(safe_suffix) > 10 or not re.fullmatch(r"\.[a-z0-9]+", safe_suffix):
            safe_suffix = ".bin"
        raw_path = os.path.join(self.raw_root, f"{document.ocr_id}{safe_suffix}")
        with open(raw_path, "wb") as handle:
            handle.write(content)
        persisted = replace(document, original_path=raw_path, input_bytes=len(content))
        document_path = os.path.join(self.document_root, f"{document.ocr_id}.json")
        with open(document_path, "w", encoding="utf-8") as handle:
            json.dump(persisted.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return persisted
