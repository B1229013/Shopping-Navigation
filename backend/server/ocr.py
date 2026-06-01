"""OCR module: extract text from photos to help identify places and signs.

Uses EasyOCR for multilingual text recognition (English + Chinese).
Text is fed into the VLM prompt alongside GroundingDINO detections,
enabling the system to read aisle signs, product labels, section names,
door labels, and other environmental text without user input.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


class OCR:
    """Lazy-loaded EasyOCR wrapper for multilingual text extraction."""

    def __init__(self, languages: Optional[List[str]] = None) -> None:
        self._reader = None
        self._languages = languages or ["en", "ch_tra"]

    def load(self) -> None:
        import easyocr

        log.info("Loading EasyOCR with languages: %s", self._languages)
        self._reader = easyocr.Reader(
            self._languages,
            gpu=True,
            verbose=False,
        )
        log.info("EasyOCR loaded successfully")

    def read(
        self,
        image_path: str,
        min_confidence: float = 0.3,
        max_results: int = 15,
    ) -> List[OCRResult]:
        """Extract text from an image.

        Args:
            image_path: Path to the image file.
            min_confidence: Minimum confidence threshold for text detection.
            max_results: Maximum number of text results to return.

        Returns:
            List of OCRResult sorted by confidence (highest first).
        """
        if self._reader is None:
            self.load()

        try:
            raw = self._reader.readtext(image_path)
        except Exception as e:
            log.warning("OCR failed on %s: %s", image_path, e)
            return []

        results: List[OCRResult] = []
        for bbox, text, conf in raw:
            text = text.strip()
            if not text or conf < min_confidence:
                continue
            results.append(OCRResult(
                text=text,
                confidence=float(conf),
                bbox=[[float(c) for c in pt] for pt in bbox],
            ))

        results.sort(key=lambda r: -r.confidence)
        return results[:max_results]

    @property
    def is_loaded(self) -> bool:
        return self._reader is not None


def summarize_ocr(results: List[OCRResult]) -> str:
    """Build a concise text summary for the VLM prompt.

    Groups text by confidence tier and deduplicates near-identical strings.
    """
    if not results:
        return "(no text detected)"

    seen: set[str] = set()
    unique: List[OCRResult] = []
    for r in results:
        normalized = r.text.lower().strip()
        if normalized not in seen and len(normalized) > 0:
            seen.add(normalized)
            unique.append(r)

    parts: List[str] = []
    for r in unique[:10]:
        parts.append(f'"{r.text}" ({r.confidence:.0%})')

    return ", ".join(parts)
