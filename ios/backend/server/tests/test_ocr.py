"""Tests for the OCR module."""
from __future__ import annotations

from server.ocr import OCR, OCRResult, summarize_ocr


class TestSummarizeOcr:
    def test_empty_results(self):
        assert summarize_ocr([]) == "(no text detected)"

    def test_single_result(self):
        results = [OCRResult(text="Dairy", confidence=0.92, bbox=[[0, 0], [100, 0], [100, 30], [0, 30]])]
        s = summarize_ocr(results)
        assert '"Dairy"' in s
        assert "92%" in s

    def test_deduplication(self):
        results = [
            OCRResult(text="Exit", confidence=0.95, bbox=[]),
            OCRResult(text="exit", confidence=0.80, bbox=[]),  # same text, different case
            OCRResult(text="Aisle 3", confidence=0.88, bbox=[]),
        ]
        s = summarize_ocr(results)
        # "exit" should appear only once (deduplicated)
        assert s.count("Exit") + s.count("exit") == 1
        assert "Aisle 3" in s

    def test_max_results_capped(self):
        results = [
            OCRResult(text=f"Text {i}", confidence=0.9 - i * 0.05, bbox=[])
            for i in range(20)
        ]
        s = summarize_ocr(results)
        # Should cap at 10 unique entries
        assert s.count('"') <= 20  # at most 10 pairs of quotes

    def test_empty_text_filtered(self):
        results = [
            OCRResult(text="", confidence=0.99, bbox=[]),
            OCRResult(text="  ", confidence=0.99, bbox=[]),
            OCRResult(text="Real Text", confidence=0.85, bbox=[]),
        ]
        s = summarize_ocr(results)
        assert "Real Text" in s
        # Should not contain empty entries
        assert '""' not in s


class TestOCRInit:
    def test_default_languages(self):
        ocr = OCR()
        assert "en" in ocr._languages
        assert not ocr.is_loaded

    def test_custom_languages(self):
        ocr = OCR(languages=["en", "ja"])
        assert ocr._languages == ["en", "ja"]
        assert not ocr.is_loaded
