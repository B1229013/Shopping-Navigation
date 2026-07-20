"""Run EasyOCR over the staged store photos and save per-photo text.

Sign text (aisle numbers, category names) is the strongest zone-distinguishing
signal in a real store — far more than GroundingDINO's generic object labels.
Output: store_ocr.json  ->  {filename: [[text, confidence], ...], ...}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server.ocr import OCR

DEFAULT_PHOTOS = Path(__file__).resolve().parent / "store_photos"


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PHOTOS
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else folder.parent / "store_ocr.json"
    ocr = OCR(languages=["en", "ch_tra"])
    ocr.load()
    out = {}
    for p in sorted(folder.glob("*.jpg")):
        results = ocr.read(str(p), min_confidence=0.4, max_results=25)
        out[p.name] = [[r.text, round(r.confidence, 3)] for r in results]
        texts = " | ".join(r.text for r in results[:14])
        print(f"{p.name:22s} {texts}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_path} ({len(out)} photos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
