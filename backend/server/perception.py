"""GroundingDINO + SAM perception. Loaded once at server startup."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from PIL import Image

from server.config import (
    GROUNDINGDINO_CONFIG, GROUNDINGDINO_WEIGHTS, SAM_WEIGHTS,
    GROUNDINGDINO_BOX_THRESHOLD, GROUNDINGDINO_TEXT_THRESHOLD,
    GROUNDINGDINO_BOX_THRESHOLD_FALLBACK, SAM_TOP_K_BOXES,
)

log = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    box: List[float]  # [x1,y1,x2,y2] absolute pixels
    score: float


def verify_goal_detection(
    image_pil: Image.Image,
    box_xyxy: List[float],
    goal_label: str,
    ask_fn: Callable[[Image.Image, str], str],
) -> bool:
    """TASK 3 — crop a goal detection and ask the VLM to confirm it.

    Store shelves are visually dense, so GroundingDINO false positives are common.
    Before trusting a detection of the goal product, crop its bounding box and ask
    the VLM a yes/no question about just that region. ``ask_fn`` takes a PIL image
    and a prompt and returns the VLM's text reply. Any failure is treated as "not
    confirmed" (returns False) so a verification outage can never fake a positive.
    """
    try:
        x1, y1, x2, y2 = [int(c) for c in box_xyxy]
        if x2 <= x1 or y2 <= y1:
            return False
        cropped = image_pil.crop((x1, y1, x2, y2))
        prompt = f"Does this image clearly show '{goal_label}'? Answer only: yes or no."
        response = ask_fn(cropped, prompt)
        return "yes" in (response or "").lower()
    except Exception as e:  # best-effort: never let verification crash a turn
        log.warning("goal verification failed (%s) — treating as unconfirmed", e)
        return False


class Perception:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._gd_model = None
        self._sam_predictor = None

    def load(self) -> None:
        from groundingdino.util.inference import load_model

        log.info("loading GroundingDINO from %s", GROUNDINGDINO_WEIGHTS)
        self._gd_model = load_model(str(GROUNDINGDINO_CONFIG), str(GROUNDINGDINO_WEIGHTS))
        # SAM is intentionally not loaded: detect() only needs GroundingDINO boxes,
        # and SAM ViT-H + llama3.2-vision together exceed the 8 GB VRAM budget.
        self._sam_predictor = None
        log.info("perception loaded on %s (GroundingDINO only)", self.device)

    def detect(self, image_path: str, prompt_classes: List[str]) -> List[Detection]:
        """Detect ``prompt_classes`` in the image (TASK 1 — adaptive thresholds).

        Runs GroundingDINO at the primary (strict) box threshold first to keep
        false positives down on busy shelves. Only if that finds nothing does it
        retry at the lower fallback threshold, so empty scenes still get a best
        effort. Returns at most ``SAM_TOP_K_BOXES`` highest-scoring detections.
        """
        if not prompt_classes:
            return []
        from groundingdino.util.inference import load_image
        text_prompt = " . ".join(prompt_classes) + " ."
        image_source, image = load_image(image_path)

        results = self._predict(image, image_source, text_prompt, GROUNDINGDINO_BOX_THRESHOLD)
        if not results:
            results = self._predict(image, image_source, text_prompt, GROUNDINGDINO_BOX_THRESHOLD_FALLBACK)

        results.sort(key=lambda d: -d.score)
        return results[:SAM_TOP_K_BOXES]

    def _predict(self, image, image_source, text_prompt: str, box_threshold: float) -> List[Detection]:
        from groundingdino.util.inference import predict
        boxes, logits, phrases = predict(
            model=self._gd_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=GROUNDINGDINO_TEXT_THRESHOLD,
            device=self.device,
        )

        h, w = image_source.shape[:2]
        results: List[Detection] = []
        for box_cxcywh, score, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box_cxcywh.tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            results.append(Detection(label=phrase, box=[x1, y1, x2, y2], score=float(score)))
        return results
