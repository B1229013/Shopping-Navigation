"""Shared client for the Chang Gung University OpenAI-compatible LLM gateway.

One API key (``CGU_API_KEY``) unlocks every model on the gateway; only the
*endpoint* differs by model type. This module exposes a single OpenAI client
pointed at the gateway plus thin helpers that route each model to the right
endpoint, so the comparison layer and the perception pipeline share one client:

    chat / vision models       -> /v1/chat/completions   chat()
    OCR models (deepseek/glm)   -> /v1/chat/completions   ocr()   (image input)
    embedding models (bge-m3…)  -> /v1/embeddings         embed()
    model discovery             -> /v1/models             list_models()

The base URL override is mandatory: without it the SDK hits api.openai.com and
returns "invalid key" for a CGU key.
"""
from __future__ import annotations

import base64
import io
import mimetypes
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

from server.config import CGU_API_KEY, CGU_BASE_URL

# Route a model id to its endpoint. The gateway groups models by name; these
# substrings are stable across the current CGU catalog. Extend as it grows.
_EMBED_HINTS = ("embedding", "bge-m3")
_OCR_HINTS = ("ocr",)  # deepseek-ocr:latest, glm-ocr:latest

_DEFAULT_OCR_PROMPT = (
    "Extract all text visible in this image exactly as written, preserving "
    "Traditional Chinese characters. Return only the extracted text."
)


@lru_cache(maxsize=1)
def get_client():
    """The one shared OpenAI client, pointed at the CGU gateway.

    Lazily imported so importing this module never forces the ``openai`` dep, and
    lru-cached so every caller reuses a single client/connection pool.
    """
    from openai import OpenAI  # lazy
    if not CGU_API_KEY:
        raise RuntimeError(
            "CGU_API_KEY is empty — add it to APPNAV-main/.env. "
            "Without it the gateway returns HTTP 401."
        )
    return OpenAI(api_key=CGU_API_KEY, base_url=CGU_BASE_URL)


def model_kind(model: str) -> str:
    """Classify a model id for routing: 'embedding' | 'ocr' | 'chat'."""
    m = model.lower()
    if any(h in m for h in _EMBED_HINTS):
        return "embedding"
    if any(h in m for h in _OCR_HINTS):
        return "ocr"
    return "chat"


def _token_kwarg(model: str, max_tokens: int) -> dict:
    """Pick the right output-length param for the model.

    The gpt-5.x family (newer OpenAI reasoning models) rejects ``max_tokens``
    with HTTP 400 and requires ``max_completion_tokens``; older and local models
    (gpt-4o, gpt-oss:20b, …) take ``max_tokens``. Verified against the gateway.
    """
    if model.lower().startswith("gpt-5"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def list_models() -> List[str]:
    """Every model id the gateway currently offers (drives the UI dropdown)."""
    return sorted(m.id for m in get_client().models.list().data)


def _image_data_url(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f"data:{mime};base64,{b64}"


def _pil_data_url(image_pil: Any) -> str:
    """Encode an in-memory PIL image (e.g. a detection crop) as a JPEG data URL."""
    buf = io.BytesIO()
    image_pil.convert("RGB").save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(
    model: str,
    prompt: str,
    image_path: Optional[str] = None,
    image_pil: Optional[Any] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout: int = 120,
) -> str:
    """Chat / vision completion via /v1/chat/completions.

    Pass ``image_path`` (a file) or ``image_pil`` (an in-memory PIL image, e.g. a
    GroundingDINO detection crop) to send an image alongside the prompt — works
    for any vision-capable chat model on the gateway, and for the OCR models.
    """
    data_url = None
    if image_pil is not None:
        data_url = _pil_data_url(image_pil)
    elif image_path:
        data_url = _image_data_url(image_path)

    if data_url:
        content: list = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]
        message = {"role": "user", "content": content}
    else:
        message = {"role": "user", "content": prompt}

    resp = get_client().chat.completions.create(
        model=model,
        messages=[message],
        temperature=temperature,
        timeout=timeout,
        **_token_kwarg(model, max_tokens),
    )
    return (resp.choices[0].message.content or "") if resp.choices else ""


def ocr(
    model: str,
    image_path: Optional[str] = None,
    image_pil: Optional[Any] = None,
    prompt: str = _DEFAULT_OCR_PROMPT,
    max_tokens: int = 1024,
    timeout: int = 120,
) -> str:
    """Run an OCR model (deepseek-ocr / glm-ocr) on an image crop.

    Thin wrapper over chat() at temperature 0 — the gateway serves OCR models
    through /v1/chat/completions with image input, same as vision chat.
    """
    return chat(model, prompt, image_path=image_path, image_pil=image_pil,
                temperature=0.0, max_tokens=max_tokens, timeout=timeout)


def run_on_image(
    models: Iterable[str],
    prompt: str,
    image_path: Optional[str] = None,
    image_pil: Optional[Any] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: int = 120,
) -> List[Dict[str, Any]]:
    """Send the SAME image + prompt through several models and collect their replies.

    The backend twin of compare.html's side-by-side run: pass e.g.
    ``["deepseek-ocr:latest", "glm-ocr:latest"]`` to compare OCR text extraction on
    a detection crop, or chat models for scene/navigation reasoning. Each model is
    routed by kind; a per-model failure is captured, never raised, so one dead model
    (e.g. the OCR 502s) can't sink the others.
    """
    results: List[Dict[str, Any]] = []
    for m in models:
        entry: Dict[str, Any] = {"model": m, "kind": model_kind(m), "text": "", "error": None}
        try:
            entry["text"] = chat(m, prompt, image_path=image_path, image_pil=image_pil,
                                 temperature=temperature, max_tokens=max_tokens, timeout=timeout)
        except Exception as e:  # keep comparing the rest
            entry["error"] = str(e)
        results.append(entry)
    return results


def embed(model: str, texts: Iterable[str]) -> List[List[float]]:
    """Embedding models (bge-m3, text-embedding-3-*) via /v1/embeddings."""
    resp = get_client().embeddings.create(model=model, input=list(texts))
    return [d.embedding for d in resp.data]
