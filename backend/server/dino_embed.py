"""dino_embed.py — DINOv2 影像 embedding（lazy 載入，供正式 server 即時使用）。

第一次用到才載入模型（不拖慢啟動）。載入失敗（沒裝 transformers/torch）時
回傳 None，讓呼叫端安全退回原本的定位方法。
"""
from __future__ import annotations
import threading
from pathlib import Path
from typing import Optional

import numpy as np

_MODEL_NAME = "facebook/dinov2-base"
_lock = threading.Lock()
_state = {"ready": False, "ok": False, "proc": None, "model": None, "torch": None}


def _ensure_loaded() -> bool:
    if _state["ready"]:
        return _state["ok"]
    with _lock:
        if _state["ready"]:
            return _state["ok"]
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
            proc = AutoImageProcessor.from_pretrained(_MODEL_NAME)
            model = AutoModel.from_pretrained(_MODEL_NAME)
            model.eval()
            _state.update(proc=proc, model=model, torch=torch, ok=True)
        except Exception as e:  # noqa: BLE001
            _state["ok"] = False
            try:
                import logging
                logging.getLogger(__name__).warning("DINOv2 載入失敗，融合定位將退回：%s", e)
            except Exception:
                pass
        _state["ready"] = True
    return _state["ok"]


def available() -> bool:
    return _ensure_loaded()


def embed_image(path: str) -> Optional[np.ndarray]:
    """回傳 L2-normalized 的 768 維向量；失敗回 None。"""
    if not _ensure_loaded():
        return None
    try:
        from PIL import Image, ImageOps
        torch = _state["torch"]
        im = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
        inp = _state["proc"](images=im, return_tensors="pt")
        with torch.no_grad():
            out = _state["model"](**inp)
        v = out.last_hidden_state[:, 0]
        v = v / v.norm(dim=-1, keepdim=True)
        return v[0].numpy().astype(np.float32)
    except Exception as e:  # noqa: BLE001
        try:
            import logging
            logging.getLogger(__name__).warning("DINOv2 embed 失敗：%s", e)
        except Exception:
            pass
        return None
