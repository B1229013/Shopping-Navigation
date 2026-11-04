"""OpenAI text embeddings for object labels, cached on disk — the vectors the server's
hybrid_max similarity uses when its embedding cache is on (visual_localization:
_emb_cache, EMBEDDING_MODEL), and what legacy/server/locate_v2._embedding_sim used.

  emb = LabelEmbeddings(cache_dir / "label_embeddings.npz")
  emb.ensure(labels)            # fetches only the missing ones (paid; ~$0.02 / 1M tokens)
  emb.matrix(labels)            # (n × d) L2-normalised rows

Same endpoint and model as server/visual_localization._compute_query_embeddings.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

BATCH = 256


class LabelEmbeddings:
    def __init__(self, path: Path):
        from server import config as C
        self.path, self.model = Path(path), C.EMBEDDING_MODEL
        self.vec: dict = {}
        self.tokens = 0
        if self.path.is_file():
            z = np.load(self.path, allow_pickle=False)
            if str(z["model"]) == self.model:
                self.vec = dict(zip(z["labels"].tolist(), z["vectors"].astype(np.float32)))

    def ensure(self, labels):
        from server import config as C
        import requests
        missing = sorted({l for l in labels if l and l not in self.vec})
        if not missing:
            return 0
        if not C.OPENAI_API_KEY:
            sys.exit("OPENAI_API_KEY is not set")
        for i in range(0, len(missing), BATCH):
            batch = missing[i:i + BATCH]
            for attempt in range(3):
                try:
                    r = requests.post("https://api.openai.com/v1/embeddings",
                                      json={"input": batch, "model": self.model},
                                      headers={"Authorization": f"Bearer {C.OPENAI_API_KEY}",
                                               "Content-Type": "application/json"}, timeout=60)
                    r.raise_for_status()
                    break
                except Exception as e:
                    if attempt == 2:
                        self.save()
                        sys.exit(f"embedding request failed: {e}")
                    time.sleep(2 * (attempt + 1))
            body = r.json()
            self.tokens += body.get("usage", {}).get("total_tokens", 0)
            for item in body["data"]:
                v = np.asarray(item["embedding"], np.float32)
                self.vec[batch[item["index"]]] = v / np.linalg.norm(v)
            print(f"\r  embeddings {min(i + BATCH, len(missing))}/{len(missing)} ({self.tokens} tokens)",
                  end="", file=sys.stderr)
        print(file=sys.stderr)
        self.save()
        return len(missing)

    def save(self):
        labels = list(self.vec)
        np.savez_compressed(self.path, model=np.array(self.model), labels=np.array(labels),
                            vectors=np.stack([self.vec[l] for l in labels]).astype(np.float16))

    def matrix(self, labels):
        m = np.stack([self.vec[l] for l in labels]).astype(np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)   # stored as float16
