"""Local text embeddings with an on-disk cache. No network after the first model download."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path

import numpy as np

ORIGINAL_MODEL = "BAAI/bge-small-en-v1.5"
MODEL = os.environ.get("POCKET_RECALL_EMBED_MODEL", ORIGINAL_MODEL)


# Models fastembed does not list but that ship an ONNX export, registered on first use.
CUSTOM_MODELS = {
    "BAAI/bge-m3": dict(pooling="CLS", normalization=True, dim=1024, model_file="onnx/model.onnx",
                        additional_files=["onnx/model.onnx_data"], size_in_gb=2.27,
                        description="Multilingual (100+ languages), dense CLS embedding, 8192 tokens"),
}
# Long-window models only see this many characters of a passage (~512 tokens): retrieval works on
# short turns and excerpt chunks, and attention cost grows with the square of the length.
MAX_CHARS = {"BAAI/bge-m3": 2000}
_registered: set[str] = set()


def _register(model: str) -> None:
    if model not in CUSTOM_MODELS or model in _registered:
        return
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType

    spec = dict(CUSTOM_MODELS[model])
    spec["pooling"] = getattr(PoolingType, spec["pooling"])
    try:
        TextEmbedding.add_custom_model(model=model, sources=ModelSource(hf=model), **spec)
    except ValueError:
        pass  # already registered in this process
    _registered.add(model)


def _key(text: str, model: str) -> str:
    # Keys for the original default model stay text-only so existing caches remain valid.
    salt = "" if model == ORIGINAL_MODEL else model + "\0"
    return hashlib.sha1((salt + text).encode()).hexdigest()


class Embedder:
    def __init__(self, cache_path: Path, model: str = MODEL):
        self.model_name = model
        self._model = None
        self._queries: dict[str, np.ndarray] = {}  # one recall ranks many excerpts against the same query
        self._vectors: dict[str, np.ndarray] = {}  # passages already read from disk, for a long-running server
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            # A fixed cache dir: fastembed's default lives under the temp dir and gets re-downloaded.
            cache = Path(os.environ.get("POCKET_RECALL_MODEL_DIR", "~/.pocket-recall/models")).expanduser()
            _register(self.model_name)
            self._model = TextEmbedding(self.model_name, cache_dir=str(cache))
        return self._model

    def passages(self, texts: list[str]) -> np.ndarray:
        """Unit-normalised vectors for passages, computing only the ones not cached yet."""
        keys = [_key(t, self.model_name) for t in texts]
        found: dict[str, np.ndarray] = {k: self._vectors[k] for k in keys if k in self._vectors}
        unread = [k for k in dict.fromkeys(keys) if k not in found]
        for i in range(0, len(unread), 900):
            chunk = unread[i : i + 900]
            q = f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(chunk))})"
            found.update((k, np.frombuffer(v, dtype=np.float32)) for k, v in self.db.execute(q, chunk))
        missing = list({k: t for k, t in zip(keys, texts) if k not in found}.items())
        if missing:
            limit = MAX_CHARS.get(self.model_name)
            # A batch is padded to its longest text; for long-window models that costs more than
            # batching saves, so they go one passage at a time (measured: 140 s vs 270 s per 500 turns).
            texts_in = [t[:limit] for _, t in missing] if limit else [t for _, t in missing]
            vecs = self.model.passage_embed(texts_in, batch_size=1 if limit else 256)
            rows = [(k, np.asarray(v, dtype=np.float32).tobytes()) for (k, _), v in zip(missing, vecs)]
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self.db.commit()
            found.update((k, np.frombuffer(b, dtype=np.float32)) for k, b in rows)
        self._vectors.update(found)
        return _normalise(np.stack([found[k] for k in keys])) if keys else np.zeros((0, 0), np.float32)

    def query(self, text: str) -> np.ndarray:
        if text not in self._queries:
            if len(self._queries) > 256:
                self._queries.clear()
            capped = text[: MAX_CHARS.get(self.model_name) or len(text)]
            vec = np.asarray(next(iter(self.model.query_embed([capped]))), dtype=np.float32)
            self._queries[text] = _normalise(vec[None])[0]
        return self._queries[text]


def _normalise(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9)


class Reranker:
    """Local cross-encoder: reads question and passage together, slower but sharper than embeddings."""

    def __init__(self, model: str):
        self.model_name = model
        self._model = None
        self._queries: dict[str, np.ndarray] = {}  # one recall ranks many excerpts against the same query

    @property
    def model(self):
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            cache = Path(os.environ.get("POCKET_RECALL_MODEL_DIR", "~/.pocket-recall/models")).expanduser()
            self._model = TextCrossEncoder(self.model_name, cache_dir=str(cache))
        return self._model

    def scores(self, query: str, texts: list[str]) -> list[float]:
        return [float(x) for x in self.model.rerank(query, texts, batch_size=32)]
