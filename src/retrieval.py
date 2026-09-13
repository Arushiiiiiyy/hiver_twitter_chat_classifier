"""Embedding index over historical resolved pairs, used to ground reply generation in
how this brand actually resolved similar issues before (not just generic LLM
knowledge).
Qwen 3 is used because it is uses less GPU and better at retrieval
Qwen 3 is an instruction aware model, hence the query should be given a "query" tag and 
documents should not use that
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Lighter fallback if you don't have GPU access — see README "If using CPU only".
CPU_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RetrievalIndex:
    def __init__(self, pairs: list[dict], embeddings: np.ndarray, model_name: str = DEFAULT_MODEL):
        self.pairs = pairs
        self.embeddings = embeddings  # (N, D), L2-normalized, encoded WITHOUT the query prompt
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def query(self, text: str, k: int = 3) -> list[dict]:
        model = self._get_model()
        # Qwen3-Embedding models ship a built-in "query" prompt; other models (e.g. the
        # CPU fallback) silently ignore prompt_name if they don't define one.
        try:
            vec = model.encode([text], prompt_name="query", normalize_embeddings=True)[0]
        except ValueError:
            vec = model.encode([text], normalize_embeddings=True)[0]
        sims = self.embeddings @ vec
        top_idx = np.argsort(-sims)[:k]
        return [
            {**self.pairs[i], "similarity": float(sims[i])} for i in top_idx
        ]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"pairs": self.pairs, "embeddings": self.embeddings,
                         "model_name": self.model_name}, f)

    @classmethod
    def load(cls, path: str) -> "RetrievalIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["pairs"], data["embeddings"], data.get("model_name", DEFAULT_MODEL))


def build(pairs_path: str, index_path: str, model_name: str = DEFAULT_MODEL) -> None:
    pairs = [json.loads(line) for line in Path(pairs_path).open()]
    model = SentenceTransformer(model_name)
    texts = [p["customer_text"] for p in pairs]
    # Documents: no prompt. Only queries get the "query" instruction at search time.
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    RetrievalIndex(pairs, np.array(embeddings), model_name).save(index_path)
    print(f"Indexed {len(pairs)} pairs with {model_name} -> {index_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pairs", required=True)
    b.add_argument("--index", required=True)
    b.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Embedding model. Default {DEFAULT_MODEL} (GPU recommended). "
                        f"Pass {CPU_FALLBACK_MODEL} if you're CPU-only.")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.pairs, args.index, args.model)
