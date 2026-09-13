"""Embedding index over historical resolved pairs, used to ground reply generation in
how this brand *actually* resolved similar issues before (not just generic LLM
knowledge).

Uses sentence-transformers locally (no API cost for the index build, only for
generation) with plain cosine similarity — no need for a real vector DB at this scale
(tens of thousands of pairs fits in memory fine; documented in decision_log.md as a
deliberate simplification that would need revisiting past ~1M pairs).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"


class RetrievalIndex:
    def __init__(self, pairs: list[dict], embeddings: np.ndarray):
        self.pairs = pairs
        self.embeddings = embeddings  # (N, D), L2-normalized
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(_MODEL_NAME)
        return self._model

    def query(self, text: str, k: int = 3) -> list[dict]:
        vec = self._get_model().encode([text], normalize_embeddings=True)[0]
        sims = self.embeddings @ vec
        top_idx = np.argsort(-sims)[:k]
        return [
            {**self.pairs[i], "similarity": float(sims[i])} for i in top_idx
        ]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"pairs": self.pairs, "embeddings": self.embeddings}, f)

    @classmethod
    def load(cls, path: str) -> "RetrievalIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["pairs"], data["embeddings"])


def build(pairs_path: str, index_path: str) -> None:
    pairs = [json.loads(line) for line in Path(pairs_path).open()]
    model = SentenceTransformer(_MODEL_NAME)
    texts = [p["customer_text"] for p in pairs]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    RetrievalIndex(pairs, np.array(embeddings)).save(index_path)
    print(f"Indexed {len(pairs)} pairs -> {index_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pairs", required=True)
    b.add_argument("--index", required=True)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.pairs, args.index)
