
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Lighter fallback when no GPU is available.
CPU_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RetrievalIndex:
    def __init__(self, pairs: list[dict], embeddings: np.ndarray, model_name: str = DEFAULT_MODEL):
        self.pairs = pairs
        self.embeddings = embeddings  # (N, D), L2-normalized, encoded without the query prompt
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def query(self, text: str, k: int = 3, exclude_pair_id: str | None = None) -> list[dict]:
        
        model = self._get_model()
        # Qwen3 models define a "query" prompt; models that don't will raise, so fall back.
        try:
            vec = model.encode([text], prompt_name="query", normalize_embeddings=True)[0]
        except ValueError:
            vec = model.encode([text], normalize_embeddings=True)[0]
        sims = self.embeddings @ vec

        order = np.argsort(-sims)
        results = []
        for i in order:
            if exclude_pair_id is not None and self.pairs[i].get("pair_id") == exclude_pair_id:
                continue
            results.append({**self.pairs[i], "similarity": float(sims[i])})
            if len(results) >= k:
                break
        return results

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"pairs": self.pairs, "embeddings": self.embeddings,
                         "model_name": self.model_name}, f)

    @classmethod
    def load(cls, path: str) -> "RetrievalIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["pairs"], data["embeddings"], data.get("model_name", DEFAULT_MODEL))


def build(pairs_path: str, index_path: str, model_name: str | None = None) -> None:
    import torch
    if model_name is None:
        model_name = DEFAULT_MODEL if torch.cuda.is_available() else CPU_FALLBACK_MODEL
        print(f"[Retrieval] Auto-selected embedding model: {model_name}")
    pairs = [json.loads(line) for line in Path(pairs_path).open()]
    model = SentenceTransformer(model_name)
    texts = [p["customer_text"] for p in pairs]
    # Documents get no prompt; only queries do, at search time.
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    RetrievalIndex(pairs, np.array(embeddings), model_name).save(index_path)
    print(f"Indexed {len(pairs)} pairs with {model_name} -> {index_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pairs", required=True)
    b.add_argument("--index", required=True)
    b.add_argument("--model", default=None,
                   help=f"Embedding model. Defaults to {DEFAULT_MODEL} if a GPU is "
                        f"detected, otherwise {CPU_FALLBACK_MODEL}.")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.pairs, args.index, args.model)
