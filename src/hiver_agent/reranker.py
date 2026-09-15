
from __future__ import annotations

from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"

_MODEL_CACHE: dict[str, CrossEncoder] = {}


def _get_model(model_name: str) -> CrossEncoder:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = CrossEncoder(model_name)
    return _MODEL_CACHE[model_name]


def rerank(query: str, candidates: list[dict], top_k: int = 5,
           model_name: str = DEFAULT_MODEL) -> list[dict]:
    
    if not candidates:
        return []
    model = _get_model(model_name)
    pairs = [(query, c["customer_text"]) for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_k]
