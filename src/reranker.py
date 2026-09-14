"""Second-stage reranking over the embedding index's top-N candidates.

Retrieval (retrieval.py) is fast but approximate — it ranks by embedding cosine
similarity alone. A cross-encoder reranker looks at the (query, candidate) pair
jointly and is typically meaningfully more accurate at the top of the ranking, at the
cost of being too slow to run over the whole index (hence the two-stage design: cheap
embedding search narrows N=20 candidates down, the reranker only has to score those 20).

This is also a genuine ablation for the report: "retrieval only" vs. "retrieval +
rerank" is a real, defensible comparison — not just architectural decoration. See
report/REPORT.md §2.

Requires transformers>=4.51.0 and a recent sentence-transformers with generative
CrossEncoder support for Qwen3-Reranker (pip install -U sentence-transformers if you
hit a KeyError: 'qwen3' or the CrossEncoder constructor rejects the model).
"""
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
    """Re-scores and re-sorts `candidates` (dicts with a 'customer_text' field, as
    returned by RetrievalIndex.query) using the cross-encoder, returning the top_k.

    Keeps the original embedding `similarity` alongside the new `rerank_score` on each
    candidate so you can compare the two rankings directly in your eval/report.
    """
    if not candidates:
        return []
    model = _get_model(model_name)
    pairs = [(query, c["customer_text"]) for c in candidates]
    scores = model.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_k]
