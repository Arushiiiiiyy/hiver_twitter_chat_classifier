"""Draft a reply grounded in how this brand historically resolved similar issues,
using the retrieval index built in retrieval.py.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import OpenAI

from llm_utils import safe_chat_completion
from reranker import rerank as rerank_candidates
from retrieval import RetrievalIndex

load_dotenv()

_CLIENT = None


def _client() -> OpenAI | None:
    global _CLIENT
    if os.environ.get("LLM_PROVIDER", "").lower() == "local":
        return None
    if _CLIENT is None:
        _CLIENT = OpenAI(
            api_key=os.environ.get("LLM_API_KEY", "dummy"),
            base_url=os.environ.get("LLM_BASE_URL"),
        )
    return _CLIENT


PROMPT_TEMPLATE = """You are a customer-support agent for this brand. Draft a reply to
the customer's message below. Match the brand's tone and typical resolution pattern
shown in the past examples — don't invent policies (refund amounts, timelines) that
aren't grounded in those examples or the customer's message itself. Keep it to 1-3
sentences, Twitter-reply length.

Past similar cases this brand has resolved:
{examples}

Customer's message now: "{message}"

Reply:"""


def generate_reply(message: str, index: RetrievalIndex, k: int = 3,
                    model: str | None = None, use_reranker: bool = True,
                    retrieve_n: int = 20) -> dict:
    """Two-stage retrieval when use_reranker=True: embedding search pulls `retrieve_n`
    candidates cheaply, the cross-encoder reranks them and keeps the top `k` for the
    prompt. Set use_reranker=False to fall back to embedding-only ranking — useful for
    the "retrieval only" vs. "retrieval + rerank" ablation in the report.
    """
    model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")

    if use_reranker:
        candidates = index.query(message, k=retrieve_n)
        retrieved = rerank_candidates(message, candidates, top_k=k)
        ranking_method = "embedding+rerank"
    else:
        retrieved = index.query(message, k=k)
        ranking_method = "embedding_only"

    examples_block = "\n".join(
        f'- Customer: "{r["customer_text"]}" -> Brand: "{r["brand_reply"]}"'
        for r in retrieved
    )
    prompt = PROMPT_TEMPLATE.format(examples=examples_block, message=message)
    resp = safe_chat_completion(
        _client(),
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return {
        "reply": resp.choices[0].message.content.strip(),
        "grounded_on": [r["pair_id"] for r in retrieved],
        "top_similarity": retrieved[0]["similarity"] if retrieved else 0.0,
        "top_rerank_score": retrieved[0].get("rerank_score") if retrieved else None,
        "ranking_method": ranking_method,
    }
