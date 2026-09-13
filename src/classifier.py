from __future__ import annotations

import json
import os
from dotenv import load_dotenv

from openai import OpenAI

from intents import INTENTS

load_dotenv()

_CLIENT = None


def _client() -> OpenAI:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_BASE_URL"),
        )
    return _CLIENT


SYSTEM_PROMPT = f"""You classify customer-support tweets into exactly one intent from
this fixed list: {INTENTS}.

Respond ONLY with JSON, no markdown fences, no preamble:
{{"intent": "<one of the list>", "confidence": <float 0-1>, "rationale": "<<=15 words>"}}

If nothing fits well, use "other" with a low confidence rather than forcing a fit."""

# A handful of few-shot examples improves consistency a lot for near-zero cost.
# Replace/extend these with real examples from YOUR brand once you've looked at the
# data — generic examples are a starting point, not a substitute.
FEW_SHOT = [
    {"text": "I've been trying to log in for an hour and it keeps saying wrong password!!",
     "label": "account_access"},
    {"text": "you charged me twice for the same order, I want my money back", "label": "billing_refund"},
    {"text": "my package says delivered but it's not here", "label": "order_delivery"},
    {"text": "app has been down all morning, is this a known issue?", "label": "service_outage"},
    {"text": "my driver crashed and I hit my head", "label": "safety_incident"},
    {"text": "this is the third time this has happened, absolutely unacceptable service",
     "label": "complaint_escalation"},
]


def classify(text: str, model: str | None = None) -> dict:
    model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    examples_block = "\n".join(f'- "{e["text"]}" -> {e["label"]}' for e in FEW_SHOT)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nExamples:\n" + examples_block},
        {"role": "user", "content": text},
    ]
    resp = _client().chat.completions.create(model=model, messages=messages, temperature=0)
    raw = resp.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Defensive fallback — log and mark low-confidence "other" rather than crash
        # a batch eval run over one bad JSON response.
        parsed = {"intent": "other", "confidence": 0.0, "rationale": f"unparseable: {raw[:80]}"}
    if parsed.get("intent") not in INTENTS:
        parsed["intent"] = "other"
    return parsed
