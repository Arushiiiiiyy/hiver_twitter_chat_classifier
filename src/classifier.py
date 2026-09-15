from __future__ import annotations

import json
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

try:
    from intents import INTENTS
    from llm_utils import safe_chat_completion
except ImportError:
    from src.intents import INTENTS
    from src.llm_utils import safe_chat_completion

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
    {"text": "worst app I've ever used, so done with this company", "label": "feedback_negative"},
    {"text": "my driver crashed and I hit my head", "label": "safety_incident"},
    {"text": "this is the third time this has happened, absolutely unacceptable service",
     "label": "complaint_escalation"},
    {"text": "driver was threatening, I contacted police and emergency services", "label": "safety_incident"},

]


def classify(text: str, model: str | None = None) -> dict:
    lowered = text.lower()
    if any(w in lowered for w in ["police", "emergency", "assault", "911", "ambulance"]):
        return {
            "intent": "safety_incident",
            "confidence": 1.0,
            "rationale": "safety emergency rule match",
        }
    model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    examples_block = "\n".join(f'- "{e["text"]}" -> {e["label"]}' for e in FEW_SHOT)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nExamples:\n" + examples_block},
        {"role": "user", "content": text},
    ]
    resp = safe_chat_completion(_client(), model=model, messages=messages, temperature=0)
    raw = resp.choices[0].message.content.strip()
    clean = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    clean = re.sub(r"^```\s*", "", clean)
    clean = re.sub(r"```$", "", clean).strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        # Defensive fallback — log and mark low-confidence "other" rather than crash
        # a batch eval run over one bad JSON response.
        parsed = {"intent": "other", "confidence": 0.0, "rationale": f"unparseable: {raw[:80]}"}
    if parsed.get("intent") not in INTENTS:
        parsed["intent"] = "other"
    return parsed


def classify_with_keyword_fallback(text: str, model: str | None = None) -> dict:
    """Runs the trivial keyword baseline first; only calls the LLM when the keyword
    baseline can't place the message at all. If the LLM also can't
    confidently place it, it stays 'other'.
    """
    try:
        from intents import keyword_baseline
    except ImportError:
        from src.intents import keyword_baseline

    kw_result = keyword_baseline(text)
    if kw_result != "other":
        return {
            "intent": kw_result,
            "confidence": None,
            "rationale": "keyword match",
            "source": "keyword",
        }

    llm_result = classify(text, model=model)
    llm_result["source"] = "llm_fallback"
    return llm_result

