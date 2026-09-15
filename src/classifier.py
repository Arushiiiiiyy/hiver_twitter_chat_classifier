"""Few-shot LLM intent classifier. This is the system the two baselines in intents.py
are compared against.
"""
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

# Hard rule ahead of the LLM. If a message names an emergency explicitly we route it
# without waiting on a model call that could return something softer.
_SAFETY_TRIGGERS = ["police", "emergency", "assault", "911", "ambulance"]


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

# Second-pass prompt, only used when the first pass returns "other". Keeps the
# catch-all bucket from absorbing messages that do have a usable category.
RECONSIDER_PROMPT = f"""That message was tentatively marked "other". Look again before
that is final. A message does not need to match a category perfectly to be better
served by it than by "other". Reconsider against: {INTENTS}.

If a genuine fit exists, pick it. If nothing fits, keep "other".

Respond ONLY with JSON, no markdown fences:
{{"intent": "<one of the list>", "confidence": <float 0-1>, "rationale": "<<=15 words>"}}"""

FEW_SHOT = [
    {"text": "I've been trying to log in for an hour and it keeps saying wrong password!!",
     "label": "account_access"},
    {"text": "you charged me twice for the same ride, I want my money back",
     "label": "billing_refund"},
    {"text": "driver still hasn't arrived and the app says he's 2 mins away for 20 mins",
     "label": "order_delivery"},
    {"text": "app has been down all morning, is this a known issue?", "label": "service_outage"},
    {"text": "my driver crashed and I hit my head", "label": "safety_incident"},
    {"text": "driver was threatening, I contacted police and emergency services",
     "label": "safety_incident"},
    {"text": "left my water bottle in the car, how do I get it back?", "label": "general_inquiry"},
    {"text": "this is the third time this has happened, absolutely unacceptable service",
     "label": "complaint_escalation"},
]


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences the model adds despite instructions, then parse."""
    clean = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    clean = re.sub(r"^```\s*", "", clean)
    clean = re.sub(r"```$", "", clean).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # One bad response shouldn't kill a 188-row batch run.
        return {"intent": "other", "confidence": 0.0, "rationale": f"unparseable: {raw[:80]}"}


def classify(text: str, model: str | None = None, reconsider_other: bool = True) -> dict:
    """reconsider_other sends one extra call when the first pass lands on "other",
    giving the model a second look before the message is written off."""
    lowered = text.lower()
    if any(w in lowered for w in _SAFETY_TRIGGERS):
        return {
            "intent": "safety_incident",
            "confidence": 1.0,
            "rationale": "safety emergency rule match",
            "source": "safety_rule",
        }

    model = model or os.environ.get("LLM_MODEL", "gemini-3.6-flash")
    examples_block = "\n".join(f'- "{e["text"]}" -> {e["label"]}' for e in FEW_SHOT)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\nExamples:\n" + examples_block},
        {"role": "user", "content": text},
    ]
    resp = safe_chat_completion(_client(), model=model, messages=messages, temperature=0)
    raw = resp.choices[0].message.content.strip()
    parsed = _parse_json_response(raw)

    if reconsider_other and parsed.get("intent") == "other":
        second_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": RECONSIDER_PROMPT},
        ]
        resp2 = safe_chat_completion(_client(), model=model, messages=second_messages,
                                     temperature=0)
        parsed2 = _parse_json_response(resp2.choices[0].message.content.strip())
        if parsed2.get("intent") in INTENTS:
            parsed2["reconsidered"] = True
            parsed = parsed2

    if parsed.get("intent") not in INTENTS:
        parsed["intent"] = "other"
    return parsed


def classify_with_keyword_fallback(text: str, model: str | None = None) -> dict:
    """Keyword baseline first, LLM only when the keyword pass cannot place the message.
    Used in the report to measure how many queries the LLM layer rescues from 'other'."""
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
