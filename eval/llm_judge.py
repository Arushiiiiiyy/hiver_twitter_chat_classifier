"""LLM-as-judge for reply quality, scored 1-5 on four rubric dimensions. Deliberately
uses a *different* prompt style (explicit rubric + reasoning-before-score) than the
reply generator itself to reduce the chance the judge just rewards its own generation
style. Still: validate against eval/human_judge_check.jsonl before trusting this.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from llm_utils import safe_chat_completion

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


JUDGE_PROMPT = """Rate this customer-support reply on 4 dimensions, 1 (poor) to 5
(excellent) each:
- relevance: does it actually address the customer's message?
- correctness: does it avoid inventing policies/facts not supported by context?
- tone: is it appropriate for the brand (polite, not robotic, not dismissive)?
- completeness: does it give the customer a clear next step if one is needed?

Customer message: "{message}"
Reply being rated: "{reply}"

Think briefly, then respond ONLY with JSON (no markdown fences):
{{"relevance": <1-5>, "correctness": <1-5>, "tone": <1-5>, "completeness": <1-5>,
"overall": <1-5>, "rationale": "<<=20 words>"}}"""


def judge_reply(message: str, reply: str, model: str | None = None) -> dict:
    import re
    model = model or os.environ.get("LLM_MODEL", "gemini-3.6-flash")
    prompt = JUDGE_PROMPT.format(message=message, reply=reply)
    resp = safe_chat_completion(
        _client(),
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    clean = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    clean = re.sub(r"^```\s*", "", clean)
    clean = re.sub(r"```$", "", clean).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"relevance": None, "correctness": None, "tone": None,
                 "completeness": None, "overall": None, "rationale": f"unparseable: {raw[:80]}"}
