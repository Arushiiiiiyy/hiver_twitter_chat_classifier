"""Decide auto-handle vs. escalate-to-human, with a stated reason.

Deliberately simple and legible (a small rule table over three signals) rather than
another learned model — for a decision that carries real cost (a bad auto-reply on a
billing dispute), an auditable rule set beats an opaque classifier, and it's something
you can actually defend line-by-line live. Document the threshold choices in
decision_log.md — they're guesses that should be tuned against your golden set.
"""
from __future__ import annotations

HIGH_RISK_INTENTS = {"billing_refund", "complaint_escalation", "account_access", "safety_incident"}
MIN_CONFIDENCE_FOR_AUTO = 0.75
MIN_GROUNDING_SIMILARITY = 0.55
# Intents that should never get an AI-drafted reply at all, regardless of confidence or
# grounding — the reply itself is the risk here, not just the auto-handle decision.
# safety_incident is the clear case: drafting any reply to a sexual-assault or accident
# report — even a good one — is a product/ethics call this system shouldn't make alone.
NEVER_AUTO_DRAFT_INTENTS = {"safety_incident"}


def decide(intent_result: dict, reply_result: dict) -> dict:
    intent = intent_result["intent"]
    confidence = intent_result.get("confidence", 0.0)
    similarity = reply_result.get("top_similarity", 0.0)

    reasons = []
    escalate = False

    if intent in HIGH_RISK_INTENTS:
        escalate = True
        reasons.append(f"high-risk intent ({intent}) always routed to a human")

    if confidence < MIN_CONFIDENCE_FOR_AUTO:
        escalate = True
        reasons.append(f"low classifier confidence ({confidence:.2f} < {MIN_CONFIDENCE_FOR_AUTO})")

    if similarity < MIN_GROUNDING_SIMILARITY:
        escalate = True
        reasons.append(
            f"no closely similar past resolution found (top similarity {similarity:.2f} "
            f"< {MIN_GROUNDING_SIMILARITY}) — reply may not be well grounded"
        )

    if not reasons:
        reasons.append("high-confidence intent, low-risk category, well-grounded reply")

    return {"escalate": escalate, "reasons": reasons}
