"""Intent taxonomy + two baselines to compare the real system against.

The taxonomy below is a reasonable *starting point* covering intents that show up
across most brands in this dataset (billing, outage, delivery, etc.) — but the
assignment wants intents defined *from your data*, so treat this as a draft: run
scripts/sample_for_labeling.py, skim ~50 examples, and adjust names/boundaries/add an
"OTHER" catch-all threshold before finalizing. Document any change in decision_log.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

INTENTS = [
    "account_access",       # login, password, locked out, 2FA
    "billing_refund",       # charges, refunds, payment disputes
    "product_defect",       # broken/faulty product or bug in the app/service
    "order_delivery",       # shipping status, late/lost package
    "service_outage",       # can't connect, app/site down, service disruption
    "safety_incident",      # accidents, harassment, assault, police involvement — never auto-drafted
    "complaint_escalation", # explicit anger, "unacceptable", demands a manager
    "general_inquiry",      # how-to, feature question, information request
    "feedback_praise",      # compliments, thanks, unsolicited feedback
    "feedback_negative",    # negative feedback
    "other",                # catch-all — should stay a small % if taxonomy is good
]

# --- Baseline 1 (trivial): keyword matching -------------------------------------
_KEYWORDS: dict[str, list[str]] = {
    "account_access": ["password", "login", "log in", "locked out", "can't sign in", "2fa"],
    "billing_refund": ["refund", "charged", "charge", "billing", "payment", "money back"],
    # safety_incident checked before product_defect: for a rideshare-style brand,
    # "crash"/"crashed" almost always means a vehicle collision, not an app crash.
    # Reorder or adjust per-brand — this is exactly the kind of taxonomy detail that
    # needs eyeballing on real data, not a one-size-fits-all default.
    "safety_incident": ["assault", "accident", "unsafe", "harass", "police", "injur", "threat",
                         "crashed", "sexually"],
    "product_defect": ["broken", "defective", "doesn't work", "not working", "bug", "glitch"],
    "order_delivery": ["delivery", "shipped", "shipping", "package", "tracking", "order"],
    "service_outage": ["down", "outage", "can't connect", "not loading", "offline"],
    "complaint_escalation": ["unacceptable", "manager", "furious", "worst", "disgusted", "lawsuit"],
    "feedback_praise": ["thank you", "thanks", "great job", "love", "awesome"],
    "feedback_negative": ["worst", "hate this", "so done", "never again", "ridiculous",
                           "pathetic", "sick of"]
}


def keyword_baseline(text: str) -> str:
    """Trivial baseline: first keyword match wins, else 'other'. This is deliberately
    dumb — it exists as the floor your real system must clear, not a serious attempt."""
    lowered = text.lower()
    for intent, kws in _KEYWORDS.items():
        if any(kw in lowered for kw in kws):
            return intent
    return "other"


@dataclass
class TfidfLogRegBaseline:
    """Simple baseline #2: TF-IDF + logistic regression, trained on your labeled
    (or a bootstrap-labeled) set. Meaningfully better than keyword matching without
    touching an LLM at all — this is the bar an LLM-based system needs to clear to
    justify its cost/latency."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> None:
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self.fitted = True

    def predict(self, texts: list[str]) -> list[str]:
        if not self.fitted:
            raise RuntimeError("Call .fit() before .predict()")
        X = self.vectorizer.transform(texts)
        return list(self.clf.predict(X))
