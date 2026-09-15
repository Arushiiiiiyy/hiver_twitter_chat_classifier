
from __future__ import annotations

from dataclasses import dataclass

INTENTS = [
    "account_access",        # login, password, locked out, payment method rejected
    "billing_refund",        # fare disputes, double charges, cancellation fees
    "product_defect",        # app bugs, broken features, bad routing
    "order_delivery",        # ride or Uber Eats order status, driver not arriving
    "service_outage",        # app down, cannot connect, service unavailable in area
    "safety_incident",       # collision, harassment, assault, police. Never auto-drafted
    "complaint_escalation",  # dissatisfaction, including venting and explicit demands
    "general_inquiry",       # how-to, policy question, lost item, information request
    "feedback_praise",       # compliments and thanks
    "other",                 # catch-all, should stay a small share of traffic
]


_KEYWORDS: dict[str, list[str]] = {
    "account_access": ["password", "login", "log in", "locked out", "can't sign in", "2fa"],
    "billing_refund": ["refund", "charged", "charge", "billing", "payment", "money back",
                       "fare", "cancellation fee"],
    "safety_incident": ["assault", "accident", "unsafe", "harass", "police", "injur",
                        "threat", "crashed", "sexually"],
    "product_defect": ["broken", "defective", "doesn't work", "not working", "bug", "glitch"],
    "order_delivery": ["delivery", "shipped", "shipping", "package", "tracking", "order",
                       "driver not", "still waiting"],
    "service_outage": ["down", "outage", "can't connect", "not loading", "offline"],
    "complaint_escalation": ["unacceptable", "manager", "furious", "worst", "disgusted",
                             "lawsuit", "ridiculous", "pathetic", "never again"],
    "feedback_praise": ["thank you", "thanks", "great job", "love", "awesome"],
}


def keyword_baseline(text: str) -> str:
    """Trivial baseline: first keyword match wins, else 'other'. Deliberately weak, it
    is the floor the real system has to clear."""
    lowered = text.lower()
    for intent, kws in _KEYWORDS.items():
        if any(kw in lowered for kw in kws):
            return intent
    return "other"


@dataclass
class TfidfLogRegBaseline:
    """Simple baseline: TF-IDF plus logistic regression, no LLM involved. This is the
    bar an LLM system has to beat to justify its cost and latency."""

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
