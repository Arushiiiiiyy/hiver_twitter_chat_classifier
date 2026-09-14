"""End-to-end pipeline: for each customer message, classify intent, retrieve grounding
examples, draft a reply, and decide auto-handle vs. escalate.

Run against the golden set for evaluation, or against arbitrary input for a demo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Ensure src/ is in sys.path when running from any working directory
sys.path.insert(0, str(Path(__file__).parent))

from classifier import classify
from escalation import decide, NEVER_AUTO_DRAFT_INTENTS
from reply_generator import generate_reply
from retrieval import RetrievalIndex

load_dotenv()


def run_one(message: str, index: RetrievalIndex, use_reranker: bool = True) -> dict:
    intent_result = classify(message)

    if intent_result["intent"] in NEVER_AUTO_DRAFT_INTENTS:
        # Route directly to a human without generating any draft reply — for an intent
        # like safety_incident, the risk is in the act of AI-drafting a response at
        # all, not just in whether it gets auto-sent. See decision_log.md.
        reply_result = {"reply": None, "grounded_on": [], "top_similarity": 0.0}
        escalation_result = {
            "escalate": True,
            "reasons": [f"intent '{intent_result['intent']}' is never auto-drafted — "
                        f"routed directly to a human agent"],
        }
    else:
        reply_result = generate_reply(message, index, use_reranker=use_reranker)
        escalation_result = decide(intent_result, reply_result)

    return {
        "message": message,
        "intent": intent_result["intent"],
        "intent_confidence": intent_result.get("confidence"),
        "intent_rationale": intent_result.get("rationale"),
        "reply": reply_result["reply"],
        "grounded_on": reply_result["grounded_on"],
        "top_similarity": reply_result["top_similarity"],
        "escalate": escalation_result["escalate"],
        "escalation_reasons": escalation_result["reasons"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="JSONL with a 'customer_text' field per line")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-rerank", action="store_true",
                    help="Skip cross-encoder reranking (recommended on CPU for 100x faster speed)")
    args = ap.parse_args()

    index = RetrievalIndex.load(args.index)
    rows = [json.loads(line) for line in Path(args.golden).open()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for existing completed predictions to resume seamlessly
    processed_ids = set()
    if out_path.exists():
        for line in out_path.open():
            line = line.strip()
            if not line:
                continue
            try:
                processed_ids.add(json.loads(line).get("pair_id"))
            except Exception:
                pass
        if processed_ids:
            print(f"[pipeline] Resuming: found {len(processed_ids)} already completed items.")

    with out_path.open("a" if processed_ids else "w") as f:
        for row in tqdm(rows, desc="Running pipeline"):
            pid = row.get("pair_id")
            if pid and pid in processed_ids:
                continue
            result = run_one(row["customer_text"], index, use_reranker=not args.no_rerank)
            result["pair_id"] = pid
            f.write(json.dumps(result) + "\n")
            f.flush()

    print(f"Finished! Output written to {out_path}")


if __name__ == "__main__":
    main()
