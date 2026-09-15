"""End-to-end pipeline: classify intent, retrieve grounding examples, draft a reply,
then decide auto-handle vs escalate.

Run against the golden set for evaluation, or against arbitrary input for a demo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Keep src/ importable regardless of the working directory the script is run from.
sys.path.insert(0, str(Path(__file__).parent))

from classifier import classify
from escalation import decide, NEVER_AUTO_DRAFT_INTENTS
from reply_generator import generate_reply
from retrieval import RetrievalIndex

load_dotenv()


def run_one(message: str, index: RetrievalIndex, use_reranker: bool = True,
            exclude_pair_id: str | None = None) -> dict:
    intent_result = classify(message)

    if intent_result["intent"] in NEVER_AUTO_DRAFT_INTENTS:
        # No draft at all for these. The risk is in AI-drafting a reply to something
        # like an assault report, not just in whether it gets auto-sent.
        reply_result = {"reply": None, "grounded_on": [], "top_similarity": 0.0,
                        "ranking_method": "skipped"}
        escalation_result = {
            "escalate": True,
            "reasons": [f"intent '{intent_result['intent']}' is never auto-drafted, "
                        f"routed directly to a human agent"],
        }
    else:
        reply_result = generate_reply(message, index, use_reranker=use_reranker,
                                      exclude_pair_id=exclude_pair_id)
        escalation_result = decide(intent_result, reply_result)

    return {
        "message": message,
        "intent": intent_result["intent"],
        "intent_confidence": intent_result.get("confidence"),
        "intent_rationale": intent_result.get("rationale"),
        "reply": reply_result["reply"],
        "grounded_on": reply_result["grounded_on"],
        "top_similarity": reply_result["top_similarity"],
        "ranking_method": reply_result.get("ranking_method"),
        "escalate": escalation_result["escalate"],
        "escalation_reasons": escalation_result["reasons"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="JSONL with a 'customer_text' field per line")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-rerank", action="store_true",
                    help="Skip cross-encoder reranking (much faster on CPU)")
    ap.add_argument("--allow-self-retrieval", action="store_true",
                    help="Do not exclude a row's own pair from its retrieval results. "
                         "Off by default: leaving it on leaks the ground-truth reply.")
    args = ap.parse_args()

    index = RetrievalIndex.load(args.index)
    rows = [json.loads(line) for line in Path(args.golden).open()]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support, so a rate-limited or interrupted run can be restarted.
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
            result = run_one(
                row["customer_text"],
                index,
                use_reranker=not args.no_rerank,
                exclude_pair_id=None if args.allow_self_retrieval else pid,
            )
            result["pair_id"] = pid
            f.write(json.dumps(result) + "\n")
            f.flush()

    print(f"Finished. Output written to {out_path}")


if __name__ == "__main__":
    main()
