"""Runs the full evaluation: intent accuracy vs. two baselines, escalation agreement,
LLM-judge reply scores, and (if eval/human_judge_check.jsonl exists) judge-vs-human
agreement.

Usage:
    python eval/run_eval.py --golden eval/golden_set.jsonl \
        --predictions eval/predictions.jsonl --out eval/report.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from hiver_agent.intents import keyword_baseline, TfidfLogRegBaseline

# Siblings in eval/ -- resolved because Python puts the script's own directory on
# sys.path. Kept here rather than in the package so the whole harness lives in eval/.
from metrics import escalation_agreement, intent_accuracy, judge_human_agreement
from llm_judge import judge_reply

load_dotenv()


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).open()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--human-judge-check", default="eval/human_judge_check.jsonl")
    ap.add_argument("--skip-judge", action="store_true",
                    help="Recompute every metric except LLM-judge reply quality, which "
                         "needs one API call per reply. Lets the harness run with no API "
                         "key or quota; judge numbers stay in the committed report.json.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    golden = {row["pair_id"]: row for row in load_jsonl(args.golden)}
    preds = {row["pair_id"]: row for row in load_jsonl(args.predictions)}
    common_ids = [pid for pid in golden if pid in preds]
    if not common_ids:
        raise SystemExit("No overlapping pair_ids between golden set and predictions.")

    gold_intents = [golden[pid]["gold_intent"] for pid in common_ids]
    gold_escalate = [bool(golden[pid]["gold_escalate"]) for pid in common_ids]
    sys_intents = [preds[pid]["intent"] for pid in common_ids]
    sys_escalate = [bool(preds[pid]["escalate"]) for pid in common_ids]
    texts = [golden[pid]["customer_text"] for pid in common_ids]

    # Baseline 1: trivial keyword matching
    kw_preds = [keyword_baseline(t) for t in texts]

    # Baseline 2: TF-IDF + LogReg, trained/evaluated with 5-fold-ish split by just
    # fitting on 70% and testing on 30% of the golden set (small-n, so treat this
    # number as directional, not a headline claim  say so in the report).
    split = int(len(texts) * 0.7)
    tfidf = TfidfLogRegBaseline()
    if split >= 5 and len(set(gold_intents[:split])) > 1:
        tfidf.fit(texts[:split], gold_intents[:split])
        tfidf_preds_test = tfidf.predict(texts[split:])
        tfidf_metrics = intent_accuracy(gold_intents[split:], tfidf_preds_test)
    else:
        tfidf_metrics = {"note": "golden set too small / too few classes for a held-out TF-IDF split"}

    report = {
        "n_examples": len(common_ids),
        "intent_accuracy": {
            "your_system": intent_accuracy(gold_intents, sys_intents),
            "baseline_keyword": intent_accuracy(gold_intents, kw_preds),
            "baseline_tfidf_logreg_heldout": tfidf_metrics,
        },
        "escalation_agreement": escalation_agreement(gold_escalate, sys_escalate),
    }

    # LLM-judge reply quality (only over examples where a reply exists).
    # One API call per reply, so --skip-judge exists for reviewers without quota.
    if args.skip_judge:
        report["reply_quality_llm_judge"] = {
            "note": "skipped via --skip-judge; see the committed eval/report.json for "
                    "these numbers, produced by a full run with API access."
        }
    else:
        judge_scores = []
        for pid in common_ids:
            reply = preds[pid].get("reply")
            if not reply:
                continue
            score = judge_reply(golden[pid]["customer_text"], reply)
            judge_scores.append(score.get("overall"))
        valid_scores = [s for s in judge_scores if s is not None]
        if valid_scores:
            report["reply_quality_llm_judge"] = {
                "mean_overall_score": sum(valid_scores) / len(valid_scores),
                "n_scored": len(valid_scores),
                "n_unparseable": len(judge_scores) - len(valid_scores),
            }

    # Judge-vs-human agreement, if the human check file exists
    check_path = Path(args.human_judge_check)
    if check_path.exists():
        check_rows = load_jsonl(str(check_path))
        judge_vals = [r["judge_overall"] for r in check_rows if r.get("human_overall") is not None]
        human_vals = [r["human_overall"] for r in check_rows if r.get("human_overall") is not None]
        if judge_vals:
            report["judge_human_agreement"] = judge_human_agreement(judge_vals, human_vals)
    else:
        report["judge_human_agreement"] = {
            "note": f"{check_path} not found  you must hand-score a ~30-example subsample "
                     "yourself and compare to the judge's scores before trusting the judge numbers above."
        }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
