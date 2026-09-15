"""Prints every number the report cites, straight from the eval artifacts.

Run after pipeline.py and run_eval.py so the report is never transcribed by hand.

Usage: python scripts/report_numbers.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.metrics import cohen_kappa_score


def load(p):
    return [json.loads(l) for l in Path(p).open()]


def main():
    gold = {r["pair_id"]: r for r in load("eval/golden_set.jsonl")}
    print(f"golden set n = {len(gold)}")
    print("intent distribution:", Counter(r["gold_intent"] for r in gold.values()).most_common())
    esc = sum(1 for r in gold.values() if r["gold_escalate"])
    print(f"gold escalate: {esc}/{len(gold)} ({esc/len(gold):.1%})")

    rep = Path("eval/report.json")
    if rep.exists():
        print("\n--- eval/report.json ---")
        print(json.dumps(json.loads(rep.read_text()), indent=2))
    else:
        print("\neval/report.json missing. Run eval/run_eval.py first.")

    pred_path = Path("eval/predictions.jsonl")
    if pred_path.exists():
        preds = {r["pair_id"]: r for r in load(pred_path)}
        common = [p for p in gold if p in preds]
        print(f"\npredictions n = {len(preds)}, overlapping = {len(common)}")

        leak = sum(1 for p in common if p in (preds[p].get("grounded_on") or []))
        print(f"self-retrieval leak: {leak}/{len(common)} (should be 0)")

        sims = [preds[p].get("top_similarity", 0) for p in common if preds[p].get("reply")]
        if sims:
            print(f"top_similarity mean {sum(sims)/len(sims):.3f}, max {max(sims):.3f}")

        zero = sum(1 for p in common if not preds[p].get("reply"))
        print(f"zero-draft (safety routed, no reply generated): {zero}/{len(common)}")

        print("\ntop gold -> predicted confusions:")
        conf = Counter((gold[p]["gold_intent"], preds[p]["intent"]) for p in common)
        for (g, s), n in conf.most_common(12):
            print(f"  {'OK ' if g == s else 'ERR'} {g:22} -> {s:22} {n}")

    hj = Path("eval/human_judge_check.jsonl")
    if hj.exists():
        rows = load(hj)
        j = [r["judge_overall"] for r in rows if r.get("human_overall") is not None]
        h = [r["human_overall"] for r in rows if r.get("human_overall") is not None]
        print(f"\njudge vs human n={len(j)}")
        print(f"  exact  {sum(1 for a, b in zip(j, h) if a == b)/len(j):.1%}")
        print(f"  within1 {sum(1 for a, b in zip(j, h) if abs(a-b) <= 1):.0f}/{len(j)}")
        print(f"  kappa  {cohen_kappa_score(j, h):.4f}")
        print(f"  judge mean {sum(j)/len(j):.2f}, human mean {sum(h)/len(h):.2f}")


if __name__ == "__main__":
    main()
