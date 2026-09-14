# Report — AI Support Agent for [BRAND]

*Fill in every [bracketed] section using your actual eval run output
(`eval/report.json`) and real examples from `eval/predictions.jsonl`. This skeleton
only has the required structure — none of the content.*

## 1. Problem framing

- What does "good" mean for this brand specifically? [e.g., for a billing-heavy brand,
  correctness on money-related claims matters more than reply fluency]
- What did you choose NOT to build, and why? [e.g., no multi-turn context beyond the
  first reply; no handling of non-English tweets; no fine-tuning, only few-shot]

## 2. Results vs. baselines

[Pull from `eval/report.json` -> `intent_accuracy`. Present as a small table:
your_system vs. baseline_keyword vs. baseline_tfidf_logreg_heldout, with n and macro-F1
not just accuracy — accuracy alone is misleading on an imbalanced intent distribution.]

## 3. Failure analysis — top 5 failure modes

For each: 1-2 real examples (pair_id + text), your hypothesis for *why* it fails, and
whether it's a classifier failure, a retrieval/grounding failure, or an escalation
policy failure.

1. [failure mode] — example: [pair_id]. Hypothesis: [...]
2. ...
3. ...
4. ...
5. ...

## 4. What is misleading about my headline number?

*(Mandatory — do not skip.)* Be specific and self-critical. Candidates worth
considering honestly:
- Your golden set size (150-250) has wide confidence intervals — quote them.
- The TF-IDF baseline split is small-n and directional only (see run_eval.py note).
- The LLM judge may be biased toward the reply generator's own style/verbosity —
  report the judge-human kappa from `eval/human_judge_check.jsonl` and be honest if
  it's mediocre.
- Escalation "accuracy" is meaningless if the base rate of escalation is skewed — you
  reported Cohen's kappa for this reason; discuss what it actually shows.
- Retrieval grounding quality was only sanity-checked, not rigorously validated against
  a separate faithfulness metric.

## 5. What you'd do next with one more week

[e.g., multi-turn context handling, active-learning loop on judge disagreements,
per-intent escalation threshold tuning against golden-set base rates, testing a second
brand to check taxonomy generalizes]
