# Hiver SDE Intern — AI Support Agent

Built an AI customer-support agent for Uber from the [Customer Support on
Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset.
Pipeline: intent classification → grounded reply drafting (RAG over the brand's own
historical resolutions) → escalate-or-auto-handle decision with a stated reason.

## 0. Why this brand / why these choices

Uber is a large brand providing users with daily commute options. It has a huge database. Hence it is important for the company to use AI as a helping hand to reply to their customers efficiently, at the same time making sure that proper queries are escalated to humans in time. Hence building a chatbot helps in quickly resolving the queries of the customers and saves the time of the employees to focus on more important and critical queries.



## 1. Setup (~2 min)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export LLM_API_KEY=sk-...          # any OpenAI-compatible endpoint
export LLM_BASE_URL=https://api.openai.com/v1   # or your provider
export LLM_MODEL=gpt-4o-mini       # cheap model is fine, this isn't the bottleneck
```

## 2. Get the data (~2 min)

1. Download `twcs.csv` from Kaggle (link above) — requires a Kaggle account.
   (Confirmed working against the real 3M-row file: `tweet_id, author_id, inbound,
   created_at, text, response_tweet_id, in_response_to_tweet_id`, matching the schema
   assumed in `src/data_prep.py`.)
2. Place it at `data/raw/twcs.csv`.
3. Pick a brand handle present in the data (check with
   `python scripts/list_brands.py data/raw/twcs.csv | head -20`) and set it:
   ```bash
   export BRAND_HANDLE=Uber_Support
   ```


**Hardware**: retrieval uses Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B (see
`decision_log.md` #18) — both 0.6B models, ~1.2GB each in fp16, ~2.5GB VRAM combined.
This comfortably fits on a **free Colab T4**; you do not need a bigger GPU or a
dedicated cluster for this assignment. If you're on CPU only, pass
`--model sentence-transformers/all-MiniLM-L6-v2` to `retrieval.py build` and skip the
reranker (`use_reranker=False` in `reply_generator.generate_reply`).

## 3. Build the pipeline artifacts (~5 min on a subsample)

```bash
python src/data_prep.py --brand $BRAND_HANDLE --sample 20000 \
    --out data/processed/pairs.jsonl
python src/retrieval.py build --pairs data/processed/pairs.jsonl \
    --index data/processed/index.pkl
```

`--sample 20000` subsamples the raw tweets *before* pairing — the grader explicitly does
not run this on the full 3M-row dataset, and neither should you for iteration.

## 4. Build your golden set (~this is on you, budget real time for it)

```bash
python scripts/sample_for_labeling.py --pairs data/processed/pairs.jsonl \
    --n 200 --out eval/golden_set_unlabeled.jsonl
```

This stratifies the sample across rough intent clusters (via embedding k-means) plus a
slice of outliers/edge cases, so you're not just labeling 200 easy examples. Open the
output file and fill in the `gold_intent`, `gold_reply_ok` (does *some* reasonable reply
exist that a human agent would send), and `gold_escalate` fields by hand. Save as
`eval/golden_set.jsonl`. See `eval/golden_set_schema.md` for the field spec and
`eval/golden_set.example.jsonl` for 5 worked examples showing the labeling standard —
those 5 are illustrations only, they do not count toward your 150–250.

## 5. Run the agent + evaluation (~5 min)

```bash
python src/pipeline.py --golden eval/golden_set.jsonl \
    --index data/processed/index.pkl --out eval/predictions.jsonl

python eval/run_eval.py --golden eval/golden_set.jsonl \
    --predictions eval/predictions.jsonl --out eval/report.json
```

`run_eval.py` prints: intent accuracy vs. two baselines (keyword-match trivial baseline,
TF-IDF+LogReg simple baseline), retrieval precision@3, LLM-judge reply scores, and
judge-vs-human agreement (Cohen's kappa) on a 30-example subsample you label separately
in `eval/human_judge_check.jsonl`.

## Repo layout

```
src/
  data_prep.py       brand filter + thread reconstruction + cleaning
  intents.py          intent taxonomy + baselines (keyword, TF-IDF+LogReg)
  classifier.py        few-shot LLM intent classifier (the "real" system)
  retrieval.py         embedding index over historical resolved pairs
  reply_generator.py   RAG reply drafting
  escalation.py         escalation policy + reason string
  pipeline.py           orchestrates the above, CLI entrypoint
  llm_utils.py        LLM utilities for local model usage
  reranker.py   
eval/
  golden_set_schema.md
  golden_set.example.jsonl   (5 illustrative examples, NOT part of your 150-250)
  golden_set_unlabeled.jsonl
  golden_set.jsonl
  human_judge_check.jsonl
  metrics.py
  predictions.py (after running the pipeline)
  report.json
  llm_judge.py
  run_eval.py
scripts/
  list_brands.py
  sample_for_labeling.py
report/REPORT.md
decision_log.md
```
