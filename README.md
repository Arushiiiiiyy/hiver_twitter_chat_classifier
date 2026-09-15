# AI Support Agent for Uber_Support

An AI customer-support agent built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, scoped to a single brand: `@Uber_Support`.

The pipeline classifies an incoming message into an intent, drafts a reply grounded in how
Uber has historically resolved similar issues, and decides whether to auto-handle or
escalate to a human with a stated reason.

- Evaluation results and analysis: [`report/REPORT.md`](report/REPORT.md)
- Decisions and why they were made: [`decision_log.md`](decision_log.md)
- Golden set sampling and labelling note: [`eval/golden_set_schema.md`](eval/golden_set_schema.md)

## Why Uber_Support

Uber has 56,307 brand replies in this dataset, enough history for retrieval grounding to
have something real to draw on. It also has a property most brands here do not: a
safety-critical tail. Around 2% of inbound messages reference collisions, harassment or
police involvement. That forces the escalation policy to be an actual policy rather than a
confidence threshold, and it is the reason `safety_incident` never gets an AI-drafted
reply at all.

## Quickstart

Reproduces the headline numbers in under 15 minutes on a subsample. Every command below is
run from the repo root.

### 1. Install

```bash
git clone <repo-url>
cd hiver-support-agent
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure the LLM

This project uses a **Google AI Studio** key. Google exposes an OpenAI-compatible
endpoint, so the standard `openai` SDK is used, pointed at Google's base URL. No
Google-specific client library is needed.

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then
create a `.env` file in the repo root:

```bash
cat > .env << 'EOF'
LLM_PROVIDER=api
LLM_API_KEY=your_google_ai_studio_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-2.0-flash
BRAND_HANDLE=Uber_Support
EOF
```

Every script loads `.env` automatically via `python-dotenv`. `.env` is gitignored.

<details>
<summary>Running fully locally instead, with no API key</summary>

```bash
cat > .env << 'EOF'
LLM_PROVIDER=local
LOCAL_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
BRAND_HANDLE=Uber_Support
EOF
```

`src/llm_utils.py` loads the model through Hugging Face `transformers`. It uses float16 on
CUDA and float32 on CPU. CPU works but is roughly 5 to 10 times slower.
</details>

### 3. Get the data

`twcs.csv` is 493 MB and is not in this repo.

```bash
mkdir -p data/raw
# Download twcs.csv from the Kaggle link above and place it at data/raw/twcs.csv
```

Confirm the brand handle is present:

```bash
python scripts/list_brands.py data/raw/twcs.csv | head -20
```

`Uber_Support` should appear in the list.

### 4. Build the pairs and the retrieval index

```bash
python src/data_prep.py --brand Uber_Support --sample 30000 --out data/processed/pairs.jsonl
python src/retrieval.py build --pairs data/processed/pairs.jsonl --index data/processed/index.pkl
```

`--sample 30000` subsamples the raw CSV before pairing. The subsample keeps all rows
belonging to the target brand, so a brand with lower volume does not vanish.

`retrieval.py` picks `Qwen/Qwen3-Embedding-0.6B` when a GPU is visible and falls back to
`all-MiniLM-L6-v2` on CPU automatically.

### 5. Run the agent over the golden set

```bash
python src/pipeline.py \
    --golden eval/golden_set.jsonl \
    --index data/processed/index.pkl \
    --out eval/predictions.jsonl \
    --no-rerank
```

Drop `--no-rerank` to enable the Qwen3 cross-encoder reranker. It improves retrieval
ordering but is slow on CPU.

The run is resumable. If it stops partway, rerun the same command and it continues from
where it left off.

Each row's own pair is excluded from its retrieval results. Golden-set messages are also
in the index, so without that exclusion the top hit is the message itself at similarity
1.0 and the generator sees the ground-truth reply it is meant to produce. `--allow-self-retrieval`
disables the exclusion, but only for deliberately measuring that leak.

### 6. Evaluate

```bash
python eval/run_eval.py \
    --golden eval/golden_set.jsonl \
    --predictions eval/predictions.jsonl \
    --out eval/report.json
```

Prints and writes: intent accuracy and macro-F1 for the system and both baselines,
escalation accuracy and Cohen's kappa, LLM-judge reply scores, and judge-vs-human
agreement measured against the 30 replies hand-scored in `eval/human_judge_check.jsonl`.

### Rebuilding the golden set from scratch (optional)

The labelled set is already committed. To regenerate the unlabelled sample:

```bash
python scripts/sample_for_labeling.py \
    --pairs data/processed/pairs.jsonl \
    --n 200 \
    --out eval/golden_set_unlabeled.jsonl
```

Labels then have to be filled in by hand. See `eval/golden_set_schema.md`.

## Hardware

The embedding model (0.6B) and reranker (0.6B) need about 2.5 GB of VRAM combined, which
fits a free Colab T4 with room to spare. Generation goes through the API, so it costs no
local GPU. Everything also runs on CPU with the automatic fallback, just slower.

On a shared cluster, point the Hugging Face cache at scratch to avoid home quota errors:

```bash
export HF_HOME=/scratch/$USER/huggingface
```

## Repo layout

```
src/
  data_prep.py         brand filter, thread reconstruction, cleaning
  intents.py           intent taxonomy, keyword and TF-IDF baselines
  classifier.py        few-shot LLM classifier, safety pre-filter, "other" reconsideration
  retrieval.py         embedding index over historical resolved pairs
  reranker.py          Qwen3 cross-encoder reranker, optional
  reply_generator.py   RAG reply drafting
  escalation.py        escalation rules and stated reasons
  pipeline.py          orchestration, CLI entrypoint
  llm_utils.py         API calls with retry, plus local GPU inference path
eval/
  golden_set.jsonl           188 hand-labelled examples
  golden_set_unlabeled.jsonl the sample before labelling
  golden_set_schema.md       sampling and labelling note
  human_judge_check.jsonl    30 replies scored by both the judge and a human
  metrics.py                 accuracy, macro-F1, Cohen's kappa
  llm_judge.py               4-dimension reply quality rubric
  run_eval.py                evaluation entrypoint
  predictions.jsonl          generated by pipeline.py
  report.json                generated by run_eval.py
scripts/
  list_brands.py             list brand handles by volume
  sample_for_labeling.py     stratified sampler for the golden set
report/REPORT.md
decision_log.md
```
