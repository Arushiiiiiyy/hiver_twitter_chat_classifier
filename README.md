# AI Support Agent for Uber_Support

An AI customer-support agent built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, scoped to a single brand: `@Uber_Support`.

The pipeline classifies an incoming message into an intent, drafts a reply grounded in how
Uber has historically resolved similar issues, and decides whether to auto-handle or
escalate to a human with a stated reason.

- Full analysis and failure modes: [`report.pdf`](report.pdf)
- Decisions and why they were made: [`decision_log.md`](decision_log.md)
- Golden set sampling and labelling note: [`eval/golden_set_schema.md`](eval/golden_set_schema.md)

## Results

Measured on 188 hand-labelled examples in `eval/golden_set.jsonl`.

**Intent classification**

| System | Accuracy | Macro-F1 |
|---|---|---|
| Few-shot LLM classifier (this system) | 36.2% | 0.316 |
| Baseline 2 — TF-IDF + LogReg, held-out | 35.1% | 0.206 |
| Baseline 1 — keyword matching (trivial) | 24.5% | 0.236 |

**Escalation decision** — 59.6% accuracy, Cohen's κ 0.191 against human labels.

**Reply quality** — LLM judge mean 2.99 / 5 across 179 drafted replies, 0 unparseable.

**Judge trustworthiness** — against 30 replies scored by hand: 66.7% exact agreement,
100% within one point, Cohen's κ 0.513.

On accuracy the system is barely ahead of the TF-IDF baseline. The macro-F1 gap is the
more meaningful one, and the TF-IDF number comes from a 70/30 split of 188 examples, so
it is directional rather than a firm claim. `report.pdf` covers what these numbers hide,
including a mandatory section on what is misleading about the headline figure.

## Reproduce the results (under 15 minutes, no API key, no dataset download)

Every number in the table above is recomputed from committed artifacts. No Kaggle
download, no GPU, no API quota.

```bash
git clone <repo-url>
cd hiver_twitter_chat_classifier

python3 -m venv venv
source venv/bin/activate          # Windows PowerShell: venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .

python eval/run_eval.py \
    --golden eval/golden_set.jsonl \
    --predictions eval/predictions.jsonl \
    --skip-judge \
    --out /tmp/verify_report.json

python scripts/report_numbers.py
```

The install is about 25 seconds — the eval path needs only numpy, scikit-learn, openai and
python-dotenv. PyTorch and sentence-transformers are in the optional `full` extra, needed
only to rebuild the index from raw data.

`--skip-judge` skips LLM-judge reply scoring, which costs one API call per reply. Every
other metric — intent accuracy, macro-F1, escalation agreement, Cohen's κ, judge-vs-human
agreement — is recomputed from scratch. The judge numbers are in the committed
`eval/report.json`, produced by a full run with API access.

Drop `--skip-judge` if you have a key with quota and want to regenerate the judge scores
too. Note that Google's free tier allows 20 requests per day for `gemini-3.6-flash`, which
is not enough for the ~179 calls this makes.

<details>
<summary>Per-platform setup details</summary>

**macOS** — `brew install python@3.12 git`, or the installer from
[python.org](https://www.python.org/downloads/).

**Ubuntu** — `sudo apt update && sudo apt install -y python3 python3-venv python3-pip git`.
`python3-venv` is a separate package; without it `python3 -m venv` fails with
`ensurepip is not available`.

**Windows** — install Python from [python.org](https://www.python.org/downloads/) with
**"Add python.exe to PATH"** ticked, plus
[Git for Windows](https://git-scm.com/download/win). Activate with
`venv\Scripts\Activate.ps1` in PowerShell or `venv\Scripts\activate.bat` in `cmd.exe`. If
PowerShell blocks activation, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

Confirm the right interpreter is active before continuing:

```bash
python -c "import sys; print(sys.executable)"
```

The printed path must be inside your project folder.
</details>

## Rebuilding from raw data (optional, much slower)

Needed only to regenerate `pairs.jsonl`, the retrieval index and `predictions.jsonl`. This
requires the 493 MB dataset and either a paid API key or a local GPU. Budget an hour, not
15 minutes.

### 1. Install the full dependency set

```bash
pip install -e ".[full]"
```

Adds pandas, torch, transformers and sentence-transformers. On Linux without an NVIDIA
GPU, install the CPU-only torch build first to avoid an 800 MB CUDA download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[full]"
```

### 2. Configure the LLM

This project uses a **Google AI Studio** key. Google exposes an OpenAI-compatible
endpoint, so the standard `openai` SDK is used, pointed at Google's base URL. No
Google-specific client library is needed.

Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then create
`.env` in the repo root:

```bash
cat > .env << 'EOF'
LLM_PROVIDER=api
LLM_API_KEY=your_google_ai_studio_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.6-flash
BRAND_HANDLE=Uber_Support
EOF
```

On Windows PowerShell, use a here-string instead — `@'` and `'@` must each sit alone at
the start of their line:

```powershell
@'
LLM_PROVIDER=api
LLM_API_KEY=your_google_ai_studio_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.6-flash
BRAND_HANDLE=Uber_Support
'@ | Set-Content -Path .env -Encoding utf8
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

`src/hiver_agent/llm_utils.py` loads the model through Hugging Face `transformers`, using
float16 on CUDA and float32 on CPU. This sidesteps API quotas entirely, but downloads
roughly 3 GB of weights on first run and is 5 to 10 times slower on CPU.
</details>

### 3. Get the data

Download `twcs.csv` (493 MB) from the
[Kaggle dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
and place it at `data/raw/twcs.csv`.

```bash
mkdir -p data/raw
python scripts/list_brands.py data/raw/twcs.csv | head -20
```

`Uber_Support` should appear in the list. On Windows PowerShell, use
`New-Item -ItemType Directory -Force -Path data\raw` and `Select-Object -First 20`.

### 4. Build the pairs and the retrieval index

```bash
python -m hiver_agent.data_prep --brand Uber_Support --sample 30000 --out data/processed/pairs.jsonl
python -m hiver_agent.retrieval build --pairs data/processed/pairs.jsonl --index data/processed/index.pkl
```

`--sample 30000` subsamples the raw CSV before pairing, keeping all rows belonging to the
target brand so a lower-volume brand does not vanish.

**The embedding model depends on your hardware.** `retrieval.py` picks
`Qwen/Qwen3-Embedding-0.6B` when a GPU is visible and falls back to `all-MiniLM-L6-v2` on
CPU. The committed results were produced with the CPU fallback, so rebuilding on a GPU
gives a different index and different numbers. Pass `--model-name all-MiniLM-L6-v2`
explicitly to match.

### 5. Run the agent over the golden set

One API call per example, 188 total.

```bash
python -m hiver_agent.pipeline --golden eval/golden_set.jsonl --index data/processed/index.pkl --out eval/predictions.jsonl --no-rerank
```

Written on one line so it works in every shell. On macOS and Ubuntu you can split it with
a trailing `\`; in PowerShell use a trailing backtick `` ` ``.

The run is resumable — including after a quota limit. Rerun the same command and it
continues where it left off.

Drop `--no-rerank` to enable the Qwen3 cross-encoder reranker. It improves retrieval
ordering but is slow on CPU.

Each row's own pair is excluded from its retrieval results. Golden-set messages are also
in the index, so without that exclusion the top hit is the message itself at similarity
1.0 and the generator sees the ground-truth reply it is meant to produce.
`--allow-self-retrieval` disables the exclusion, but only for deliberately measuring that
leak.

### 6. Re-evaluate

```bash
python eval/run_eval.py --golden eval/golden_set.jsonl --predictions eval/predictions.jsonl --out eval/report.json
```

Without `--skip-judge` this also regenerates LLM-judge reply scores, roughly 179
additional API calls.

### Rebuilding the golden set from scratch (optional)

The labelled set is already committed. To regenerate the unlabelled sample:

```bash
python scripts/sample_for_labeling.py --pairs data/processed/pairs.jsonl --n 200 --out eval/golden_set_unlabeled.jsonl
```

Labels then have to be filled in by hand. See `eval/golden_set_schema.md`.

## Why Uber_Support

Uber has 56,307 brand replies in this dataset, enough history for retrieval grounding to
have something real to draw on. It also has a property most brands here do not: a
safety-critical tail. Around 2% of inbound messages reference collisions, harassment or
police involvement. That forces the escalation policy to be an actual policy rather than a
confidence threshold, and it is the reason `safety_incident` never gets an AI-drafted
reply at all.

## Troubleshooting

**`ModuleNotFoundError: No module named 'hiver_agent'`** — the editable install did not
run, or you are in a different environment than the one you installed into. Reactivate the
venv and rerun `pip install -e .`.

**`ensurepip is not available` on Ubuntu** — install `python3-venv`, delete the
half-created `venv/` folder, and create it again.

**`running scripts is disabled on this system` on Windows** — see the execution policy
note in the setup details above.

**`RESOURCE_EXHAUSTED` / HTTP 429** — Google free-tier daily quota. Use `--skip-judge`, or
switch `LLM_MODEL` to a different Gemini model for a separate quota bucket.

**Editor shows unresolved imports but the code runs** — point your editor at
`venv/bin/python` (`venv\Scripts\python.exe` on Windows) rather than a system Python.

## Hardware

The embedding model (0.6B) and reranker (0.6B) need about 2.5 GB of VRAM combined, which
fits a free Colab T4 with room to spare. Generation goes through the API, so it costs no
local GPU. Everything also runs on CPU with the automatic fallback, just slower.

On a shared cluster, point the Hugging Face cache at scratch to avoid home quota errors:

```bash
export HF_HOME=/scratch/$USER/huggingface
export PIP_CACHE_DIR=/scratch/$USER/pip
```

## Repo layout

```
src/
  hiver_agent/
    __init__.py          package marker and version
    data_prep.py         brand filter, thread reconstruction, cleaning
    intents.py           intent taxonomy, keyword and TF-IDF baselines
    classifier.py        few-shot LLM classifier, safety pre-filter, "other" reconsideration
    retrieval.py         embedding index over historical resolved pairs
    reranker.py          Qwen3 cross-encoder reranker, optional
    reply_generator.py   RAG reply drafting
    escalation.py        escalation rules and stated reasons
    pipeline.py          orchestration, CLI entrypoint
    llm_utils.py         API calls with retry, plus local inference path
eval/
  golden_set.jsonl           188 hand-labelled examples
  golden_set_unlabeled.jsonl the sample before labelling
  golden_set_schema.md       sampling and labelling note
  human_judge_check.jsonl    30 replies scored by both the judge and a human
  metrics.py                 accuracy, macro-F1, Cohen's kappa
  llm_judge.py               4-dimension reply quality rubric
  run_eval.py                evaluation entrypoint
  predictions.jsonl          committed, so the eval runs without an API key
  report.json                committed headline metrics
  stale_pre_fix/             superseded results from before the self-retrieval fix
scripts/
  list_brands.py             list brand handles by volume
  sample_for_labeling.py     stratified sampler for the golden set
  report_numbers.py          prints every number cited in the report
pyproject.toml               package metadata and dependency extras
pyrefly.toml                 type-checker config
requirements.txt             eval-path dependencies (light)
requirements-full.txt        rebuild dependencies (torch, transformers, pandas)
report.pdf
decision_log.md
```
