# AI Support Agent for Uber_Support

An AI customer-support agent built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset, scoped to a single brand: `@Uber_Support`.

The pipeline classifies an incoming message into an intent, drafts a reply grounded in how
Uber has historically resolved similar issues, and decides whether to auto-handle or
escalate to a human with a stated reason.

- Evaluation results and analysis: [`report.pdf`](report.pdf)
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

Reproduces the headline numbers on a subsample. Every command is run from the repo root.

Commands are given for macOS, Ubuntu and Windows. Where only one block is shown, it is
identical on all three.

> **No API key, or hit a rate limit?** `eval/predictions.jsonl` is committed. You can skip
> steps 6 and 7 entirely and run step 8 against it. Google's free tier allows 20 requests
> per day for `gemini-3.6-flash`, which is not enough to regenerate all 188 predictions.
> A local GPU or a paid key is needed for a full end-to-end rerun.

### 0. Prerequisites

Python 3.10 to 3.14, `git`, about 3 GB of free disk for dependencies and models, plus
500 MB for the dataset.

**macOS**

```bash
brew install python@3.12 git
```

Or use the installer from [python.org](https://www.python.org/downloads/). Xcode command
line tools (`xcode-select --install`) are needed if `git` is not already present.

**Ubuntu**

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

`python3-venv` is a separate package on Debian and Ubuntu. Without it, `python3 -m venv`
fails with `ensurepip is not available`.

**Windows**

Install Python from [python.org](https://www.python.org/downloads/) and tick
**"Add python.exe to PATH"** during setup. Install
[Git for Windows](https://git-scm.com/download/win). The commands below assume
**PowerShell**; `cmd.exe` variants are noted where they differ.

### 1. Clone

```bash
git clone <repo-url>
cd hiver_twitter_chat_classifier
```

### 2. Create and activate a virtual environment

**macOS and Ubuntu**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -m venv venv
venv\Scripts\Activate.ps1
```

If PowerShell refuses with an execution policy error, allow it for this session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows (cmd.exe)**

```bat
py -m venv venv
venv\Scripts\activate.bat
```

Your prompt should now start with `(venv)`. Confirm the interpreter is the one inside the
project, not a system-wide Python:

```bash
python -c "import sys; print(sys.executable)"
```

The path printed must contain your project folder. Every remaining step assumes this venv
is active. If you open a new terminal, activate it again.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `hiver_agent` package in editable mode. Without it,
`python -m hiver_agent.pipeline` and the imports in `eval/` will fail with
`ModuleNotFoundError`.

The bulk of the download is PyTorch. On Linux, the default wheel bundles CUDA and is
roughly 800 MB; if you have no NVIDIA GPU, install the CPU-only build first and the rest
afterwards:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

Verify the install:

```bash
python -c "import hiver_agent, torch, sentence_transformers; print('setup ok')"
```

### 4. Configure the LLM

This project uses a **Google AI Studio** key. Google exposes an OpenAI-compatible
endpoint, so the standard `openai` SDK is used, pointed at Google's base URL. No
Google-specific client library is needed.

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), then
create a `.env` file in the repo root.

**macOS and Ubuntu**

```bash
cat > .env << 'EOF'
LLM_PROVIDER=api
LLM_API_KEY=your_google_ai_studio_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.6-flash
BRAND_HANDLE=Uber_Support
EOF
```

**Windows (PowerShell)**

```powershell
@'
LLM_PROVIDER=api
LLM_API_KEY=your_google_ai_studio_key_here
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.6-flash
BRAND_HANDLE=Uber_Support
'@ | Set-Content -Path .env -Encoding utf8
```

The `@'` and `'@` markers must each sit alone at the very start of their line, with no
leading spaces, or PowerShell will not recognise the here-string.

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

`src/hiver_agent/llm_utils.py` loads the model through Hugging Face `transformers`. It
uses float16 on CUDA and float32 on CPU. This avoids API quotas entirely, but downloads
roughly 3 GB of weights on first run and is 5 to 10 times slower on CPU.
</details>

### 5. Get the data

`twcs.csv` is 493 MB and is not in this repo. Download it from the
[Kaggle dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
and place it at `data/raw/twcs.csv`.

**macOS and Ubuntu**

```bash
mkdir -p data/raw
python scripts/list_brands.py data/raw/twcs.csv | head -20
```

**Windows (PowerShell)**

```powershell
New-Item -ItemType Directory -Force -Path data\raw
python scripts\list_brands.py data\raw\twcs.csv | Select-Object -First 20
```

`Uber_Support` should appear in the list.

### 6. Build the pairs and the retrieval index

```bash
python -m hiver_agent.data_prep --brand Uber_Support --sample 30000 --out data/processed/pairs.jsonl
python -m hiver_agent.retrieval build --pairs data/processed/pairs.jsonl --index data/processed/index.pkl
```

These work unchanged on Windows; forward slashes in the argument values are handled by
Python's `pathlib`.

`--sample 30000` subsamples the raw CSV before pairing. The subsample keeps all rows
belonging to the target brand, so a brand with lower volume does not vanish.

`retrieval.py` picks `Qwen/Qwen3-Embedding-0.6B` when a GPU is visible and falls back to
`all-MiniLM-L6-v2` on CPU automatically.

### 7. Run the agent over the golden set

This step makes one API call per example. See the rate-limit note at the top before
starting.

```bash
python -m hiver_agent.pipeline --golden eval/golden_set.jsonl --index data/processed/index.pkl --out eval/predictions.jsonl --no-rerank
```

Written on one line so it works in every shell. On macOS and Ubuntu you can split it
across lines with a trailing `\`; in PowerShell use a trailing backtick `` ` `` instead.

Drop `--no-rerank` to enable the Qwen3 cross-encoder reranker. It improves retrieval
ordering but is slow on CPU.

The run is resumable. If it stops partway — including when it hits a quota limit — rerun
the same command and it continues from where it left off.

Each row's own pair is excluded from its retrieval results. Golden-set messages are also
in the index, so without that exclusion the top hit is the message itself at similarity
1.0 and the generator sees the ground-truth reply it is meant to produce.
`--allow-self-retrieval` disables the exclusion, but only for deliberately measuring that
leak.

### 8. Evaluate

```bash
python eval/run_eval.py --golden eval/golden_set.jsonl --predictions eval/predictions.jsonl --out eval/report.json
```

Prints and writes: intent accuracy and macro-F1 for the system and both baselines,
escalation accuracy and Cohen's kappa, LLM-judge reply scores, and judge-vs-human
agreement measured against the 30 replies hand-scored in `eval/human_judge_check.jsonl`.

To print every number cited in the report straight from the artifacts:

```bash
python scripts/report_numbers.py
```

### Rebuilding the golden set from scratch (optional)

The labelled set is already committed. To regenerate the unlabelled sample:

```bash
python scripts/sample_for_labeling.py --pairs data/processed/pairs.jsonl --n 200 --out eval/golden_set_unlabeled.jsonl
```

Labels then have to be filled in by hand. See `eval/golden_set_schema.md`.

### Troubleshooting

**`ModuleNotFoundError: No module named 'hiver_agent'`** — the editable install did not
run, or you are in a different environment than the one you installed into. Reactivate the
venv and rerun `pip install -e .`.

**`ensurepip is not available` on Ubuntu** — install `python3-venv`, then delete the
half-created `venv/` folder and create it again.

**`running scripts is disabled on this system` on Windows** — the execution policy fix in
step 2.

**`RESOURCE_EXHAUSTED` / HTTP 429** — Google free-tier daily quota. Wait for the reset,
switch `LLM_MODEL` to a different Gemini model for a separate quota bucket, or use the
committed `eval/predictions.jsonl` and skip to step 8.

**Editor shows unresolved imports but the code runs** — point your editor at
`venv/bin/python` (`venv\Scripts\python.exe` on Windows) rather than a system Python.

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
  predictions.jsonl          generated by pipeline.py
  report.json                generated by run_eval.py
  stale_pre_fix/             superseded predictions and metrics, kept for comparison
scripts/
  list_brands.py             list brand handles by volume
  sample_for_labeling.py     stratified sampler for the golden set
  report_numbers.py          prints every number cited in the report
pyproject.toml               package metadata, makes hiver_agent importable
pyrefly.toml                 type-checker config (search paths, interpreter)
requirements.txt             pinned runtime dependencies
report.pdf
decision_log.md
```
