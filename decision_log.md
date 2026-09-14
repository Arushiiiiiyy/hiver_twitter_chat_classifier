# Decision log

Non-obvious decisions made in this codebase and why. Review each one — several are
genuine judgment calls you should be ready to defend or change, not settled facts.

1. **Multi-reply threads truncated to first reply** (`data_prep.py`) — simplifies
   pairing at the cost of losing later back-and-forth; flagged in report §4.
2. **Cleaning strips @handles/URLs but keeps punctuation/case** — tone signal (e.g.
   ALL CAPS, "!!!") is useful for intent/urgency, so we don't lowercase everything.
3. **Subsample biased toward the target brand, not uniform random** (`data_prep.py
   --sample`) — a brand with low tweet volume would nearly disappear under uniform
   subsampling of a 3M-row dataset.
4. **Intent taxonomy is a draft, not final** (`intents.py`) — must be adjusted after
   looking at real data for your chosen brand; ship the taxonomy you actually used, not
   this generic starting point, and note what changed.
5. **Two baselines chosen: keyword match (trivial) and TF-IDF+LogReg (simple)** — the
   assignment explicitly wants two baselines of different sophistication, not two
   variants of the same approach.
6. **Escalation policy is rule-based, not learned** — auditability and live-defensibility
   mattered more than squeezing out marginal accuracy on a decision with real cost.
7. **Escalation thresholds (confidence 0.75, similarity 0.55) are unvalidated guesses**
   — tune these against your golden set's actual escalate/no-escalate base rate before
   reporting a final number.
8. **Retrieval uses local sentence-transformers, not an LLM API, for embeddings** — no
   per-call cost when re-indexing during iteration; only generation/judging hits the
   paid API.
9. **In-memory cosine similarity instead of a vector DB** — fine at the dataset sizes
   this assignment expects; would need revisiting at real production scale.
10. **Golden set is stratified via k-means + explicit outlier slice, not random** — a
    purely random sample of typical support tweets underrepresents the edge cases that
    actually reveal system weaknesses.
11. **LLM judge scores on 4 named rubric dimensions, not one holistic score** — makes
    disagreements with the human check diagnosable (e.g. judge overrates tone but
    underrates correctness) instead of one opaque number.
12. **Judge prompt is deliberately written differently from the generation prompt** —
    to reduce self-preference bias (a judge grading against its own style tends to
    reward verbosity/hedging it was itself prompted to produce).
13. **Trailing agent-signature tags stripped from brand replies** (`data_prep.py`,
    `AGENT_SIGNATURE_RE`) — on the real dataset, ~91% of AmazonHelp replies end with a
    2-3 letter agent initial like "^SJ". Left in, both the retrieval embeddings and the
    reply generator would treat these as meaningful content; they're pure noise.
14. **Language filter uses character-script ratio, not langdetect** (`is_probably_english`)
    — langdetect was tried first and rejected: it misclassified short, unambiguous
    English phrases (e.g. "my order never arrived" as Danish) non-deterministically
    run-to-run. A non-ASCII character ratio catches the dominant real pattern in this
    dataset (Japanese, Hindi — ~7% of AmazonHelp customer messages) deterministically.
    Known gap: it won't catch other Latin-script languages (spot-checked: some Spanish
    slips through) — worth re-checking per brand and calling out in the report.
15. **Added `safety_incident` as a dedicated intent, picked up while inspecting real
    Uber_Support data** (`intents.py`, `escalation.py`, `pipeline.py`) — ~1.9% of real
    customer messages (1,055/55,655) reference accidents, harassment, or police
    involvement, with a visibly distinct brand response pattern ("we take this very
    seriously... DM immediately") vs. generic complaints. Added to `HIGH_RISK_INTENTS`
    (always escalate) AND `NEVER_AUTO_DRAFT_INTENTS` (pipeline skips reply generation
    entirely, not just auto-send) — drafting any AI reply to a sexual-assault report is
    a product/ethics call this system shouldn't make unsupervised, independent of
    whether the draft would've been auto-sent.
16. **Known trivial-baseline ambiguity: "crash"** (`intents.py`) — the keyword baseline
    can't distinguish "the app crashed" from "my driver crashed," both hit
    `safety_incident`. Left as-is deliberately: the trivial baseline is supposed to be
    weak, and this exact ambiguity is a good example for the report's "why the real
    system beats the baselines" argument, not something to over-engineer away.
17. **Switched retrieval to Qwen3-Embedding-0.6B + added Qwen3-Reranker-0.6B as a
    second stage** (`retrieval.py`, `reranker.py`, `reply_generator.py`) — with
    Colab/GPU access confirmed available, the marginal cost of a stronger embedding
    model plus a reranking stage is low, and it gives the report a genuine third
    comparison point ("retrieval only" vs. "retrieval + rerank"), not just a bigger
    model for its own sake. Both are 0.6B variants specifically — deliberately sized to
    fit comfortably on a free Colab T4 (~2.5GB VRAM combined) rather than requiring a
    dedicated GPU cluster; a CPU-only fallback (`all-MiniLM-L6-v2`, no reranker) is
    still supported via `--model` for anyone without GPU access at all.
18. **Qwen3-Embedding requires asymmetric query/document encoding** (`retrieval.py`)
    — the query (new customer message) is encoded with the model's built-in "query"
    instruction prompt; historical documents being indexed are encoded plain. Getting
    this backwards is a documented, easy-to-miss mistake for this model family that
    measurably hurts retrieval quality — not a stylistic choice.
19. **Generator stays API-based, not self-hosted, despite having GPU access** — the
    embedding/reranker models are small enough to load directly in a script (fits the
    <15-minute reproduction requirement); a competent local LLM generator (7B+) would
    need a serving layer (vLLM/similar), which adds setup complexity disproportionate
    to what the assignment is actually grading (evaluation rigor, not self-hosting).
20. **[ADD YOUR OWN]**
