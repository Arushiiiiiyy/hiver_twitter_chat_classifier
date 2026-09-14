# Golden set schema

One JSON object per line (JSONL). Fields marked (you fill in) require manual labeling.

| field              | type   | description |
|--------------------|--------|-------------|
| `pair_id`          | str    | from data_prep.py output, for traceability |
| `customer_text`    | str    | the cleaned customer message |
| `brand_reply`      | str    | the brand's actual historical reply (for reference only — not ground truth for generation quality, since a real agent's actual reply is one valid reply, not the only valid one) |
| `gold_intent`      | str (you fill in) | one of the taxonomy in `src/intents.py`, or a new label if none fit — track any new labels you add |
| `gold_escalate`    | bool (you fill in) | would a competent human agent escalate this, based on the message alone? |
| `gold_escalate_reason` | str (you fill in) | one short phrase — used to sanity-check the system's stated reasons, not for exact match |
| `notes`            | str, optional | anything ambiguous about this example — these are gold for your failure-analysis section |

## Labeling guidance

- Label independently before looking at what the brand actually did — the brand's own
  reply is sometimes itself a bad resolution (that's realistic and worth noting in the
  report if you see it).
- If a message is multi-intent, pick the primary one and note the secondary in `notes`.
- Deliberately include some ambiguous/edge cases (sarcasm, non-English fragments,
  incomplete threads) — a golden set of only clean examples will make your system look
  better than it is.
