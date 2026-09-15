# Golden evaluation set

`eval/golden_set.jsonl` holds 188 hand-labelled Uber_Support messages. This note covers
how they were sampled, how they were labelled, and what the resulting set looks like.

## How they were sampled

Random sampling of support tweets over-represents the common cases (fare disputes, "where
is my driver") and under-represents the edge cases that actually break a classifier. So
sampling was stratified instead, via `scripts/sample_for_labeling.py`:

1. Every candidate pair from `data/processed/pairs.jsonl` was embedded.
2. K-means clustered those embeddings, and examples were drawn evenly across clusters so
   each distinct "shape" of message is represented.
3. A further 15% of the sample was reserved for outliers, the points furthest from their
   cluster centroid. These are the very short messages, the sarcastic ones and the
   fragments of broken threads.

The sampler wrote `eval/golden_set_unlabeled.jsonl` with the label fields blank. Target
was 200; 188 survived labelling after dropping rows that were unreadable or had no
customer content at all.

## How they were labelled

Labelling was done by hand, one row at a time, before looking at the system's output on
any of them. Two rules were applied throughout:

- **Label from the customer message alone.** The brand's historical reply is present in
  the file for reference, but it was not treated as ground truth. Uber's own agents
  sometimes gave a poor or evasive answer, and copying that would bake their mistakes
  into the evaluation.
- **Escalation is judged on what a competent human agent would do**, not on what the
  system happens to do. A message can be correctly classified and still need a human.

Multi-intent messages were labelled with the primary intent, with the secondary one noted
in `notes` where it was worth recording.

## Fields

| field | type | description |
|---|---|---|
| `pair_id` | str | traceability back to `data/processed/pairs.jsonl` |
| `customer_text` | str | the cleaned customer message |
| `brand_reply` | str | Uber's actual historical reply, reference only |
| `gold_intent` | str | hand-assigned, one of the taxonomy in `src/intents.py` |
| `gold_escalate` | bool | would a competent human agent escalate this |
| `gold_escalate_reason` | str | short phrase, present on 178 of 188 rows |
| `notes` | str | ambiguity worth recording |

## What the set looks like

188 examples. Median message length 99 characters, range 4 to 278.

| intent | count |
|---|---:|
| complaint_escalation | 56 |
| billing_refund | 33 |
| general_inquiry | 31 |
| product_defect | 16 |
| other | 14 |
| safety_incident | 11 |
| account_access | 11 |
| order_delivery | 11 |
| feedback_praise | 5 |

Escalation is close to balanced: 94 of 188 labelled escalate, 94 not. That balance is why
the evaluation reports Cohen's kappa alongside raw accuracy.

`service_outage` exists in the taxonomy but drew zero labels in this sample. It is kept
because it is a real Uber failure mode, but any metric for it in the results is computed
over an empty support set and means nothing.
