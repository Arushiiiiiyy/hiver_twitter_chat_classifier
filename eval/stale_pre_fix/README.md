# Stale results, kept for reference only

These were generated before two fixes landed:

1. Retrieval self-exclusion. 176 of 188 predictions here retrieved the query's own pair
   at similarity 1.0, so the generator saw the ground-truth reply.
2. Removal of `feedback_negative`, which the golden set was never labelled against.

Do not cite these numbers. Regenerate with `src/pipeline.py` then `eval/run_eval.py`.
