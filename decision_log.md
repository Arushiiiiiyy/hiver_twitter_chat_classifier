# Decision log

The non-obvious calls made while building this, and the reasoning behind each. Several
are judgment calls rather than settled facts, and a few were made only after something
broke.

**1. Picked Uber_Support over the higher-volume brands.**
AmazonHelp has three times the data. I went with Uber anyway because it has a
safety-critical tail that Amazon does not, which makes the escalation decision genuinely
interesting rather than a confidence threshold with extra steps.

**2. Threads truncated to the first brand reply.**
The raw data links tweets by `in_response_to_tweet_id`, so full threads are
reconstructable. I only keep the first reply. Most Uber threads immediately redirect to
DM, so later turns are mostly logistics, not resolution. This limitation shows up
directly in failure mode 5.

**3. Stripped handles and URLs, kept case and punctuation.**
Lowercasing everything would have thrown away signal. "FIVE WHOLE POUNDS" and "why??" carry
urgency that the classifier can use.

**4. Subsample biased toward the target brand instead of uniform random.**
A uniform 30k-row sample of a 3M-row file yields almost no Uber conversations. The
sampler keeps every Uber row and randomly fills the remainder.

**5. Dropped langdetect for a character-script check.**
langdetect was the obvious choice and it failed on this data. It classified "my order
never arrived" as Danish, and non-deterministically, giving different answers run to run
on short text. A non-ASCII character ratio catches what actually appears here, Japanese
and Hindi, and gives the same answer every time. It will not catch Spanish, which I found
in the data and am not handling.

**6. Agent signature stripping is dead code for this brand.**
I wrote `AGENT_SIGNATURE_RE` after finding that 91% of AmazonHelp replies end in a two
letter agent initial like "^SJ". Uber does not do this at all. I kept the code because it
costs nothing and matters if anyone reruns this on another brand, but it does nothing
here and I would rather say so than let it look load-bearing.

**7. Added safety_incident after reading the data, not before.**
Roughly 2% of Uber messages, 1,055 of 55,655, mention collisions, harassment or police.
Uber's own replies to those look visibly different from its replies to complaints. That
is a category, not a variant of "angry customer".

**8. safety_incident gets no draft at all, not just no auto-send.**
It is in both `HIGH_RISK_INTENTS` and `NEVER_AUTO_DRAFT_INTENTS`. The pipeline skips
generation entirely. Drafting a reply to an assault report and putting it in front of an
agent to approve is still the system composing that reply, and that is not a call it
should make unsupervised.

**9. A hard keyword rule sits in front of the LLM for emergencies.**
If a message contains "police", "911", "ambulance", "assault" or "emergency" it routes to
`safety_incident` without a model call. A model that returns something softer on those
words is a worse failure than a few false escalations.

**10. Escalation is a rule table, not a learned model.**
Three signals: intent risk, classifier confidence, retrieval grounding strength. I can
explain any individual decision line by line. A learned escalation model would probably
score slightly better and would be much harder to defend when it gets one wrong.

**11. The escalation thresholds, 0.75 confidence and 0.55 similarity, are guesses.**
They were never tuned. The golden set splits 94/94 on escalate, so there was a clean
opportunity to tune them and I did not take it. Reported as-is.

**12. Removed feedback_negative after it broke the evaluation.**
I had split "venting with no demand" from "complaint with a demand" as a separate intent.
The problem was that I added it after labelling the golden set, so the gold labels never
use it while the system predicted it 55 times. That mismatch alone accounted for 44 of
138 errors and dragged accuracy from 0.40 down to 0.27. The taxonomy has to match what
the golden set was labelled against, so the split is gone. It is a real distinction and
worth revisiting, but only alongside a relabelling pass.

**13. Retrieval excludes the query's own pair, added after finding a leak.**
Golden-set messages are in the retrieval index. The first version retrieved them, so on
94% of examples the top hit was the message itself at similarity 1.0 and the generator
was shown the exact reply it was supposed to write. Reply quality and the grounding
signal were both meaningless. `exclude_pair_id` now drops the row's own pair before
ranking. The `--allow-self-retrieval` flag exists only to measure the leak deliberately.

**14. Qwen3-Embedding needs asymmetric encoding.**
Queries get the model's built-in "query" prompt, indexed documents get none. This is easy
to get backwards and doing so measurably hurts retrieval.

**15. Both retrieval models are 0.6B on purpose.**
The 4B and 8B variants exist. 0.6B keeps the whole thing inside a free Colab T4 at about
2.5 GB, which matters more for reproducibility than a marginal retrieval gain.

**16. Generation goes through the API even though a GPU was available.**
A self-hosted 7B generator needs a serving layer to be usable. That is setup complexity
this assignment is not asking about, and it works against the 15-minute reproduction
requirement.

**17. In-memory cosine similarity, no vector database.**
At a few thousand pairs a FAISS index or a hosted vector DB adds a dependency and buys
nothing. This would need revisiting well before production scale.

**18. Golden set stratified by k-means with a deliberate outlier slice.**
15% of the sample is drawn from the points furthest from their cluster centroid. Those are
the one-word messages and broken fragments. A golden set of only clean examples would have
made every number look better and taught me nothing.

**19. The LLM judge scores four named dimensions, not one holistic number.**
Relevance, correctness, tone, completeness. When the judge and I disagreed I wanted to see
which dimension caused it, and it was consistently correctness on vague replies.

**20. The judge prompt is written differently from the generation prompt.**
Same model on both sides invites self-preference bias. Different phrasing and an explicit
rubric reduce it. The judge-human check exists because this mitigation is partial, and it
did show the judge running lenient on non-committal replies.

**21. Banking77 was not used.**
It is offered as optional and only for intent work. Its 77 banking intents do not map onto
rideshare traffic, and the assignment asks for intents defined from the data. Borrowing a
banking taxonomy would have worked against that.
