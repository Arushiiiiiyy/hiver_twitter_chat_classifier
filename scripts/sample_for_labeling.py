"""Stratified sample of pairs for hand-labeling: k-means over embeddings so the sample
spans different message "shapes" rather than just the most common ones, plus a slice
of outliers (low density regions) deliberately included as edge cases.

Outputs an *unlabeled* JSONL with empty gold_* fields for you to fill in by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outlier-fraction", type=float, default=0.15,
                     help="fraction of the sample reserved for low-density outliers")
    args = ap.parse_args()

    pairs = [json.loads(line) for line in Path(args.pairs).open()]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [p["customer_text"] for p in pairs]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    n_outliers = int(args.n * args.outlier_fraction)
    n_clustered = args.n - n_outliers
    n_clusters = min(20, max(2, n_clustered // 5))

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(embeddings)
    per_cluster = max(1, n_clustered // n_clusters)

    chosen_idx: set[int] = set()
    rng = np.random.default_rng(42)
    for c in range(n_clusters):
        idxs = np.where(km.labels_ == c)[0]
        if len(idxs) == 0:
            continue
        pick = rng.choice(idxs, size=min(per_cluster, len(idxs)), replace=False)
        chosen_idx.update(pick.tolist())

    # Outliers: points farthest from their assigned cluster centroid.
    dists = np.linalg.norm(embeddings - km.cluster_centers_[km.labels_], axis=1)
    remaining = [i for i in range(len(pairs)) if i not in chosen_idx]
    remaining_sorted = sorted(remaining, key=lambda i: -dists[i])
    chosen_idx.update(remaining_sorted[:n_outliers])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for i in sorted(chosen_idx)[: args.n]:
            p = pairs[i]
            f.write(json.dumps({
                "pair_id": p["pair_id"],
                "customer_text": p["customer_text"],
                "brand_reply": p["brand_reply"],
                "gold_intent": "",
                "gold_escalate": None,
                "gold_escalate_reason": "",
                "notes": "",
            }) + "\n")

    print(f"Sampled {min(len(chosen_idx), args.n)} examples -> {out_path}")
    print("Now open this file and fill in the gold_* fields by hand. "
          "See eval/golden_set_schema.md for guidance.")


if __name__ == "__main__":
    main()
