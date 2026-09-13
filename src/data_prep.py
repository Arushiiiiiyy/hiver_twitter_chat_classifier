import argparse
import json
import re
from pathlib import Path

import pandas as pd

HANDLE_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
WS_RE = re.compile(r"\s+")
AGENT_SIGNATURE_RE = re.compile(r"\s*\^[A-Za-z]{1,3}\s*$")  # trailing "^SJ" style sign-off


def clean_text(text: str, is_reply: bool = False) -> str:
    text = HANDLE_RE.sub("", text)
    text = URL_RE.sub("", text)
    if is_reply:
        text = AGENT_SIGNATURE_RE.sub("", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def is_probably_english(text: str, non_ascii_threshold: float = 0.3) -> bool:
    """Here, we are stripping away comments in different languages by detecting the script, ie any 
    script which is not english. This might pass for languages with english scripts like Spanish, but we 
    will look into it later"""
    if not text:
        return True
    non_ascii = sum(1 for c in text if ord(c) > 0x2FF)  
    return (non_ascii / len(text)) <= non_ascii_threshold


def build_pairs(df: pd.DataFrame, brand: str, filter_english: bool = True) -> list[dict]:
    """This function helps me build pairs of customer tweet and company tweet. Recursive tweets not
     handled as of now"""
    df = df.set_index("tweet_id", drop=False)
    brand_tweets = df[(df["author_id"] == brand) & (df["inbound"] == False)]  # noqa: E712

    pairs = []
    dropped_non_english = 0
    for _, brand_row in brand_tweets.iterrows():
        parent_id = brand_row["in_response_to_tweet_id"]
        if pd.isna(parent_id):
            continue
        parent_id = int(parent_id)
        if parent_id not in df.index:
            continue
        customer_row = df.loc[parent_id]
        if isinstance(customer_row, pd.DataFrame):  # duplicate ids, take first
            customer_row = customer_row.iloc[0]
        if not customer_row.get("inbound", False):
            continue  # parent wasn't actually a customer message

        customer_text = clean_text(str(customer_row["text"]))
        brand_text = clean_text(str(brand_row["text"]), is_reply=True)
        if len(customer_text) < 3 or len(brand_text) < 3:
            continue

        if filter_english and not is_probably_english(customer_text):
            dropped_non_english += 1
            continue

        pairs.append(
            {
                "pair_id": f"{customer_row['tweet_id']}_{brand_row['tweet_id']}",
                "customer_tweet_id": int(customer_row["tweet_id"]),
                "brand_tweet_id": int(brand_row["tweet_id"]),
                "customer_text": customer_text,
                "brand_reply": brand_text,
                "created_at": customer_row.get("created_at"),
            }
        )
    if dropped_non_english:
        print(f"[data_prep] dropped {dropped_non_english} non-English customer messages "
              f"(pass --keep-non-english to disable)")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/twcs.csv")
    ap.add_argument("--brand", required=True, help="author_id of the brand, e.g. AmazonHelp")
    ap.add_argument("--sample", type=int, default=None,
                     help="Row-count subsample of the raw CSV before pairing (perf)")
    ap.add_argument("--keep-non-english", action="store_true",
                     help="Disable the English-only filter (kept off by default; see decision_log.md)")
    ap.add_argument("--out", default="data/processed/pairs.jsonl")
    args = ap.parse_args()

    usecols = ["tweet_id", "author_id", "inbound", "created_at", "text",
               "response_tweet_id", "in_response_to_tweet_id"]
    df = pd.read_csv(args.raw, usecols=usecols)
    if args.sample:
        # Bias the subsample toward rows involving the target brand so we don't
        # throw away most of its (rare) conversations in a random subsample.
        brand_mask = df["author_id"] == args.brand
        brand_rows = df[brand_mask]
        other_rows = df[~brand_mask].sample(
            n=min(args.sample, (~brand_mask).sum()), random_state=42
        )
        df = pd.concat([brand_rows, other_rows]).drop_duplicates("tweet_id")

    pairs = build_pairs(df, args.brand, filter_english=not args.keep_non_english)
    if not pairs:
        raise SystemExit(
            f"No pairs found for brand '{args.brand}'. Check the handle with "
            f"scripts/list_brands.py, or increase --sample."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(pairs)} (customer, brand_reply) pairs to {out_path}")


if __name__ == "__main__":
    main()
