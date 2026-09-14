"""List candidate brand handles (non-inbound authors) by tweet volume.

Usage: python scripts/list_brands.py data/raw/twcs.csv
"""
import sys
import pandas as pd


def main(csv_path: str) -> None:
    # twcs.csv columns: tweet_id, author_id, inbound, created_at, text,
    # response_tweet_id, in_response_to_tweet_id
    df = pd.read_csv(csv_path, usecols=["author_id", "inbound"])
    brands = df[df["inbound"] == False]  # noqa: E712 (kaggle stores literal bools)
    counts = brands["author_id"].value_counts()
    print(f"{'brand_handle':<25} tweet_count")
    for handle, count in counts.head(40).items():
        print(f"{handle:<25} {count}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/list_brands.py <path_to_twcs.csv>")
        sys.exit(1)
    main(sys.argv[1])
